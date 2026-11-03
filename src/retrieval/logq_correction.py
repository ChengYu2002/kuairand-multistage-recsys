"""In-batch + LogQ Correction（plan §16.5）。

## 它修的是什么

In-batch 负采样按**正样本频率**抽负样本：一个视频越热门，就越频繁地成为别人的
负样本，于是被压低的次数也越多。这不是模型学到的东西，是采样过程带进来的偏差。

本项目实测到的后果（§7.2 频次分层，3 seeds）：

    R@100 倍随机猜      tail      mid      warm
    Random              5.7x     15.3x    36.8x
    In-batch           10.6x     12.9x     7.6x   <- warm 层崩在这里

以及物品向量里「与用户无关的那一份得分」与 train_freq 的 Spearman ρ，
从 Random 的 +0.549 掉到 In-batch 的 +0.202 —— 热度分量被削弱而非强化。

## 做法：sampled softmax 的标准校正

    z(u, i) = <u, v_i> / T - log Q(i)

然后照常走同一个 softmax。**候选集合里每个 logit 都要减**，正样本也不例外 ——
给所有 logit 加同一个常数不改 softmax，但 log Q 是**逐 item** 的，
只校正负样本会让正样本相对位移一个与它自身热度相关的量，那是另一个偏差。

直觉：Q 大的 item（热门）减得多 -> 它在分母里的权重变小 -> 每次出现被压得更轻，
恰好抵消「出现次数多」。

## Q 的口径：全局训练正样本频率

    Q(i) = 视频 i 在**可训练正样本行**中出现的次数 / 可训练正样本总行数

「可训练正样本」严格沿用 `two_tower.train_rows` 的定义，一个字不改：

    (is_click OR long_view) 且 target 落在 Protocol A 候选库内

**不能**用的几种口径，每一种都会悄悄算出另一个实验：

    所有曝光次数         —— 那是曝光分布，不是 in-batch 的 proposal 分布
    catalog.train_freq   —— 它数的是曝光，且含非正向行为
    当前 batch 的经验频率 —— 1,024 行里绝大多数视频计数为 0，方差极大；
                             而我们有固定的离线训练集，可以精确算出全局 Q
    valid / test         —— 直接泄露

Q=0 的 item **直接报错，不用 epsilon 掩盖**。in-batch 的负样本全部来自 batch 内的
正样本，正样本按定义也在训练集里，所以进入 loss 的每个 item 都必然 Q>0；
真出现 0 说明 Q 的来源和训练行不是同一套，那是必须炸出来的错，不是边界情况。

## 只作用于训练 loss

推理、建索引、全库 Top-K 一律用**未校正**的点积。校正是为了纠正训练时的采样偏差，
不是模型打分的一部分 —— 线上打分时没有「采样」这回事。

## 它不解决假负样本

accidental hit（负样本恰好等于正样本）仍由 `loss()` 里的 mask 处理。
logQ 只管「热门视频被更频繁抽中」这一个频率偏差，两者正交。

用法：
    table = build_log_q(data.target, rows, lo, hi)
    lq_pos, lq_neg = table.lookup(pos_ids), table.lookup(neg_ids)
"""

from __future__ import annotations

import numpy as np
import torch


class LogQTable:
    """候选库区间上的 log Q(i)。Q=0 的位置存 NaN，查到就报错。"""

    __slots__ = ("hi", "lo", "log_q", "n_rows", "n_support")

    def __init__(
        self, log_q: np.ndarray, lo: int, hi: int, n_rows: int, n_support: int
    ) -> None:
        self.log_q = np.ascontiguousarray(log_q, np.float32)
        self.lo, self.hi = int(lo), int(hi)
        self.n_rows, self.n_support = int(n_rows), int(n_support)
        if self.log_q.shape != (self.hi - self.lo + 1,):
            raise ValueError(
                f"log_q 长度应为 {self.hi - self.lo + 1}，实际 {self.log_q.shape}"
            )

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """词表行号 -> log Q。越界或 Q=0 直接报错。"""
        if int(ids.min()) < self.lo or int(ids.max()) > self.hi:
            raise ValueError(
                f"item id 超出候选库区间 [{self.lo}, {self.hi}]："
                f"实测 [{int(ids.min())}, {int(ids.max())}]。"
                "logQ 的 Q 只在候选库上定义，区间外说明正负样本不同空间。"
            )
        out = torch.from_numpy(self.log_q)[ids - self.lo]
        bad = torch.isnan(out)
        if bool(bad.any()):
            ex = ids[bad].flatten()[:5].tolist()
            raise ValueError(
                f"{int(bad.sum())} 个 item 的 Q=0，无法取对数（例：{ex}）。"
                "in-batch 的负样本全部来自 batch 内正样本，正样本按定义也在训练行里，"
                "所以这不该发生 —— 说明 Q 的来源与训练行不是同一套。"
                "**不要加 epsilon 掩盖**，那会把一个口径错误变成一个静默的偏置。"
            )
        return out

    def stats(self) -> dict:
        q = np.exp(self.log_q[~np.isnan(self.log_q)])
        n = self.hi - self.lo + 1
        return {
            "n_train_rows": self.n_rows,
            "n_catalog": n,
            "n_items_with_support": self.n_support,
            "coverage": self.n_support / n if n else 0.0,
            # 相对均匀分布的倍数 —— 看偏斜有多大
            "q_x_uniform_min": float(q.min() * n) if q.size else None,
            "q_x_uniform_p50": float(np.median(q) * n) if q.size else None,
            "q_x_uniform_max": float(q.max() * n) if q.size else None,
        }


def build_log_q(target: np.ndarray, rows: np.ndarray, lo: int, hi: int) -> LogQTable:
    """由**可训练正样本行**统计 Q 并取对数。

    target  全部样本的 target_idx（RetrievalData.target）
    rows    train_rows() 选出的行号 —— 必须是训练时真正用的那一份，
            否则 Q 描述的是另一个 proposal 分布
    """
    t = np.asarray(target)[np.asarray(rows)]
    if t.size == 0:
        raise ValueError("可训练正样本为 0 行，无法估计 Q")
    if t.min() < lo or t.max() > hi:
        raise ValueError(
            f"训练正样本的 target 超出候选库区间 [{lo}, {hi}]："
            f"实测 [{int(t.min())}, {int(t.max())}]。"
            "说明传进来的 rows 不是 train_target_scope=catalog 口径下的那一份。"
        )
    counts = np.bincount(t - lo, minlength=hi - lo + 1).astype(np.float64)
    q = counts / counts.sum()
    with np.errstate(divide="ignore"):
        log_q = np.where(q > 0, np.log(q), np.nan)
    return LogQTable(log_q, lo, hi, int(t.size), int((counts > 0).sum()))
