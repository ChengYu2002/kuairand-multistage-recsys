"""曝光负采样（plan §16.3）。

拿「这个用户**确实被推荐过、但从未正向过**」的视频当负样本。它与 random / in-batch
的本质差别在于负样本的**条件分布**：

    random    从候选库均匀抽 —— 与这个用户无关
    in-batch  按正样本频率抽 —— 与这个用户无关，偏热门
    exposure  从这个用户自己的曝光里抽 —— 已经通过了线上推荐系统的筛选

曝光负样本是"难负样本"：它们本来就是线上模型认为这个用户可能喜欢的东西。
模型不能再靠"这个视频和这个用户八竿子打不着"蒙对，必须学更细的区分。
代价是分布偏移 —— 它学的是「在线上已筛过的集合里排序」，而评估要它在 19.4 万
候选库上**召回**。这个错配正是本研究要量出来的东西，不是实现缺陷。

## 口径：与 plan 字面不同，这里的负样本是「未正向」而非「未点击」

plan §16.3 的原文是「标准日志中未点击曝光」。这里用的是

    positive        = is_click OR long_view
    exposure 负样本 = 曝光过，且该 (user, item) 在整个 train 段**从未**成为 positive

理由：召回任务的正样本定义就是 `eval.positive_signal: [is_click, long_view]`
（configs/retrieval.yaml）。若负样本池只排除 is_click，那么一个"没点但长播"的视频
会同时是训练正样本、又是该用户的负样本 —— 与标签直接矛盾。两处口径必须同源，
否则 ΔRecall 分不清来自采样策略还是来自两套正样本定义。

## (user, item) 去重，且「任意一次正向」就整对排除

同一个视频可能被同一用户曝光几十次。不去重的话重复曝光的视频获得成比例更高的
采样概率，exposure 会悄悄变成"该用户的热度采样"，混进一个本研究没打算研究的变量。

排除必须**先按 (user, item) 聚合再判定**：逐行判定的话，"这次没点"的那条曝光
会把一个该用户点过的视频放进负样本池。

## 为什么可以读整个 train 段的标签

负样本池是**训练样本的构造方式**，不是喂给模型的输入特征，所以它不受 T-1 快照
约束 —— 训练算法本来就看得见 train 段的标签。但它也**不是**严格逐曝光的
"截至此刻未正向"，README 必须写明这是一个**静态训练负样本池**。
valid / test 一个字都不读。

## 必须限制在候选库区间内

与 random 同一个理由（random_negative.py 的"为什么从候选库抽"）：区间外的 item
在评估时永远不会被打分。更要紧的是 configs/retrieval.yaml 里
`train_target_scope: catalog` 的那条注释 —— 正负样本若不在同一个空间，
"不在候选库"就成了完美预测正样本的捷径，而候选库成员身份恰好由 index 区间编码。

## 一处必须披露的策略间不对称

因为池子排除了该用户的全部正向 item，exposure 的 accidental hit
（负样本恰好等于正样本）概率为 0；random 每个负样本约 1/194310；in-batch 至少
1/batch_size（抽到自己那一行），且热门 item 会让同批出现重复目标，真实屏蔽率
更高 —— 这个数**尚未实测**，不要在报告里写一个推导值当实测值。
损失函数把 accidental hit 屏蔽成 -inf，于是**有效负样本数**在三种策略间并不
完全相同。这是"排除矛盾负样本"的必然代价，README 中与结论一并披露。

## 接口

与 RandomNegative / InBatchNegative 完全一致：

    sample(batch, n_neg, generator) -> LongTensor (B, n_neg)

额外依赖 `batch["user_id"]` —— 不知道是谁，就无从取"他自己的曝光"。

用法：
    pool = build_exposure_pool(proc / "samples_train_main.parquet", n_users, lo, hi)
    sampler = ExposureNegative(lo, hi, pool)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch

POSITIVE_SIGNALS = ("is_click", "long_view")


class ExposurePool:
    """每个用户的「曝光过但从未正向」候选库内 item 集合，CSR 布局。

    为什么是 CSR 而不是 {user: array} 的字典：采样要在 torch 里一次性对整个 batch
    做 gather，字典会退化成逐样本 Python 循环（1024 条 x 2 万步 = 2 千万次）。

    内存（1K 实测由 verify 脚本记录）：indptr 8 KB + items 约 4 MB，可忽略。
    """

    __slots__ = ("hi", "indptr", "items", "lo", "n_pairs_in_range", "n_users")

    def __init__(
        self,
        indptr: np.ndarray,
        items: np.ndarray,
        n_users: int,
        lo: int,
        hi: int,
        n_pairs_in_range: int,
    ) -> None:
        self.indptr = np.ascontiguousarray(indptr, np.int64)
        self.items = np.ascontiguousarray(items, np.int64)
        self.n_users, self.lo, self.hi = int(n_users), int(lo), int(hi)
        self.n_pairs_in_range = int(n_pairs_in_range)
        if self.indptr.shape != (self.n_users + 1,):
            raise ValueError(
                f"indptr 长度应为 n_users+1={self.n_users + 1}，"
                f"实际 {self.indptr.shape}"
            )
        if self.indptr[0] != 0 or self.indptr[-1] != self.items.size:
            raise ValueError("indptr 两端与 items 长度不一致，CSR 结构已损坏")

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.indptr)

    def pool_of(self, user_id: int) -> np.ndarray:
        return self.items[self.indptr[user_id] : self.indptr[user_id + 1]]

    def coverage(self) -> dict:
        s = self.sizes
        return {
            "n_users": self.n_users,
            "pool_pairs": int(self.items.size),
            "exposed_pairs_in_range": self.n_pairs_in_range,
            "users_empty": int((s == 0).sum()),
            "users_nonempty_ratio": float((s > 0).mean()) if self.n_users else 0.0,
            "pool_size_min": int(s.min()) if self.n_users else 0,
            "pool_size_p50": int(np.median(s)) if self.n_users else 0,
            "pool_size_max": int(s.max()) if self.n_users else 0,
        }


def build_exposure_pool(
    samples: Path | str | pl.DataFrame,
    n_users: int,
    lo: int,
    hi: int,
    positive_signals: tuple[str, ...] = POSITIVE_SIGNALS,
) -> ExposurePool:
    """从 train 段曝光表构造 CSR 曝光负样本池。

    只读 4 列（user_id / target_idx / 两个标签）—— samples 表里 hist_idx 是逐样本的
    列表列，整表读进来装载峰值 3.77 GB，而这里只要约 70 MB。
    """
    cols = ["user_id", "target_idx", *positive_signals]
    lf = (
        pl.scan_parquet(samples) if isinstance(samples, (str, Path)) else samples.lazy()
    ).select(cols)
    # ① 先砍到候选库区间（区间外的 item 不参与打分，见模块 docstring）
    lf = lf.filter((pl.col("target_idx") >= lo) & (pl.col("target_idx") <= hi))
    pos = pl.col(positive_signals[0]) > 0
    for c in positive_signals[1:]:
        pos = pos | (pl.col(c) > 0)
    # ② 按 (user, item) 聚合：去重 + 「任意一次正向则整对排除」一次做完
    agg = (
        lf.select("user_id", "target_idx", pos.alias("pos"))
        .group_by("user_id", "target_idx")
        .agg(pl.col("pos").any().alias("pos"))
    )
    tot = int(agg.select(pl.len()).collect().item())
    neg = (
        agg.filter(~pl.col("pos"))
        .select("user_id", "target_idx")
        # 必须按 user_id 排序：CSR 的 items 段落顺序由它决定。
        # 二级键 target_idx 只为让池子内容可复现、可逐项比对。
        .sort("user_id", "target_idx")
        .collect()
    )

    u = neg.get_column("user_id").to_numpy().astype(np.int64)
    items = neg.get_column("target_idx").to_numpy().astype(np.int64)
    if u.size and (u.min() < 0 or u.max() >= n_users):
        raise ValueError(
            f"user_id 超出 [0, {n_users}) ：实测 [{u.min()}, {u.max()}]。"
            "池子按 user_id 直接下标索引，越界会静默取到别人的曝光。"
        )
    indptr = np.zeros(n_users + 1, np.int64)
    np.cumsum(np.bincount(u, minlength=n_users), out=indptr[1:])
    return ExposurePool(indptr, items, n_users, lo, hi, tot)


class ExposureNegative:
    name = "exposure"
    # 从「这个用户自己看过但没正向过」的视频里抽负样本

    def __init__(self, catalog_lo: int, catalog_hi: int, pool: ExposurePool) -> None:
        if catalog_hi < catalog_lo:
            raise ValueError(f"候选库区间非法：[{catalog_lo}, {catalog_hi}]")
        self.lo, self.hi = int(catalog_lo), int(catalog_hi)
        if (pool.lo, pool.hi) != (self.lo, self.hi):
            raise ValueError(
                f"池子按 [{pool.lo}, {pool.hi}] 构造，采样器要 [{self.lo}, {self.hi}]。"
                "两者不一致意味着池子里可能有打不到分的 item。"
            )
        self.pool = pool
        self._indptr = torch.from_numpy(pool.indptr)
        self._items = torch.from_numpy(pool.items)
        self.n_rows = 0  # 累计采过的样本行数
        self.n_fallback_rows = 0  # 其中空池子、退回 random 的行数

    def sample(
        self, batch: dict, n_neg: int, generator: torch.Generator
    ) -> torch.Tensor:
        if "user_id" not in batch:
            raise KeyError(
                "batch 里没有 user_id —— exposure 负采样要定位「这个用户曝光过什么」。"
                "RetrievalData.batch() 必须返回 user_id。"
            )
        u = batch["user_id"].long()
        b = u.shape[0]
        start, cnt = self._indptr[u], self._indptr[u + 1] - self._indptr[u]

        # 空池子的行退回 random。两条路都先算出来再 where 选 —— 按需只抽一部分会让
        # 随机数流依赖"本批有几个空池子用户"，同 seed 下仍可复现，但调试时无从对齐。
        fb = torch.randint(
            self.lo, self.hi + 1, (b, n_neg), generator=generator, dtype=torch.long
        )
        if self._items.numel() == 0:
            neg = fb
        else:
            r = torch.rand((b, n_neg), generator=generator)
            # off = floor(r * cnt)，均匀落在 [0, cnt-1]。
            # 这里依赖一个性质：float32 下 floor(r*cnt) < cnt 恒成立。证明：
            # r 的最大值是 1-2^-24，真值 cnt-cnt*2^-24 与 cnt 的距离 cnt*2^-24
            # 大于该处的半个 ULP（cnt >= 2^e 时半 ULP = 2^(e-24) <= cnt*2^-24），
            # 故必定向下舍入。verify 脚本对池子里实际出现的每个 cnt 直接断言这一条，
            # 而不是靠这段推导 —— 推导写错过一次。
            off = (r * cnt.unsqueeze(1).to(r.dtype)).long()
            # 这个 clamp 是承重的，但管的是**空池子**：cnt=0 的行 start 可能正好等于
            # items 的长度（尾部用户），下标会越出数组末尾而抛异常。算出来的值属于
            # 下一个用户，但它随后被 torch.where 丢掉，所以只需保证不越界。
            idx = (start.unsqueeze(1) + off).clamp(max=self._items.numel() - 1)
            neg = self._items[idx]

        empty = cnt == 0
        self.n_rows += b
        self.n_fallback_rows += int(empty.sum())
        return torch.where(empty.unsqueeze(1), fb, neg)

    def stats(self) -> dict:
        return {
            "strategy": self.name,
            "sampled_rows": self.n_rows,
            "fallback_rows": self.n_fallback_rows,
            "fallback_ratio": (self.n_fallback_rows / self.n_rows)
            if self.n_rows
            else 0.0,
            **self.pool.coverage(),
        }
