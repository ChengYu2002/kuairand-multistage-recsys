"""全库检索：分块打分 + Top-K（plan §16）。

双塔训练完之后有两堆向量 —— 每个查询时刻一个用户向量、每个候选视频一个物品向量，
打分就是点积。问题在于朴素写法 `U @ V.T` 会物化一个 51,746 x 194,310 的矩阵，
**100 亿个数、float32 下 40 GB**，32 GB 的机器上当场就炸。

这里按查询分块：一次算 block 行的分数（默认 1,024 行约 800 MB），取完 Top-K 立刻丢掉，
峰值内存由块大小决定而不是由查询总数决定。

## 为什么是暴力而不是 ANN

19.4 万候选，暴力点积够快（实测 1,024 x 50,000 约 0.074s，外推到全库约 15 秒）。
ANN 是近似算法，会额外引入「召回率损失」这个变量 —— 而本项目要比较的是四种负采样
策略之间的差异，不该让近似误差混进来。文件名沿用 ann_index，内容是精确检索。

## 并列必须显式处理

`np.argpartition` 只保证前 k 个位置装着前 k 大的值，**边界上并列取哪几个是任意的**。
ItemCF 那轮就栽在这里：候选库第 500 名有 17 个并列，导致结果既不可复现、也复现不出
热度基线的冻结值。这里统一规则：**分数相同时取下标较小者**（候选库按热度降序排列，
即偏向热门，与热度基线的 tie-break 一致）。

## Top-K 用 torch 而不是 numpy

profile 结果很反直觉：矩阵乘法只占 3%，argpartition + partition 占 93%。而且这两个
调用在做同一件事 —— 第 k 大的值可以直接从 argpartition 选中的 k 个里取最小值得到，
不必再 partition 一遍。

即便去掉冗余，np.argpartition 仍然慢，且它**返回整行的 int64 排列**（1024 x 194,310
就是 1.5 GB）。torch.topk 只返回 k 个，实测同一子块 12.7 ms vs 214.4 ms，快 17 倍，
内存也更省。torch 本来就是 Week 2 的依赖，因此这里直接用它。

    单个子块 (128 x 194,310) 取 top-500
        np.argpartition + np.partition   388.9 ms
        只用 np.argpartition             214.4 ms
        torch.topk                        12.7 ms

## 输入必须是有限值 —— 这道检查不是形式主义

NaN / Inf 不会让 torch.topk 报错，它会**静默返回一组看起来完全正常的 Top-K**。
实测：把整个用户向量矩阵设成 NaN，返回的是 `[[0,1,2], [0,1,2], ...]`。
而候选库是按热度降序排的，下标 0,1,2 恰好是最热门的三个 —— 也就是说，
**一个训练发散的模型会输出酷似热度基线的结果**，Recall 算得出来、数字也"合理"，
于是人会去怀疑模型结构，而不是怀疑权重已经是 NaN。

因此在检索入口一次性校验输入的有限性。代价是扫一遍两个矩阵（194,310 x 64 约 12M 个数，
毫秒级），换掉一整类极难定位的排查。

## 不支持「排除已看」

协议要求所有召回方法的后处理完全一致，一律不排除用户历史视频（见 README §7.1）。
不提供这个开关是刻意的 —— 多一条用不到的代码路径就多一处可能被误用的地方。

用法（库）：
    idx, sc = topk_scores(user_vecs, item_vecs, k=500)
"""

from __future__ import annotations

import numpy as np
import torch

# 实测 block=256 既更快也更省：8,192 查询下 2.13s / 0.66 GB，而 block=1024 是 2.47s / 2.49 GB。
# 更大的块并不能让矩阵乘法更快（它只占 3% 的时间），却让所有临时量按比例膨胀。
DEFAULT_BLOCK = 256

# 实测倍率：真实峰值稳定在「分数块大小」的 3.4 倍（block=256/512/1024 三档都是 3.4x）。
# 这个数是量出来的、不是推出来的 —— torch.topk 的内部缓冲与 CPU 缓存分配器都不透明，
# 前两版按第一性原理推的公式分别低估了 6 倍和 3 倍。27K 上要靠它定块大小，宁可用实测值。
PEAK_MULTIPLIER = 3.4


def _topk_block(scores: torch.Tensor, k: int) -> np.ndarray:
    """对一块分数逐行取 Top-k 下标，按 (分数降序, 下标升序) 排好。

    torch.topk 只返回 k 个（不像 np.argpartition 要吐出整行排列），因此这里不需要再切子块。
    但它在并列处取谁同样是未定义的，所以仍要自己修正。
    """
    vals, idx = torch.topk(scores, k, dim=1)          # 已按分数降序
    kth = vals[:, -1]

    # 判定必须用 >= ：第 k 大的值前面**最多**只有 k-1 个严格更大的元素，
    # 因此 (scores > kth).sum() < k 对每一行都成立 —— 用它做条件等于没有条件。
    # 真正的判据是「>= 门槛的元素超过 k 个」，即门槛值上存在跨边界的并列组。
    n_ge = (scores >= kth[:, None]).sum(dim=1)
    tie_rows = torch.nonzero(n_ge > k, as_tuple=False).flatten().tolist()

    part = idx.numpy().astype(np.int32)
    vv = vals.numpy()
    if tie_rows:
        sn = scores.numpy()
        for r in tie_rows:
            row = sn[r]
            strict = np.flatnonzero(row > kth[r].item())
            ties = np.flatnonzero(row == kth[r].item())
            # torch.topk 在并列处的选择是任意的；规则要求取下标较小者
            part[r] = np.concatenate([strict, np.sort(ties)[: k - len(strict)]])
            vv[r] = row[part[r]]

    order = np.lexsort((part, -vv))                   # 先分数降序，再下标升序
    return np.take_along_axis(part, order, axis=1).astype(np.int32)


def topk_scores(
    user_vecs: np.ndarray,
    item_vecs: np.ndarray,
    k: int,
    block: int = DEFAULT_BLOCK,
    return_scores: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """全库检索。返回 (n_queries, k) 的候选下标，以及可选的对应分数。

    下标是 item_vecs 的行号；调用方负责把它映射回 video_id。
    """
    u = np.ascontiguousarray(user_vecs, dtype=np.float32)
    v = np.ascontiguousarray(item_vecs, dtype=np.float32)
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError(f"需要二维数组，收到 {u.shape} 和 {v.shape}")
    if u.shape[1] != v.shape[1]:
        raise ValueError(f"维度不匹配：用户 {u.shape[1]} 维，物品 {v.shape[1]} 维")
    n, m = u.shape[0], v.shape[0]
    if k < 1:
        raise ValueError(f"k 必须 >= 1，收到 {k}")
    if k > m:
        raise ValueError(f"k={k} 超过候选数 {m}")
    if block < 1:
        raise ValueError(f"block 必须 >= 1，收到 {block}")
    # NaN/Inf 检查必须在这里做：往下走就再也不会报错了，只会得到一组假装正常的 Top-K。
    for name, arr in (("user_vecs", u), ("item_vecs", v)):
        if not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).sum())
            raise ValueError(
                f"{name} 含 {n_bad:,} 个 NaN/Inf。检索不会因此报错，只会静默返回"
                "一组看似正常的 Top-K（全 NaN 时甚至酷似热度基线），因此在这里拦下。"
                "最常见的原因是训练发散或学习率过大。"
            )
    # 点积溢出需要量级到 1e19，正常训练不可能；真到了这一步说明上游已经出问题。
    big = max(float(np.abs(u).max(initial=0.0)), float(np.abs(v).max(initial=0.0)))
    if big > 1e18:
        raise ValueError(f"输入向量的最大绝对值 {big:.3g} 过大，点积可能溢出")

    tu = torch.from_numpy(u)
    tvt = torch.from_numpy(np.ascontiguousarray(v.T))
    idx = np.empty((n, k), dtype=np.int32)
    sc = np.empty((n, k), dtype=np.float32) if return_scores else None
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        s = tu[lo:hi] @ tvt                                 # (hi-lo, m)，用完即弃
        top = _topk_block(s, k)
        idx[lo:hi] = top
        if sc is not None:
            sc[lo:hi] = np.take_along_axis(s.numpy(), top, axis=1)
        # 必须显式释放：不 del 的话，下一轮算新块时旧块仍被 s 引用，
        # 两个 760 MB 的分数块会短暂同时存在，峰值近乎翻倍（实测 0.44 GB vs 0.25 GB）。
        del s, top
    return idx, sc


def peak_total_bytes(
    n_queries: int, n_items: int, dim: int, k: int, block: int = DEFAULT_BLOCK
) -> int:
    """整个检索过程的总峰值 —— 扩到 27K 前要按这个数规划，而不是只看单块。

    除了分块产生的临时量，还有三份必须**全程常驻**的数组：物品向量、查询向量、输出下标。
    1K 上它们加起来才 166 MB，所以单块估算够用；27K 上物品向量本身就会成为大头。
    """
    resident = (n_items * dim + n_queries * dim) * 4 + n_queries * k * 4
    return resident + peak_block_bytes(n_items, block)


def peak_block_bytes(n_items: int, block: int = DEFAULT_BLOCK) -> int:
    """估算峰值常驻字节数 —— 27K 上要靠它来定块大小。

    用实测倍率而不是逐项相加：分数块之外还有 `scores >= kth` 的布尔数组、torch.topk 的
    内部缓冲、以及 CPU 缓存分配器不立即归还的部分，后两者都不透明。逐项推导的版本
    先后低估了 6 倍和 3 倍，实测倍率反而稳定（见 PEAK_MULTIPLIER）。
    """
    return int(block * n_items * 4 * PEAK_MULTIPLIER)
