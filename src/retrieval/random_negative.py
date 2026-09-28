"""随机负采样（plan §16）。

从候选库区间里均匀抽。与 in-batch 的区别只在"从哪儿抽"，其余（损失、塔、数据、
评估）完全相同 —— 这正是 §16.6 要求的：四种策略之间除采样外不得有任何差异。

## 接口契约（Week 3 插 exposure / hybrid 时不得改动其他代码）

    sample(batch, n_neg, generator) -> LongTensor (B, n_neg)

返回的是**词表行号**，取值一律落在候选库区间。不负责规避"抽到正样本本身"——
那由损失函数统一屏蔽（accidental hit）。四种策略规避的难度不同，放在损失里
才能保证它们面对完全一样的处理。

## 为什么从候选库抽而不是整个词表

词表 897,503 个里只有 194,310 个是候选。拿非候选当负样本，等于让模型去区分
一批它在评估时根本不会遇到的东西，训练信号与评估任务错位。
"""

from __future__ import annotations

import torch


class RandomNegative:
    name = "random"

    def __init__(self, catalog_lo: int, catalog_hi: int) -> None:
        if catalog_hi < catalog_lo:
            raise ValueError(f"候选库区间非法：[{catalog_lo}, {catalog_hi}]")
        self.lo, self.hi = int(catalog_lo), int(catalog_hi)

    def sample(self, batch: dict, n_neg: int, generator: torch.Generator) -> torch.Tensor:
        b = batch["target"].shape[0]
        return torch.randint(
            self.lo, self.hi + 1, (b, n_neg), generator=generator, dtype=torch.long
        )
