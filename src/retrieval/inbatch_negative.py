"""In-batch 负采样（plan §16）。

拿同一批里**其他样本的正样本**当负样本。它与随机采样的本质差别是分布：
被抽中的概率正比于该 item 在正样本里出现的频率，因此天然偏向热门 item ——
这正是 logQ 校正（P1）要修的东西，也是这两种策略值得对比的原因。

接口与 RandomNegative 完全一致：

    sample(batch, n_neg, generator) -> LongTensor (B, n_neg)

同样不负责规避"抽到自己"：第 i 行有 1/B 的概率抽到自己的正样本，
统一由损失函数屏蔽（accidental hit）。

## 一个必须留意的性质

batch_size 小于 n_neg+1 时可抽的来源不足，只能有放回地重复抽。这不是错误，
但会让负样本重复度升高，等效负样本数下降。构造时会检查并给出提示。
"""

from __future__ import annotations

import torch


class InBatchNegative:
    name = "inbatch"

    def __init__(self, catalog_lo: int, catalog_hi: int) -> None:
        # 区间只用于断言：batch 里的正样本按 train_target_scope 已经限定在候选库内。
        self.lo, self.hi = int(catalog_lo), int(catalog_hi)

    def sample(self, batch: dict, n_neg: int, generator: torch.Generator) -> torch.Tensor:
        tgt = batch["target"]
        b = tgt.shape[0]
        if b < 2:
            raise ValueError("in-batch 负采样至少需要 2 条样本")
        # 有放回地从同批目标里抽。自己抽到自己的情况交给损失屏蔽，
        # 在这里规避会改变分布（这正是 in-batch 要研究的那个分布）。
        idx = torch.randint(0, b, (b, n_neg), generator=generator, dtype=torch.long)
        return tgt[idx]
