"""Hybrid 负采样（plan §16.4，P0）：Random + Exposure。

两种负样本各占一半：

    每行 10 个负样本 = 前 5 个 random（覆盖整个候选库） + 后 5 个 exposure（难负样本）

## 为什么是 Random + Exposure，而不是 In-batch + Exposure

plan §16.4 给的是"例如"，两种都允许。选 Random 的理由是**它们互补的维度是正交的**：

    random    均匀、与用户无关      —— 提供"整个候选库"的覆盖
    exposure  条件于用户、已被线上筛过 —— 提供难度
    in-batch  偏热门、与用户无关      —— 提供的是热度先知识，与 exposure 的"该用户看过"
                                        高度重叠（看过的本来就偏热门）

In-batch + Exposure 的两半会争夺同一个方向，混合比例的效果会和各自单独的效果纠缠，
四策略对比读不出"混合"本身带来了什么。

## 为什么是固定 5+5，而不是每个位置按概率二选一

不是为了"才能复现" —— 概率混合给定 seed 同样可复现。真正的理由是**每一行的负样本
组成完全一致**：按概率混合时每行实际拿到的 exposure 个数是个 Binomial(10, 0.5)
随机量（std 1.58），于是"混合比例"自己成了一个逐行波动的变量。研究要量的是
比例 0.5 的效果，不是比例在 0.5 附近抖动的效果。

列的布局固定（前 random 后 exposure）不影响损失：accidental hit 是逐元素屏蔽的，
softmax 对负样本的顺序也无关。

## 接口

与其余三种完全一致：

    sample(batch, n_neg, generator) -> LongTensor (B, n_neg)
"""

from __future__ import annotations

import torch

from src.retrieval.exposure_negative import ExposureNegative, ExposurePool
from src.retrieval.random_negative import RandomNegative

DEFAULT_MIX = {"random": 0.5, "exposure": 0.5}


class HybridNegative:
    name = "hybrid"
    # 一半从整个候选库随便抽，一半从这个用户看过但没正向过的视频里抽

    def __init__(
        self,
        catalog_lo: int,
        catalog_hi: int,
        pool: ExposurePool,
        mix: dict | None = None,
    ) -> None:
        self.lo, self.hi = int(catalog_lo), int(catalog_hi)
        self.random = RandomNegative(catalog_lo, catalog_hi)
        self.exposure = ExposureNegative(catalog_lo, catalog_hi, pool)
        self.mix = dict(DEFAULT_MIX if mix is None else mix)
        if set(self.mix) != set(DEFAULT_MIX):
            raise ValueError(
                f"hybrid_mix 的键必须是 {sorted(DEFAULT_MIX)}，收到 {sorted(self.mix)}"
            )
        if abs(sum(self.mix.values()) - 1.0) > 1e-9:
            raise ValueError(f"hybrid_mix 之和必须为 1，收到 {self.mix}")

    def split(self, n_neg: int) -> tuple[int, int]:
        """(n_random, n_exposure)。固定切分，与 batch 内容无关。"""
        n_exp = round(n_neg * self.mix["exposure"])
        n_rand = n_neg - n_exp
        if n_rand <= 0 or n_exp <= 0:
            raise ValueError(
                f"n_neg={n_neg} 按 {self.mix} 切分得到 random={n_rand} / "
                f"exposure={n_exp}，有一侧为 0 —— 那就不是 hybrid 了，"
                "应当直接用对应的单一策略跑，否则实验名与实际内容不符。"
            )
        return n_rand, n_exp

    def sample(
        self, batch: dict, n_neg: int, generator: torch.Generator
    ) -> torch.Tensor:
        n_rand, n_exp = self.split(n_neg)
        return torch.cat(
            [
                self.random.sample(batch, n_rand, generator),
                self.exposure.sample(batch, n_exp, generator),
            ],
            dim=1,
        )

    def stats(self) -> dict:
        return {
            "strategy": self.name,
            "mix": dict(self.mix),
            **{k: v for k, v in self.exposure.stats().items() if k != "strategy"},
        }
