"""用户塔（plan §13.1）。

不使用 user_id embedding：只有 1,000 个用户，ID embedding 会被直接背下来而无法泛化。
用户由「最近看过什么」+ T-1 统计 + 静态画像 + context 表示。

## 掩码不是可选项

历史序列定长 50，短的补 PAD。torch 的 padding_idx=0 只保证 PAD 那一行恒为零向量且
不回传梯度，**不会**把它从平均的分母里去掉 —— 直接 mean(dim=1) 会让历史 5 条和 50 条
差 10 倍尺度。而实测训练时平均 49.67/50 条可编码、评估时只有 12.05/50，
尺度会系统性错位约 4 倍：模型在"历史很长"上训练，却在"历史很短"上评估。

因此这里用 masked sum / valid_len。OOV（index 1）同样被 mask：评估时 82.5% 的历史
条目是 OOV，让它们参与只会让池化结果被同一个向量主导。

## 空历史

valid_len = 0 时历史通道输出**零向量**（clamp(min=1) 只是防除零，分子本来就是 0）。
模型靠 hist_len 与 user_has_history 知道"这不是一个内容为零的兴趣，而是没有兴趣可读"。
实测评估时刻 0.37% 为空、训练 0.005%。
"""

from __future__ import annotations

import torch
from torch import nn

from src.retrieval.item_tower import mlp


class UserTower(nn.Module):
    def __init__(
        self,
        item_emb: nn.Embedding,
        n_t1: int,
        n_static_num: int,
        static_cat_sizes: list[int],
        n_tabs: int = 16,
        hidden: list[int] | None = None,
        cat_dim: int = 8,
    ) -> None:
        super().__init__()
        self.item_emb = item_emb
        self.cat_emb = nn.ModuleList(
            [nn.Embedding(c, min(cat_dim, max(2, c // 2))) for c in static_cat_sizes]
        )
        self.tab_emb = nn.Embedding(n_tabs, 4)
        self.hour_emb = nn.Embedding(24, 4)
        d = (
            item_emb.embedding_dim            # 历史池化
            + n_t1                            # T-1 统计
            + n_static_num                    # 静态数值
            + sum(e.embedding_dim for e in self.cat_emb)
            + 4 + 4                           # tab, hour
            + 1                               # hist_len（池化向量背后有多少证据）
        )
        self.mlp = mlp(d, hidden or [256, 128, 64])
        self.out_dim = (hidden or [256, 128, 64])[-1]

    def pool_history(self, hist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """masked sum / valid_len。padding_idx 不能替代这一步，理由见模块 docstring。"""
        m = mask.unsqueeze(-1).float()
        s = (self.item_emb(hist) * m).sum(dim=1)
        return s / m.sum(dim=1).clamp(min=1.0)     # 全 mask 时分子为 0 -> 零向量

    def forward(self, b: dict) -> torch.Tensor:
        cat = torch.cat(
            [e(b["user_static_cat"][:, i]) for i, e in enumerate(self.cat_emb)], dim=-1
        )
        x = torch.cat(
            [
                self.pool_history(b["hist"], b["hist_mask"]),
                b["user_t1"],
                b["user_static_num"],
                cat,
                self.tab_emb(b["tab"]),
                self.hour_emb(b["hour"]),
                # hist_len 做 log1p：长度是重尾的（中位 50，但评估时中位 12）
                torch.log1p(b["hist_len"]).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.mlp(x)
