"""物品塔（plan §13.2）。

    Config A   仅 item_id embedding
    Config B   item_id + author_id + tag + duration + video_age

两个 Config **共用同一个 MLP 结构**，只有输入特征不同。若 A 直接输出 embedding、
B 才过 MLP，两者就同时差了特征与深度，§7.2 那张表的对比会被混淆。

## embedding 与用户塔共享

同一个视频在「候选」和「历史」两种角色下用同一份向量。这正是词表取并集的理由：
只出现在历史里的 70.3 万个 item 也能通过用户塔吃到梯度，等它作为候选出现时已经学过。
不共享的话这部分梯度就白丢了，参数量还翻倍（897,504 x 64 = 5,740 万）。

## T-1 统计特征不进物品塔（§13.2）

物品向量要离线算一次建索引，依赖每日变化的量会让召回复杂化。video_age 严格说也随
日期变化 —— 它进 Config B 是 plan 的要求，代价是物品向量必须按评估日重算一遍，
且训练/评估的 age 分布有系统性差异（train 中位 2 天 / test 23 天），报告里要披露。
"""

from __future__ import annotations

import torch
from torch import nn


def mlp(in_dim: int, hidden: list[int]) -> nn.Sequential:
    """塔的 MLP。**最后一层不加激活。**

    末层跟 ReLU 会把输出限制在非负象限，L2 归一化后所有向量落在单位球的正卦限，
    两两余弦相似度恒为正 —— 模型只能表达「不太相似」，无法表达「不相似」。
    实测（600 步的 checkpoint）：60.4% 的用户向量坐标被压成恰好 0，
    全部两两相似度落在 [0.2582, 0.6901]，64 维里只有约四成在工作。
    """
    layers: list[nn.Module] = []
    d = in_dim
    for i, h in enumerate(hidden):
        layers.append(nn.Linear(d, h))
        if i < len(hidden) - 1:
            layers.append(nn.ReLU())
        d = h
    return nn.Sequential(*layers)


class ItemTower(nn.Module):
    def __init__(
        self,
        item_emb: nn.Embedding,
        config: str = "id_only",
        n_authors: int = 2,
        n_tags: int = 2,
        hidden: list[int] | None = None,
        side_dim: int = 16,
    ) -> None:
        super().__init__()
        if config not in ("id_only", "id_side"):
            raise ValueError(f"item_tower_config 必须是 id_only 或 id_side，收到 {config!r}")
        self.config = config
        self.item_emb = item_emb
        d = item_emb.embedding_dim
        if config == "id_side":
            # padding_idx=0：tag 补位不参与梯度。author 的 0 号同样是保留位。
            self.author_emb = nn.Embedding(n_authors, side_dim, padding_idx=0)
            self.tag_emb = nn.Embedding(n_tags, side_dim, padding_idx=0)
            d += side_dim * 2 + 4            # + duration, has_duration, age, has_upload
        self.mlp = mlp(d, hidden or [256, 128, 64])
        self.out_dim = (hidden or [256, 128, 64])[-1]

    def forward(self, b: dict) -> torch.Tensor:
        x = self.item_emb(b["item_idx"])
        if self.config == "id_side":
            tags = b["tags"]
            tm = (tags != 0).unsqueeze(-1).float()
            # tag 是多值（1~5 个），按有效个数平均；全为 PAD 时得零向量而不是 NaN
            tag_vec = (self.tag_emb(tags) * tm).sum(1) / tm.sum(1).clamp(min=1.0)
            x = torch.cat(
                [
                    x,
                    self.author_emb(b["author"]),
                    tag_vec,
                    b["duration"].unsqueeze(-1),
                    b["has_duration"].unsqueeze(-1),
                    b["age"].unsqueeze(-1),
                    b["has_upload"].unsqueeze(-1),
                ],
                dim=-1,
            )
        return self.mlp(x)
