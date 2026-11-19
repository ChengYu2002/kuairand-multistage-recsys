"""排序侧公共输入层（plan §20/§27.1）。batch dict -> 一个扁平特征向量 x。

## 为什么这是独立一层，而不是写在各模型里

§27.1 的 ΔAUC = AUC_MTL − AUC_ST。这个差值只有在「除共享机制外一切相同」时才可归因
（与 §16.6 对负采样的要求是同一件事）。如果每个模型自己拼输入，四份代码必然在拼接
顺序、embedding 初始化、缺失处理上产生差异，而 ΔAUC 会把这些差异一起吸收 ——
表格看上去完全正常。所以 Single-Task / MMoE / PLE / Selective Sharing 共用这一层。

分工：**本层只负责「把 batch 变成 x」，不含任何 MLP。** 共享机制（单塔 / 专家+门控 /
专家分组）作用在 x 之上，那才是四个模型唯一允许不同的地方。

`fingerprint()` 把通道名、维度、词表大小压成一个哈希写进 results。报告阶段比对四个
模型的指纹，不一致就报错 —— 靠"我记得配置一样"是不够的。

## 两张 embedding 表，不共享

历史用召回词表（897,505 行），目标用排序词表（194,312 行），**各一张表**。

不共享的理由是 P0 修复：召回词表 =「候选库 ∪ train 正向视频」，成员身份部分由训练
标签决定 —— train 段 OOV 样本的 is_click / long_view 正样本率恰好为 0，于是「在词表内」
单独就有 train AUC 0.6981 / test 0.5190。目标 ID 必须换成只由 train 曝光构建的词表。
而历史换不了（samples 表里超词表的历史项已压成 OOV，原始 id 丢了），也不必换
（hist_len 单特征 AUC train 0.4983 / test 0.5942，方向相反，不构成捷径）。

两侧共享 embedding 是双塔为了离线建索引才有的约束（§13.2），排序没有。

`item_idx_retrieval`（召回词表行号）在 batch 里只作诊断，**绝不进 x** —— 验算会断言
这一点，因为误用不会报错，只会让 train 上多出一条 AUC 0.70 的捷径。

## 排序为什么不是「用户塔 + 物品塔」

双塔把两侧分开，是为了物品向量能离线算一次建索引（§13.2）。排序没这个约束，它的
价值恰恰在于 user × item 早融合看到一起，所以这里是一次扁平拼接。但历史池化的口径
（masked sum / valid_len）与双塔逐字一致，item embedding 也共用同一张表。

## 通道清单（顺序即拼接顺序，改了指纹就变）

    hist_pool     item_dim     历史聚合（默认 masked mean；DIN 实验改为 target-aware
                               attention），历史 ID 始终走**召回词表**
    item_id       item_dim     目标视频 ID（**排序词表**表，只由 train 曝光构建）。
                               测试段 89.2% 是 OOV，由 video_id_known 标注
    author        author_dim   作者（排序专用词表，test 覆盖 64%）
    tag_pool      tag_dim      标签 masked mean（test 覆盖 97%）
    user_t1       52           用户过去 1/3/7 天的行为统计
    item_t1       52           视频过去 1/3/7 天的统计（双塔刻意不要，排序要）
    pair          15           user x author 过去 3/7 天的偏好（§8.4）。加它是因为
                               is_follow 的 GAUC 只有 0.49014 —— 用户内部排序≈随机，
                               而单特征 ua_imp_7d 就有 0.53784。
    user_static   4 + Σcat     静态画像：数值 + 25 个类别字段的 embedding
    tab, hour     4 + 4        曝光上下文
    numeric       4            duration(已标准化), age, log1p(hist_len), tag_len
    flags         5            author/tag/duration/upload/video_id 是否已知

## 缺失一律显式

五个 *_known 标志和 hist_len / tag_len 都进 x。没有它们，模型分不清「时长是 0」和
「不知道时长」、「没有兴趣」和「没有兴趣可读」—— 而排序侧测试段一多半的物品信息
本来就是缺的，这件事必须让模型知道。
"""

from __future__ import annotations

import hashlib
import json

import torch
from torch import nn

from src.ranking.dataset import KNOWN_FLAGS, OOV, RankingData, RankItemStatic

# 进 x 的标志位，顺序固定（改顺序 = 改指纹）。video_id_known 已在 KNOWN_FLAGS 内。
FLAG_KEYS = KNOWN_FLAGS

# 只作诊断、绝不进 x 的键。验算据此断言编码器没读它们。
FORBIDDEN_IN_X = ("item_idx_retrieval",)


def masked_mean(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """masked sum / valid_len。padding_idx 不能替代：mean 仍会把 PAD 算进分母。

    全 mask 时分子本来就是 0，clamp 只防除零 -> 输出零向量（不是 OOV 向量）。
    模型靠 hist_len / tag_len 知道「这不是内容为零的兴趣，而是没有兴趣可读」。
    """
    m = mask.unsqueeze(-1).to(emb.dtype)
    return (emb * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


# 历史中只平均有效视频
# 标签中只平均有效标签
# PAD 和被 mask 的历史 OOV 不参与


class DINHistoryPool(nn.Module):
    """候选目标条件化的历史聚合；支持受控比较两种权重语义。

    历史向量 ``h`` 与由目标 ID / 作者 / 标签 / 数值 / known flags 拼出的 query ``q``
    先投到同一维度，再用 ``[h, q, h-q, h*q]`` 计算逐位置相关性。

    ``masked_softmax`` 是已经完成的 DIN-softmax 受控变体：输出与 masked mean 同尺度。
    ``raw`` 对应原论文“不要求权重和为 1”的聚合语义，保留候选相关的激活强度。注意
    ``hist_len`` 只能表达候选无关的历史数量，不能替代 raw 权重和所表达的候选相关强度。
    两种模式仍使用 ReLU 且没有 MBA regularization，因此都不冒充完整论文复现。

    全 mask 行不能直接对全 ``-inf`` 做 softmax（会产生 NaN）。实现先用有限极小值，
    softmax 后再乘 mask 并重新归一；全空行因此严格输出零向量。
    """

    def __init__(
        self,
        hist_dim: int,
        query_dim: int,
        attention_hidden: list[int],
        normalization: str = "masked_softmax",
    ) -> None:
        super().__init__()
        if hist_dim <= 0 or query_dim <= 0:
            raise ValueError(
                f"DIN 维度必须为正，收到 hist={hist_dim}, query={query_dim}"
            )
        if not attention_hidden or any(int(h) <= 0 for h in attention_hidden):
            raise ValueError(
                f"DIN attention_hidden 必须是正整数列表，收到 {attention_hidden}"
            )
        self.hist_dim = int(hist_dim)
        self.query_dim = int(query_dim)
        self.attention_hidden = [int(h) for h in attention_hidden]
        self.normalization = str(normalization)
        if self.normalization not in {"masked_softmax", "raw"}:
            raise ValueError(
                "DIN normalization 只支持 'masked_softmax' / 'raw'，"
                f"收到 {self.normalization!r}"
            )
        self.query_proj = nn.Linear(self.query_dim, self.hist_dim)
        layers: list[nn.Module] = []
        d = 4 * self.hist_dim
        for h in self.attention_hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.local_activation = nn.Sequential(*layers)

    def fingerprint(self) -> dict:
        return {
            "kind": "din_style_target_attention",
            "hist_dim": self.hist_dim,
            "query_dim": self.query_dim,
            "attention_hidden": self.attention_hidden,
            "interaction": "[h,q,h-q,h*q]",
            "normalization": self.normalization,
            "empty_history": "zero_vector",
        }

    def forward(
        self,
        history: torch.Tensor,
        mask: torch.Tensor,
        query: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if history.ndim != 3 or history.shape[-1] != self.hist_dim:
            raise ValueError(
                f"history 应为 (B,L,{self.hist_dim})，收到 {tuple(history.shape)}"
            )
        if mask.shape != history.shape[:2]:
            raise ValueError(
                f"mask 应为 {tuple(history.shape[:2])}，收到 {tuple(mask.shape)}"
            )
        if query.ndim != 2 or query.shape != (history.shape[0], self.query_dim):
            raise ValueError(
                f"query 应为 ({history.shape[0]},{self.query_dim})，收到 {tuple(query.shape)}"
            )

        q = self.query_proj(query).unsqueeze(1).expand(-1, history.shape[1], -1)
        interaction = torch.cat([history, q, history - q, history * q], dim=-1)
        score = self.local_activation(interaction).squeeze(-1)
        valid = mask.to(torch.bool)
        if self.normalization == "masked_softmax":
            score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
            weights = torch.softmax(score, dim=1) * valid.to(score.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
        else:
            # 原始 DIN 放松 sum(weights)=1 的约束。mask 后直接加权求和，让权重总和
            # 保留“当前候选激活了多少历史兴趣”的强度；全空历史自然得到严格零向量。
            weights = score * valid.to(score.dtype)
        pooled = torch.bmm(weights.unsqueeze(1), history).squeeze(1)
        if return_weights:
            return pooled, weights
        return pooled


class RankingEncoder(nn.Module):
    def __init__(self, data: RankingData, cfg: dict) -> None:
        super().__init__()
        st: RankItemStatic = data.item_static
        inner = data.inner
        ic = cfg["input"]
        item_dim = int(cfg["embedding_dim"])
        # 目标 ID 通道的两个开关。两者都进指纹，四个模型必须一致。
        # id_dropout：训练时以该概率把目标 ID 换成 OOV，同时把 video_id_known 置 0。
        #   两件事一起做才自洽 —— 只换 ID 会教模型「known=1 也可能配 OOV 向量」。
        #   它同时解决两个问题：(a) ID 被背下来导致从第 1 轮就过拟合；
        #   (b) test 段 89.2% 的行要用 OOV 向量，而不做 dropout 时 OOV 行在 train
        #       只占 47.5% 且分布不同。plan 把它列在 P1。
        # item_id_channel=False：整条目标 ID 通道（含 video_id_known）不进 x，用于消融。
        self.id_dropout = float(ic.get("id_dropout", 0.0))
        if not 0.0 <= self.id_dropout < 1.0:
            raise ValueError(
                f"input.id_dropout 必须在 [0,1) 内，收到 {self.id_dropout}"
            )
        self.use_item_id = bool(ic.get("item_id_channel", True))
        self.history_pooling = str(ic.get("history_pooling", "masked_mean"))
        if self.history_pooling not in {"masked_mean", "din"}:
            raise ValueError(
                "input.history_pooling 只支持 'masked_mean' / 'din'，"
                f"收到 {self.history_pooling!r}"
            )
        self.author_dim = int(ic["author_dim"])
        self.tag_dim = int(ic["tag_dim"])
        cat_dim = int(ic["cat_dim"])

        # 两张表：历史走召回词表，目标走排序词表（理由见模块 docstring 的 P0 一节）
        n_hist = inner.item_tags.shape[0]  # 召回 video 词表行数
        self.hist_emb = nn.Embedding(n_hist, item_dim, padding_idx=0)
        # hist_emb：历史视频，使用召回词表
        self.item_emb = nn.Embedding(st.n_videos_vocab, item_dim, padding_idx=0)
        # item_emb：当前目标视频，使用排序词表
        self.author_emb = nn.Embedding(st.n_authors, self.author_dim, padding_idx=0)
        self.tag_emb = nn.Embedding(st.n_tags_vocab, self.tag_dim, padding_idx=0)
        self.cat_emb = nn.ModuleList(
            [
                nn.Embedding(c, min(cat_dim, max(2, c // 2)))
                for c in inner.user_static_cat_sizes
            ]
        )
        self.tab_emb = nn.Embedding(16, 4)
        self.hour_emb = nn.Embedding(24, 4)
        for e in (
            self.hist_emb,
            self.item_emb,
            self.author_emb,
            self.tag_emb,
            *self.cat_emb,
            self.tab_emb,
            self.hour_emb,
        ):
            nn.init.normal_(e.weight, std=0.01)
            if e.padding_idx is not None:
                with torch.no_grad():
                    e.weight[e.padding_idx].zero_()

        # 装载器口径（max_hist / mask_oov_in_history）必须进指纹：它们改了 x 的含义，
        # 而 config 里写了却没接线的 bug 不会报错，只会让 YAML 变成装饰。
        self.data_params = dict(data.params)
        want = {
            "max_hist": int(ic["max_hist"]),
            "mask_oov_in_history": bool(ic["mask_oov_in_history"]),
            "pair_features": bool(ic["pair_features"]),
        }
        if self.data_params != want:
            raise ValueError(
                f"装载器口径 {self.data_params} 与 config input 段 {want} 不一致 —— "
                "说明 RankingData 没有按 config 构造。config 写了却不生效比写错更危险。"
            )
        self.n_user_t1 = inner.user_t1.shape[1]
        self.n_item_t1 = inner.item_t1.shape[1]
        self.n_pair = 0 if data.pair is None else int(data.pair.shape[1])
        self.pair_cols = list(data.pair_cols)
        self.n_static_num = inner.user_static_num.shape[1]
        self.n_cat = sum(e.embedding_dim for e in self.cat_emb)
        # 顺序即拼接顺序。写成表是为了 fingerprint 能把它序列化。
        # item_id 通道关掉时，video_id_known 也一并去掉 —— 没有 ID 向量的话这个标志
        # 只剩「这视频 train 期见过吗」，而它恰是 train/valid 强度差最大的通道（0.1174）。
        self.flag_keys = (
            tuple(FLAG_KEYS)
            if self.use_item_id
            else tuple(k for k in FLAG_KEYS if k != "video_id_known")
        )
        # DIN query 不只用目标 ID：test 段目标 ID 覆盖率只有 10.8%，只用它会让 89.2%
        # 的候选共用 OOV query，attention 在候选之间静默退化。作者 / 标签 / 时长 / 年龄
        # 在推荐发生前已存在，known flags 让模型区分“数值为 0”和“该字段缺失”。
        self.din: DINHistoryPool | None = None
        self.din_query_fields: tuple[str, ...] = ()
        self.din_query_flag_keys: tuple[str, ...] = ()
        if self.history_pooling == "din":
            dc = ic.get("din")
            if not isinstance(dc, dict):
                raise ValueError("history_pooling='din' 时必须提供 input.din 配置段")
            # video_id_known 与 item_idx 是否为 OOV 完全等价，而 item_vec 已经进 query；
            # 再放一次只会给其轻微热度漂移开第二条路径。其余 flags 仍用于区分元数据缺失。
            self.din_query_flag_keys = tuple(
                k for k in self.flag_keys if k != "video_id_known"
            )
            self.din_query_fields = (
                *(("item_id",) if self.use_item_id else ()),
                "author",
                "tag_pool",
                "duration",
                "age",
                "known_flags",
            )
            query_dim = (
                (item_dim if self.use_item_id else 0)
                + self.author_dim
                + self.tag_dim
                + 2
                + len(self.din_query_flag_keys)
            )
            self.din = DINHistoryPool(
                item_dim,
                query_dim,
                [int(h) for h in dc.get("attention_hidden", [])],
                normalization=str(dc.get("normalization", "masked_softmax")),
            )
        # 323 维输入的目录
        self.channels: list[tuple[str, int]] = [
            ("hist_pool", item_dim),
            *([("item_id", item_dim)] if self.use_item_id else []),
            ("author", self.author_dim),
            ("tag_pool", self.tag_dim),
            ("user_t1", self.n_user_t1),
            ("item_t1", self.n_item_t1),
            *([("pair", self.n_pair)] if self.n_pair else []),
            ("user_static_num", self.n_static_num),
            ("user_static_cat", self.n_cat),
            ("tab", 4),
            ("hour", 4),
            ("numeric", 4),
            ("flags", len(self.flag_keys)),
        ]
        self.out_dim = sum(d for _, d in self.channels)

    def fingerprint(self) -> dict:
        """输入口径的指纹。四个模型必须一致，否则 ΔAUC 不可归因。"""
        body = {
            "channels": self.channels,
            "out_dim": self.out_dim,
            "vocabs": {
                "hist": self.hist_emb.num_embeddings,
                "item": self.item_emb.num_embeddings,
                "author": self.author_emb.num_embeddings,
                "tag": self.tag_emb.num_embeddings,
            },
            "cat_sizes": [e.num_embeddings for e in self.cat_emb],
            "flags": list(self.flag_keys),
            "data_params": self.data_params,
            "pair_cols": self.pair_cols,
            "forbidden_in_x": list(FORBIDDEN_IN_X),
            "id_dropout": self.id_dropout,
            "item_id_channel": self.use_item_id,
        }
        # masked_mean 是历史正式结果的既有口径。默认路径不新增字段，确保旧配置的
        # fingerprint（df312ebad132ea41）逐位不变；DIN 才显式写入新结构。
        if self.din is not None:
            body["history_pooling"] = self.history_pooling
            body["din"] = {
                **self.din.fingerprint(),
                "query_fields": list(self.din_query_fields),
                "query_known_flags": list(self.din_query_flag_keys),
            }
        h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
        return {"sha": h, **body}

    def _target_embeddings(
        self, b: dict
    ) -> tuple[
        dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor, torch.Tensor
    ]:
        """目标侧 embedding 只查一次；ID dropout 同时服务主通道与 DIN query。"""
        flags = {k: b[k] for k in self.flag_keys}
        item_vec = None
        if self.use_item_id:
            idx = b["item_idx"]  # 排序词表行号，不是召回的
            if self.training and self.id_dropout > 0:
                drop = torch.rand(idx.shape, device=idx.device) < self.id_dropout
                idx = torch.where(drop, torch.full_like(idx, OOV), idx)
                if "video_id_known" in flags:
                    flags["video_id_known"] = torch.where(
                        drop,
                        torch.zeros_like(flags["video_id_known"]),
                        flags["video_id_known"],
                    )
            item_vec = self.item_emb(idx)
        author_vec = self.author_emb(b["author"])
        tag_vec = masked_mean(self.tag_emb(b["tags"]), b["tag_mask"])
        return flags, item_vec, author_vec, tag_vec

    def _history_pool(
        self,
        b: dict,
        flags: dict[str, torch.Tensor],
        item_vec: torch.Tensor | None,
        author_vec: torch.Tensor,
        tag_vec: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        history = self.hist_emb(b["hist"])
        if self.din is None:
            if return_weights:
                raise ValueError("masked_mean 没有 attention weights")
            return masked_mean(history, b["hist_mask"])
        query_parts = []
        if item_vec is not None:
            query_parts.append(item_vec)
        query_parts.extend(
            [
                author_vec,
                tag_vec,
                b["duration"].unsqueeze(-1),
                b["age"].unsqueeze(-1),
                torch.stack([flags[k] for k in self.din_query_flag_keys], dim=-1),
            ]
        )
        return self.din(
            history,
            b["hist_mask"],
            torch.cat(query_parts, dim=-1),
            return_weights=return_weights,
        )

    def attention_weights(self, b: dict) -> torch.Tensor:
        """返回 DIN 权重供验算/诊断；正式 forward 仍只返回统一的 x。"""
        if self.din is None:
            raise ValueError("当前编码器使用 masked_mean，没有 attention weights")
        flags, item_vec, author_vec, tag_vec = self._target_embeddings(b)
        _, weights = self._history_pool(
            b, flags, item_vec, author_vec, tag_vec, return_weights=True
        )
        return weights

    def forward(self, b: dict) -> torch.Tensor:
        flags, item_vec, author_vec, tag_vec = self._target_embeddings(b)
        item_channel = [item_vec] if item_vec is not None else []
        hist_pool = self._history_pool(b, flags, item_vec, author_vec, tag_vec)
        x = torch.cat(
            [
                # 历史 ID 转成 embedding
                hist_pool,
                *item_channel,
                author_vec,
                tag_vec,
                b["user_t1"],
                b["item_t1"],
                *([b["pair"]] if self.n_pair else []),
                b["user_static_num"],
                torch.cat(
                    [e(b["user_static_cat"][:, i]) for i, e in enumerate(self.cat_emb)],
                    dim=-1,
                ),
                self.tab_emb(b["tab"]),
                self.hour_emb(b["hour"]),
                torch.stack(
                    [
                        b["duration"],  # 已按 train 统计标准化
                        b["age"],  # 已 log1p
                        # 长度是重尾的（训练均值 49.7/50，测试只有 8.5/50），取 log1p
                        torch.log1p(b["hist_len"]),
                        b["tag_len"],
                    ],
                    dim=-1,
                ),
                torch.stack([flags[k] for k in self.flag_keys], dim=-1),
            ],
            dim=-1,
        )
        if not torch.isfinite(x).all():
            # 静默的 NaN 会让 loss 变 NaN，然后你会去怀疑学习率
            bad = (~torch.isfinite(x)).any(0).nonzero().flatten().tolist()
            raise ValueError(f"输入含 NaN/Inf，列号 {bad[:10]}（共 {len(bad)} 列）")
        return x


def mlp(
    in_dim: int, hidden: list[int], out_dim: int | None = None, dropout: float = 0.0
) -> nn.Sequential:
    """ReLU MLP。最后一层不接激活；out_dim 给定时再接一个线性头。

    与 src/retrieval/item_tower.mlp 分开是因为排序要 dropout 与可选输出头，
    而那个已经被双塔的 checkpoint 固定住，不能改。
    """
    layers: list[nn.Module] = []
    d = in_dim
    # hidden = [256, 128, 64]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        if dropout > 0:
            # Dropout 会在训练时随机把一部分神经元输出变成 0
            layers.append(nn.Dropout(dropout))
        d = h
    if out_dim is not None:
        # 最终linear输出层
        layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def count_params(m: nn.Module) -> dict:
    """按 embedding / 其余 分别计数（§25.1 的 parameter budget 要分段报）。"""
    emb = sum(
        p.numel()
        for mod in m.modules()
        if isinstance(mod, nn.Embedding)
        for p in mod.parameters()
    )
    total = sum(p.numel() for p in m.parameters())
    return {"embedding": emb, "dense": total - emb, "total": total}


# 参数分段的归类前缀。§25.1 要求 MMoE / PLE / Selective 之间参数预算可比，而它们的结构
# 完全不同（共享专家 / 分层专家 / shared+task_specific 维度），只报一个 dense 总数没法比。
_PARAM_GROUPS = (
    ("expert", ("experts.", "shared_experts.", "task_experts.")),
    ("gate", ("gates.",)),
    ("tower", ("towers.", "tower.")),
)


def count_params_grouped(m: nn.Module) -> dict:
    """embedding / expert / gate / tower / other 五段计数。

    归类按参数名前缀。未命中任何前缀的落进 other（Single-Task 的整个塔叫 tower，
    所以它会落在 tower 段；编码器里的非 embedding 参数落 other）。
    """
    emb_ids = {
        id(p)
        for mod in m.modules()
        if isinstance(mod, nn.Embedding)
        for p in mod.parameters()
    }
    out = dict.fromkeys(("embedding", "expert", "gate", "tower", "other"), 0)
    for name, p in m.named_parameters():
        if id(p) in emb_ids:
            out["embedding"] += p.numel()
            continue
        for group, prefixes in _PARAM_GROUPS:
            if any(k in name for k in prefixes):
                out[group] += p.numel()
                break
        else:
            out["other"] += p.numel()
    out["total"] = sum(p.numel() for p in m.parameters())
    return out
