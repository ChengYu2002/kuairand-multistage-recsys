"""双塔召回：模型、损失与训练（plan §16）。

## 损失用 sampled softmax / InfoNCE，不用 BCE

换负采样策略时只有 negatives 集合变，公式一个字不动 —— 四种策略才真正可比（§16.6）。
BCE 要为每个负样本单独定标签与权重，换策略时容易不知不觉改掉别的东西。

    s_pos = <u, v_pos> / T
    s_neg = <u, v_neg> / T                      （B x n_neg）
    loss  = CrossEntropy([s_pos, s_neg...], 0)

## accidental hit 由损失统一屏蔽

负样本恰好等于正样本时，把它的 logit 置为 -inf。**不让采样器自己规避**：
四种策略规避的难度不同（in-batch 规避会改变它要研究的那个分布），
放在损失里才能保证它们面对完全一样的处理。

## L2 归一化 + temperature

归一化后点积落在 [-1, 1]，直接进 softmax 会让 logits 过平、梯度过小，
因此除以 temperature（config，默认 0.05）。

## 训练正样本的范围

只取目标落在**候选库区间**的正样本（config: train_target_scope）。
与负样本同空间是硬要求：若正样本可在词表任意位置而负样本只来自候选库，
「不在候选库」就完美预测「是正样本」（占 42%），而成员身份恰好由 index 区间编码。
代价是正样本从 208 万降到 121 万，但那 42% 的 item 仍通过用户塔的历史池化吃到梯度。

## 评估：物品向量对 Config B 是**按日**的

age = 请求日 − 上传日，所以 id_side 下同一个 item 在不同评估日的向量不同，必须按日各算
一份（测试窗 4 天 = 4 份）。id_only 与日期无关，只算一次。这正是 §13.2 把随日期变化的
量排除出物品塔的理由 —— video_age 是 plan 明确要求进 Config B 的例外，代价就在这里。

评估口径与 ItemCF、热度基线完全一致：同一个候选库、同一份考题、同一个 evaluate()。
查询按 (user_id, time_ms) 去重（66,536 条考题 -> 51,746 个时刻），算完再映射回去。

用法：
    python -m src.retrieval.two_tower --config configs/retrieval.yaml --negative random
    python -m src.retrieval.two_tower --negative random --eval-only
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.retrieval.dataset import RetrievalData
from src.retrieval.inbatch_negative import InBatchNegative
from src.retrieval.item_tower import ItemTower
from src.retrieval.random_negative import RandomNegative
from src.retrieval.user_tower import UserTower
from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)

SAMPLERS = {"random": RandomNegative, "inbatch": InBatchNegative}


class TwoTower(nn.Module):
    def __init__(self, data: RetrievalData, cfg: dict) -> None:
        super().__init__()
        # 读取配置 维度
        tt = require(cfg, "two_tower")
        dim = tt["embedding_dim"]

        # 创建贡献embadding表，根据 video_idx 查找视频向量的字典
        n_vocab = data.item_tags.shape[0]
        self.item_emb = nn.Embedding(n_vocab, dim, padding_idx=0)

        nn.init.normal_(self.item_emb.weight, std=0.01)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

        # ItemTower
        # video_idx
           # ↓
        # 共享 item embedding
           # ↓
        # MLP：256 → 128 → 64
           # ↓
        # 物品向量 v
        self.item_tower = ItemTower(
            self.item_emb, tt["item_tower_config"],
            n_authors=int(data.item_author.max()) + 2,
            n_tags=int(data.item_tags.max()) + 2,
            hidden=tt["item_tower_hidden"],
        )
        # UserTower
        self.user_tower = UserTower(
            self.item_emb, n_t1=data.user_t1.shape[1],
            n_static_num=data.user_static_num.shape[1],
            static_cat_sizes=data.user_static_cat_sizes,
            hidden=tt["user_tower_hidden"],
        )
        self.temperature = float(tt["temperature"])
        # 除零会得到 inf/NaN，负温度会把 softmax 的方向整个反过来 —— 两者都不报错。
        if not self.temperature > 0:
            raise ValueError(f"two_tower.temperature 必须 > 0，收到 {self.temperature}")

    def user_vec(self, b: dict) -> torch.Tensor:
        # 把Tower的各种MLP输出 归一化。模型比较的是两个向量的方向是否接近，而不是谁的数值更大。
        return F.normalize(self.user_tower(b), dim=-1)

    def item_vec(self, b: dict) -> torch.Tensor:
        return F.normalize(self.item_tower(b), dim=-1)

    def loss(self, u: torch.Tensor, v_pos: torch.Tensor, v_neg: torch.Tensor,
             pos_ids: torch.Tensor, neg_ids: torch.Tensor) -> torch.Tensor:
        """让每个用户在「1 个正视频 + n 个负视频」中选出正视频。"""

        # 用户分别和正视频、负视频做点积；向量已经 L2 归一化，所以点积就是余弦相似度。
        # 除以 temperature 只放大分数差距，不改变视频之间的排序。
        s_pos = (u * v_pos).sum(-1, keepdim=True) / self.temperature      # (B, 1)
        s_neg = torch.einsum("bd,bnd->bn", u, v_neg) / self.temperature   # (B, n_neg)

        # 如果负采样误抽到正视频本身，就把该位置设为 -inf，让它不参与 softmax。(e^{-inf}=0)
        s_neg = s_neg.masked_fill(neg_ids == pos_ids.unsqueeze(1), float("-inf"))

        # 每行是一道分类题：[正视频, 负视频1, ..., 负视频N]。
        # 正视频固定放在第 0 列，所以每行的正确答案都是 0。
        logits = torch.cat([s_pos, s_neg], dim=1)
        target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # 推高正视频分数，同时压低负视频分数；返回整个 batch 的平均 loss。
        return F.cross_entropy(logits, target)


def train_rows(data: RetrievalData, lo: int, hi: int, scope: str) -> np.ndarray:
    """训练用的样本行号：正向事件，且目标在指定空间内。"""
    if scope not in ("catalog", "vocab"):
        raise ValueError(f"scope 必须是 catalog 或 vocab，收到 {scope!r}")
    pos = np.zeros(len(data), bool)
    for k in ("is_click", "long_view"):
        pos |= data.labels[k] > 0
    t = data.target
    ok = (t >= lo) & (t <= hi) if scope == "catalog" else (t > 1)
    return np.flatnonzero(pos & ok)


@torch.no_grad()
def item_matrix(model: TwoTower, data: RetrievalData, lo: int, hi: int,
                date: int | None, block: int = 8192) -> np.ndarray:
    """候选库全部 item 的向量。date 为 None 表示与日期无关（id_only）。"""
    idx = np.arange(lo, hi + 1, dtype=np.int64)
    out = None
    for s in range(0, len(idx), block):
        chunk = idx[s : s + block]
        d = None if date is None else np.full(len(chunk), date, np.int64)
        v = model.item_vec(data.item_features(chunk, d)).numpy()
        if out is None:
            out = np.empty((len(idx), v.shape[1]), np.float32)
        out[s : s + block] = v
    return out

# 训练好的双塔模型
#         ↓
# 读取测试集的每个请求时刻
#         ↓
# 用户塔生成测试用户向量 U
#         ↓
# 物品塔生成候选库向量 V
#         ↓
# 计算分数 U × Vᵀ
#         ↓
# 每个请求取 Top-K 视频
#         ↓
# 映射成原始 video_id
#         ↓
# 传给 evaluate()
#         ↓
# 计算 Recall@K 和 NDCG@K

@torch.no_grad()
def evaluate_checkpoint(model: TwoTower, cfg: dict, proc, protocol: str):
    """加载好的模型 -> 每个查询时刻的 Top-K -> 统一的 evaluate()。返回 polars.DataFrame。"""
    import polars as pl

    from src.evaluation.retrieval_metrics import evaluate
    from src.retrieval.ann_index import topk_scores

    tt = require(cfg, "two_tower")
    k_list = require(cfg, "eval", "k_list")
    # Top XX
    k = max(k_list)
    # 读取候选库和测试考题
    meta = json.loads((proc / f"vocab_meta_{protocol}.json").read_text("utf-8"))
    lo, hi = meta["video_catalog_index_min"], meta["video_catalog_index_max"]
    catalog = pl.read_parquet(proc / f"catalog_{protocol}.parquet")
    reqs = pl.read_parquet(proc / f"eval_requests_{protocol}.parquet")

    data = RetrievalData(proc, protocol, "test")
    # 找到每道考题对应的样本行
    samples = pl.read_parquet(
        proc / f"samples_test_{protocol}.parquet", columns=["user_id", "video_id", "time_ms"]
    ).with_row_index("srow")
    # 显式保序：polars 的 left join 不保证输出顺序与左表一致
    linked = (
        reqs.select("user_id", "video_id", "time_ms", "date")
        .with_row_index("_i")
        .join(samples, on=["user_id", "video_id", "time_ms"], how="left")
        .sort("_i")
    )
    if linked["srow"].null_count():
        raise AssertionError("有考题在 samples_test 里找不到对应行")
    # 合并相同请求时刻
    mom = (
        linked.select("user_id", "time_ms", "srow", "date")
        .unique(subset=["user_id", "time_ms"], keep="first")
        .sort(["user_id", "time_ms"])
        .with_row_index("mrow")
    )
    # 记录考题属于哪个请求时刻
    mid = (
        linked.select("user_id", "time_ms")
        .with_row_index("_j")
        .join(mom.select("user_id", "time_ms", "mrow"), on=["user_id", "time_ms"], how="left")
        .sort("_j")
        .get_column("mrow").to_numpy()
    )
    log.info("考题 %s 条 -> 唯一查询时刻 %s 个", f"{len(reqs):,}", f"{len(mom):,}")

    # 1. 用训练好的用户塔，为每个测试请求时刻生成用户向量。
    srows = mom["srow"].to_numpy()
    parts = []
    for s in range(0, len(srows), 8192):
        parts.append(model.user_vec(data.batch(srows[s : s + 8192])).numpy())
    uv = np.vstack(parts)

    # 2. 用训练好的物品塔生成候选库向量；id_side 依赖请求日，必须按日各算一份。
    per_day = tt["item_tower_config"] == "id_side"
    dates = np.unique(mom["date"].to_numpy()) if per_day else [None]
    topk = np.empty((len(mom), k), np.int32)
    for d in dates:
        sel = np.arange(len(mom)) if d is None else np.flatnonzero(mom["date"].to_numpy() == d)
        iv = item_matrix(model, data, lo, hi, d)
        # 3. 用户向量与全候选库计算点积分数，为每个请求保留 Top-K。
        idx, _ = topk_scores(uv[sel], iv, k=k)
        topk[sel] = idx
        del iv
    log.info("物品向量算了 %d 份（%s）", len(dates),
             "按评估日，因为 id_side 的 video_age 随日期变" if per_day else "与日期无关")

    # 映射回原始 video_id
    # 4. Top-K 目前是候选矩阵行号，先映射回原始 video_id。
    cat_ids = catalog.get_column("video_id").to_numpy()
    # 5. 把预测 Top-K 和测试集真实答案交给统一 evaluate()，计算 Recall/NDCG。
    return evaluate(cat_ids[topk][mid], reqs.get_column("video_id"),
                    reqs.get_column("user_id").to_numpy(), k_list)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--negative", default=None, choices=tuple(SAMPLERS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=None, help="覆盖 config，用于烟测")
    ap.add_argument("--eval-only", action="store_true", help="不训练，直接评估已有 checkpoint")
    ap.add_argument("--no-eval", action="store_true", help="只训练，跳过评估")
    ap.add_argument("--tag", default=None, help="覆盖产物名（默认由实验维度拼成）")
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))

    strategy = args.negative or require(cfg, "negative_sampling", "strategy")
    if strategy not in SAMPLERS:
        raise NotImplementedError(
            f"负采样策略 {strategy!r} 尚未实现（当前只有 {sorted(SAMPLERS)}）。"
            "exposure / hybrid 是 Week 3 的内容。"
        )
    tt = require(cfg, "two_tower")
    scope = tt["train_target_scope"]
    if scope not in ("catalog", "vocab"):
        raise ValueError(
            f"two_tower.train_target_scope 必须是 catalog 或 vocab，收到 {scope!r}。"
            "拼错会静默走另一条分支：vocab 口径下正负样本不同空间，"
            "「不在候选库」将完美预测「是正样本」（占 42%）。"
        )
    steps = args.max_steps or tt["max_steps"]
    bs, n_neg = tt["batch_size"], tt["negatives_per_positive"]
    meta = json.loads((proc / f"vocab_meta_{args.protocol}.json").read_text("utf-8"))
    lo, hi = meta["video_catalog_index_min"], meta["video_catalog_index_max"]

    # 产物名必须带上**定义一次实验的全部维度**：否则 §7.2 的 id_only / id_side 两组会
    # 互相覆盖，辅助协议也会覆盖主协议。--max-steps 是烟测专用，带上它以免覆盖正式结果。
    tag = args.tag or (
        f"{strategy}_{tt['item_tower_config']}_{args.protocol}_seed{args.seed}"
        + (f"_smoke{steps}" if args.max_steps else "")
    )
    set_seed(args.seed)
    out = project_path("experiments") / f"two_tower_{tag}.pt"
    if args.eval_only:
        ck = torch.load(out, weights_only=False)
        d0 = RetrievalData(proc, args.protocol, "test")
        model = TwoTower(d0, cfg)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        del d0
        _report(evaluate_checkpoint(model, cfg, proc, args.protocol), tag, ck)
        return 0

    data = RetrievalData(proc, args.protocol, "train")
    rows = train_rows(data, lo, hi, scope)
    log.info("训练正样本 %s 条（目标范围 %s，候选库 index [%d, %d]）",
             f"{len(rows):,}", scope, lo, hi)

    model = TwoTower(data, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=tt["lr"])
    sampler = SAMPLERS[strategy](lo, hi)
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model.train()
    t0, losses = time.perf_counter(), []
    for step in range(1, steps + 1):
        # 1. 从全部训练正样本中随机抽一个 batch，再把行号对应的数据组装出来。
        pick = rows[rng.integers(0, len(rows), bs)]
        b = data.batch(pick)

        # 2. 每条正样本抽 n_neg 个负视频。Random 和 In-batch 只在这里来源不同。
        neg_ids = sampler.sample(b, n_neg, gen)

        # 3. 取出负视频特征。先拉平为 B*n_neg 条，日期也为每个负视频重复一次；
        #    id_side 要用「请求日期 - 上传日期」计算 video_age。
        neg_feat = data.item_features(
            neg_ids.numpy().ravel(), np.repeat(data.date[pick], n_neg)
        )

        # 4. 两座塔前向：得到用户、正视频和负视频的 64 维单位向量。
        u = model.user_vec(b)
        v_pos = model.item_vec(b)
        v_neg = model.item_vec(neg_feat).view(bs, n_neg, -1)

        # 5. loss 只负责判分；backward 算梯度，step 才真正更新两座塔和 embedding。
        loss = model.loss(u, v_pos, v_neg, b["target"], neg_ids)
        # 清理当前梯度
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Adam跟新update
        opt.step()

        # 6. 保存当前 loss；每完成约 10% 时打印最近 100 步均值，减少单个 batch 的波动。
        losses.append(float(loss))
        if step % max(1, steps // 10) == 0 or step == steps:
            log.info("step %6d/%d  loss %.4f  (%.1fs)",
                     step, steps, float(np.mean(losses[-100:])), time.perf_counter() - t0)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "strategy": strategy,
                "item_tower_config": tt["item_tower_config"], "protocol": args.protocol,
                "seed": args.seed, "steps": steps,
                "catalog_range": [lo, hi], "loss_tail": float(np.mean(losses[-100:]))}, out)
    log.info("已保存 %s（最后 100 步均值 loss %.4f，总耗时 %.1fs）",
             out.name, float(np.mean(losses[-100:])), time.perf_counter() - t0)
    if not args.no_eval:
        del data
        model.eval()
        _report(evaluate_checkpoint(model, cfg, proc, args.protocol), tag,
                {"steps": steps, "loss_tail": float(np.mean(losses[-100:])),
                 "strategy": strategy, "item_tower_config": tt["item_tower_config"],
                 "protocol": args.protocol, "seed": args.seed})
    return 0


def _report(df, tag: str, ck: dict) -> None:
    from src.evaluation.retrieval_metrics import format_table

    print(f"\nTwoTower-{tag}（{ck.get('steps', '?')} 步，"
          f"loss {ck.get('loss_tail', float('nan')):.4f}）")
    print(format_table(df))
    print()
    dst = project_path("results") / f"two_tower_{tag}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({
        "tag": tag, "strategy": ck.get("strategy"), "seed": ck.get("seed"),
        "item_tower_config": ck.get("item_tower_config"), "protocol": ck.get("protocol"),
        "steps": ck.get("steps"), "loss_tail": ck.get("loss_tail"),
        "metrics": {f"{r['metric']}@{r['k']}": {"per_request": r["per_request"],
                                                "per_user": r["per_user"]}
                    for r in df.iter_rows(named=True)},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("结果已写出 %s", dst.name)


if __name__ == "__main__":
    raise SystemExit(main())
