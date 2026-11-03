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
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.retrieval.dataset import RetrievalData
from src.retrieval.full_inbatch import full_inbatch_loss, off_diagonal_duplicate_mask
from src.retrieval.item_tower import ItemTower
from src.retrieval.logq_correction import build_log_q
from src.retrieval.negative_sampling import (
    FULL_INBATCH,
    SAMPLERS,
    USES_LOGQ,
    build_sampler,
)
from src.retrieval.user_tower import UserTower
from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)

# SAMPLERS / build_sampler 见 src/retrieval/negative_sampling.py


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
            self.item_emb,
            tt["item_tower_config"],
            n_authors=int(data.item_author.max()) + 2,
            n_tags=int(data.item_tags.max()) + 2,
            hidden=tt["item_tower_hidden"],
        )
        # UserTower
        self.user_tower = UserTower(
            self.item_emb,
            n_t1=data.user_t1.shape[1],
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

    def loss(
        self,
        u: torch.Tensor,
        v_pos: torch.Tensor,
        v_neg: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        log_q_pos: torch.Tensor | None = None,
        log_q_neg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """让每个用户在「1 个正视频 + n 个负视频」中选出正视频。

        log_q_* 不为 None 时做 sampled-softmax 的 logQ 校正（§16.5）：
        从**每个** logit 里减去该 item 被采为候选的对数概率。两个都传或都不传 ——
        只校正负样本会让正样本相对位移一个与它自身热度相关的量，那是另一个偏差。
        默认 None，此时整条路径与校正引入前逐字相同（scripts/prove_noop.py 实测）。
        """

        # 用户分别和正视频、负视频做点积；向量已经 L2 归一化，所以点积就是余弦相似度。
        # 除以 temperature 只放大分数差距，不改变视频之间的排序。
        s_pos = (u * v_pos).sum(-1, keepdim=True) / self.temperature  # (B, 1)
        s_neg = torch.einsum("bd,bnd->bn", u, v_neg) / self.temperature  # (B, n_neg)

        if (log_q_pos is None) != (log_q_neg is None):
            raise ValueError(
                "log_q_pos 与 log_q_neg 必须同时给或同时不给 —— "
                "只校正一侧会引入一个与热度相关的系统性位移。"
            )
        if log_q_pos is not None:
            # log Q < 0，所以减去它把**每个** logit 都抬高 —— 不是「给热门减掉一块」。
            # 热门 item 的 Q 大、log Q 更接近 0，抬得**更少**，于是它在 softmax 分母里
            # 的相对权重变小，每次作为负样本受到的梯度更轻，恰好抵消「出现次数多」。
            # 这是相对关系；verify_logq.py 的 B 段用 (-d_hot) < (-d_cold) 把它钉成可证伪的。
            # 推理与建索引**不做**这一步。
            s_pos = s_pos - log_q_pos.unsqueeze(1)
            s_neg = s_neg - log_q_neg

        # 如果负采样误抽到正视频本身，就把该位置设为 -inf，让它不参与 softmax。(e^{-inf}=0)
        # logQ 不解决假负样本，两者正交：先校正再屏蔽，-inf 减任何有限值仍是 -inf。
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
def item_matrix(
    model: TwoTower,
    data: RetrievalData,
    lo: int,
    hi: int,
    date: int | None,
    block: int = 8192,
) -> np.ndarray:
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


def eval_moments(proc, protocol: str):
    """考题表、唯一查询时刻、以及「考题 -> 时刻」的下标。

    从 request_topk 里**抽出**而非复制：scripts/global_component_rho.py 要的 ū 必须
    在同一组查询时刻上取平均，否则它和 Recall 不是同一口径。抄第二份的话，改一处
    另一处不会跟着改 —— 两个数仍然都能算出来，只是其中一个对不上任何东西。
    """
    import polars as pl

    reqs = pl.read_parquet(proc / f"eval_requests_{protocol}.parquet")
    # 找到每道考题对应的样本行
    samples = pl.read_parquet(
        proc / f"samples_test_{protocol}.parquet",
        columns=["user_id", "video_id", "time_ms"],
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
        .join(
            mom.select("user_id", "time_ms", "mrow"),
            on=["user_id", "time_ms"],
            how="left",
        )
        .sort("_j")
        .get_column("mrow")
        .to_numpy()
    )
    log.info("考题 %s 条 -> 唯一查询时刻 %s 个", f"{len(reqs):,}", f"{len(mom):,}")
    return reqs, mom, mid


@torch.no_grad()
def user_matrix(
    model: TwoTower, data: RetrievalData, srows, block: int = 8192
) -> np.ndarray:
    """给定样本行的用户向量。与 item_matrix 对称，供 request_topk 与 ρ 脚本共用。"""
    parts = []
    for s in range(0, len(srows), block):
        parts.append(model.user_vec(data.batch(srows[s : s + block])).numpy())
    return np.vstack(parts)


@torch.no_grad()
def request_topk(model: TwoTower, cfg: dict, proc, protocol: str, data=None):
    """逐 request 的 Top-K（已映射回 video_id），外加考题表。

    从 evaluate_checkpoint 里**抽出**而非复制：§7.2 的频次分层分析要的是逐 request
    的 Top-K 而不是聚合指标，而两处各算一份的话，一处改了另一处不会跟着改 ——
    两张表仍然都能出，只是其中一张是错的。
    """
    import polars as pl

    from src.retrieval.ann_index import topk_scores

    tt = require(cfg, "two_tower")
    k_list = require(cfg, "eval", "k_list")
    # Top XX
    k = max(k_list)
    # 读取候选库和测试考题
    meta = json.loads((proc / f"vocab_meta_{protocol}.json").read_text("utf-8"))
    lo, hi = meta["video_catalog_index_min"], meta["video_catalog_index_max"]
    catalog = pl.read_parquet(proc / f"catalog_{protocol}.parquet")
    reqs, mom, mid = eval_moments(proc, protocol)

    # data 可由调用方传入：分层分析要对每个 checkpoint 逐个算 Top-K，每次重新
    # 装载 test 段每次约 60 秒，策略数一多就是十几分钟，而这份数据是只读的、可共用。
    # 默认 None 时行为与改动前完全一致。
    if data is None:
        data = RetrievalData(proc, protocol, "test")

    # 1. 用训练好的用户塔，为每个测试请求时刻生成用户向量。
    uv = user_matrix(model, data, mom["srow"].to_numpy())

    # 2. 用训练好的物品塔生成候选库向量；id_side 依赖请求日，必须按日各算一份。
    per_day = tt["item_tower_config"] == "id_side"
    dates = np.unique(mom["date"].to_numpy()) if per_day else [None]
    topk = np.empty((len(mom), k), np.int32)
    for d in dates:
        sel = (
            np.arange(len(mom))
            if d is None
            else np.flatnonzero(mom["date"].to_numpy() == d)
        )
        iv = item_matrix(model, data, lo, hi, d)
        # 3. 用户向量与全候选库计算点积分数，为每个请求保留 Top-K。
        idx, _ = topk_scores(uv[sel], iv, k=k)
        topk[sel] = idx
        del iv
    log.info(
        "物品向量算了 %d 份（%s）",
        len(dates),
        "按评估日，因为 id_side 的 video_age 随日期变" if per_day else "与日期无关",
    )

    # 映射回原始 video_id
    # 4. Top-K 目前是候选矩阵行号，先映射回原始 video_id。
    cat_ids = catalog.get_column("video_id").to_numpy()
    return cat_ids[topk][mid], reqs


@torch.no_grad()
def evaluate_checkpoint(model: TwoTower, cfg: dict, proc, protocol: str):
    """加载好的模型 -> 每个查询时刻的 Top-K -> 统一的 evaluate()。返回 polars.DataFrame。"""
    from src.evaluation.retrieval_metrics import evaluate

    # 5. 把预测 Top-K 和测试集真实答案交给统一 evaluate()，计算 Recall/NDCG。
    topk, reqs = request_topk(model, cfg, proc, protocol)
    return evaluate(
        topk,
        reqs.get_column("video_id"),
        reqs.get_column("user_id").to_numpy(),
        require(cfg, "eval", "k_list"),
    )


def artifact_tag(
    strategy: str,
    item_tower_config: str,
    protocol: str,
    seed: int,
    *,
    smoke_steps: int | None = None,
    n_neg_override: int | None = None,
) -> str:
    """产物名必须带上**定义一次实验的全部维度** —— 少一维就是静默覆盖。

    这是一个函数而不是 main() 里的一行 f-string，因为 verify 脚本要校的就是**这条
    规则本身**：原先 verify 里把名字按手抄的字面量重拼了一遍，命名规则改了它照样通过，
    等于没校。

    `negatives_per_positive` 在 §7.3 之后也成了一个会变的维度，但它只在被**显式扫描**
    时进名字：不传 `--negatives-per-positive` 的调用拼出的名字与改动前逐字相同，
    所以已有的 15 份产物不用改名，§7.1~§7.3 的表也不会突然对不上。
    """
    tag = f"{strategy}_{item_tower_config}_{protocol}_seed{seed}"
    if n_neg_override is not None:
        tag += f"_neg{n_neg_override}"
    if smoke_steps is not None:
        tag += f"_smoke{smoke_steps}"
    return tag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--negative", default=None, choices=tuple(SAMPLERS))
    ap.add_argument(
        "--item-tower-config",
        default=None,
        choices=("id_only", "id_side"),
        help="覆盖 config 中的物品塔输入，用于保持 ID-only / ID+Side 实验入口稳定",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--negatives-per-positive",
        type=int,
        default=None,
        help="覆盖 config 的 negatives_per_positive；传了就把 _neg{n} 拼进产物名，"
        "否则负样本数量扫描会写到 n_neg=10 基线的文件名上，静默覆盖 §7.2/§7.3 的基线",
    )
    ap.add_argument("--max-steps", type=int, default=None, help="覆盖 config，用于烟测")
    ap.add_argument(
        "--eval-only", action="store_true", help="不训练，直接评估已有 checkpoint"
    )
    ap.add_argument("--no-eval", action="store_true", help="只训练，跳过评估")
    ap.add_argument("--tag", default=None, help="覆盖产物名（默认由实验维度拼成）")
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))

    strategy = args.negative or require(cfg, "negative_sampling", "strategy")
    if strategy not in SAMPLERS:
        raise NotImplementedError(
            f"负采样策略 {strategy!r} 不存在（当前有 {sorted(SAMPLERS)}）"
        )
    tt = require(cfg, "two_tower")
    if args.item_tower_config is not None:
        tt["item_tower_config"] = args.item_tower_config
    if args.negatives_per_positive is not None:
        if args.negatives_per_positive < 1:
            raise ValueError(f"负样本数必须 >= 1，收到 {args.negatives_per_positive}")
        if strategy in FULL_INBATCH:
            # 完整 in-batch 的位置数由 batch_size 决定（B-1），config 里那个值根本不参与。
            # 允许这个组合，产物名里会出现一个对训练毫无影响的 _neg{n}，等于撒谎。
            raise ValueError(
                f"{strategy} 的负样本位置是 B-1，--negatives-per-positive 对它无意义"
            )
        tt["negatives_per_positive"] = args.negatives_per_positive
    scope = tt["train_target_scope"]
    if scope not in ("catalog", "vocab"):
        raise ValueError(
            f"two_tower.train_target_scope 必须是 catalog 或 vocab，收到 {scope!r}。"
            "拼错会静默走另一条分支：vocab 口径下正负样本不同空间，"
            "「不在候选库」将完美预测「是正样本」（占 42%）。"
        )
    steps = args.max_steps if args.max_steps is not None else tt["max_steps"]
    if steps < 1:
        raise ValueError(f"训练步数必须 >= 1，收到 {steps}")
    bs, n_neg = tt["batch_size"], tt["negatives_per_positive"]
    negative_mode = "full_inbatch" if strategy in FULL_INBATCH else "sampled"
    negative_positions = bs - 1 if strategy in FULL_INBATCH else n_neg
    meta = json.loads((proc / f"vocab_meta_{args.protocol}.json").read_text("utf-8"))
    lo, hi = meta["video_catalog_index_min"], meta["video_catalog_index_max"]

    # 命名规则本身见 artifact_tag()，verify 脚本校的是那个函数。
    tag = args.tag or artifact_tag(
        strategy,
        tt["item_tower_config"],
        args.protocol,
        args.seed,
        smoke_steps=steps if args.max_steps else None,
        n_neg_override=args.negatives_per_positive,
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
    log.info(
        "训练正样本 %s 条（目标范围 %s，候选库 index [%d, %d]）",
        f"{len(rows):,}",
        scope,
        lo,
        hi,
    )

    # logQ 校正（§16.5）。Q 由**这一份 rows** 统计 —— 它就是训练时真正用的那份，
    # 换成曝光次数或 catalog.train_freq 都会描述另一个 proposal 分布。
    logq = None
    if strategy in USES_LOGQ:
        logq = build_log_q(data.target, rows, lo, hi)
        s = logq.stats()
        log.info(
            "logQ：由 %s 条训练正样本统计，覆盖候选库 %.1f%%（%s 个 item）；"
            "Q/均匀 min/p50/max = %.3f/%.3f/%.1f",
            f"{s['n_train_rows']:,}",
            100 * s["coverage"],
            f"{s['n_items_with_support']:,}",
            s["q_x_uniform_min"],
            s["q_x_uniform_p50"],
            s["q_x_uniform_max"],
        )

    model = TwoTower(data, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=tt["lr"])
    sampler = build_sampler(
        strategy, lo, hi, data=data, cfg=cfg, proc=proc, protocol=args.protocol
    )
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if strategy in FULL_INBATCH:
        # 守卫放在循环之前：它是实验臂的构造前提，不是每步都要重新确认的东西。
        if logq is None:
            raise AssertionError("完整 in-batch 实验臂必须启用 logQ")
        log.info(
            "完整 in-batch：每行使用 batch 其余 %d 个候选位置；"
            "ID+Side 按请求日期分组编码物品，config 的 negatives_per_positive=%d "
            "在本实验臂中不参与训练",
            bs - 1,
            n_neg,
        )

    model.train()
    t0, losses = time.perf_counter(), []
    for step in range(1, steps + 1):
        # 1. 从全部训练正样本中随机抽一个 batch，再把行号对应的数据组装出来。
        pick = rows[rng.integers(0, len(rows), bs)]
        b = data.batch(pick)

        if strategy in FULL_INBATCH:
            # batch 的 B 个 target 同时构成每行的 1 个正样本与最多 B-1 个负样本。
            # ID+Side 的 age 依赖请求日期，不能直接复用 v_pos.T；专用实现按日期
            # 分组编码，并只屏蔽同 ID 的非对角 accidental hit。
            if step == 1:
                # B-1 只是上限：同 ID 的重复会被屏蔽掉。记录实测值，别让报告里
                # 的「1023 个负样本」是一个没人量过的数。
                dup = int(off_diagonal_duplicate_mask(b["target"]).sum())
                log.info(
                    "首步实测：B=%d，同 ID 重复占 %d 个非对角位置，"
                    "平均每行有效负样本 %.2f（上限 %d）",
                    bs,
                    dup,
                    bs - 1 - dup / bs,
                    bs - 1,
                )
            loss = full_inbatch_loss(
                model,
                data,
                b,
                data.date[pick],
                logq.lookup(b["target"]),
            )
        else:
            # 每条正样本抽 n_neg 个负视频。五臂主实验继续走原路径，一个字义不变。
            neg_ids = sampler.sample(b, n_neg, gen)

            # 取出负视频特征。先拉平为 B*n_neg 条，日期也为每个负视频重复一次；
            # id_side 要用「请求日期 - 上传日期」计算 video_age。
            neg_feat = data.item_features(
                neg_ids.numpy().ravel(), np.repeat(data.date[pick], n_neg)
            )

            # 两座塔前向：得到用户、正视频和负视频的 64 维单位向量。
            u = model.user_vec(b)
            v_pos = model.item_vec(b)
            v_neg = model.item_vec(neg_feat).view(bs, n_neg, -1)

            # loss 只负责判分；backward 算梯度，step 才真正更新两座塔和 embedding。
            lq_pos = lq_neg = None
            if logq is not None:
                lq_pos, lq_neg = logq.lookup(b["target"]), logq.lookup(neg_ids)
            loss = model.loss(u, v_pos, v_neg, b["target"], neg_ids, lq_pos, lq_neg)
        # 清理当前梯度
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Adam跟新update
        opt.step()

        # 6. 保存当前 loss；每完成约 10% 时打印最近 100 步均值，减少单个 batch 的波动。
        losses.append(float(loss.detach()))
        if step % max(1, steps // 10) == 0 or step == steps:
            log.info(
                "step %6d/%d  loss %.4f  (%.1fs)",
                step,
                steps,
                float(np.mean(losses[-100:])),
                time.perf_counter() - t0,
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    sampler_stats = sampler.stats() if hasattr(sampler, "stats") else None
    logq_stats = logq.stats() if logq is not None else None
    torch.save(
        {
            "state_dict": model.state_dict(),
            "strategy": strategy,
            "item_tower_config": tt["item_tower_config"],
            "protocol": args.protocol,
            "seed": args.seed,
            "steps": steps,
            "negative_mode": negative_mode,
            "negative_positions_per_positive": negative_positions,
            "catalog_range": [lo, hi],
            "loss_tail": float(np.mean(losses[-100:])),
            # exposure/hybrid 的空池子 fallback 比例必须随结果一起落盘：
            # 比例偏高时 "exposure" 其实掺了 random，读结论时必须看得见。
            "sampler": sampler_stats,
            "logq": logq_stats,
        },
        out,
    )
    log.info(
        "已保存 %s（%s，总耗时 %.1fs）",
        out.name,
        _loss_txt(float(np.mean(losses[-100:])), negative_positions),
        time.perf_counter() - t0,
    )
    if not args.no_eval:
        del data
        model.eval()
        _report(
            evaluate_checkpoint(model, cfg, proc, args.protocol),
            tag,
            {
                "steps": steps,
                "negative_mode": negative_mode,
                "negative_positions_per_positive": negative_positions,
                "loss_tail": float(np.mean(losses[-100:])),
                "strategy": strategy,
                "item_tower_config": tt["item_tower_config"],
                "protocol": args.protocol,
                "seed": args.seed,
                "sampler": sampler_stats,
                "logq": logq_stats,
            },
        )
    return 0


def _loss_txt(loss: float, positions: int | None) -> str:
    """loss 带上随机猜水平 —— 它**不可跨实验臂直接比**。

    n_neg=10 是 11 分类（随机猜 ln(11)=2.398），完整 in-batch 是 B 分类
    （ln(1024)=6.931）。完整 in-batch 的 loss 必然大一个量级，那是题从「11 选 1」
    变成了「1024 选 1」，不是训练变差。可比的是归一化量 loss/ln(C)。
    不写出来的话，三个月后看 results/*.json 的第一反应一定是「完整 in-batch 更差」。
    """
    if not positions:
        return f"loss {loss:.4f}"
    c = positions + 1
    chance = math.log(c)
    return (
        f"loss {loss:.4f}（{c} 分类，随机猜 {chance:.3f}，归一化 {loss / chance:.3f}）"
    )


def _report(df, tag: str, ck: dict) -> None:
    from src.evaluation.retrieval_metrics import format_table

    print(
        f"\nTwoTower-{tag}（{ck.get('steps', '?')} 步，"
        f"{_loss_txt(ck.get('loss_tail', float('nan')), ck.get('negative_positions_per_positive'))}）"
    )
    print(format_table(df))
    print()
    dst = project_path("results") / f"two_tower_{tag}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        json.dumps(
            {
                "tag": tag,
                "strategy": ck.get("strategy"),
                "seed": ck.get("seed"),
                "item_tower_config": ck.get("item_tower_config"),
                "protocol": ck.get("protocol"),
                "steps": ck.get("steps"),
                "negative_mode": ck.get("negative_mode"),
                "negative_positions_per_positive": ck.get(
                    "negative_positions_per_positive"
                ),
                "loss_tail": ck.get("loss_tail"),
                # 派生值随结果落盘：跨臂引用 loss 时必须用这个，而不是 loss_tail。
                "loss_tail_normalized": (
                    ck["loss_tail"]
                    / math.log(ck["negative_positions_per_positive"] + 1)
                    if ck.get("negative_positions_per_positive")
                    and ck.get("loss_tail") is not None
                    else None
                ),
                "sampler": ck.get("sampler"),
                "logq": ck.get("logq"),
                "metrics": {
                    f"{r['metric']}@{r['k']}": {
                        "per_request": r["per_request"],
                        "per_user": r["per_user"],
                    }
                    for r in df.iter_rows(named=True)
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("结果已写出 %s", dst.name)


if __name__ == "__main__":
    raise SystemExit(main())
