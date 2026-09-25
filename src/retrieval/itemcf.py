"""ItemCF 召回基线（plan §16）。

定位：给双塔提供一个经典、非神经的对照。有了它才能说清「双塔学到的是协同信号，
而不只是热度」。它不进 candidate union，只作为独立基线评估。

## 两个矩阵必须分开

M（学相似度）  train 段全部正向行为，(user, item) 去重。**不能**限制成最近 50 ——
               那会把共现从 12.07 亿对压到 124 万对，绝大多数 item 对零共现。
H（查询历史）  request 时刻严格之前的行为。主线取最近 50，与双塔输入对齐，使
               ItemCF vs 双塔的差异只来自模型。

## 为什么不会爆内存

朴素实现要物化 item-item 矩阵：每用户在候选库内平均 1,209 个正向 item，两两组合
共 12.07 亿对，上限约 14.5 GB。这里不物化 —— 按行分块算相似度，每块算完立刻截断到
top-100 再丢弃其余，峰值由块大小决定。

截断之后全程稀疏：每个 request 的候选上限 = |H| x topk_similar = 5,000，所以也不存在
51,746 x 194,310（约 100 亿）的稠密打分矩阵。all_before 那版 |H| 中位 4,271，候选接近
全库，靠按查询时刻分块控制内存。

## request 单位

66,536 条考题只落在 51,746 个不同的 (user_id, time_ms) 上。同一时刻的多条考题共享
同一份查询历史，因此只打 51,746 次分再映射回去 —— 省 22% 算力，且天然保证同一时刻
产出完全相同的 Top-K。

## 已知偏差（README 必须披露）

不排除用户已曝光过的 item，以保持与热度基线、双塔的后处理口径一致。代价是实测
热度 Top-50 中有 26.3% 是该用户训练段已曝光的项，这些坑位对命中没有贡献。

用法：
    python -m src.retrieval.itemcf --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl
from scipy import sparse

from src.evaluation.retrieval_metrics import evaluate, format_table
from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)


def select_topk(cols: np.ndarray, vals: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """取前 k 大，并列时取 cols 下标较小者。

    不能直接用 np.argpartition：它只保证前 k 个位置装的是前 k 小的值，**边界上并列的
    元素取哪几个是任意的**。候选库第 500 名恰好有 17 个并列（train_freq=114，排名
    488~504），交给 argpartition 会让结果既不可复现、也复现不出热度基线的冻结值。
    """
    if len(vals) <= k:
        return cols, vals
    thresh = np.partition(-vals, k - 1)[k - 1]
    strict = np.flatnonzero(-vals < thresh)          # 严格优于门槛，必定入选
    ties = np.flatnonzero(-vals == thresh)           # 并列，按 cols 升序取足
    need = k - len(strict)
    if need < len(ties):
        ties = ties[np.argsort(cols[ties], kind="stable")][:need]
    sel = np.concatenate([strict, ties])
    return cols[sel], vals[sel]


def build_profile_matrix(
    proc, splits: list[str], signals: list[str]
) -> tuple[sparse.csr_matrix, np.ndarray, dict[int, int]]:
    """M：(用户 x item) 二值矩阵，用于学习 item-item 相似度。

    (user, item) 去重 —— 同一个用户重复看同一个视频不应放大共现强度。
    """
    positive = pl.any_horizontal([pl.col(c) == 1 for c in signals])
    df = (
        pl.scan_parquet(proc / "logs_split.parquet")
        .filter(pl.col("split").is_in(splits) & positive)
        .select("user_id", "video_id")
        .unique()
        .collect()
    )
    users = np.sort(df.get_column("user_id").unique().to_numpy())
    items = np.sort(df.get_column("video_id").unique().to_numpy())
    u_idx = {int(u): i for i, u in enumerate(users)}
    i_idx = {int(v): i for i, v in enumerate(items)}
    r = np.fromiter((u_idx[u] for u in df.get_column("user_id")), np.int32, len(df))
    c = np.fromiter((i_idx[v] for v in df.get_column("video_id")), np.int32, len(df))
    # float64：相似度的并列结构必须由数学决定，不能由精度决定（见 sim_decimals）。
    M = sparse.csr_matrix(
        (np.ones(len(df), np.float64), (r, c)), shape=(len(users), len(items))
    )
    M.data[:] = 1.0
    return M, items, i_idx


def item_similarity(
    M: sparse.csr_matrix,
    row_cols: np.ndarray,
    cat_cols: np.ndarray,
    topk: int,
    *,
    iuf: bool,
    zero_self: bool,
    block: int,
    decimals: int = 9,
) -> sparse.csr_matrix:
    """(历史 item x 候选 item) 的余弦相似度，每行只保留 top-k。

    分块沿行方向进行：每块算出约 |block| x 194,310 的稀疏结果，截断后丢弃其余，
    因此完整的 item-item 矩阵从不存在。
    """
    W = M
    if iuf:
        # 活跃用户看过的东西太多，任意两个 item 都容易在他那里共现。IUF 用
        # w_v = 1/log(1+|Iv|) 压低这种贡献，使共现更多来自「专门去看这两个」。
        #
        # 必须开方：分子是 WᵀW，行权重 a 会让用户 v 的贡献变成 a²。要让分子恰好是
        #   numerator(i,j) = Σ_v w_v · M[v,i] · M[v,j]
        # （Breese et al. 的标准定义），行权重就得取 sqrt(w_v)。直接乘 w_v 会得到 w_v²，
        # 与 config 和注释声明的口径不符。
        deg = np.asarray(M.sum(axis=1)).ravel()
        W = sparse.diags(np.sqrt(1.0 / np.log1p(deg))) @ M

    # 归一化用的模长要在**加权后**的矩阵上算，否则余弦不再是余弦。
        # 假设视频A的用户列是：
        # A = [1, 1, 0, 1]
        # 说明4个用户中，有3个正向看过A。
        # 模长为：
        # ||A|| = sqrt(1² + 1² + 0² + 1²)
        #     = sqrt(3)
    norms = np.sqrt(np.asarray(W.multiply(W).sum(axis=0)).ravel())
    norms[norms == 0] = 1.0
    Wc = W.tocsc()

    n_row, n_cat = len(row_cols), len(cat_cols)
    cat_rank = np.full(int(M.shape[1]), -1, np.int32)
    cat_rank[cat_cols] = np.arange(n_cat, dtype=np.int32)

    idx_parts, val_parts = [], []
    # 分块
    for lo in range(0, n_row, block):
        hi = min(lo + block, n_row)
        left = Wc[:, row_cols[lo:hi]].T.tocsr()          # (b x 用户)
        C = (left @ Wc[:, cat_cols]).tocsr()             # (b x 候选)
        C = sparse.diags(1.0 / norms[row_cols[lo:hi]]) @ C @ sparse.diags(1.0 / norms[cat_cols])
        C = C.tocsr()
        for r in range(hi - lo):
            s, e = C.indptr[r], C.indptr[r + 1]
            cols, vals = C.indices[s:e], C.data[s:e]
            if zero_self:
                # 自相似恒为 1，会让历史 item 自己霸占榜首。它不是「排除已看」的
                # 后处理，而是「一个 item 不是自己的邻居」这一相似度定义。
                self_col = cat_rank[row_cols[lo + r]]
                if self_col >= 0:
                    keep = cols != self_col
                    cols, vals = cols[keep], vals[keep]
            # 先量化再截断：否则数学上相等的相似度会因 1 个 ULP 的差异被判为非并列，
            # 「谁进 top-k」就成了浮点意外而非声明的规则。
            vals = np.round(vals, decimals)
            cols, vals = select_topk(cols, vals, topk)
            order = np.lexsort((cols, -vals))            # 全序键，保证可复现
            idx_parts.append(cols[order])
            val_parts.append(vals[order])

    lens = np.fromiter((len(v) for v in val_parts), np.int64, n_row)
    indptr = np.concatenate([[0], np.cumsum(lens)])
    return sparse.csr_matrix(
        (np.concatenate(val_parts), np.concatenate(idx_parts), indptr), shape=(n_row, n_cat)
    )


def topk_from_scores(
    scores: sparse.csr_matrix, k: int, n_cat: int, decimals: int = 9,
    pad_order: str = "catalog_asc",
) -> tuple[np.ndarray, int]:
    """每行取 Top-k 候选下标；非零候选不足 k 时补位。

    候选库按 (train_freq desc, video_id asc) 排序，所以下标越小越热门。

    补位规则必须与并列规则一致：得不到相似度的候选本质上**全部并列在 0 分**，
    而本模块声明的并列规则是「取下标较小者」。早先的实现反过来从最冷门补起
    （理由是「补位不该撞中正确答案」），那等于对同一种情形用了两套规则，
    并且会系统性低估 —— 实测影响 13.28% 的查询时刻，Recall@500 差约 5%。
    pad_order 保留 catalog_desc 以便报告这一选择的敏感性。
    """
    n = scores.shape[0]
    out = np.empty((n, k), np.int32)
    if pad_order == "catalog_asc":
        tail = np.arange(n_cat, dtype=np.int32)          # 与并列规则一致：热门优先
    elif pad_order == "catalog_desc":
        tail = np.arange(n_cat - 1, -1, -1, dtype=np.int32)
    else:
        raise ValueError(f"未知 pad_order: {pad_order!r}")
    n_padded = 0
    for r in range(n):
        s, e = scores.indptr[r], scores.indptr[r + 1]
        cols, vals = scores.indices[s:e], scores.data[s:e]
        vals = np.round(vals, decimals)      # 同上：得分是相似度之和，并列同样普遍
        cols, vals = select_topk(cols, vals, k)
        order = np.lexsort((cols, -vals))
        cols = cols[order]
        if len(cols) < k:
            n_padded += 1
            pad = tail[~np.isin(tail, cols)][: k - len(cols)]
            cols = np.concatenate([cols, pad])
        out[r] = cols
    return out, n_padded


def iter_history_blocks(
    mode: str,
    moments: pl.DataFrame,
    proc,
    row_pos: dict[int, int],
    signals: list[str],
    max_len: int,
    block: int,
):
    """按查询时刻分块产出 H 的 (行下标, 列下标)。

    分块是必须的：all_before 口径下总非零约 2.62 亿（约 3.1 GB），一次性构造会爆内存。
    两种口径都只取 request 时刻**严格之前**的行为 —— recent50 直接复用
    user_history.parquet（其严格性已由 verify_history.py 独立验算），
    all_before 用 searchsorted 定位前缀，同刻事件整批排除。
    """
    n = len(moments)
    if mode == "recent50":
        h = (
            pl.scan_parquet(proc / "user_history.parquet")
            .select("user_id", "time_ms", "hist")
            .unique(subset=["user_id", "time_ms"])
            .collect()
        )
        long = (
            moments.with_row_index("mid")
            .join(h, on=["user_id", "time_ms"], how="left")
            .with_columns(pl.col("hist").list.head(max_len))
            .explode("hist")
            .drop_nulls("hist")
        )
        mid = long.get_column("mid").to_numpy()
        vid = long.get_column("hist").to_numpy()
        col = np.fromiter((row_pos.get(int(v), -1) for v in vid), np.int32, len(vid))
        ok = col >= 0
        mid, col = mid[ok], col[ok]
        for lo in range(0, n, block):
            hi = min(lo + block, n)
            m = (mid >= lo) & (mid < hi)
            yield lo, hi, (mid[m] - lo).astype(np.int32), col[m]
        return

    # all_before：每个用户的正向序列排好后，第 q 个查询时刻的历史就是它的一个前缀。
    positive = pl.any_horizontal([pl.col(c) == 1 for c in signals])
    pos = (
        pl.scan_parquet(proc / "logs_split.parquet")
        .filter(positive)
        .select("user_id", "video_id", "time_ms")
        .collect()
        .sort(["user_id", "time_ms", "video_id"])
    )
    starts = {}
    uid = pos.get_column("user_id").to_numpy()
    bounds = np.searchsorted(uid, np.unique(uid), side="left")
    for u, b in zip(np.unique(uid), bounds):
        starts[int(u)] = int(b)
    ptimes = pos.get_column("time_ms").to_numpy()
    pitems = pos.get_column("video_id").to_numpy()
    ends = {u: (starts[u] + int((uid == u).sum())) for u in starts}

    m_user = moments.get_column("user_id").to_numpy()
    m_time = moments.get_column("time_ms").to_numpy()
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        rows_l, cols_l = [], []
        for r in range(lo, hi):
            u = int(m_user[r])
            s = starts.get(u)
            if s is None:
                continue
            e = ends[u]
            # side="left" -> 严格早于 t，同一毫秒的整批事件被排除
            cut = s + int(np.searchsorted(ptimes[s:e], m_time[r], side="left"))
            vids = pitems[s:cut]
            if not len(vids):
                continue
            c = np.fromiter((row_pos.get(int(v), -1) for v in vids), np.int32, len(vids))
            c = c[c >= 0]
            rows_l.append(np.full(len(c), r - lo, np.int32))
            cols_l.append(c)
        if rows_l:
            yield lo, hi, np.concatenate(rows_l), np.concatenate(cols_l)
        else:
            yield lo, hi, np.empty(0, np.int32), np.empty(0, np.int32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    # 必须限定取值：任何非 recent50 的字符串都会落进 all_before 分支，
    # 而那条路径重得多（历史中位 4,271，曾触发 OOM），拼错不该静默进去。
    ap.add_argument("--history", default=None, choices=("recent50", "all_before"))
    ap.add_argument("--iuf", default=None, choices=("true", "false"))
    ap.add_argument("--tag", default=None, help="结果标签，默认按口径自动生成")
    # all_before 口径下 H 稠密得多（中位 4,271 条历史），H @ Sim 的结果接近全库，
    # 必须把打分块调小，否则单块就要几个 GB。
    ap.add_argument("--score-block", type=int, default=None)
    ap.add_argument("--sim-block", type=int, default=None)
    ap.add_argument("--save-topk", default=None, help="把 (n_requests, topk_recall) 的 Top-K 存成 .npy")
    args = ap.parse_args()
    cfg = load_config(args.config)

    cf = require(cfg, "itemcf")
    mode = args.history or require(cfg, "itemcf", "history")
    if mode not in ("recent50", "all_before"):
        raise ValueError(f"itemcf.history 必须是 recent50 或 all_before，收到 {mode!r}")
    iuf = (args.iuf == "true") if args.iuf else bool(require(cfg, "itemcf", "iuf"))
    k_list = require(cfg, "eval", "k_list")
    k_recall = require(cfg, "itemcf", "topk_recall")
    if k_recall < max(k_list):
        raise ValueError(f"topk_recall({k_recall}) 必须 >= max(k_list)={max(k_list)}")
    if require(cfg, "itemcf", "similarity") != "cosine":
        raise NotImplementedError(f"仅支持 cosine，收到 {cf['similarity']!r}")
    tag = args.tag or f"ItemCF-{'50' if mode == 'recent50' else 'All'}{'-IUF' if iuf else ''}"

    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    catalog = pl.read_parquet(proc / f"catalog_{args.protocol}.parquet")
    reqs = pl.read_parquet(proc / f"eval_requests_{args.protocol}.parquet")

    t0 = time.perf_counter()
    M, items, i_idx = build_profile_matrix(
        proc, require(cfg, "itemcf", "profile_splits"), require(cfg, "eval", "positive_signal")
    )
    log.info("M: %d 用户 x %s item，非零 %s", M.shape[0], f"{M.shape[1]:,}", f"{M.nnz:,}")

    # 候选库来自 train 段**曝光**，而 M 来自 train 段**正向行为**，后者是前者的子集。
    # 这 6,758 个 item 拿不到任何相似度分，但它们**必须留在候选空间里**：协议要求
    # ItemCF / 双塔 / 热度基线在完全相同的 194,310 个候选上检索。把它们从候选空间
    # 删掉会让其中 491 个曾是考题答案的 item 永远不可召回（影响 622 条考题，0.935%），
    # 那是协议违规而非模型能力差异。正确做法是让它们得 0 分，与其他零分候选一同并列。
    cat_cols = np.fromiter(
        (i_idx.get(int(v), -1) for v in catalog.get_column("video_id")), np.int32, len(catalog)
    )
    valid_mask = cat_cols >= 0
    valid_positions = np.flatnonzero(valid_mask).astype(np.int32)   # 在完整候选库中的下标
    if not valid_mask.all():
        log.info("候选库 %s 个中，%s 个在训练段无正向行为（保留在候选空间内，恒得 0 分）",
                 f"{len(catalog):,}", f"{(~valid_mask).sum():,}")

    # 唯一查询时刻：同刻的多条考题共享历史，只需打一次分
    moments = reqs.select("user_id", "time_ms").unique().sort(["user_id", "time_ms"])
    log.info("考题 %s 条 -> 唯一查询时刻 %s 个", f"{len(reqs):,}", f"{len(moments):,}")

    # Sim 只需要为「真正出现在查询历史里」的 item 建行
    if mode == "recent50":
        hist_items = (
            pl.scan_parquet(proc / "user_history.parquet")
            .filter(pl.col("split") == "test")
            .select(pl.col("hist").explode().unique())
            .collect()
            .to_series()
            .drop_nulls()
            .to_numpy()
        )
    else:
        hist_items = items
    row_vids = np.array(sorted({int(v) for v in hist_items if int(v) in i_idx}), np.int64)
    row_cols = np.fromiter((i_idx[int(v)] for v in row_vids), np.int32, len(row_vids))
    row_pos = {int(v): i for i, v in enumerate(row_vids)}
    log.info("需要建相似度行的历史 item: %s 个", f"{len(row_vids):,}")

    Sim = item_similarity(
        M, row_cols, cat_cols[valid_mask], require(cfg, "itemcf", "topk_similar"),
        iuf=iuf, zero_self=bool(require(cfg, "itemcf", "zero_self_similarity")),
        block=args.sim_block or require(cfg, "itemcf", "sim_block"),
        decimals=require(cfg, "itemcf", "sim_decimals"),
    )
    # 列下标从「有效子集内的位置」映射回「完整候选库内的位置」，使打分在 194,310 上进行。
    Sim = sparse.csr_matrix(
        (Sim.data, valid_positions[Sim.indices], Sim.indptr),
        shape=(Sim.shape[0], len(catalog)),
    )
    log.info("Sim: %s x %s，非零 %s (%.1fs)",
             f"{Sim.shape[0]:,}", f"{Sim.shape[1]:,}", f"{Sim.nnz:,}", time.perf_counter() - t0)

    # 打分：分块 + 即时取 Top-K，稀疏全程
    cat_ids = catalog.get_column("video_id").to_numpy()
    n_cat = len(catalog)
    topk_idx = np.empty((len(moments), k_recall), np.int32)
    padded = 0
    for lo, hi, r, c in iter_history_blocks(
        mode, moments, proc, row_pos, require(cfg, "eval", "positive_signal"),
        require(cfg, "itemcf", "history_max_len"),
        args.score_block or require(cfg, "itemcf", "score_block"),
    ):
        H = sparse.csr_matrix(
            (np.ones(len(r), np.float64), (r, c)), shape=(hi - lo, Sim.shape[0])
        )
        H.data[:] = 1.0                                   # 同一 item 重复出现只算一次
        blk, npad = topk_from_scores(
            (H @ Sim).tocsr(), k_recall, n_cat, require(cfg, "itemcf", "sim_decimals"),
            require(cfg, "itemcf", "pad_order"),
        )
        topk_idx[lo:hi] = blk
        padded += npad
    if padded:
        log.warning(
            "有 %s 个查询时刻的非零候选不足 %d（%.2f%%），已按 pad_order=%s 补位 —— "
            "这部分 Top-K 不由 ItemCF 决定，报告中必须披露",
            f"{padded:,}", k_recall, 100 * padded / len(moments),
            require(cfg, "itemcf", "pad_order"),
        )

    # 映射回 66,536 条考题
    # 显式保序：polars 的 left join 不保证输出顺序与左表一致，顺序一乱每条考题就会
    # 拿到别人的 Top-K，且不会报错。
    mid = (
        reqs.select("user_id", "time_ms")
        .with_row_index("_i")
        .join(moments.with_row_index("mid"), on=["user_id", "time_ms"], how="left")
        .sort("_i")
        .get_column("mid").to_numpy()
    )
    topk = cat_ids[topk_idx][mid]

    if args.save_topk:
        np.save(args.save_topk, topk)
        log.info("Top-K 已存至 %s %s", args.save_topk, topk.shape)

    df = evaluate(topk, reqs.get_column("video_id"), reqs.get_column("user_id").to_numpy(), k_list)
    print(f"\n{tag}（history={mode}, iuf={iuf}, topk_similar={cf['topk_similar']}）")
    print(format_table(df))
    print()

    # 结果必须落盘：三个版本的数字后面要进报告、要和双塔对比，只打印在终端等于没存。
    out = project_path("results") / f"itemcf_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tag": tag, "protocol": args.protocol,
        "history": mode, "iuf": iuf,
        "topk_similar": require(cfg, "itemcf", "topk_similar"),
        "pad_order": require(cfg, "itemcf", "pad_order"),
        "sim_decimals": require(cfg, "itemcf", "sim_decimals"),
        "catalog_size": len(catalog),
        "catalog_without_train_positive": int((~valid_mask).sum()),
        "n_requests": len(reqs), "n_moments": len(moments),
        "n_moments_padded": int(padded),
        "metrics": {f"{r['metric']}@{r['k']}": {"per_request": r["per_request"],
                                                "per_user": r["per_user"]}
                    for r in df.iter_rows(named=True)},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("结果已写出 %s（总耗时 %.1fs）", out, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
