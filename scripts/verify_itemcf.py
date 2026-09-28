"""独立验算 ItemCF（plan §16）。

ItemCF 在本数据上**低于热度基线**。这种结果有两种完全不同的成因：
  (a) 实现错了；(b) 共现信号确实太稀疏。
二者在指标上长得一模一样，所以必须有能把它们切开的检查 —— 这是本脚本存在的唯一理由。

检查分三层：
  A. 手算比对：4 用户 5 item 的小例子，余弦相似度与最终得分全部由定义手推。
  B. 独立路径：抽样若干 item，用**稠密逐列**的方式重算相似度，与分块+截断的实现逐值
     比对；另在小规模子问题上让 topk_similar = 候选数（等价于不截断），与结合律版
     Scores = (H D^-½ Mᵀ)(M D^-½) 比对 —— 后者完全不碰 item-item 矩阵。
  C. 打分链路：把相似度换成 sim(j,i) = train_freq(i)（与 j 无关），则
     score(u,i) ∝ train_freq(i)，排序必然是热度序，Recall 必须**精确复现**热度基线
     的冻结值。这条能证明「打分 -> 排序 -> 取 TopK」是对的，从而把 (a) 排除掉。

注意：不能用「相似度全设为常数应退化成热度基线」——全常数时所有候选同分，Top-K
由 tie-break 决定而非热度，该断言本身不成立。

用法：
    python scripts/verify_itemcf.py --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.retrieval_metrics import evaluate
from src.retrieval.itemcf import (
    build_profile_matrix,
    item_similarity,
    iter_history_blocks,
    topk_from_scores,
)
from src.utils.config import load_config, project_path, require

TOL = 1e-6


def dense_cosine(M: sparse.csr_matrix, j: int, cat_cols: np.ndarray) -> np.ndarray:
    """独立路径：直接按定义算第 j 个 item 对所有候选的余弦相似度。

    只把**单列**取成稠密（993 维）。整张 M 是 993 x 890,745，稠密化要 7 GB。
    """
    col = np.asarray(M[:, j].todense()).ravel().astype(np.float64)
    sub_m = M[:, cat_cols].astype(np.float64)
    num = np.asarray(sub_m.T @ col).ravel()
    den = np.linalg.norm(col) * np.sqrt(np.asarray(sub_m.multiply(sub_m).sum(axis=0)).ravel())
    den[den == 0] = 1.0
    return num / den


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--n-sample", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))

    failures: list[str] = []
    checks = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        if ok:
            print(f"  OK   {msg}")
        else:
            failures.append(msg)
            print(f"  FAIL {msg}")

    # ---------- A. 手算比对 ----------
    print("\n=== A. 手算比对（4 用户 x 5 item）===")
    #        i0 i1 i2 i3 i4
    # u0      1  1  0  0  0
    # u1      1  1  1  0  0
    # u2      0  0  1  1  0
    # u3      1  0  0  1  1
    A = np.array([[1, 1, 0, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0],
                  [1, 0, 0, 1, 1]], dtype=np.float32)
    Ms = sparse.csr_matrix(A)
    allc = np.arange(5, dtype=np.int32)
    S = item_similarity(Ms, allc, allc, topk=5, iuf=False, zero_self=True, block=2)
    Sd = np.asarray(S.todense())
    # |U0|=3, |U1|=2, |U2|=2, |U3|=2, |U4|=1
    # sim(0,1) = |{u0,u1}| / sqrt(3*2) = 2/sqrt(6)
    check(abs(Sd[0, 1] - 2 / math.sqrt(6)) < TOL, f"sim(i0,i1) = 2/√6 = {2 / math.sqrt(6):.6f}")
    # sim(0,3) = |{u3}| / sqrt(3*2) = 1/sqrt(6)
    check(abs(Sd[0, 3] - 1 / math.sqrt(6)) < TOL, f"sim(i0,i3) = 1/√6 = {1 / math.sqrt(6):.6f}")
    # sim(1,4) = 0（u1 与 u4 无共同用户）
    check(abs(Sd[1, 4]) < TOL, "sim(i1,i4) = 0（无共同用户）")
    check(all(abs(Sd[i, i]) < TOL for i in range(5)), "自相似已置零")

    # 查询历史 H = {i0, i1} 时，对 i2 的得分 = sim(0,2)+sim(1,2)
    Hs = sparse.csr_matrix(([1.0, 1.0], ([0, 0], [0, 1])), shape=(1, 5))
    sc = np.asarray((Hs @ S).todense()).ravel()
    want_i2 = 1 / math.sqrt(3 * 2) + 1 / math.sqrt(2 * 2)
    check(abs(sc[2] - want_i2) < TOL, f"H={{i0,i1}} 时 score(i2) = {want_i2:.6f}（手算）")

    # ---------- B. 独立路径 ----------
    print("\n=== B. 独立路径比对 ===")
    M, _items, i_idx = build_profile_matrix(
        proc, require(cfg, "itemcf", "profile_splits"), require(cfg, "eval", "positive_signal")
    )
    catalog = pl.read_parquet(proc / "catalog_main.parquet")
    cat_cols = np.fromiter(
        (i_idx.get(int(v), -1) for v in catalog.get_column("video_id")), np.int32, len(catalog)
    )
    cat_cols = cat_cols[cat_cols >= 0]

    rng = np.random.default_rng(args.seed)
    deg = np.asarray(M.sum(axis=0)).ravel()
    cand = np.flatnonzero(deg >= 3)                     # 太冷门的 item 相似度全是 0，测不出东西
    sample = rng.choice(cand, size=min(args.n_sample, len(cand)), replace=False)
    sample = np.sort(sample).astype(np.int32)
    topk = require(cfg, "itemcf", "topk_similar")
    dec = require(cfg, "itemcf", "sim_decimals")
    S1 = item_similarity(M, sample, cat_cols, topk, iuf=False, zero_self=True,
                         block=3, decimals=dec)

    worst = 0.0
    for r, j in enumerate(sample):
        ref = dense_cosine(M, int(j), cat_cols)
        self_col = int(np.flatnonzero(cat_cols == j)[0]) if (cat_cols == j).any() else -1
        if self_col >= 0:
            ref[self_col] = 0.0
        # 量化是算法规格的一部分（见 config 的 sim_decimals），参考路径同样要应用：
        # 否则 3/√27 与 2/√12 这类数学上相等的值会因 1 ULP 被判为可区分，
        # 边界上谁进 top-k 就成了浮点意外。stable argsort 的并列规则与 select_topk 一致。
        ref = np.round(ref, dec)
        keep = np.argsort(-ref, kind="stable")[:topk]
        keep = keep[ref[keep] > 0]
        got = dict(zip(S1.indices[S1.indptr[r]:S1.indptr[r + 1]],
                       S1.data[S1.indptr[r]:S1.indptr[r + 1]]))
        if len(got) != len(keep):
            worst = float("inf")
            break
        for c in keep:
            worst = max(worst, abs(got.get(int(c), 0.0) - ref[c]))
    check(worst < 1e-5, f"抽样 {len(sample)} 个 item：分块+截断 vs 稠密逐列重算，最大偏差 {worst:.2e}")

    # 小规模子问题上，不截断时必须等于结合律版（后者完全不碰 item-item 矩阵）
    sub = np.sort(rng.choice(cand, size=60, replace=False)).astype(np.int32)
    S2 = item_similarity(M, sub, sub, topk=len(sub), iuf=False, zero_self=False,
                         block=7, decimals=dec)
    nrm = np.sqrt(np.asarray(M.multiply(M).sum(axis=0)).ravel())
    nrm[nrm == 0] = 1.0
    Hsub = sparse.csr_matrix(
        (np.ones(len(sub), np.float32), (np.arange(len(sub)), np.arange(len(sub)))),
        shape=(len(sub), len(sub)),
    )
    Mn = M[:, sub] @ sparse.diags(1.0 / nrm[sub])
    assoc = np.asarray(((Hsub @ Mn.T) @ Mn).todense())   # 结合律：中间量是 用户 x 用户
    diff = np.abs(np.asarray(S2.todense()) - np.round(assoc, dec)).max()
    check(diff < 1e-5, f"60 个 item 的子问题：分块版 vs 结合律版，最大偏差 {diff:.2e}")

    # ---------- C. 打分链路 ----------
    print("\n=== C. 打分链路（热度替换）===")
    reqs = pl.read_parquet(proc / "eval_requests_main.parquet")
    # 必须用**完整**候选库：主流程在 194,310 上打分。早先这里取 catalog 的前 n_cat 行，
    # 与主流程实际使用的有效子集并不是同一批 item；之所以仍然通过，只是因为热度
    # Top-500 恰好都落在有效子集里 —— 那不能证明完整打分链路无误。
    n_cat = len(catalog)
    k_recall = require(cfg, "itemcf", "topk_recall")
    k_list = require(cfg, "eval", "k_list")
    # sim(j, i) := train_freq(i)，与 j 无关 -> score(u,i) ∝ train_freq(i) -> 排序 = 热度序
    freq = catalog.get_column("train_freq").to_numpy().astype(np.float64)
    # 所有 request 的得分行完全相同，算一行再平铺 —— 直接乘会得到
    # 66,536 x 187,552 的全非零矩阵（125 亿个数）。
    one = sparse.csr_matrix(freq.reshape(1, -1))
    idx1, npad = topk_from_scores(one, k_recall, n_cat,
                                  pad_order=require(cfg, "itemcf", "pad_order"))
    idx = np.tile(idx1, (len(reqs), 1))
    check(npad == 0, "热度替换下无需补位")
    topk_ids = catalog.get_column("video_id").to_numpy()[idx]
    df = evaluate(topk_ids, reqs.get_column("video_id"),
                  reqs.get_column("user_id").to_numpy(), k_list)
    golden = require(cfg, "eval", "expected_popularity_baseline")
    for k in k_list:
        got = df.filter((pl.col("metric") == "recall") & (pl.col("k") == k))["per_request"].item()
        want = golden[f"recall@{k}"]
        check(abs(got - want) < 1e-6,
              f"热度替换后 recall@{k} = {got:.6f}（热度基线冻结值 {want:.6f}）")

    # ---------- D. 历史构建 ----------
    print("\n=== D. 历史构建（两种口径，含可证伪）===")
    positive = pl.any_horizontal(
        [pl.col(c) == 1 for c in require(cfg, "eval", "positive_signal")]
    )
    allpos = (
        pl.scan_parquet(proc / "logs_split.parquet")
        .filter(positive).select("user_id", "video_id", "time_ms").collect()
    )
    moments = reqs.select("user_id", "time_ms").unique().sort(["user_id", "time_ms"])
    probe = moments.sample(n=6, seed=args.seed).sort(["user_id", "time_ms"])
    row_vids = np.array(sorted({int(v) for v in _items}), np.int64)
    row_pos = {int(v): i for i, v in enumerate(row_vids)}

    for mode in ("recent50", "all_before"):
        _lo, _hi, r, c = next(iter_history_blocks(
            mode, probe, proc, row_pos,
            require(cfg, "eval", "positive_signal"),
            require(cfg, "itemcf", "history_max_len"), len(probe),
        ))
        ok_strict, ok_content, ok_relax = True, True, False
        for q in range(len(probe)):
            u = probe["user_id"][q]
            t = probe["time_ms"][q]
            got = {int(row_vids[i]) for i in c[r == q]}
            # 独立路径：直接过滤该用户严格早于 t 的正向行为
            ref_all = allpos.filter((pl.col("user_id") == u) & (pl.col("time_ms") < t))
            if mode == "recent50":
                ref = ref_all.sort(["time_ms", "video_id"], descending=[True, False])
                want = {int(v) for v in ref["video_id"].head(
                    require(cfg, "itemcf", "history_max_len"))}
            else:
                want = {int(v) for v in ref_all["video_id"]}
            want = {v for v in want if v in row_pos}
            if mode == "all_before" and got != want:
                ok_content = False
            # 严格性：历史里不能出现 time >= t 的 item（同刻整批必须排除）
            at_or_after = {int(v) for v in allpos.filter(
                (pl.col("user_id") == u) & (pl.col("time_ms") >= t))["video_id"]}
            leaked = got & (at_or_after - {int(v) for v in ref_all["video_id"]})
            if leaked:
                ok_strict = False
            # 可证伪：把条件放宽成 <= 后，至少有一个时刻的结果应当改变
            relaxed = {int(v) for v in allpos.filter(
                (pl.col("user_id") == u) & (pl.col("time_ms") <= t))["video_id"] if int(v) in row_pos}
            if mode == "all_before" and relaxed != want:
                ok_relax = True
        check(ok_strict, f"{mode}: 历史中无 time >= t 的 item（同刻整批排除）")
        if mode == "all_before":
            check(ok_content, "all_before: 与独立过滤逐元素一致")
            check(ok_relax, "all_before: 放宽为 <= 后结果确实改变（严格性可证伪）")

    # ---------- E. IUF 定义 ----------
    print("\n=== E. IUF 权重（必须是一次而非平方）===")
    A = np.array([[1, 1, 0, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0],
                  [1, 0, 0, 1, 1]], dtype=np.float64)
    Ms2 = sparse.csr_matrix(A)
    allc2 = np.arange(5, dtype=np.int32)
    Siuf = np.asarray(item_similarity(
        Ms2, allc2, allc2, topk=5, iuf=True, zero_self=True, block=5, decimals=12
    ).todense())
    w = 1.0 / np.log1p(A.sum(axis=1))
    num01 = float((w * A[:, 0] * A[:, 1]).sum())          # Σ_v w_v·M[v,0]·M[v,1]
    den0 = math.sqrt(float((w * A[:, 0] ** 2).sum()))
    den1 = math.sqrt(float((w * A[:, 1] ** 2).sum()))
    want_iuf = round(num01 / (den0 * den1), 12)
    check(abs(Siuf[0, 1] - want_iuf) < 1e-9,
          f"IUF 下 sim(i0,i1) = {want_iuf:.9f}（分子为一次权重 Σ w_v·M·M）")
    sq = float((w**2 * A[:, 0] * A[:, 1]).sum()) / (
        math.sqrt(float((w**2 * A[:, 0] ** 2).sum())) * math.sqrt(float((w**2 * A[:, 1] ** 2).sum()))
    )
    check(abs(Siuf[0, 1] - sq) > 1e-9 or abs(want_iuf - sq) < 1e-12,
          "与「权重被平方」的版本可区分（否则该检查无效）")

    # ---------- F. 考题 -> 查询时刻的映射 ----------
    print("\n=== F. 考题映射 ===")
    mid = (
        reqs.select("user_id", "time_ms")
        .join(moments.with_row_index("mid"), on=["user_id", "time_ms"], how="left")
        .get_column("mid")
    )
    check(mid.null_count() == 0, "每条考题都映射到了一个查询时刻")
    g = (
        reqs.select("user_id", "time_ms").with_columns(mid.alias("mid"))
        .group_by("user_id", "time_ms").agg(pl.col("mid").n_unique().alias("u"))
    )
    check(int(g["u"].max()) == 1, "同一 (user_id, time_ms) 的考题共享同一份 Top-K")

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("ItemCF 验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ItemCF 验算通过（手算 / 独立路径 / 打分链路 / 历史 / IUF / 映射）。")
    print("=> 本脚本覆盖的范围内未发现实现错误。这不等于「实现无误」——")
    print("   未覆盖的部分（如打分分块的边界、补位规则的影响）仍需单独讨论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
