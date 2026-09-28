"""独立验算召回指标（plan §15）。

指标实现错了是**静默**的：ItemCF、双塔、四种负采样共用这一个入口，它一旦算错，
所有模型会**一起**错到同一个方向，对比表看上去完全正常，而结论已经不成立。
因此这里不依赖「跑出来的数好不好看」，只依赖可以手算或有闭式解的定点。

检查分三层：
  A. 手算比对：一个 2 request 的小例子，Recall / HitRate / NDCG 三个值全部由定义
     独立推导（不调用被测实现的任何中间结果）。该例子刻意构造成多目标，使
     Recall ≠ HitRate，从而真正覆盖多目标分支与 IDCG 的计算。
  B. 可证伪：完美预测必须得 1.0；随机预测必须掉到 K/|catalog| 量级；K 增大时指标
     必须单调不减；把每行顺序倒过来后 Recall 必须**不变**而 NDCG 必须**变小** ——
     最后一条是唯一能证明 NDCG 真的在用排名信息的检查。
  C. 校准靶：热度基线的六个值与 config 中冻结的实测值逐项比对（容差 1e-6）。

用法：
    python scripts/verify_retrieval_metrics.py --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.retrieval_metrics import evaluate, popularity_topk
from src.utils.config import load_config, project_path, require

TOL = 1e-6


def pick(df: pl.DataFrame, metric: str, k: int, col: str = "per_request") -> float:
    return df.filter((pl.col("metric") == metric) & (pl.col("k") == k)).get_column(col).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)

    k_list = require(cfg, "eval", "k_list")
    k_max = max(k_list)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    catalog = pl.read_parquet(proc / "catalog_main.parquet")
    reqs = pl.read_parquet(proc / "eval_requests_main.parquet")
    targets = reqs.get_column("video_id")
    users = reqs.get_column("user_id").to_numpy()

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

    def close(a: float, b: float, tol: float = TOL) -> bool:
        return abs(a - b) <= tol

    # ---------- A. 手算比对 ----------
    print("\n=== A. 手算比对（多目标小例子）===")
    # request 0: 正确答案 {10,20,30}，Top-4 = [10,99,20,98] -> 命中位置 0 与 2
    # request 1: 正确答案 {40}，    Top-4 = [97,96,95,94] -> 全不中
    topk_s = np.array([[10, 99, 20, 98], [97, 96, 95, 94]])
    tgt_s = [[10, 20, 30], [40]]
    got = evaluate(topk_s, tgt_s, np.array([0, 1]), [2, 4])

    # 下面的期望值全部由指标定义直接写出，不引用实现的任何中间量。
    d1 = 1.0 / math.log2(3)  # 位置 1（0 起）的折扣
    exp = {
        # @4：req0 命中 2/3；DCG=1/log2(2)+1/log2(4)=1.5；IDCG=前 min(3,4)=3 个位置
        ("recall", 4): (2 / 3 + 0) / 2,
        ("hitrate", 4): (1 + 0) / 2,
        ("ndcg", 4): ((1.0 + 0.5) / (1.0 + d1 + 0.5) + 0) / 2,
        # @2：req0 只剩位置 0 命中；IDCG=前 min(3,2)=2 个位置
        ("recall", 2): (1 / 3 + 0) / 2,
        ("hitrate", 2): (1 + 0) / 2,
        ("ndcg", 2): (1.0 / (1.0 + d1) + 0) / 2,
    }
    for (metric, k), want in exp.items():
        check(close(pick(got, metric, k), want), f"{metric}@{k} = {want:.6f}（手算）")
    check(
        not close(pick(got, "recall", 4), pick(got, "hitrate", 4)),
        f"多目标下 Recall({pick(got, 'recall', 4):.4f}) ≠ HitRate({pick(got, 'hitrate', 4):.4f})",
    )

    # 两种平均必须走不同的计算路径：构造一个用户活跃度不均的例子。
    # 用户 0 有 2 个 request（中、不中），用户 1 有 1 个（中）。
    #   按 request 平均 = (1+0+1)/3 = 2/3   按用户平均 = ((1+0)/2 + 1)/2 = 0.75
    # 三行 Top-2 各自互不重复（新的唯一性守卫会拒绝 [0,0] 这类无效夹具）。
    skew = evaluate(
        np.array([[1, 7], [8, 9], [3, 6]]), [1, 5, 3], np.array([0, 0, 1]), [2]
    )
    check(close(pick(skew, "recall", 2), 2 / 3), "按 request 平均 = 2/3（活跃度不均例）")
    check(close(pick(skew, "recall", 2, "per_user"), 0.75), "按用户平均 = 0.75（同例）")

    # ---------- B. 可证伪 ----------
    print("\n=== B. 可证伪 ===")
    n = len(reqs)
    tgt_np = targets.to_numpy()
    # 填充值取一个比任何真实 id 都大的数，保证不会与正确答案重复
    # （Top-K 内重复出现正确答案会让 Recall > 1，evaluate 对此有硬校验）。
    filler = int(max(tgt_np.max(), catalog.get_column("video_id").max())) + 1

    # 填充列必须两两不同，否则本身就是非法 Top-K。filler 起步保证不与任何正确答案相撞。
    perfect = np.broadcast_to(filler + np.arange(k_max), (n, k_max)).astype(np.int64).copy()
    perfect[:, 0] = tgt_np
    df_p = evaluate(perfect, targets, users, k_list)
    check(
        all(close(pick(df_p, m, k), 1.0) for m in ("recall", "hitrate", "ndcg") for k in k_list),
        "完美预测（正确答案置于第 1 位）下全部指标 = 1.0",
    )

    rng = np.random.default_rng(args.seed)
    ids = catalog.get_column("video_id").to_numpy()
    # 行内必须无放回。做法是在一个固定随机排列上取长度 K 的随机窗口：窗口内天然互不
    # 重复，且对任一正确答案而言落入窗口的概率恰为 K/N。各行共用同一排列会带来轻微
    # 相关性，但期望值不受影响，落在 0.5x~2x 的容差带内绰绰有余。
    perm = rng.permutation(len(ids))
    starts = rng.integers(0, len(ids), size=n)
    rand_topk = ids[perm[(starts[:, None] + np.arange(k_max)) % len(ids)]]
    df_r = evaluate(rand_topk, targets, users, k_list)
    theory = k_max / len(ids)
    got_r = pick(df_r, "recall", k_max)
    check(
        0.5 * theory <= got_r <= 2 * theory,
        f"随机预测 Recall@{k_max} = {got_r:.6f}，理论值 {theory:.6f}（允许 0.5x~2x）",
    )

    df_pop = evaluate(popularity_topk(catalog, n, k_max), targets, users, k_list)
    ks = sorted(k_list)
    # Recall / HitRate 随 K 单调不减是无条件成立的：分母与 K 无关，分子只会增加。
    for metric in ("recall", "hitrate"):
        vals = [pick(df_pop, metric, k) for k in ks]
        check(
            all(a <= b + TOL for a, b in itertools.pairwise(vals)),
            f"{metric} 随 K 单调不减：{' <= '.join(f'{v:.5f}' for v in vals)}",
        )
    # NDCG 的单调性**只在单目标下成立**：多目标时 IDCG 会随 K 增大（前 min(K, n_targets)
    # 个位置），若新增的位置没命中，NDCG 反而下降。所以先确认当前协议确实是单目标，
    # 否则这条断言本身就是错的。多目标的正确性由 A 层手算例负责。
    single = all(
        not isinstance(t, (list, tuple, set, np.ndarray)) for t in targets.to_list()[:1000]
    )
    check(single, "当前协议为单目标（NDCG 单调性断言的前提）")
    if single:
        vals = [pick(df_pop, "ndcg", k) for k in ks]
        check(
            all(a <= b + TOL for a, b in itertools.pairwise(vals)),
            f"ndcg 随 K 单调不减（单目标下）：{' <= '.join(f'{v:.5f}' for v in vals)}",
        )
    # 反过来把多目标的非单调性钉住：这不是 bug，是 NDCG 的定义使然。
    nm = evaluate(np.array([[1, 9]]), [[1, 2]], np.array([0]), [1, 2])
    check(
        close(pick(nm, "ndcg", 1), 1.0)
        and close(pick(nm, "ndcg", 2), 1.0 / (1.0 + 1.0 / math.log2(3))),
        f"多目标下 NDCG 随 K 下降：@1={pick(nm, 'ndcg', 1):.4f} -> "
        f"@2={pick(nm, 'ndcg', 2):.4f}（IDCG 增大而未新增命中）",
    )

    # 倒序：先用一个有闭式解的合成例，再在真实数据上确认。
    syn = np.array([[7, 8, 9, 10]])
    syn_fwd = evaluate(syn, [7], np.array([0]), [4])
    syn_rev = evaluate(syn[:, ::-1].copy(), [7], np.array([0]), [4])
    check(
        close(pick(syn_fwd, "ndcg", 4), 1.0)
        and close(pick(syn_rev, "ndcg", 4), 1.0 / math.log2(5)),
        f"合成例倒序：NDCG@4 由 1.0 变为 1/log2(5)={1 / math.log2(5):.6f}",
    )
    df_rev = evaluate(popularity_topk(catalog, n, k_max)[:, ::-1].copy(), targets, users, k_list)
    check(
        close(pick(df_rev, "recall", k_max), pick(df_pop, "recall", k_max)),
        "真实数据倒序：Recall 不变（Recall 不看顺序）",
    )
    check(
        pick(df_rev, "ndcg", k_max) < pick(df_pop, "ndcg", k_max) - TOL,
        f"真实数据倒序：NDCG 变小（{pick(df_pop, 'ndcg', k_max):.6f} -> "
        f"{pick(df_rev, 'ndcg', k_max):.6f}）",
    )

    # 输入校验：下面每一项在修复前都会**静默算出一个数**，而不是报错。
    print("\n=== B2. 非法输入必须报错（修复前均为静默算错）===")
    bad_cases = [
        ("Top-K 内重复正确答案", (np.array([[5, 5, 6, 7]]), [5], np.array([0]), [4])),
        ("Top-K 内重复非正确答案", (np.array([[9, 9, 1]]), [1], np.array([0]), [3])),
        ("target 数少于 request 数", (np.array([[1, 2], [3, 4]]), np.array([1]), np.array([0, 1]), [2])),
        ("同一 request 的 target 重复", (np.array([[1, 2, 3]]), [[1, 1]], np.array([0]), [3])),
        ("k_list 为空", (np.array([[1, 2]]), [1], np.array([0]), [])),
        ("k_list 含重复", (np.array([[1, 2]]), [1], np.array([0]), [2, 2])),
        ("k_list 含 0", (np.array([[1, 2]]), [1], np.array([0]), [0, 2])),
        ("k_list 超过 K_max", (np.array([[1, 2]]), [1], np.array([0]), [5])),
    ]
    for label, argv in bad_cases:
        try:
            evaluate(*argv)
            check(False, f"{label} —— 应报错但未报错")
        except ValueError as e:
            check(True, f"{label} —— 已拦截（{str(e)[:36]}…）")

    # ---------- C. 校准靶 ----------
    print("\n=== C. 校准靶（热度基线冻结值）===")
    golden = (cfg.get("eval") or {}).get("expected_popularity_baseline")
    if not golden:
        check(False, "未配置 eval.expected_popularity_baseline")
    else:
        for key, want in golden.items():
            metric, _, k = key.partition("@")
            got_v = pick(df_pop, metric, int(k))
            check(close(got_v, want), f"{key} = {got_v:.6f}（冻结值 {want:.6f}）")

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("召回指标验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("召回指标验算通过（手算比对 / 可证伪 / 校准靶 三层）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
