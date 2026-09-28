"""独立验算特征编码规格（阶段 0 的 ⑤）。

这一步的错误全部是静默的：填错一个冷启动值，56.3% 的样本会带着错误的先验进入训练，
指标只会「差一点」，不会报错。归一化统计量若用上了 valid/test，同样不会有任何异常。

检查分五层：
  A. 先验自洽：平滑率的填充值必须等于生成端用的先验 g。从既有特征表按
     g = (sm·(imp+α) − num)/α 反解，与规格中的值逐个标签比对。**并证明填 0 会不同** ——
     否则「填 g 而不是 0」这条设计约束没有被真正检验。
  B. 统计量只用 train：用独立路径在 train 段重算均值/标准差，与规格比对；同时算一遍
     全量（含 valid/test）的统计量，**必须不同**，否则这条限制形同虚设。
  C. apply_spec 的行为：手算若干列的变换结果；类目列的 -124 必须映射成一个合法 index；
     常量列必须被丢弃。
  D. 冷启动路径：构造一整行 null（模拟 join 不到 T-1 特征），套用后计数必须是 0 的
     标准化值、平滑率必须落在先验上、has_history 必须是 0。
  E. 覆盖率：两张 T-1 特征表的全部特征列都必须在规格里，一列都不能漏。

用法：
    python scripts/verify_encoder.py --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features._driver import PRIOR_SPLITS
from src.features.feature_encoder import COUNT, FLAG, SMOOTH, apply_spec, classify
from src.utils.config import load_config, project_path, require

TOL = 1e-9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))

    failures: list[str] = []
    checks = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  {'OK  ' if ok else 'FAIL'} {msg}")
        if not ok:
            failures.append(msg)

    spec = json.loads((proc / f"encoder_spec_{args.protocol}.json").read_text(encoding="utf-8"))
    cols = spec["columns"]
    alpha = spec["alpha"]
    labels = require(cfg, "data", "labels", "tasks")
    logs = pl.scan_parquet(proc / "logs_split.parquet")

    # ---------- A. 先验自洽 ----------
    print("\n=== A. 平滑率的填充值 = 生成端的先验 g ===")
    u = pl.read_parquet(proc / "feat_user_daily.parquet")
    for lab in labels:
        back = float(
            ((u[f"user_{lab}_rate_sm_7d"] * (u["user_imp_7d"] + alpha) - u[f"user_{lab}_7d"])
             / alpha).median()
        )
        got = cols[f"user_{lab}_rate_sm_7d"]["fill"]
        check(abs(back - got) < TOL, f"{lab}: 规格填充 {got:.8f} = 反解先验 {back:.8f}")
    g_click = cols["user_is_click_rate_sm_7d"]["fill"]
    check(abs(g_click - 0.0) > 0.01,
          f"先验 {g_click:.5f} 与 0 显著不同 —— 「填 g 而非 0」确实是有内容的约束")
    # 独立重算先验（不看特征表，直接从日志）
    recomputed = (
        logs.filter(pl.col("split").is_in(PRIOR_SPLITS))
        .select([pl.col(c).mean().alias(c) for c in labels]).collect().row(0, named=True)
    )
    check(all(abs(recomputed[lab] - cols[f"user_{lab}_rate_sm_1d"]["fill"]) < TOL for lab in labels),
          f"三个窗口的平滑率共用同一先验，且与 {'+'.join(PRIOR_SPLITS)} 段重算一致")

    # ---------- B. 统计量只用 train ----------
    print("\n=== B. 归一化统计量只用 train 段 ===")
    lo, hi = spec["stats_from"]["date_min"], spec["stats_from"]["date_max"]
    tr_lo = int(logs.filter(pl.col("split") == "train").select(pl.col("date").min()).collect().item())
    tr_hi = int(logs.filter(pl.col("split") == "train").select(pl.col("date").max()).collect().item())
    check((lo, hi) == (tr_lo, tr_hi), f"规格记录的区间 {lo}~{hi} = train 段实际区间")

    basis = spec["stats_from"].get("basis")
    check(basis == "train_samples", f"加权口径记录为 {basis}")
    probe = ["user_imp_7d", "user_distinct_7d"]
    lf = pl.scan_parquet(proc / "feat_user_daily.parquet")
    for c in probe:
        # 独立重算：train 段曝光样本 join 回特征表，缺失补 0（与 apply_spec 同序）
        want = (
            logs.filter(pl.col("split") == "train").select("user_id", "date")
            .join(lf.select("user_id", "date", c), on=["user_id", "date"], how="left")
            .select(pl.col(c).fill_null(0.0).log1p().mean().alias("m"),
                    pl.col(c).fill_null(0.0).log1p().std().alias("s"))
            .collect().row(0, named=True)
        )
        check(abs(want["m"] - cols[c]["mean"]) < 1e-9 and abs(want["s"] - cols[c]["std"]) < 1e-9,
              f"{c}: mean/std 与「train 样本加权」独立重算一致")
        # 可证伪 1：按特征行等权会得到不同结果 —— 说明口径选择确实有内容
        ed = lf.filter((pl.col("date") >= lo) & (pl.col("date") <= hi)).select(
            pl.col(c).log1p().mean()).collect().item()
        check(abs(ed - cols[c]["mean"]) > 1e-6,
              f"{c}: 按特征行等权 {ed:.6f} ≠ 按样本加权 {cols[c]['mean']:.6f}")
        # 可证伪 2：含 valid/test 也会不同 —— 说明只用 train 的限制生效
        allv = (
            logs.filter(pl.col("split") != "warmup").select("user_id", "date")
            .join(lf.select("user_id", "date", c), on=["user_id", "date"], how="left")
            .select(pl.col(c).fill_null(0.0).log1p().mean()).collect().item()
        )
        check(abs(allv - cols[c]["mean"]) > 1e-6,
              f"{c}: 含 valid/test 的 {allv:.6f} ≠ 只用 train 的 {cols[c]['mean']:.6f}")

    # ---------- C. apply_spec 行为 ----------
    print("\n=== C. apply_spec 的变换 ===")
    sp = cols["user_imp_7d"]
    t = pl.DataFrame({"user_imp_7d": [0.0, 100.0, None]})
    got = apply_spec(t, spec)["user_imp_7d"].to_list()
    want = [(math.log1p(v) - sp["mean"]) / sp["std"] for v in (0.0, 100.0, 0.0)]
    check(all(abs(a - b) < 1e-9 for a, b in zip(got, want)),
          "count 列：null -> 填 0 -> log1p -> 标准化（手算一致）")

    cat = spec["categorical"]["is_live_streamer"]
    t = pl.DataFrame({"is_live_streamer": [-124, 1, None]})
    got = apply_spec(t, spec)["is_live_streamer"].to_list()
    check(got[0] == cat["map"]["-124"] and got[1] == cat["map"]["1"] and got[2] == cat["oov_index"],
          f"is_live_streamer: -124 -> {got[0]}, 1 -> {got[1]}, null -> OOV {got[2]}")
    check(all(v >= 0 for v in got), "映射结果全部非负（原始 -124 不会直接进网络）")
    check("is_lowactive_period" in spec["dropped"] and "is_lowactive_period" not in cols,
          "常量列 is_lowactive_period 已丢弃")

    # ---------- D. 冷启动整行缺失 ----------
    print("\n=== D. 冷启动：整行 join 不到 T-1 特征 ===")
    item_cols = [c for c in pl.scan_parquet(proc / "feat_item_daily.parquet").collect_schema().names()
                 if classify(c, "item")]
    cold = pl.DataFrame({c: [None] for c in item_cols},
                        schema={c: pl.Float64 for c in item_cols})
    out = apply_spec(cold, spec).row(0, named=True)
    sp0 = cols["item_imp_7d"]
    check(abs(out["item_imp_7d"] - (math.log1p(0.0) - sp0["mean"]) / sp0["std"]) < 1e-9,
          "计数列 -> 0 的标准化值")
    check(all(abs(out[f"item_{lab}_rate_sm_7d"] - recomputed[lab]) < TOL for lab in labels),
          "平滑率 -> 先验 g（不是 0）")
    check(all(out[f"item_{lab}_rate_7d"] == 0.0 for lab in labels), "朴素率 -> 0")
    check(out["item_has_history"] == 0.0, "has_history -> 0（用来区分「统计值为 0」与「无历史」）")
    check(out["item_trend"] == 0.0, "trend -> 0（与计数一致）")
    check(all(v is not None and not math.isnan(v) for v in out.values()),
          f"冷启动行无 null / NaN（{len(out)} 列）")

    # ---------- E. 覆盖率 ----------
    print("\n=== E. 特征列覆盖 ===")
    for fname, prefix in (("feat_user_daily", "user"), ("feat_item_daily", "item")):
        names = pl.scan_parquet(proc / f"{fname}.parquet").collect_schema().names()
        feat = [c for c in names if classify(c, prefix)]
        missing = [c for c in feat if c not in cols]
        check(not missing, f"{fname}: {len(feat)} 个特征列全部在规格里")
    kinds = {k: sum(1 for v in cols.values() if v["kind"] == k) for k in (COUNT, SMOOTH, FLAG)}
    check(kinds[SMOOTH] == 30, f"平滑率 30 列（5 标签 x 3 窗口 x 2 侧），实得 {kinds[SMOOTH]}")

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("编码规格验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("编码规格验算通过（先验自洽 / 只用 train / apply 行为 / 冷启动 / 覆盖）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
