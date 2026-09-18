"""独立验算 T-1 特征（plan §8 的正确性保障）。

特征泄漏与统计口径错误都是**静默**的：不会抛异常，只会让线下指标虚高或让特征
失去意义。因此这里对每一类特征都做检查，而不只是基础计数 —— 首版只验了计数，
恰好漏掉的 distinct 就是唯一出错的那一项。

检查分三层：
  A. 原始量：用独立代码路径（按日期区间直接过滤）重算 曝光/标签计数、
     活跃天数、真实 distinct，与特征表逐值比对。
  B. 派生量：校验比率、平滑比率、趋势、has_history 是否与同行的计数自洽。
  C. 全局不变量：(key,date) 唯一、窗口单调、无 NaN/Inf、时间口径排除当天。

用法：
    python scripts/verify_t1_features.py --n 6
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features._driver import PRIOR_SPLITS  # noqa: E402
from src.utils.config import load_config, project_path, require  # noqa: E402

FMT = "%Y%m%d"
TOL = 1e-9


def day(d: int, delta: int) -> int:
    return int((datetime.strptime(str(d), FMT) + timedelta(days=delta)).strftime(FMT))


class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []

    def check(self, ok: bool, msg: str) -> None:
        if ok:
            self.passed += 1
        else:
            self.failures.append(msg)

    def close(self, ok_msg: str, n: int | None = None) -> None:
        if not self.failures:
            print(f"  {ok_msg}" + (f"（{n} 项）" if n else ""))


def verify_primitives(rep, logs, feat, key, prefix, labels, windows, n, seed) -> None:
    """A 层：与独立重算比对。"""
    other = "video_id" if key == "user_id" else "user_id"
    w_max = max(windows)
    pool = feat.filter(pl.col(f"{prefix}_imp_{w_max}d") > 5)
    rows = pool.sample(n=min(n, len(pool)), seed=seed)
    excludes_today = 0

    for r in rows.iter_rows(named=True):
        kv, t = r[key], r["date"]
        for w in windows:
            sub = logs.filter((pl.col(key) == kv) & pl.col("date").is_between(day(t, -w), day(t, -1)))
            rep.check(len(sub) == r[f"{prefix}_imp_{w}d"],
                      f"{key}={kv} date={t} w={w} imp: 表={r[f'{prefix}_imp_{w}d']} 重算={len(sub)}")
            for c in labels:
                rep.check(int(sub[c].sum()) == r[f"{prefix}_{c}_{w}d"],
                          f"{key}={kv} date={t} w={w} {c}: 表={r[f'{prefix}_{c}_{w}d']} 重算={int(sub[c].sum())}")
        win = logs.filter((pl.col(key) == kv) & pl.col("date").is_between(day(t, -w_max), day(t, -1)))
        rep.check(win["date"].n_unique() == r[f"{prefix}_active_days_{w_max}d"],
                  f"{key}={kv} date={t} active_days: 表={r[f'{prefix}_active_days_{w_max}d']} 重算={win['date'].n_unique()}")
        rep.check(win[other].n_unique() == r[f"{prefix}_distinct_{w_max}d"],
                  f"{key}={kv} date={t} distinct: 表={r[f'{prefix}_distinct_{w_max}d']} 重算={win[other].n_unique()}")
        # 排除当天必须是实质性的：含当天后计数应当变化
        incl = logs.filter((pl.col(key) == kv) & pl.col("date").is_between(day(t, -w_max), t))
        if len(incl) > r[f"{prefix}_imp_{w_max}d"]:
            excludes_today += 1

    print(f"  A 原始量：抽查 {len(rows)} 个 (key,date)，其中 {excludes_today} 个可证伪「已排除当天」")


def verify_derived(rep, feat, prefix, labels, windows, alpha, prior, n, seed) -> None:
    """B 层：派生列与同行计数是否自洽。"""
    w_max = max(windows)
    s = feat.sample(n=min(n * 50, len(feat)), seed=seed)
    for w in windows:
        imp = pl.col(f"{prefix}_imp_{w}d")
        for c in labels:
            num = pl.col(f"{prefix}_{c}_{w}d")
            raw, sm = f"{prefix}_{c}_rate_{w}d", f"{prefix}_{c}_rate_sm_{w}d"
            bad_raw = s.filter((imp > 0) & ((pl.col(raw) - num / imp).abs() > TOL)).height
            rep.check(bad_raw == 0, f"{raw} 与 计数/曝光 不符：{bad_raw} 行")
            bad_null = s.filter((imp == 0) & pl.col(raw).is_not_null()).height
            rep.check(bad_null == 0, f"{raw} 在曝光为 0 时应为 null：{bad_null} 行")
            expect_sm = (num + alpha * prior[c]) / (imp + alpha)
            bad_sm = s.filter((pl.col(sm) - expect_sm).abs() > TOL).height
            rep.check(bad_sm == 0, f"{sm} 与平滑公式不符：{bad_sm} 行")
    if 1 in windows:
        base = pl.col(f"{prefix}_imp_{w_max}d") / w_max
        bad = s.filter((base > 0) & ((pl.col(f"{prefix}_trend") - pl.col(f"{prefix}_imp_1d") / base).abs() > TOL)).height
        rep.check(bad == 0, f"{prefix}_trend 与公式不符：{bad} 行")
    rep.check(s.filter(pl.col(f"{prefix}_has_history") != 1).height == 0,
              f"{prefix}_has_history 在特征表内应恒为 1")
    print(f"  B 派生量：{len(s):,} 行全列公式自洽")


def verify_invariants(rep, feat, prefix, labels, windows) -> None:
    """C 层：全局不变量。"""
    rep.check(feat.select(["date"]).height == feat.unique(subset=[feat.columns[0], "date"]).height,
              "(key, date) 存在重复行")
    for c in labels + ["imp"]:
        for a, b in zip(windows, windows[1:]):
            bad = feat.filter(pl.col(f"{prefix}_{c}_{a}d") > pl.col(f"{prefix}_{c}_{b}d")).height
            rep.check(bad == 0, f"{prefix}_{c}: {a}d > {b}d 的行有 {bad} 条")
    nan_cols = []
    for c, dt in feat.schema.items():
        if dt in (pl.Float32, pl.Float64):
            if feat.select(pl.col(c).is_nan().sum() + pl.col(c).is_infinite().sum()).item() > 0:
                nan_cols.append(c)
    rep.check(not nan_cols, f"存在 NaN/Inf 的列：{nan_cols}")
    print(f"  C 不变量：唯一性、窗口单调、无 NaN/Inf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    proc = project_path(require(cfg, "dataset", "processed_dir"))
    windows = sorted(require(cfg, "features", "windows"))
    labels = require(cfg, "labels", "tasks")

    full = pl.read_parquet(proc / "logs_split.parquet",
                           columns=["user_id", "video_id", "date", "split", *labels])
    prior = (
        full.filter(pl.col("split").is_in(PRIOR_SPLITS))
        .select([pl.col(c).mean().alias(c) for c in labels])
        .row(0, named=True)
    )
    logs = full.drop("split")

    rep = Report()
    for key, prefix, fname in (
        ("user_id", "user", "feat_user_daily.parquet"),
        ("video_id", "item", "feat_item_daily.parquet"),
    ):
        print(f"\n=== {prefix} 维度 ===")
        feat = pl.read_parquet(proc / fname)
        verify_primitives(rep, logs, feat, key, prefix, labels, windows, args.n, args.seed)
        verify_derived(rep, feat, prefix, labels, windows, args.alpha, prior, args.n, args.seed)
        verify_invariants(rep, feat, prefix, labels, windows)

    print(f"\n共校验 {rep.passed} 项，失败 {len(rep.failures)} 项。")
    if rep.failures:
        for f in rep.failures[:20]:
            print(f"  FAIL {f}")
        return 1
    print("T-1 特征验算通过（原始量 / 派生量 / 不变量 三层）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
