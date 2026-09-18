"""独立验算 T-1 特征（plan §8 的正确性保障）。

特征泄漏是**静默**的：不会抛异常，只会让线下指标虚高。因此这里用一条
与 src/features/rolling.py 完全不同的代码路径重算若干样本 ——
直接按日期区间过滤原始日志再求和 —— 并与特征表逐值比对。

同时显式检查两件事：
  * 「过去 w 天」严格取 [target-w, target-1]，不含 target 当天；
  * 把区间改成含当天后结果确实会变，以证明上面那条不是因为恰好相等而通过。

用法：
    python scripts/verify_t1_features.py --n 5
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, project_path, require  # noqa: E402

FMT = "%Y%m%d"


def day(d: int, delta: int) -> int:
    return int((datetime.strptime(str(d), FMT) + timedelta(days=delta)).strftime(FMT))


def recompute(logs: pl.DataFrame, key: str, kval: int, target: int, w: int, labels: list[str]) -> dict:
    """完全独立的实现：直接按 [target-w, target-1] 过滤求和。"""
    lo, hi = day(target, -w), day(target, -1)
    sub = logs.filter((pl.col(key) == kval) & pl.col("date").is_between(lo, hi))
    return {"imp": len(sub), **{c: int(sub[c].sum()) for c in labels}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--n", type=int, default=5, help="每个维度抽查多少个 (key, date)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    proc = project_path(require(cfg, "dataset", "processed_dir"))
    windows = require(cfg, "features", "windows")
    labels = require(cfg, "labels", "tasks")
    logs = pl.read_parquet(proc / "logs_split.parquet", columns=["user_id", "video_id", "date", *labels])
    random.seed(args.seed)

    failures = 0
    checked = 0
    for key, prefix, fname in (
        ("user_id", "user", "feat_user_daily.parquet"),
        ("video_id", "item", "feat_item_daily.parquet"),
    ):
        feat = pl.read_parquet(proc / fname)
        # 只抽活跃样本，避免全 0 的平凡情形掩盖错误
        pool = feat.filter(pl.col(f"{prefix}_imp_{max(windows)}d") > 5)
        rows = pool.sample(n=min(args.n, len(pool)), seed=args.seed)
        print(f"\n=== {prefix} 维度：抽查 {len(rows)} 个 (key, date) ===")

        for r in rows.iter_rows(named=True):
            kval, target = r[key], r["date"]
            for w in windows:
                exp = recompute(logs, key, kval, target, w, labels)
                got = {"imp": r[f"{prefix}_imp_{w}d"], **{c: r[f"{prefix}_{c}_{w}d"] for c in labels}}
                checked += 1
                if exp != got:
                    failures += 1
                    print(f"  FAIL {key}={kval} date={target} w={w}")
                    for k in exp:
                        if exp[k] != got[k]:
                            print(f"       {k}: 特征表={got[k]}  独立重算={exp[k]}")
            # 反向检查：把区间改成含当天，结果应当不同（否则说明当天本就无数据，抽样无效）
            w = max(windows)
            incl = logs.filter((pl.col(key) == kval) & pl.col("date").is_between(day(target, -w), target))
            same_day = len(incl) - r[f"{prefix}_imp_{w}d"]
            flag = "含当天会多 %d 行 -> 确实排除了当天" % same_day if same_day > 0 else "当天无数据（此样本无法证伪）"
            print(f"  OK   {key}={kval} date={target}  imp_7d={r[f'{prefix}_imp_{w}d']:>6}  {flag}")

    print(f"\n比对 {checked} 组计数，失败 {failures} 组。")
    if failures:
        print("T-1 特征与独立重算不一致，请检查 src/features/rolling.py")
        return 1
    print("T-1 特征验算通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
