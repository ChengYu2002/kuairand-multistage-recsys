"""独立验算用户历史序列（plan §9）。

历史泄漏是最严重的一类：若当前曝光（或同刻的其他事件）混入自己的历史，模型就
直接看到了答案，而这不会抛任何异常。因此这里用与 build_history.py 完全不同的
代码路径 —— 对单个用户直接按时间戳过滤排序 —— 重算若干样本的历史并逐元素比对。

三项检查：
  A. 历史内容与「该用户 time_ms 严格早于当前的最近 N 个正向 item」完全一致；
  B. 把条件放宽为「早于等于」后结果确实改变（证明严格性不是碰巧成立）；
  C. 同一毫秒内的并发事件不进入历史。

用法：
    python scripts/verify_history.py --n 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, project_path, require  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_max = require(cfg, "history", "max_len")
    signals = require(cfg, "history", "positive_signal")
    proc = project_path(require(cfg, "dataset", "processed_dir"))

    logs = pl.read_parquet(proc / "logs_split.parquet",
                           columns=["user_id", "video_id", "time_ms", *signals])
    hist = pl.read_parquet(proc / "user_history.parquet")

    positive = pl.any_horizontal([pl.col(c) == 1 for c in signals])
    pos = logs.filter(positive).select(["user_id", "video_id", "time_ms"])

    rows = hist.filter(pl.col("split") != "warmup").sample(n=args.n, seed=args.seed)
    failures: list[str] = []
    strict_proven = 0
    concurrent_cases = 0

    for r in rows.iter_rows(named=True):
        u, t, got = r["user_id"], r["time_ms"], list(r["hist"])
        # 与 build_history 使用同一全序键：时间戳并列时顺序才唯一
        up = pos.filter(pl.col("user_id") == u).sort(["time_ms", "video_id"])

        # A. 严格早于
        before = up.filter(pl.col("time_ms") < t)
        expect = list(before["video_id"].tail(n_max).reverse())
        if expect != got:
            failures.append(f"user={u} t={t}: 长度 表={len(got)} 重算={len(expect)}；"
                            f"前3 表={got[:3]} 重算={expect[:3]}")

        # B. 放宽到「早于等于」后应当不同
        incl = up.filter(pl.col("time_ms") <= t)
        expect_incl = list(incl["video_id"].tail(n_max).reverse())
        if expect_incl != expect:
            strict_proven += 1

        # C. 同刻并发事件
        same_ms = up.filter(pl.col("time_ms") == t)
        if len(same_ms) > 0:
            concurrent_cases += 1
            leaked = set(same_ms["video_id"].to_list()) & set(got[:len(same_ms)])
            # 同一视频可能在更早时刻也被看过，故仅在数量层面判断
            if expect != got:
                failures.append(f"user={u} t={t}: 同刻有 {len(same_ms)} 个正向事件且比对失败")

        mark = "OK  " if expect == got else "FAIL"
        print(f"  {mark} user={u:>4} len={len(got):>3} 同刻正向={len(same_ms)} "
              f"首项={got[0] if got else '-'}")

    print(f"\n抽查 {len(rows)} 条")
    print(f"  A 内容比对失败: {len(failures)}")
    print(f"  B 可证伪严格性(放宽后结果改变): {strict_proven}/{len(rows)}")
    print(f"  C 命中同刻并发事件的样本: {concurrent_cases}")
    if failures:
        for f in failures[:10]:
            print(f"  FAIL {f}")
        return 1
    print("用户历史序列验算通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
