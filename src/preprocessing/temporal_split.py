"""Global Temporal Split（plan §6）。

关键实现细节：**按 time_ms 切，而不是按 date 切**。

KuaiRand 的 `date` 是北京时间（UTC+8），`time_ms` 是 UTC 毫秒时间戳，两者在跨零点
附近有约 0.52% 的记录不一致。若按 date 切分，会出现「标为 train 的行实际发生时间
晚于标为 valid 的行」，即真实的时间泄漏。改用 time_ms 后，§6.2 要求的
max(train) < min(valid) < min(test) 是构造上成立的。

warmup 段只用于生成 T-1 特征（§8.2 的 7 天窗口），不产出训练样本。

用法：
    python -m src.preprocessing.temporal_split --config configs/data.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import polars as pl

from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)

ORDER = ["warmup", "train", "valid", "test", "unused"]


def end_of_day_ms(yyyymmdd: int, offset_hours: int) -> int:
    """返回该本地日期结束时刻（次日 00:00，本地时区）对应的 UTC 毫秒时间戳。"""
    d = datetime.strptime(str(yyyymmdd), "%Y%m%d").replace(
        tzinfo=timezone(timedelta(hours=offset_hours))
    )
    return int((d + timedelta(days=1)).timestamp() * 1000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    off = require(cfg, "split", "timezone_offset_hours")
    bounds = {k: require(cfg, "split", k) for k in ("warmup_end", "train_end", "valid_end", "test_end")}
    if not (bounds["warmup_end"] < bounds["train_end"] < bounds["valid_end"] < bounds["test_end"]):
        raise ValueError(f"切分日期不是递增的: {bounds}")
    cut = {k: end_of_day_ms(v, off) for k, v in bounds.items()}
    log.info("切分边界（UTC+%d 当日 24:00 -> UTC ms）:", off)
    for k, v in bounds.items():
        log.info("  %-11s %s -> %d", k, v, cut[k])

    proc = project_path(require(cfg, "dataset", "processed_dir"))
    src = proc / "logs_main.parquet"
    if not src.is_file():
        raise FileNotFoundError(f"{src} 不存在，请先运行 src.preprocessing.preprocess")

    t = pl.col("time_ms")
    df = pl.read_parquet(src).with_columns(
        pl.when(t < cut["warmup_end"]).then(pl.lit("warmup"))
        .when(t < cut["train_end"]).then(pl.lit("train"))
        .when(t < cut["valid_end"]).then(pl.lit("valid"))
        .when(t < cut["test_end"]).then(pl.lit("test"))
        .otherwise(pl.lit("unused"))
        .alias("split")
    )

    stats = df.group_by("split").agg(
        pl.len().alias("rows"),
        pl.col("time_ms").min().alias("t_min"),
        pl.col("time_ms").max().alias("t_max"),
        pl.col("user_id").n_unique().alias("users"),
        pl.col("video_id").n_unique().alias("items"),
        pl.col("is_click").sum().alias("clicks"),
    )
    by = {r["split"]: r for r in stats.iter_rows(named=True)}

    def local(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, timezone(timedelta(hours=off))).strftime("%m-%d %H:%M")

    print()
    print(f"{'split':<8}{'行数':>12}{'起':>13}{'止':>13}{'用户':>7}{'item':>10}{'正向事件':>12}")
    for name in ORDER:
        if name not in by:
            continue
        r = by[name]
        print(f"{name:<8}{r['rows']:>12,}{local(r['t_min']):>13}{local(r['t_max']):>13}"
              f"{r['users']:>7,}{r['items']:>10,}{r['clicks']:>12,}")
    print()

    for a, b in (("warmup", "train"), ("train", "valid"), ("valid", "test")):
        if a in by and b in by:
            if by[a]["t_max"] >= by[b]["t_min"]:
                raise AssertionError(f"时间泄漏：max({a}) >= min({b})")
            log.info("OK  max(%s) < min(%s)", a, b)

    out = proc / "logs_split.parquet"
    df.write_parquet(out, compression="zstd")
    log.info("已写出 %s (%.0f MB)", out.name, out.stat().st_size / 1024**2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
