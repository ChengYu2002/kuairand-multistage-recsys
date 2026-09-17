"""原始 CSV -> Parquet 缓存（plan §33「数据缓存」）。

做四件事：
  1. 按 `scene.tab_whitelist` 过滤场景；tab 0 语义不同，单独存为 holdout。
  2. 丢弃 `leakage.forbidden_fields` —— 曝光后产物不进入建模表，从物理上杜绝误用。
     如需做 post-hoc 分析，请回原始 CSV 读取。
  3. 收窄 dtype（int64 -> int8/int32），大幅减小体积与后续内存占用。
  4. 由 time_ms 重算 `date`，使日期与时间戳构造上一致（见下）。
  5. 落盘 Parquet：列式 + 压缩 + 带类型，后续每一步都不必再解析 CSV。

关于 date 的重算：原始 `date` 列把 23 点之后的记录算作第二天（实测 9,015,279 行中
有 57,340 行如此，占 0.64%，且 100% 发生在 23 点）。这个偏差方向一致，本身不产生
泄漏，但会让「Day T 的样本只用 <= T-1 的聚合」这句话失去唯一解释——而特征泄漏是
静默的。因此统一以 time_ms（UTC）加时区偏移重算日期，原始值保留为 `date_raw` 备查。

用法：
    python -m src.preprocessing.preprocess --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import time

import polars as pl

from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)

# 收窄后的类型；未列出的列保持推断结果。
NARROW = {
    "user_id": pl.Int32,
    "video_id": pl.Int32,
    "date": pl.Int32,
    "hourmin": pl.Int16,
    "duration_ms": pl.Int32,
    "tab": pl.Int8,
    "is_rand": pl.Int8,
}
LABEL_COLS = ["is_click", "long_view", "is_like", "is_follow", "is_comment", "is_forward", "is_hate"]


def _rebuild_date(lf: pl.LazyFrame, offset_hours: int) -> pl.LazyFrame:
    """用 time_ms 重算本地日期；保留原始值为 date_raw。"""
    cols = set(lf.collect_schema().keys())
    if not {"date", "time_ms"} <= cols:
        return lf
    local = pl.from_epoch("time_ms", time_unit="ms") + pl.duration(hours=offset_hours)
    return lf.with_columns(pl.col("date").alias("date_raw")).with_columns(
        local.dt.strftime("%Y%m%d").cast(pl.Int32).alias("date"),
        (local.dt.hour().cast(pl.Int16) * 100 + local.dt.minute().cast(pl.Int16)).alias("hourmin"),
    )


def _narrow(lf: pl.LazyFrame) -> pl.LazyFrame:
    cols = set(lf.collect_schema().keys())
    casts = [pl.col(c).cast(t) for c, t in NARROW.items() if c in cols]
    casts += [pl.col(c).cast(pl.Int8) for c in LABEL_COLS if c in cols]
    return lf.with_columns(casts) if casts else lf


def _scan(raw_dir, pattern: str) -> pl.LazyFrame:
    paths = sorted(raw_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"{raw_dir} 下没有匹配 {pattern} 的文件")
    log.info("读取 %d 个文件: %s", len(paths), ", ".join(p.name for p in paths))
    return pl.concat([pl.scan_csv(p) for p in paths], how="vertical_relaxed")


def _write(lf: pl.LazyFrame, out, label: str) -> int:
    t0 = time.perf_counter()
    df = lf.collect()
    df.write_parquet(out, compression="zstd")
    mb = out.stat().st_size / 1024**2
    log.info("%-16s %10s 行 -> %s (%.0f MB, %.1fs)", label, f"{len(df):,}", out.name, mb, time.perf_counter() - t0)
    return len(df)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    raw_dir = project_path(require(cfg, "dataset", "raw_dir"))
    out_dir = project_path(require(cfg, "dataset", "processed_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    whitelist = require(cfg, "scene", "tab_whitelist")
    holdout = cfg.get("scene", {}).get("holdout_tabs", [])
    forbidden = cfg.get("leakage", {}).get("forbidden_fields", [])

    tz = require(cfg, "split", "timezone_offset_hours")
    logs = _rebuild_date(_narrow(_scan(raw_dir, require(cfg, "dataset", "log_standard"))), tz)
    present = set(logs.collect_schema().keys())
    drop = [c for c in forbidden if c in present]
    if drop:
        log.info("丢弃曝光后字段（§7.1）: %s", ", ".join(drop))
        logs = logs.drop(drop)

    total = logs.select(pl.len()).collect().item()
    n_main = _write(logs.filter(pl.col("tab").is_in(whitelist)), out_dir / "logs_main.parquet", "主建模 tab" + str(whitelist))
    if holdout:
        _write(logs.filter(pl.col("tab").is_in(holdout)), out_dir / "logs_holdout.parquet", f"holdout tab{holdout}")
    log.info("主表覆盖原始日志的 %.1f%%（%s / %s）", 100 * n_main / total, f"{n_main:,}", f"{total:,}")

    rnd = cfg.get("dataset", {}).get("log_random")
    if rnd:
        r = _rebuild_date(_narrow(_scan(raw_dir, rnd)), tz)
        r = r.drop([c for c in forbidden if c in set(r.collect_schema().keys())])
        _write(r, out_dir / "logs_random.parquet", "随机曝光")

    _write(_scan(raw_dir, "video_features_basic*.csv"), out_dir / "video_features.parquet", "视频特征")
    _write(_scan(raw_dir, "user_features*.csv"), out_dir / "user_features.parquet", "用户特征")

    log.info("完成，输出目录 %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
