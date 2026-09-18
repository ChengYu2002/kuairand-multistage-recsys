"""原始 CSV -> Parquet 缓存（plan §33「数据缓存」）。

做五件事：
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
    # 只有日志里同时存在这两列时才能重算；用户表、视频表不需要处理。
    cols = set(lf.collect_schema().keys())
    if not {"date", "time_ms"} <= cols:
        return lf
    # Unix 时间戳按 UTC 解释，再加 8 小时得到快手日志对应的北京时间。
    local = pl.from_epoch("time_ms", time_unit="ms") + pl.duration(hours=offset_hours)
    # 先备份官方 date为date_raw，再用同一个时间戳同时生成 date 和 hourmin，避免两列互相矛盾。
    return lf.with_columns(pl.col("date").alias("date_raw")).with_columns(
        local.dt.strftime("%Y%m%d").cast(pl.Int32).alias("date"),
        (local.dt.hour().cast(pl.Int16) * 100 + local.dt.minute().cast(pl.Int16)).alias("hourmin"),
    )


def _narrow(lf: pl.LazyFrame) -> pl.LazyFrame:
    # 只转换当前文件真正拥有的列，避免不同类型的 CSV 因缺列而报错。
    cols = set(lf.collect_schema().keys())
    casts = [pl.col(c).cast(t) for c, t in NARROW.items() if c in cols]
    # 所有行为标签都是 0/1，用 Int8 足够，能明显降低内存占用。
    casts += [pl.col(c).cast(pl.Int8) for c in LABEL_COLS if c in cols]
    return lf.with_columns(casts) if casts else lf


def _scan(raw_dir, pattern: str) -> pl.LazyFrame:
    # 在指定目录中查找所有匹配的 CSV 分片。
    paths = sorted(raw_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"{raw_dir} 下没有匹配 {pattern} 的文件")
    log.info("读取 %d 个文件: %s", len(paths), ", ".join(p.name for p in paths))
    # 惰性读取：先记录“要做哪些操作”，暂时不真正读取和计算数据。
    return pl.concat([pl.scan_csv(p) for p in paths], how="vertical_relaxed")


def _write(lf: pl.LazyFrame, out, label: str) -> int:
    t0 = time.perf_counter()
    # collect() 会真正读取 CSV 并执行前面记录的过滤、删列和类型转换。
    df = lf.collect()
    # Parquet 保留数据类型，后续读取比 CSV 更快；zstd 负责压缩体积。
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

    # 主实验真正使用的数据，例如当前的 tab 1/2/4。
    whitelist = require(cfg, "scene", "tab_whitelist")
    # 暂时隔离、不参与主训练，留到以后检查跨场景表现。
    holdout = cfg.get("scene", {}).get("holdout_tabs", [])
    # 这些列是曝光后才知道的结果，必须从建模表中删除。
    forbidden = cfg.get("leakage", {}).get("forbidden_fields", [])

    # 所有日期都用同一个时区规则，避免切分和 T-1 特征使用不同日期。
    tz = require(cfg, "split", "timezone_offset_hours")
    logs = _rebuild_date(_narrow(_scan(raw_dir, require(cfg, "dataset", "log_standard"))), tz)
    # 只删除实际存在的禁用列，避免某个数据版本缺列时报错。
    present = set(logs.collect_schema().keys())
    drop = [c for c in forbidden if c in present]
    if drop:
        log.info("丢弃曝光后字段（§7.1）: %s", ", ".join(drop))
        logs = logs.drop(drop)

    # 标准日志一份用于主实验，一份作为场景 holdout；两者不会混在一起训练。
    # Polars 这种 DataFrame 框架本质上是在做向量化/列式运算，不是让你自己一行一行 for
    total = logs.select(pl.len()).collect().item()
    # 这部分写成 logs_main.parquet
    n_main = _write(logs.filter(pl.col("tab").is_in(whitelist)), out_dir / "logs_main.parquet", "主建模 tab" + str(whitelist))
    if holdout:
        # 这部分写成‘ogs_holdout.parquet
        _write(logs.filter(pl.col("tab").is_in(holdout)), out_dir / "logs_holdout.parquet", f"holdout tab{holdout}")
    log.info("主表覆盖原始日志的 %.1f%%（%s / %s）", 100 * n_main / total, f"{n_main:,}", f"{total:,}")

    rnd = cfg.get("dataset", {}).get("log_random")
    if rnd:
        # 随机曝光日志单独保存，后续只用于去偏或鲁棒性实验。
        r = _rebuild_date(_narrow(_scan(raw_dir, rnd)), tz)
        r = r.drop([c for c in forbidden if c in set(r.collect_schema().keys())])
        # 这部分写成 logs_random.parquet
        _write(r, out_dir / "logs_random.parquet", "随机曝光")

    # 用户表和视频表是静态侧信息，只做 CSV -> Parquet，不参与 tab 过滤。
    # 写用户特征信息
    _write(_scan(raw_dir, "video_features_basic*.csv"), out_dir / "video_features.parquet", "视频特征")
    _write(_scan(raw_dir, "user_features*.csv"), out_dir / "user_features.parquet", "用户特征")

    log.info("完成，输出目录 %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
