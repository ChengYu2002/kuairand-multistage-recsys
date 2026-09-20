"""用户最近 N 个正向行为序列（plan §9）。

用户塔不使用 user_id embedding —— 本数据集只有 1000 个用户，ID embedding 会被直接
记住而无法泛化（§13.1）。改用「最近看过什么」来表示用户，因此需要为每条曝光取出


该用户在**此之前**的最近 N 个正向 item。
该用户在**此之前**的最近 N 个正向 item。
该用户在**此之前**的最近 N 个正向 item。


正确性要点一 —— 严格早于：历史必须排除当前曝光本身，否则模型直接看到答案。
实现上以 time_ms - 1 作为 as-of 连接的左键，从而排除同刻及其后的一切记录。

正确性要点二 —— 时间戳是请求级的：实测 4,117,844 个正向事件只落在 1,398,316 个
(user, time_ms) 上，75.5% 的时间戳承载多个事件（一次请求返回一批视频，共用同一毫秒）。
因此批次内部无法定序，整批排除是唯一安全的口径 —— 这正是上面减 1ms 的效果。
并列还会让排序结果不唯一，进而破坏可复现性（§26 的 3 seeds 要求数据完全一致），
故一律以 (user_id, time_ms, video_id) 作为全序键。

不使用逐行 Python 循环（§9）：先在正向事件上用 shift 构造 N 列历史，再用
join_asof 把每条曝光对齐到它之前最近的那个正向事件。

用法：
    python -m src.preprocessing.build_history --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import time

import polars as pl

from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    n = require(cfg, "history", "max_len")
    signals = require(cfg, "history", "positive_signal")
    proc = project_path(require(cfg, "dataset", "processed_dir"))
    src = proc / "logs_split.parquet"
    if not src.is_file():
        raise FileNotFoundError(f"{src} 不存在，请先运行 temporal_split")

    t0 = time.perf_counter()
    logs = pl.read_parquet(src, columns=["user_id", "video_id", "time_ms", "split", *signals])

    positive = pl.any_horizontal([pl.col(c) == 1 for c in signals])
    pos = (
        logs.filter(positive)
        .select(["user_id", "video_id", "time_ms"])
        .sort(["user_id", "time_ms", "video_id"])
    )
    log.info("正向事件（%s）: %s / %s 行 (%.1f%%)",
             " or ".join(signals), f"{len(pos):,}", f"{len(logs):,}", 100 * len(pos) / len(logs))

    # shift(0) 是该正向事件自身，shift(i) 是它之前第 i 个；对齐时左键已减 1ms，
    # 因此 shift(0) 对当前曝光而言仍是严格过去的事件。
    pos = pos.with_columns(
        [pl.col("video_id").shift(i).over("user_id").alias(f"h{i}") for i in range(n)]
    ).with_columns(
        pl.concat_list([f"h{i}" for i in range(n)]).list.drop_nulls().alias("hist")
    ).select(["user_id", "time_ms", "hist"])

    out = (
        # 用 t-1 当连接键，所以只匹配严格早于 t 的事件。
        # 如果这里写成 <=，正向样本会在自己的历史里看到自己，AUC 直接虚高。这是最典型的静默泄漏。
        logs.sort(["user_id", "time_ms", "video_id"])
        .with_columns((pl.col("time_ms") - 1).alias("_key"))
        .join_asof(
            pos.rename({"time_ms": "_key"}),
            on="_key",
            by="user_id",
            strategy="backward",
        )
        .with_columns(
            pl.col("hist").fill_null(pl.lit([], dtype=pl.List(pl.Int32))),
        )
        .with_columns(pl.col("hist").list.len().cast(pl.Int16).alias("hist_len"))
        .select(["user_id", "video_id", "time_ms", "split", "hist", "hist_len"])
    )

    dst = proc / "user_history.parquet"
    out.write_parquet(dst, compression="zstd")
    log.info("%s 行 -> %s (%.0f MB, %.1fs)",
             f"{len(out):,}", dst.name, dst.stat().st_size / 1024**2, time.perf_counter() - t0)

    stat = out.filter(pl.col("split") != "warmup").select(
        pl.col("hist_len").mean().alias("mean"),
        pl.col("hist_len").median().alias("median"),
        (pl.col("hist_len") == 0).mean().alias("empty"),
        (pl.col("hist_len") == n).mean().alias("full"),
    ).row(0, named=True)
    log.info("非 warmup 样本的历史长度：均值 %.1f  中位 %.0f  空 %.2f%%  满(%d) %.1f%%",
             stat["mean"], stat["median"], 100 * stat["empty"], n, 100 * stat["full"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
