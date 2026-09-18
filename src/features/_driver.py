"""daily_user_features / daily_item_features 的共用驱动。"""

from __future__ import annotations

import argparse
import time

import polars as pl

from src.features.rolling import daily_aggregate, rolling_features
from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

# 平滑先验只能由「模型当时能看到的」数据估计，因此仅用 warmup+train。
PRIOR_SPLITS = ["warmup", "train"]


def build(key: str, prefix: str, out_name: str, module: str) -> int:
    ap = argparse.ArgumentParser(prog=module)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--alpha", type=float, default=20.0, help="贝叶斯平滑强度（虚拟曝光次数）")
    args = ap.parse_args()

    log = get_logger(module)
    cfg = load_config(args.config)
    proc = project_path(require(cfg, "dataset", "processed_dir"))
    windows = require(cfg, "features", "windows")
    labels = require(cfg, "labels", "tasks")

    src = proc / "logs_split.parquet"
    if not src.is_file():
        raise FileNotFoundError(f"{src} 不存在，请先运行 temporal_split")
    lf = pl.scan_parquet(src)

    prior_lf = lf.filter(pl.col("split").is_in(PRIOR_SPLITS))
    prior = prior_lf.select([pl.col(c).mean().alias(c) for c in labels]).collect().row(0, named=True)
    log.info("平滑先验（仅 %s 段估计，alpha=%.0f）:", "+".join(PRIOR_SPLITS), args.alpha)
    for c, v in prior.items():
        log.info("  %-12s %.5f", c, v)

    t0 = time.perf_counter()
    daily = daily_aggregate(lf, key, labels)
    log.info("日表 %s x date: %s 行", key, f"{len(daily):,}")

    feat = rolling_features(daily, key, prefix, labels, windows, args.alpha, prior)

    # 投射会生成到 max_date + w_max 的目标日；超出数据范围的行永远 join 不到样本，裁掉。
    test_end = require(cfg, "split", "test_end")
    before = len(feat)
    feat = feat.filter(pl.col("date") <= test_end)
    log.info("裁掉超出 %d 的行: %s", test_end, f"{before - len(feat):,}")

    out = proc / out_name
    feat.write_parquet(out, compression="zstd")

    n_feat = len([c for c in feat.columns if c.startswith(prefix)])
    log.info(
        "%s 行 x %d 个特征 -> %s (%.0f MB, %.1fs)",
        f"{len(feat):,}", n_feat, out.name, out.stat().st_size / 1024**2, time.perf_counter() - t0,
    )
    log.info("特征列: %s", ", ".join(c for c in feat.columns if c.startswith(prefix)))
    return 0
