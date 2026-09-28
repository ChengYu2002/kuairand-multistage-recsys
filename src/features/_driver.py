"""daily_user_features / daily_item_features 的共用驱动。"""

from __future__ import annotations

import argparse
import time

import polars as pl

from src.features.rolling import daily_aggregate, distinct_pairs, rolling_features
from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

# 平滑先验只能由「样本当时已经发生的」数据估计。warmup 段（04-08~04-14）整体早于
# train 起点（04-15），因此由它估计的先验对 train/valid/test 的每一条样本都是过去数据。
# 若改用 warmup+train，先验会包含同段内更晚日期的标签，严格意义上违反 Day T <= T-1。
PRIOR_SPLITS = ["warmup"]

# key：按哪个实体统计
# prefix：输出特征用什么开头
# out_name：结果保存成什么文件
# module：日志里显示哪个模块正在运行
def build(key: str, prefix: str, out_name: str, module: str) -> int:
    ap = argparse.ArgumentParser(prog=module)
    ap.add_argument("--config", default="configs/data.yaml")
    # 默认值改为 None：真正的取值来自 config，命令行只用于临时覆盖做实验。
    ap.add_argument("--alpha", type=float, default=None, help="覆盖 config 的 features.smooth_alpha")
    args = ap.parse_args()

    log = get_logger(module)
    cfg = load_config(args.config)
    # 读取 configs/data.yaml
    proc = project_path(require(cfg, "dataset", "processed_dir"))
    # require() 的意思是：这个配置必须存在；如果缺少，就直接报错。
    windows = require(cfg, "features", "windows")
    alpha = args.alpha if args.alpha is not None else require(cfg, "features", "smooth_alpha")
    labels = require(cfg, "labels", "tasks")

    src = proc / "logs_split.parquet"
    if not src.is_file():
        raise FileNotFoundError(f"{src} 不存在，请先运行 temporal_split")
    # LazyFrame
    lf = pl.scan_parquet(src)

    # 分别计算五个行为标签在 warmup 中的平均值。
    # 只筛选出属于 warmup 时间段的数据
    prior_lf = lf.filter(pl.col("split").is_in(PRIOR_SPLITS))

    # 根据label 各计算平均值
    # 最后只有一行结果
    prior = prior_lf.select([pl.col(c).mean().alias(c) for c in labels]).collect().row(0, named=True)
    log.info("平滑先验（仅 %s 段估计，alpha=%.0f）:", "+".join(PRIOR_SPLITS), alpha)

    for c, v in prior.items():
        log.info("  %-12s %.5f", c, v)
    
    # 记录运行时间
    t0 = time.perf_counter()
    daily = daily_aggregate(lf, key, labels)
    log.info("日表 %s x date: %s 行", key, f"{len(daily):,}")

    pairs = distinct_pairs(lf, key)
    log.info("去重 (%s, other, day) 三元组: %s 行", key, f"{len(pairs):,}")

    feat = rolling_features(daily, pairs, key, prefix, labels, windows, alpha, prior)

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
