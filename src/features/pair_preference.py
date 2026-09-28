"""User x Author 偏好特征（plan §8.4）。T-1 口径，只用过去日志。

## 为什么需要它

Single-Task 基线在 valid 上的 GAUC：is_follow 0.49014、is_comment 0.50537 —— 在**同一个
用户内部**排序时与随机无异。那两个任务漂亮的 AUC（0.826 / 0.882）几乎全部来自"哪些用户
爱关注"，而不是"什么视频值得关注"。

原因是模型完全没有 user x author 交互特征：用户侧只有历史视频的**池化均值**，物品侧只有
作者 embedding，要靠 MLP 从一个平均向量里反推"这两个是不是同一个作者"极其困难。

实测（valid，单特征 GAUC）证明信号存在、只是没进 x：

    ua_imp_7d（过去 7 天该用户看该作者几次）  is_follow 0.53784   is_comment 0.53764
    模型整体                                  is_follow 0.49014   is_comment 0.50537

一个计数特征打赢了整个 3,750 万参数的模型。

## 口径与 user/item 日表的差异（都是实测驱动的，不是随手定的）

**窗口只取 3/7 天，不取 1 天。** (user, author, date) 日表有 8,371,862 行，而原始日志只有
9,015,279 行 —— 平均每个三元组只出现 1.08 次，即"一个用户一天基本只看某作者一次"。
1 天窗口对约 96% 的样本恒为 0，是一列死特征。

**只给 is_click / long_view 算平滑比率。** pair 维度的曝光量是 0~4，给 is_follow（0.11%）
这种事件算比率是纯噪声。三个稀疏标签的**计数**也只保留 is_like，comment / follow 的计数
在 >99.9% 的行上恒为 0。

**不产出朴素比率**，只留平滑比率。imp=0 时朴素比率是 null，而平滑公式在 num=imp=0 处恰好
等于全局先验 g —— 缺失填充与公式天然一致（与 feature_encoder 同一个约定）。

## 内存：semi-join 是必须的，不是优化

投射 ×7 是 58,603,034 行、聚合后 50,956,646 行（约 1.7 GB）。若照搬 user/item 那套
"3 窗口 × 5 标签 × 2 种比率"会宽到 10 GB 量级。因此在聚合**之前**用 semi-join 砍到样本
真正需要的 (user, author, 目标日) 上（train 只有 4,168,620 个）。语义上无损：聚合本就按
该键分组，整组丢掉不影响其余组。

## 落盘按样本行，不走行号间接

user/item 特征走 feat_row_* 的行号间接，是因为一个 (user, date) 会被几百条样本共用。
pair 维度没有这个性质：train 有 4,168,620 个不同的键对 4,496,306 条样本，几乎一一对应，
间接层省不到东西。所以直接产出与样本行对齐的 float32 矩阵。

## 编码：填充与标准化只用 train

    计数        缺历史填 0（语义正确：没有曝光），再 log1p + 标准化
    平滑比率    缺历史填全局先验 g（= 公式在 num=imp=0 处的值），再标准化
    has_history 0/1，让模型区分"统计值是 0"和"完全没有近期历史"

均值/标准差全部由 **train 段样本**估计（曝光加权，与 encoder_spec 的
stats_from.basis = train_samples 同口径）。

用法：
    python -m src.features.pair_preference --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from src.features.rolling import (
    daily_aggregate,
    distinct_pairs,
    rolling_features,
    to_date,
)
from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)

PREFIX = "ua"
KEY = ["user_id", "author_id"]
OTHER = "video_id"                       # 该用户在窗口内看过该作者的多少个不同视频
# 先验只能由「样本当时已经发生的」数据估计。warmup 段整体早于 train 起点，
# 因此由它估计的先验对每一条样本都是过去数据（与 _driver.PRIOR_SPLITS 同一口径）。
PRIOR_SPLITS = ["warmup"]
SPLITS = ("train", "valid", "test")


def _feature_columns(windows: list[int], labels: list[str],
                     rate_labels: list[str], w_max: int) -> list[str]:
    """进模型的列，顺序固定 —— 改顺序等于改输入含义。"""
    cols = [f"{PREFIX}_imp_{w}d" for w in windows]
    cols += [f"{PREFIX}_{c}_{w}d" for w in windows for c in labels]
    cols += [f"{PREFIX}_{c}_rate_sm_{w}d" for w in windows for c in rate_labels]
    cols += [f"{PREFIX}_distinct_{w_max}d", f"{PREFIX}_active_days_{w_max}d",
             f"{PREFIX}_has_history"]
    return cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    t0 = time.perf_counter()

    proc = project_path(require(cfg, "dataset", "processed_dir"))
    pc = require(cfg, "pair_preference")
    windows = list(pc["windows"])
    labels = list(pc["labels"])
    rate_labels = list(pc["rate_labels"])
    alpha = float(require(cfg, "features", "smooth_alpha"))
    all_labels = require(cfg, "labels", "tasks")
    if not set(labels) <= set(all_labels) or not set(rate_labels) <= set(labels):
        raise ValueError(
            f"pair_preference.labels {labels} 必须是 labels.tasks 的子集，"
            f"rate_labels {rate_labels} 必须是 labels 的子集")
    w_max = max(windows)

    vf = pl.read_parquet(proc / "video_features.parquet").select("video_id", "author_id")
    logs = pl.scan_parquet(proc / "logs_split.parquet")

    # ---- 先验：只由 warmup 段估计 ----
    prior = (logs.filter(pl.col("split").is_in(PRIOR_SPLITS))
             .select([pl.col(c).mean().alias(c) for c in labels])
             .collect().row(0, named=True))
    log.info("平滑先验（仅 %s 段，alpha=%.0f）: %s", "+".join(PRIOR_SPLITS), alpha,
             "  ".join(f"{k} {v:.5f}" for k, v in prior.items()))

    # ---- 样本真正需要的 (user, author, 目标日) ----
    need = []
    samples = {}
    for sp in SPLITS:
        s = pl.read_parquet(proc / f"samples_{sp}_{a.protocol}.parquet",
                            columns=["user_id", "video_id", "date"])
        s = s.join(vf, on="video_id", how="left")
        samples[sp] = s
        need.append(s.select("user_id", "author_id", to_date().alias("target")))
    need_keys = pl.concat(need).unique()
    log.info("样本需要的 (user, author, 目标日) 键: %s 个", f"{len(need_keys):,}")

    # ---- 日表与滚动统计（semi-join 在 rolling_features 内部、聚合之前生效）----
    src = logs.select("user_id", "video_id", "date", *labels).join(
        vf.lazy(), on="video_id", how="left")
    daily = daily_aggregate(src, KEY, labels)
    log.info("(user, author, date) 日表: %s 行", f"{len(daily):,}")
    pairs = distinct_pairs(src, KEY, other=OTHER)
    log.info("去重 (user, author, %s, day) 四元组: %s 行", OTHER, f"{len(pairs):,}")

    feat = rolling_features(daily, pairs, KEY, PREFIX, labels, windows, alpha, prior,
                            other=OTHER, rate_labels=rate_labels, need_keys=need_keys)
    del daily, pairs
    log.info("滚动特征: %s 行 x %d 列（%.1fs）", f"{len(feat):,}", len(feat.columns),
             time.perf_counter() - t0)

    cols = _feature_columns(windows, labels, rate_labels, w_max)
    missing = [c for c in cols if c not in feat.columns]
    if missing:
        raise AssertionError(f"滚动特征缺列 {missing}（实际有 {sorted(feat.columns)}）")
    feat = feat.select(*KEY, "date", *cols)

    # ---- 按样本行 join，缺历史按语义填充 ----
    fills: dict[str, float] = {}
    for c in cols:
        if c.endswith("_has_history"):
            fills[c] = 0.0
        elif "_rate_sm_" in c:
            # 平滑公式在 num=imp=0 处恰好等于先验 g，所以缺失也填 g，天然一致
            lab = c[len(PREFIX) + 1:c.index("_rate_sm_")]
            fills[c] = float(prior[lab])
        else:
            fills[c] = 0.0                      # 没有曝光，计数就是 0

    joined = {}
    for sp in SPLITS:
        j = (samples[sp].with_row_index("_i")
             .join(feat, on=["user_id", "author_id", "date"], how="left")
             .sort("_i"))
        if len(j) != len(samples[sp]):
            raise AssertionError(f"{sp} join 后行数变了：{len(j)} vs {len(samples[sp])}")
        cov = float(j.get_column(f"{PREFIX}_has_history").is_not_null().mean())
        j = j.with_columns([pl.col(c).fill_null(fills[c]) for c in cols])
        joined[sp] = j
        log.info("%-5s %s 行，有 user-author 近 %d 天历史 %.4f",
                 sp, f"{len(j):,}", w_max, cov)

    # ---- 编码：log1p(计数) + 标准化；统计量只由 train 估计 ----
    spec_cols: dict[str, dict] = {}
    tr = joined["train"]
    for c in cols:
        x = tr.get_column(c).to_numpy().astype(np.float64)
        if c.endswith("_has_history"):
            spec_cols[c] = {"kind": "flag", "fill": 0.0, "transform": "none"}
            continue
        transform = "none" if "_rate_sm_" in c else "log1p"
        z = np.log1p(x) if transform == "log1p" else x
        mean, std = float(z.mean()), float(z.std())
        if std <= 0:
            # 恒定列没有信息，但也不能用 0 去除。记下来，编码时置 0。
            log.info("  %-28s 在 train 上是常数（std=0），编码为 0", c)
            spec_cols[c] = {"kind": "numeric", "fill": fills[c], "transform": transform,
                            "mean": mean, "std": 0.0, "constant": True}
        else:
            spec_cols[c] = {"kind": "numeric", "fill": fills[c], "transform": transform,
                            "mean": mean, "std": std}

    for sp in SPLITS:
        m = np.empty((len(joined[sp]), len(cols)), np.float32)
        for k, c in enumerate(cols):
            x = joined[sp].get_column(c).to_numpy().astype(np.float64)
            sc = spec_cols[c]
            if sc["kind"] == "flag":
                m[:, k] = x
            elif sc.get("constant"):
                m[:, k] = 0.0
            else:
                z = np.log1p(x) if sc["transform"] == "log1p" else x
                m[:, k] = (z - sc["mean"]) / sc["std"]
        if not np.isfinite(m).all():
            raise AssertionError(f"{sp} 编码后含 NaN/Inf")
        out = proc / f"feat_pair_{sp}_{a.protocol}.npy"
        np.save(out, m)
        log.info("%-5s -> %s  %s x %d  (%.0f MB)", sp, out.name,
                 f"{m.shape[0]:,}", m.shape[1], out.stat().st_size / 2**20)

    spec = {
        "protocol": a.protocol,
        "prefix": PREFIX, "key": KEY, "other": OTHER,
        "windows": windows, "labels": labels, "rate_labels": rate_labels,
        "smooth_alpha": alpha, "prior_splits": PRIOR_SPLITS, "priors": prior,
        "stats_from": {"split": "train", "basis": "train_samples"},
        "columns": cols,                     # 顺序即矩阵列序
        "encoding": spec_cols,
        "coverage": {sp: float((joined[sp].get_column(f"{PREFIX}_has_history")
                                .to_numpy() > 0).mean()) for sp in SPLITS},
    }
    (proc / f"pair_spec_{a.protocol}.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("%d 列 -> pair_spec_%s.json，总耗时 %.0fs",
             len(cols), a.protocol, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
