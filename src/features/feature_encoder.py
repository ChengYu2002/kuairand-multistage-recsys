"""特征编码规格：缺失填充 + 变换 + 归一化统计量（plan 阶段 0 的 ⑤）。

本模块**只产出一份规格文件**，不改写特征表。规格描述每一列「缺了填什么、怎么变换、
用什么均值方差标准化」，由 ⑥ 特征 join 时套用。分开的理由是：统计量必须**只用 train 段
估计**，而 join 会同时碰到 valid/test —— 把两件事写在一起很容易在某次重构里把全量统计
悄悄用上去。

## 为什么填充值不能随便定

物品侧有 56.3% 的样本 join 不到 T-1 特征（冷启动是这个数据集的常态），填错就是整批废掉。
每一类的填充值都有各自的依据，不是统一填 0：

    count          -> 0        没有历史就是没有计数，0 是真值
    naive_rate     -> 0        配合 imp_{w}d == 0（窗口内无曝光）与 has_history == 0 区分
    smoothed_rate  -> 先验 g   **不是 0**。平滑率 = (num + α·g)/(imp + α)，
                               在 num=imp=0 处的值恰好是 g，填 g 是公式的连续延拓。
                               填 0 等于告诉模型「这个冷启动物品的点击率是 0%」。
    trend          -> 0        与 count 一致：没有历史即近期无活动。构造上 trend ∈ [0, 7]
                               （imp_1d ≤ imp_7d），无需变换。
    has_history    -> 0        这一列本身就是用来区分「统计值为 0」与「没有统计值」的

先验 g 与生成端必须是同一个值。g 由 warmup 段估计（PRIOR_SPLITS），α 来自
data.yaml 的 features.smooth_alpha。本模块重算 g 之后会**从既有特征表反解**一次
（g = (sm·(imp+α) − num)/α）来确认口径一致，对不上直接报错。

## 重尾计数要 log1p

实测 train 段：user_imp_7d 中位 1,342 / 最大 29,212；item_imp_7d 中位 1 / 最大 330。
直接标准化会被长尾拉垮，因此计数列一律 log1p 后再标准化。

## 统计量按「训练样本」加权，不是按特征行等权

同一个 (实体, 日期) 的特征行会被多个样本重复读取（item 侧平均 4.2 次），而且 56.3% 的
样本根本 join 不到特征、读的是冷启动行。若按唯一特征行等权统计，模型实际吃到的分布
并不是均值 0 标准差 1 —— 实测 item_imp_7d 会变成 mean −0.32 / std 2.16。
因此这里把 train 段曝光样本 join 回特征表（缺失按各自的 fill 补齐），在**那个**分布上
估统计量。口径由 encoder.stats_basis 控制。

## 静态用户画像的两个坑（实测）

    is_lowactive_period  1,000 个用户全是 0 —— 常量列，**丢弃**
    is_live_streamer     取值是 {-124, 1} 而不是 {0, 1}。-124 来自**原始 CSV**
                         （782 个 -124 / 218 个 1），不是本项目的 bug，但它伪装成数值，
                         直接喂进网络会被当成一个很大的负数。映射成 {0, 1}。

另注（README §4）：follow/fans 这类快照字段相对更早的交互可能含少量未来信息，
使用时需在报告中披露。

用法：
    python -m src.features.feature_encoder --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse
import json

import polars as pl

from src.features._driver import PRIOR_SPLITS
from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)

COUNT, NAIVE, SMOOTH, TREND, FLAG = "count", "naive_rate", "smoothed_rate", "trend", "flag"


def classify(col: str, prefix: str) -> str | None:
    """把 T-1 特征列归入五类之一；非特征列返回 None。"""
    if not col.startswith(prefix) or col in ("user_id", "video_id"):
        return None
    if col.endswith("_has_history"):
        return FLAG
    if col.endswith("_trend"):
        return TREND
    if "_rate_sm_" in col:
        return SMOOTH
    if "_rate_" in col:
        return NAIVE
    return COUNT


def label_of(col: str) -> str | None:
    """从 `user_is_click_rate_sm_7d` 这类列名里取出标签名，用来查先验。"""
    for lab in ("is_click", "long_view", "is_like", "is_comment", "is_follow"):
        if f"_{lab}_rate" in col:
            return lab
    return None


def _stats(lf: pl.LazyFrame, cols: list[str], log1p: bool,
           fills: dict[str, float] | None = None) -> dict[str, dict[str, float]]:
    """均值/标准差。log1p 的列先填充再变换再统计 —— 顺序必须与 apply_spec 完全一致。"""
    exprs = []
    for c in cols:
        e = pl.col(c)
        if fills is not None:
            e = e.fill_null(fills.get(c, 0.0))
        e = e.log1p() if log1p else e
        exprs += [e.mean().alias(f"{c}__m"), e.std().alias(f"{c}__s")]
    row = lf.select(exprs).collect().row(0, named=True)
    out = {}
    for c in cols:
        m, s = row[f"{c}__m"], row[f"{c}__s"]
        # std 为 0（常量列）时置 1，避免除零把整列变成 inf
        out[c] = {"mean": float(m or 0.0), "std": float(s) if s and s > 1e-12 else 1.0}
    return out


def apply_spec(df: pl.DataFrame | pl.LazyFrame, spec: dict) -> pl.DataFrame | pl.LazyFrame:
    """按规格套用「填充 -> 变换」。只处理 df 里实际存在的列，缺的列静默跳过。

    顺序不能颠倒：log1p_standardize 的统计量是在**填充后的值**上估的（train 段本来就
    没有 null，所以两者一致），但类目列必须先填 OOV 再查表，否则 null 会变成 map 的未命中。
    """
    present = set(df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns)
    exprs = []
    for col, sp in spec["columns"].items():
        if col not in present:
            continue
        if sp["transform"] == "vocab":
            cat = spec["categorical"][col]
            e = (
                pl.col(col).cast(pl.Utf8).replace_strict(
                    cat["map"], default=cat["oov_index"], return_dtype=pl.Int32
                )
                .fill_null(cat["oov_index"])
            )
        elif sp["transform"] == "log1p_standardize":
            e = ((pl.col(col).fill_null(sp["fill"]).log1p() - sp["mean"]) / sp["std"])
        else:
            e = pl.col(col).fill_null(sp["fill"])
        exprs.append(e.alias(col))
    return df.with_columns(exprs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))

    labels = require(cfg, "data", "labels", "tasks")
    alpha = float(require(cfg, "data", "features", "smooth_alpha"))
    basis = require(cfg, "encoder", "stats_basis")
    if basis not in ("train_samples", "entity_date"):
        raise ValueError(f"encoder.stats_basis 必须是 train_samples 或 entity_date，收到 {basis!r}")
    logs = pl.scan_parquet(proc / "logs_split.parquet")

    # 先验：与生成端同一口径（仅 warmup 段），随后用既有特征表反解校验
    prior = (
        logs.filter(pl.col("split").is_in(PRIOR_SPLITS))
        .select([pl.col(c).mean().alias(c) for c in labels])
        .collect().row(0, named=True)
    )
    u_feat = pl.read_parquet(proc / "feat_user_daily.parquet", n_rows=20000)
    for lab in labels:
        back = (
            (u_feat[f"user_{lab}_rate_sm_7d"] * (u_feat["user_imp_7d"] + alpha)
             - u_feat[f"user_{lab}_7d"]) / alpha
        )
        if abs(float(back.median()) - prior[lab]) > 1e-9:
            raise AssertionError(
                f"先验口径不一致：{lab} 反解得 {back.median():.8f}，重算得 {prior[lab]:.8f}。"
                "生成特征时用的 alpha / PRIOR_SPLITS 与当前 config 不同。"
            )
    log.info("先验口径校验通过（alpha=%.1f，仅 %s 段）", alpha, "+".join(PRIOR_SPLITS))

    # train 段日期范围 —— 所有统计量只在这个范围内估计
    tr_lo = int(logs.filter(pl.col("split") == "train").select(pl.col("date").min()).collect().item())
    tr_hi = int(logs.filter(pl.col("split") == "train").select(pl.col("date").max()).collect().item())

    spec: dict[str, object] = {
        "protocol": args.protocol,
        "alpha": alpha,
        "prior_splits": PRIOR_SPLITS,
        "priors": {k: float(v) for k, v in prior.items()},
        "stats_from": {"split": "train", "date_min": tr_lo, "date_max": tr_hi, "basis": basis},
        "columns": {},
        "dropped": {},
        "categorical": {},
    }
    cols_spec: dict = spec["columns"]

    # ---- T-1 特征（user / item）----
    for fname, prefix in (("feat_user_daily", "user"), ("feat_item_daily", "item")):
        lf = pl.scan_parquet(proc / f"{fname}.parquet")
        names = lf.collect_schema().names()
        kinds = {c: classify(c, prefix) for c in names}
        kinds = {c: k for c, k in kinds.items() if k}
        tr = lf.filter((pl.col("date") >= tr_lo) & (pl.col("date") <= tr_hi))

        counts = [c for c, k in kinds.items() if k == COUNT]
        if basis == "train_samples":
            # 把 train 段曝光样本 join 回特征表：join 不到的行留 null，由 fills 补 0，
            # 于是统计量落在「模型真正吃到的分布」上（含 56.3% 的冷启动样本）。
            key = "user_id" if prefix == "user" else "video_id"
            src = (
                logs.filter(pl.col("split") == "train").select(key, "date")
                .join(lf.select(key, "date", *counts), on=[key, "date"], how="left")
            )
        else:
            src = tr
        st = _stats(src, counts, log1p=True, fills=dict.fromkeys(counts, 0.0))
        for c in counts:
            cols_spec[c] = {"kind": COUNT, "fill": 0.0, "transform": "log1p_standardize", **st[c]}
        for c, k in kinds.items():
            if k == SMOOTH:
                lab = label_of(c)
                cols_spec[c] = {"kind": SMOOTH, "fill": float(prior[lab]), "transform": "none"}
            elif k == NAIVE:
                cols_spec[c] = {"kind": NAIVE, "fill": 0.0, "transform": "none"}
            elif k == TREND:
                cols_spec[c] = {"kind": TREND, "fill": 0.0, "transform": "none"}
            elif k == FLAG:
                cols_spec[c] = {"kind": FLAG, "fill": 0.0, "transform": "none"}
        log.info("%s: %d 列（count %d / smooth %d / naive %d）",
                 fname, len(kinds), len(counts),
                 sum(k == SMOOTH for k in kinds.values()),
                 sum(k == NAIVE for k in kinds.values()))

    # ---- 物品静态数值 ----
    it = pl.scan_parquet(proc / f"item_static_{args.protocol}.parquet")
    dur_med = float(it.select(pl.col("duration_ms").median()).collect().item())
    st = _stats(it.filter(pl.col("duration_ms").is_not_null()), ["duration_ms"], log1p=True)
    cols_spec["duration_ms"] = {
        "kind": "numeric", "fill": dur_med, "transform": "log1p_standardize", **st["duration_ms"]
    }
    for c in ("has_upload_date", "has_tag", "has_duration"):
        cols_spec[c] = {"kind": FLAG, "fill": 0.0, "transform": "none"}

    # ---- 静态用户画像 ----
    uf = pl.read_parquet(proc / "user_features.parquet")
    numeric_static = ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]
    st = _stats(uf.lazy(), numeric_static, log1p=True)
    for c in numeric_static:
        cols_spec[c] = {
            "kind": "numeric", "fill": float(uf[c].median()),
            "transform": "log1p_standardize", **st[c],
        }
    for c in uf.columns:
        if c == "user_id" or c in numeric_static:
            continue
        s = uf[c]
        if s.n_unique() <= 1:
            spec["dropped"][c] = f"常量列（全部为 {s.drop_nulls().head(1).to_list()}）"
            continue
        vals = sorted(x for x in s.unique().to_list() if x is not None)
        # 0 号留给 OOV（缺失或未见取值），真实取值从 1 开始 —— 与 build_vocab 的约定一致。
        entry = {
            "map": {str(v): i + 1 for i, v in enumerate(vals)},
            "oov_index": 0,
            "n_null": int(s.null_count()),
            "cardinality": len(vals) + 1,
        }
        if c == "is_live_streamer":
            # 原始 CSV 用 -124 表示 false，不是 0。必须显式映射，否则网络会把它当成一个大负数。
            entry["note"] = "原始 CSV 的 false 编码为 -124（782 个），true 为 1（218 个）"
        spec["categorical"][c] = entry
        cols_spec[c] = {"kind": "categorical", "fill": "OOV", "transform": "vocab"}

    dst = proc / f"encoder_spec_{args.protocol}.json"
    dst.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    by_kind: dict[str, int] = {}
    for v in cols_spec.values():
        by_kind[v["kind"]] = by_kind.get(v["kind"], 0) + 1
    print()
    print(f"encoder_spec_{args.protocol}.json  共 {len(cols_spec)} 列")
    for k, n in sorted(by_kind.items()):
        fills = {str(v["fill"]) for v in cols_spec.values() if v["kind"] == k}
        ex = "先验 g（每个标签不同）" if k == SMOOTH else ", ".join(sorted(fills)[:3])
        print(f"  {k:<15} {n:>3} 列   填充: {ex}")
    if spec["dropped"]:
        print(f"  丢弃: {spec['dropped']}")
    print(f"  统计量来源: train 段 {tr_lo}~{tr_hi}（不含 valid/test），加权口径 {basis}")
    print()
    log.info("已写出 %s", dst.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
