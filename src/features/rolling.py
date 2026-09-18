"""T-1 日级滚动聚合的共用实现（plan §8）。

用户维度与视频维度逻辑相同，只有分组键不同，因此共用这里的实现。

核心约定：**Day T 的特征只使用 <= T-1 的日志。**

实现不铺 (key x date) 稠密网格 —— 本数据集有 437 万个 video_id，铺开是 1.4 亿行，
再乘几十个特征列会直接耗尽内存。改为「投射」：把 date=D 的日聚合结果投射到
D+1 ... D+w 这 w 个目标日，再按 (key, 目标日) 汇总。这样
  * 天然不含当天（偏移量最小为 1）；
  * 只生成真正有历史的行，无历史的 key 在 join 后为 null，由 has_history 标记区分；
  * 行数与实际活跃度成正比，而不是与 key 空间成正比。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

_FMT = "%Y%m%d"


def to_date(col: str = "date") -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.to_date(_FMT)


def shift_day(d: int, days: int = 1) -> int:
    return int((datetime.strptime(str(d), _FMT) + timedelta(days=days)).strftime(_FMT))


def daily_aggregate(lf: pl.LazyFrame, key: str, labels: list[str]) -> pl.DataFrame:
    """按 (key, date) 聚合每日计数，作为投射的输入。"""
    other = "video_id" if key == "user_id" else "user_id"
    aggs = [pl.len().alias("imp")] + [pl.col(c).sum().alias(c) for c in labels]
    aggs.append(pl.col(other).n_unique().alias("distinct_other"))
    return lf.group_by([key, "date"]).agg(aggs).sort([key, "date"]).collect()


def rolling_features(
    daily: pl.DataFrame,
    key: str,
    prefix: str,
    labels: list[str],
    windows: list[int],
    smooth_alpha: float,
    global_rates: dict[str, float],
) -> pl.DataFrame:
    """展开成各窗口的滚动统计；返回的 date 表示「适用于哪一天的样本」。"""
    w_max = max(windows)
    counts = ["imp", *labels]

    src = daily.with_columns(to_date().alias("d")).drop("date")
    # 投射：date=D 的记录计入 D+1 ... D+w_max 的窗口，offset 即「几天前」
    projected = src.join(
        pl.DataFrame({"offset": list(range(1, w_max + 1))}, schema={"offset": pl.Int32}),
        how="cross",
    ).with_columns((pl.col("d") + pl.duration(days=pl.col("offset"))).alias("target"))

    aggs: list[pl.Expr] = []
    for w in windows:
        within = pl.col("offset") <= w
        for c in counts:
            aggs.append(pl.col(c).filter(within).sum().alias(f"{prefix}_{c}_{w}d"))
    aggs.append((pl.col("imp") > 0).sum().alias(f"{prefix}_active_days_{w_max}d"))
    aggs.append(pl.col("distinct_other").sum().alias(f"{prefix}_distinct_{w_max}d"))

    out = projected.group_by([key, "target"]).agg(aggs)

    # 比率列：朴素 + 贝叶斯平滑。稀疏 key 的朴素比率几乎是噪声
    # （曝光 1 次点击 1 次 -> 1.0），平滑后被拉回全局先验。
    ratio: list[pl.Expr] = []
    for w in windows:
        imp = pl.col(f"{prefix}_imp_{w}d")
        for c in labels:
            num = pl.col(f"{prefix}_{c}_{w}d")
            ratio.append(
                pl.when(imp > 0).then(num / imp).otherwise(None).alias(f"{prefix}_{c}_rate_{w}d")
            )
            ratio.append(
                ((num + smooth_alpha * global_rates[c]) / (imp + smooth_alpha))
                .alias(f"{prefix}_{c}_rate_sm_{w}d")
            )
    out = out.with_columns(ratio)

    # 热度趋势：近 1 天曝光 相对 最长窗口的日均；>1 为上升期
    if 1 in windows and w_max > 1:
        base = pl.col(f"{prefix}_imp_{w_max}d") / w_max
        out = out.with_columns(
            pl.when(base > 0)
            .then(pl.col(f"{prefix}_imp_1d") / base)
            .otherwise(None)
            .alias(f"{prefix}_trend")
        )
    out = out.with_columns(pl.lit(1, dtype=pl.Int8).alias(f"{prefix}_has_history"))

    feat_cols = [c for c in out.columns if c.startswith(prefix) and c not in (key, "target")]
    return out.select(
        pl.col(key),
        pl.col("target").dt.strftime(_FMT).cast(pl.Int32).alias("date"),
        *feat_cols,
    ).sort([key, "date"])
