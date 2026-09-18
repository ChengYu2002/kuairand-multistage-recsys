"""T-1 日级滚动聚合的共用实现（plan §8）。

用户维度与视频维度逻辑相同，只有分组键不同，因此共用这里的实现。

核心约定：**Day T 的特征只使用 <= T-1 的日志。**

不铺 (key x date) 稠密网格 —— 本数据集有 437 万个 video_id，铺开是 1.4 亿行，
再乘几十个特征列会耗尽内存。改为「投射」：把 date=D 的记录投射到 D+1 ... D+w
这几个目标日，再按 (key, 目标日) 汇总。这样
  * 天然不含当天（偏移量最小为 1）；
  * 只生成真正有历史的行，无历史的 key 在 join 后为 null，由 has_history 区分；
  * 行数与实际活跃度成正比，而不是与 key 空间成正比。

distinct 类特征不能由「每日 distinct 求和」得到 —— 那样跨天看过的同一对象会被
重复计数。因此单独投射去重后的 (key, other, day) 三元组再做 n_unique。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

_FMT = "%Y%m%d"


def to_date(col: str = "date") -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.to_date(_FMT)


def shift_day(d: int, days: int = 1) -> int:
    return int((datetime.strptime(str(d), _FMT) + timedelta(days=days)).strftime(_FMT))


def other_key(key: str) -> str:
    return "video_id" if key == "user_id" else "user_id"


def _offsets(w_max: int) -> pl.DataFrame:
    return pl.DataFrame({"offset": list(range(1, w_max + 1))}, schema={"offset": pl.Int32})


def _project(df: pl.DataFrame, w_max: int) -> pl.DataFrame:
    """把每条日级记录投射到其后 1..w_max 个目标日。"""
    return df.join(_offsets(w_max), how="cross").with_columns(
        (pl.col("d") + pl.duration(days=pl.col("offset"))).alias("target")
    )


def daily_aggregate(lf: pl.LazyFrame, key: str, labels: list[str]) -> pl.DataFrame:
    """按 (key, date) 聚合每日计数。"""
    aggs = [pl.len().alias("imp")] + [pl.col(c).sum().alias(c) for c in labels]
    return lf.group_by([key, "date"]).agg(aggs).sort([key, "date"]).collect()


def distinct_pairs(lf: pl.LazyFrame, key: str) -> pl.DataFrame:
    """去重后的 (key, other, day) 三元组，供真实 distinct 统计使用。"""
    return (
        lf.select([key, other_key(key), to_date().alias("d")]).unique().collect()
    )


def rolling_features(
    daily: pl.DataFrame,
    pairs: pl.DataFrame,
    key: str,
    prefix: str,
    labels: list[str],
    windows: list[int],
    smooth_alpha: float,
    global_rates: dict[str, float],
) -> pl.DataFrame:
    """展开成各窗口的滚动统计；返回的 date 表示「适用于哪一天的样本」。"""
    w_max = max(windows)
    # 要统计曝光加五类行为。
    counts = ["imp", *labels]

    # 投射每日统计
    projected = _project(daily.with_columns(to_date().alias("d")).drop("date"), w_max)

    # 先把计算规则放进 aggs
    aggs: list[pl.Expr] = []
    # 窗口天数
    for w in windows:
        within = pl.col("offset") <= w

        for c in counts:
            # 只filter对应窗口天数 within的 对应的c行为，进行总数sun，构造新列特征 {prefix}_{c}_{w}d
            aggs.append(pl.col(c).filter(within).sum().alias(f"{prefix}_{c}_{w}d"))
    # 包光天数
    aggs.append((pl.col("imp") > 0).sum().alias(f"{prefix}_active_days_{w_max}d"))

    # 这里开始真正计算
    out = projected.group_by([key, "target"]).agg(aggs)

    # 用户过去7天看过多少个不同视频，或者视频过去7天触达了多少个不同用户
    # 真实的 w_max 天 distinct：对去重三元组做投射后 n_unique，
    # 而不是把每日 distinct 相加（跨天重复会被重复计数）。
    dist = (
        _project(pairs, w_max)
        .group_by([key, "target"])
        .agg(pl.col(other_key(key)).n_unique().alias(f"{prefix}_distinct_{w_max}d"))
    )

    # left join: 以原来的 out 为主，保留它的全部行，再把匹配到的 distinct 列补进来
    out = out.join(dist, on=[key, "target"], how="left")

    # 比率列：朴素 + 贝叶斯平滑。稀疏 key 的朴素比率几乎是噪声
    # （曝光 1 次点击 1 次 -> 1.0），平滑后被拉回全局先验。
    ratio: list[pl.Expr] = []
    for w in windows:
        imp = pl.col(f"{prefix}_imp_{w}d")
        for c in labels:
            num = pl.col(f"{prefix}_{c}_{w}d")
            # 原始行为率：行为次数 / 曝光次数；
            # 如果窗口内没有曝光，则行为率为 null。
            ratio.append(
                pl.when(imp > 0).then(num / imp).otherwise(None).alias(f"{prefix}_{c}_rate_{w}d")
            )

            # 平滑行为率 
            # =
            # (行为次数 + alpha × 全局行为率)
            # ÷
            # (曝光次数 + alpha）
            ratio.append(
                ((num + smooth_alpha * global_rates[c]) / (imp + smooth_alpha))
                .alias(f"{prefix}_{c}_rate_sm_{w}d")
            )
    # 把所有5种行为 × 3个窗口 × 2种行为率塞进去
    out = out.with_columns(ratio)

    # 近期趋势 = 近 1 天曝光量 / 最长窗口的日均曝光量。
    # 用户侧表示近期活跃度，物品侧表示近期热度；
    # > 1 表示高于历史日均，< 1 表示低于历史日均，历史日均为 0 时记为 null。
    if 1 in windows and w_max > 1:
        base = pl.col(f"{prefix}_imp_{w_max}d") / w_max
        out = out.with_columns(
            pl.when(base > 0)
            .then(pl.col(f"{prefix}_imp_1d") / base)
            .otherwise(None)
            .alias(f"{prefix}_trend")
        )
    
    # 当前特征表中的每一行都由最长窗口内的真实历史生成，因此标记为 1。
    # join 回样本后，未匹配到特征行的实体填 0，
    # 用于区分“统计值为 0”和“完全没有近期历史”。
    out = out.with_columns(pl.lit(1, dtype=pl.Int8).alias(f"{prefix}_has_history"))

    # 找出所有 user_ / item_ 特征列，
    # 排除实体主键和目标日期，供最终输出统一选列。
    feat_cols = [c for c in out.columns if c.startswith(prefix) and c not in (key, "target")]

    # 整理最终输出：保留实体 ID，将目标日恢复为 YYYYMMDD 格式，
    # 加入全部特征列，并按实体和日期排序。
    return out.select(
        pl.col(key),
        pl.col("target").dt.strftime(_FMT).cast(pl.Int32).alias("date"),
        *feat_cols,
    ).sort([key, "date"])
