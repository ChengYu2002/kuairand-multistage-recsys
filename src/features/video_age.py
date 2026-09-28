"""物品静态属性表（plan 阶段 0 的 ② video_age 与 ③ tag multi-hot）。

两件事合成一张表：它们同源（video_features.parquet）、同键（video_id），拆成两个文件
只会让 ⑥ 特征 join 立刻再拼回去，还要多扫一遍 359 MB 的原表。

## 关于 video_age：本模块**不落盘 age**，只落盘 upload_date

age = 请求日 − 上传日，是个**随日期变化**的量。而 §13.2 把 T-1 统计量排除出物品塔，
理由正是「物品向量要离线算一次建 ANN 索引，依赖每日变化的量会把召回搞复杂」——
同一条理由对 age 完全成立。因此这里只存 upload_date，age 由使用方在 join 时现算：

    age_days = (请求日 − upload_date)，见 age_days()

这样不锁死「age 到底进不进物品塔」这个决定：排序模型可以按 (video, 请求日) 现算；
物品塔若要用，得自己选一个参考日，并接受下面这个已知问题。

## 使用 age 前必须知道的两件事（实测）

1. **训练与评估的 age 条件分布不同**：曝光记录的中位 age 在 train 段是 2 天、
   valid 是 12 天、test 是 23 天。成因是三层叠加 —— 词表固定为 warm 集合、评估只保留
   其中可编码的部分、视频本身随时间自然变老。这是条件分布的变化，**不代表原始视频
   总体发生异常漂移**；但后果是实在的：物品塔若用 age，会在「age≈2」上训练、
   在「age≈23」上评估。
2. 在**词表可编码的 4,809,007 条曝光**上实测（非全部 901.5 万条）：age 最小 0、
   最大 1,190 天，**无负值**；40.1% 的曝光 age ≤1 天、63.9% ≤7 天 —— 典型新内容信息流。

## 缺失与 OOV（词表 897,503 个视频上实测）

    upload_dt      缺 18       (0.002%)  -> upload_date = null, has_upload_date = 0
    tag            缺 26,771   (2.983%)  -> tag_indices = [OOV],  has_tag = 0
    video_duration 缺 64,045   (7.136%)  -> duration_ms = null,   has_duration = 0
    author 不在词表 589,401    (65.7%)   -> author_index = OOV

author/tag 词表是从**候选库**建的，而本表覆盖整个词表（含 70.3 万个仅用于历史的视频），
所以后两项的 OOV 比例很高。这不是缺陷：物品塔只给候选库内的 194,310 个打分，用户塔
只用历史视频的 id embedding，不碰它们的作者与 tag。

用法：
    python -m src.features.video_age --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse

import polars as pl

from src.features.build_vocab import OOV
from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)


def age_days(request_date: pl.Expr, upload_date: pl.Expr) -> pl.Expr:
    """age = 请求日 − 上传日，单位天。两个参数都是 YYYYMMDD 的整数列。

    **不能直接相减**：YYYYMMDD 是十进制拼接，20220501 − 20220430 = 71 而不是 1。
    必须先转成真正的日期再求差。upload_date 为 null 时结果也是 null，交给 ⑤ 决定怎么填。
    """
    def to_date(e: pl.Expr) -> pl.Expr:
        return e.cast(pl.Utf8).str.to_date("%Y%m%d")

    return (to_date(request_date) - to_date(upload_date)).dt.total_days()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))

    vocab = pl.read_parquet(proc / f"vocab_video_{args.protocol}.parquet").rename(
        {"id": "video_id", "index": "video_index"}
    )
    a_vocab = pl.read_parquet(proc / f"vocab_author_{args.protocol}.parquet").rename(
        {"id": "author_id", "index": "author_index"}
    ).select("author_id", "author_index")
    t_vocab = pl.read_parquet(proc / f"vocab_tag_{args.protocol}.parquet").rename(
        {"id": "tag", "index": "tag_index"}
    ).select("tag", "tag_index")
    vf = pl.read_parquet(proc / "video_features.parquet").select(
        "video_id", "author_id", "upload_dt", "tag", "video_duration"
    )

    base = vocab.join(vf, on="video_id", how="left")
    if base.height != vocab.height:
        raise AssertionError("join 后行数变了 —— video_features 里 video_id 不唯一")

    # tag：逗号分隔 -> 去空白 -> 查词表 -> 未命中或缺失落 OOV
    tags = (
        base.select("video_id", "tag")
        .with_columns(pl.col("tag").str.split(","))
        .explode("tag", empty_as_null=True)
        .with_columns(pl.col("tag").str.strip_chars())
        .filter(pl.col("tag").is_not_null() & (pl.col("tag") != ""))
        .join(t_vocab, on="tag", how="left")
        .with_columns(pl.col("tag_index").fill_null(OOV))
        # 同一视频的重复 tag 去掉；排序保证可复现
        .unique(subset=["video_id", "tag_index"])
        .group_by("video_id")
        .agg(pl.col("tag_index").sort().alias("tag_indices"))
    )

    out = (
        base.join(a_vocab, on="author_id", how="left")
        .join(tags, on="video_id", how="left")
        .with_columns(
            pl.col("upload_dt").str.to_date("%Y-%m-%d").dt.strftime("%Y%m%d")
            .cast(pl.Int32).alias("upload_date"),
            pl.col("author_index").fill_null(OOV).cast(pl.Int32),
            # 没有任何 tag 的视频给 [OOV]，而不是空列表 —— EmbeddingBag 遇到空袋会报错或给零，
            # 而「没有 tag」本身是一种有信息的状态，应当显式表示。
            pl.col("tag_indices").fill_null([OOV]).cast(pl.List(pl.Int32)),
            pl.col("video_duration").cast(pl.Int32).alias("duration_ms"),
        )
        .with_columns(
            pl.col("upload_date").is_not_null().cast(pl.Int8).alias("has_upload_date"),
            pl.col("tag").is_not_null().cast(pl.Int8).alias("has_tag"),
            pl.col("duration_ms").is_not_null().cast(pl.Int8).alias("has_duration"),
            pl.col("tag_indices").list.len().cast(pl.Int8).alias("n_tags"),
        )
        .select(
            "video_id", "video_index", "author_index", "tag_indices", "n_tags",
            "upload_date", "duration_ms", "has_upload_date", "has_tag", "has_duration",
        )
        .sort("video_index")
    )

    dst = proc / f"item_static_{args.protocol}.parquet"
    out.write_parquet(dst, compression="zstd")

    n = len(out)
    print()
    print(f"item_static_{args.protocol}.parquet  {n:,} 行 x {len(out.columns)} 列")
    print(f"{'字段':<16}{'缺失/OOV':>12}{'占比':>9}")
    for label, expr in (
        ("upload_date", pl.col("has_upload_date") == 0),
        ("tag", pl.col("has_tag") == 0),
        ("duration_ms", pl.col("has_duration") == 0),
        ("author_index=OOV", pl.col("author_index") == OOV),
    ):
        c = out.filter(expr).height
        print(f"{label:<16}{c:>12,}{c / n:>9.3%}")
    print(f"\n每视频 tag 数: 中位 {out['n_tags'].median():.0f}  最大 {out['n_tags'].max()}")
    print()
    log.info("已写出 %s（%.0f MB）—— age 不落盘，由使用方按请求日现算，见 age_days()",
             dst.name, dst.stat().st_size / 1024**2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
