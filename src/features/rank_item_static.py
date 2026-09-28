"""排序侧专用物品元数据（plan §20，config: data.yaml 的 rank_item 段）。

## 为什么排序不能沿用召回的词表

召回的 video / author / tag 词表按 Protocol A「训练期候选库」建，对召回是正确的。
但排序训练在**曝光日志**上，实测（KuaiRand-1K，主协议）：

    test 段 83.2% 的曝光，其 video_id 不在召回词表里（→ ID 通道恒为 OOV 向量）
    其中 53.2% 连 item T-1 统计也没有（全新视频，过去没被曝光过）

也就是说排序模型有一半多的考题是「完全不知道这是什么视频」。而作者 / 标签 / 时长 /
上传日期在曝光发生**之前**就已存在，是合法的 prediction-time 特征（§7.1 禁的是
播放时长、主页停留这类曝光后行为），没有理由不用。补上之后 test 段覆盖率：

    标签 96.6%    时长 93.3%    上传日期 ~100%    作者 64.1%（门槛 >=5）
    目标视频 ID 10.8%（排序词表，门槛 >=5；旧召回口径 16.8%，但那一版带标签泄漏）

video_id 本身仍有 89.2% 对不上，这个**修不了也不该修**：昨天刚上传的视频不可能有
学过的专属向量。模型从「不知道是什么」变成「不知道是哪一个，但知道谁拍的、什么类、
多长、多新」—— 这正是真实系统处理新视频的方式。

## 排序专用 video 词表（P0 修复）

召回的 video 词表 = 「候选库 ∪ train 段正向视频」，**成员身份部分由训练标签决定**。
实测 train 段 962,054 条 OOV 样本里 is_click / long_view 正样本率**恰好为 0** ——
任何正样本都会把该视频送进词表。于是「在词表内」这一个比特在 train 上单独就有
AUC 0.6981，而 test 只有 0.5190：一条只在 train 成立的捷径。is_like 不受影响
（AUC 0.5043），因为召回词表的 positive_signal 只含 is_click 与 long_view。

所以目标视频的 ID 通道必须换成**只由 train 曝光**（完全不看标签）构建的词表。
门槛与取舍见 configs/data.yaml 的 rank_item.video_min_count。

**历史通道不动**：仍用召回词表。换大词表救不回历史（samples 表里超词表的历史项已被
压成 OOV，原始 id 丢了），换小词表会把可编码率砍半。历史自身不是捷径：hist_len 的
单特征 AUC 在 train 上只有 0.4983、test 0.5942，方向相反。

## 硬隔离

本模块只**读** video_features.parquet 与 samples_train，只**写**五个新文件：

    vocab_author_rank_{p}.parquet   vocab_tag_rank_{p}.parquet
    vocab_video_rank_{p}.parquet
    item_static_rank_{p}.parquet    rank_item_spec_{p}.json

召回的词表、静态表、checkpoint 一个字节都不动。这不是承诺而是可断言的：
`scripts/freeze_manifest.py --check` 比对改动前记录的 39 个文件哈希。
理由是 author 词表一旦重排，「第 N 行是谁」就变了，已训好的双塔 checkpoint 会**静默
作废** —— 不报错，只是指标莫名变差，然后你会去怀疑模型。

index 约定直接复用 build_vocab.build_table：PAD=0、OOV=1、实体从 2 起，
排序键 (train 曝光数 desc, id asc) 是全序，保证可复现。

## 词表只用 train 建

valid / test 里没在训练期出现过的作者 / 标签一律 OOV。用全量建词表等于让模型知道
「这个作者将来会出现」，是一种隐蔽的泄漏；而且 27K 上会凭空多出几百万行学不到的 embedding。

## 归一化统计也只用 train

时长的填充值与 log1p 标准化的 mean/std 全部由 **train 段样本**估计（曝光加权，
与 encoder_spec 的 stats_from.basis = train_samples 同口径）。表里存**原始**时长，
标准化在装载时按 spec 施加 —— 这样 spec 换了不用重建 4 百万行的表。

上传日期同理只存原始 YYYYMMDD。video_age = 请求日 − 上传日，必须在取样时现算，
不能预先算成一个固定值（同一个视频在不同请求日的 age 不同）。

## 缺失标志

数值本身之外一律给 *_known 标志，让模型能区分「真的是 0」和「不知道」：

    video_id_known      视频在**排序** video 词表内（train 曝光 >= video_min_count）。
                        注意它问的**不是**召回词表 —— 那一版由训练标签决定成员身份，
                        是已修的 P0。本表口径下 test 段 OOV 为 89.2%（旧口径 83.2%）：
                        门槛更严、覆盖更低，换来的是与标签无关。
    author_known        作者在 train 词表内（不只是元数据里有 author_id）
    tag_known           至少一个标签在 train 词表内
    duration_known      basic 元数据里有时长（实测 6.6% 缺）
    upload_date_known   basic 元数据里有上传日期（实测 <0.01% 缺）

## 表为什么覆盖全部 4,371,868 个视频

不按「test 里出现过哪些视频」裁剪。裁了就等于用测试集决定资产范围，虽然这里只影响
表的行数、不影响任何数值，但口径上要经得起问。全表 4.37M 行也不贵（约 100 MB，
装载后 ~115 MB 常驻），而 video_id 在 0~4,371,899 内几乎连续（仅 31 个空洞），
装载器可以直接按 video_id 索引，连 join 都不需要。

用法：
    python -m src.features.rank_item_static --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from src.features.build_vocab import OOV, PAD, build_table
from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)

SPLITS = ("train", "valid", "test")


def _assert_source_allowed(cfg: dict, source: str) -> None:
    """禁用文件清单里的任何一项都不许出现在来源里（§7.2 的 statistic 表）。"""
    for banned in require(cfg, "leakage", "excluded_files"):
        if banned in source:
            raise ValueError(
                f"rank_item.source = {source!r} 命中 leakage.excluded_files 的 {banned!r}。"
                "该表是「整月每日平均」，对任一天的样本都含未来信息。"
            )


def _coverage(df: pl.DataFrame) -> dict:
    return {
        "n": len(df),
        "video_id_known": float(df["video_id_known"].mean()),
        "author_known": float(df["author_known"].mean()),
        "tag_known": float(df["tag_known"].mean()),
        "duration_known": float(df["duration_known"].mean()),
        "upload_date_known": float(df["upload_date_known"].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    # 文件名带 protocol：samples_train_{p} 定义了「train 段」，换协议不能互相覆盖。
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    t0 = time.perf_counter()

    proc = project_path(require(cfg, "dataset", "processed_dir"))
    source = require(cfg, "rank_item", "source")
    _assert_source_allowed(cfg, source)
    basis = require(cfg, "rank_item", "vocab_basis")
    if basis != "train_samples":
        raise NotImplementedError(
            f"rank_item.vocab_basis 目前只支持 train_samples，收到 {basis!r}。"
            "用全量建词表会让模型知道「这个作者将来会出现」。"
        )
    a_min = int(require(cfg, "rank_item", "author_min_count"))
    t_min = int(require(cfg, "rank_item", "tag_min_count"))
    v_min = int(require(cfg, "rank_item", "video_min_count"))

    # ---- 来源：basic 元数据全表 ----
    vf = pl.read_parquet(proc / f"{source}.parquet").select(
        "video_id", "author_id", "tag", "video_duration", "upload_dt"
    )
    log.info("%s.parquet: %s 个视频", source, f"{len(vf):,}")

    # ---- train 段样本（词表与统计量的唯一依据）----
    tr = pl.read_parquet(proc / f"samples_train_{args.protocol}.parquet", columns=["video_id"])
    tr_meta = tr.join(vf, on="video_id", how="left")
    log.info("train 曝光 %s 条", f"{len(tr_meta):,}")

    # ---- author 词表：train 曝光次数 >= a_min ----
    a_cnt = (
        tr_meta.filter(pl.col("author_id").is_not_null())
        .group_by("author_id").agg(pl.len().alias("n"))
        .filter(pl.col("n") >= a_min)
        .sort(["n", "author_id"], descending=[True, False])
    )
    a_vocab = build_table(a_cnt.get_column("author_id"), a_cnt.get_column("n"), oov=True)
    log.info("author 词表 %s 个（门槛 train 曝光 >= %d，覆盖 train 曝光 %.4f）",
             f"{len(a_vocab):,}", a_min,
             float(a_cnt["n"].sum()) / max(1, len(tr_meta)))

    # ---- video 词表：只由 train 段曝光次数决定，**不读任何标签列** ----
    # tr 只 select 了 video_id，这里连标签都拿不到 —— 让"不看标签"成为结构上的保证，
    # 而不是一句注释。
    v_cnt = (
        tr.group_by("video_id").agg(pl.len().alias("n"))
        .filter(pl.col("n") >= v_min)
        .sort(["n", "video_id"], descending=[True, False])
    )
    v_vocab = build_table(v_cnt.get_column("video_id"), v_cnt.get_column("n"), oov=True)
    log.info("video 词表 %s 个（门槛 train 曝光 >= %d，覆盖 train 曝光 %.4f）",
             f"{len(v_vocab):,}", v_min,
             float(v_cnt["n"].sum()) / max(1, len(tr)))

    # ---- tag 词表：同口径。tag 是逗号分隔的多值列 ----
    t_cnt = (
        tr_meta.filter(pl.col("tag").is_not_null())
        .with_columns(pl.col("tag").str.split(","))
        # empty_as_null 显式写死 + 过滤 null 与空串：polars 2.0 要改 explode 对空列表的
        # 默认行为，不写死的话行为随版本变，而变的是词表 —— 不报错，只是 tag 集合悄悄不同。
        .explode("tag", empty_as_null=True)
        .filter(pl.col("tag").is_not_null() & (pl.col("tag") != ""))
        .group_by("tag").agg(pl.len().alias("n"))
        .filter(pl.col("n") >= t_min)
        .sort(["n", "tag"], descending=[True, False])
    )
    t_vocab = build_table(t_cnt.get_column("tag"), t_cnt.get_column("n"), oov=True)
    log.info("tag 词表 %s 个（门槛 >= %d）", f"{len(t_vocab):,}", t_min)

    # ---- 时长统计：只由 train 段样本估计（曝光加权，与 encoder_spec 同口径）----
    d_tr = tr_meta.get_column("video_duration")
    fill = float(d_tr.drop_nulls().median())
    filled = np.log1p(d_tr.fill_null(fill).to_numpy().astype(np.float64))
    d_mean, d_std = float(filled.mean()), float(filled.std())
    if d_std <= 0:
        raise ValueError("train 段时长的 log1p 标准差为 0，无法标准化")
    log.info("时长（train 曝光口径）: 缺失 %.4f  填充中位数 %.0f ms  log1p mean %.4f std %.4f",
             float(d_tr.is_null().mean()), fill, d_mean, d_std)

    # ---- 静态表：basic 全表，原始值 + 词表行号 + 缺失标志 ----
    a_map = a_vocab.select(pl.col("id").alias("author_id"), pl.col("index").alias("author_rank_idx"))
    v_map = v_vocab.select(pl.col("id").alias("video_id"), pl.col("index").alias("video_rank_idx"))
    static = (
        vf.join(a_map, on="author_id", how="left")
        .join(v_map, on="video_id", how="left")
        .with_columns(
            pl.col("video_rank_idx").fill_null(OOV).cast(pl.Int32),
            pl.col("author_rank_idx").fill_null(OOV).cast(pl.Int32),
            # tag：逐个映射成行号，未进词表的落 OOV；整个字段缺失则留空列表
            # fill_null("") 在 split 之前：tag 整个字段缺失时，split(null) 会产出 **null 列表**
            # 而不是空列表，于是 n_tags 也是 null，装载时补不成定宽矩阵（实测会直接抛
            # TypeError: NoneType is not iterable）。填空串后 split 得 [""]，过滤掉就是干净的空列表。
            pl.col("tag").fill_null("").str.split(",").list.eval(
                pl.element().filter(pl.element().is_not_null() & (pl.element() != ""))
            ).alias("_tags"),
            # upload_dt 是 'YYYY-MM-DD' 字符串，去掉横杠变成 YYYYMMDD 整数；缺失记 0
            pl.col("upload_dt").str.replace_all("-", "").cast(pl.Int32, strict=False)
            .fill_null(0).alias("upload_date"),
            pl.col("video_duration").cast(pl.Float32).alias("duration_ms"),
        )
    )
    t_map = dict(zip(t_vocab["id"].to_list(), t_vocab["index"].to_list(), strict=True))
    static = static.with_columns(
        pl.col("_tags").list.eval(
            pl.element().replace_strict(t_map, default=OOV, return_dtype=pl.Int32)
        ).alias("tag_rank_idx")
    ).with_columns(
        pl.col("tag_rank_idx").list.len().cast(pl.Int32).alias("n_tags"),
        (pl.col("author_rank_idx") > OOV).alias("author_known"),
        (pl.col("video_rank_idx") > OOV).alias("video_id_known"),
        # tag_known：至少一个标签进了词表（全是 OOV 不算"知道"）
        pl.col("tag_rank_idx").list.eval(pl.element() > OOV).list.any().fill_null(False)
        .alias("tag_known"),
        pl.col("duration_ms").is_not_null().alias("duration_known"),
        (pl.col("upload_date") > 0).alias("upload_date_known"),
    ).select(
        "video_id", "video_rank_idx", "author_rank_idx", "tag_rank_idx", "n_tags",
        "duration_ms", "upload_date",
        "video_id_known", "author_known", "tag_known", "duration_known", "upload_date_known",
    ).sort("video_id")

    max_tags = int(static["n_tags"].max())
    out_static = proc / f"item_static_rank_{args.protocol}.parquet"
    static.write_parquet(out_static, compression="zstd")
    a_vocab.write_parquet(proc / f"vocab_author_rank_{args.protocol}.parquet")
    t_vocab.write_parquet(proc / f"vocab_tag_rank_{args.protocol}.parquet")
    v_vocab.write_parquet(proc / f"vocab_video_rank_{args.protocol}.parquet")

    # ---- 覆盖率：按 split 报，写进 spec 供 README 与验算引用 ----
    cov = {}
    for sp in SPLITS:
        s = pl.read_parquet(proc / f"samples_{sp}_{args.protocol}.parquet",
                            columns=["video_id", "target_idx"])
        j = s.join(static, on="video_id", how="left")
        c = _coverage(j)
        # 对照：召回词表口径下的覆盖率。只用于记录「修复前后差了多少」，不进特征。
        c["video_id_known_retrieval_vocab"] = float((j["target_idx"] != OOV).mean())
        cov[sp] = c
        log.info("%-5s n=%-9s video %.4f (旧口径 %.4f)  author %.4f  tag %.4f  "
                 "duration %.4f  upload %.4f",
                 sp, f"{c['n']:,}", c["video_id_known"],
                 c["video_id_known_retrieval_vocab"], c["author_known"], c["tag_known"],
                 c["duration_known"], c["upload_date_known"])

    spec = {
        "protocol": args.protocol,
        "source": f"{source}.parquet",
        "excluded_files": require(cfg, "leakage", "excluded_files"),
        "vocab_basis": basis,
        "stats_from": {"split": "train", "basis": "train_samples"},
        "pad_index": PAD, "oov_index": OOV,
        "n_videos": len(static),
        "max_tags": max_tags,
        "vocabs": {
            "author": {"min_count": a_min, "n_entities": len(a_vocab),
                       "embedding_rows": len(a_vocab) + 2},
            "tag": {"min_count": t_min, "n_entities": len(t_vocab),
                    "embedding_rows": len(t_vocab) + 2},
            "video": {"min_count": v_min, "n_entities": len(v_vocab),
                      "embedding_rows": len(v_vocab) + 2,
                      "built_from": "train_exposures_only_no_labels"},
        },
        "columns": {
            "duration_ms": {"kind": "numeric", "fill": fill,
                            "transform": "log1p_standardize", "mean": d_mean, "std": d_std},
            "video_age_days": {"kind": "numeric", "fill": 0.0, "transform": "log1p",
                               "note": "请求日 - 上传日，取样时现算；upload_date_known=0 时记 0"},
            **{f: {"kind": "flag", "fill": 0.0, "transform": "none"} for f in
               ("author_known", "tag_known", "duration_known", "upload_date_known",
                "video_id_known")},
        },
        "coverage": cov,
    }
    (proc / f"rank_item_spec_{args.protocol}.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("%s 行 x %d 列 -> %s (%.0f MB)，max_tags=%d，耗时 %.1fs",
             f"{len(static):,}", len(static.columns), out_static.name,
             out_static.stat().st_size / 1024**2, max_tags, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
