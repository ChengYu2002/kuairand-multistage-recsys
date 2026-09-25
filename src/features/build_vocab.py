"""ID 词表构建（plan §13）—— video_id / author_id / tag → 连续行号。

## 为什么需要这一步

embedding 层本质是一张表，每行一个向量，模型按**行号**查表。而 video_id 的取值范围是
0~437 万，直接拿它当行号就得建 437 万行，其中 95% 永远学不到东西。所以先把用得上的
id 压成连续行号。

**这一步不产生任何 embedding**，只产生对照关系。embedding 在物品塔里随机初始化、由训练
学出来；词表是确定性的、跑一次就固定。二者必须分开：词表固定，3 个 seed 的结果才可比；
词表一变，「第 N 行」就换了实体，checkpoint 静默作废 —— 不报错，只是指标莫名变差。

## 候选集合 ≠ 历史可编码集合

候选库回答「模型可以召回哪些视频」，词表还要回答「用户历史里的视频怎么编码」。
前者的 freq>=5 门槛是为了让负采样对比可测，拿它去限制历史编码毫无道理 —— 实测那样
只有 54.8% 的训练历史槽位、11.1% 的测试历史槽位能编码。因此 video 词表取
**候选库 ∪ train 段正向视频**（897,503 个），覆盖率升到 99.4% / 17.5%。

布局上把候选库放在前面：index 2~194,311 依候选库顺序，其余按 video_id 升序接在后面。
物品塔打分时直接切 [2 : 194,312] 连续区间即可，**召回范围不因词表扩大而改变**。
index 1 留给 OOV —— 不留的话 `mapping.get(vid, 1)` 会把未知视频静默当成 index 1 那个
真实视频。OOV 只用于编码历史，绝不作为训练目标。

## 三张表的约定

    index 0  PAD  历史序列补齐用，不对应任何实体。
                  torch 的 padding_idx=0 会把该行锁成全零且不回传梯度，但**不替代 mask**：
                  mean(dim=1) 仍会把 PAD 算进分母。用户塔必须自己做 masked sum / valid_len。
    index 1  OOV  该字段缺失或未进词表。video / author / tag 都保留该位置。
    index 2+ 实体

video 的顺序沿用 catalog_main 已冻结的 (train_freq desc, video_id asc)，因此行号越小越热门；
author / tag 按 (候选库内视频数 desc, id asc) 排序 —— 都是全序键，保证可复现。

用法：
    python -m src.features.build_vocab --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse
import json

import polars as pl

from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)

PAD, OOV = 0, 1


def build_table(ids: pl.Series, counts: pl.Series | None, oov: bool) -> pl.DataFrame:
    """把一列 id 编成 index。

    counts 为 None 时按给定顺序原样编号（video 用，顺序来自候选库）；
    否则按 (counts desc, id asc) 排序后编号。
    """
    df = pl.DataFrame({"id": ids}) if counts is None else (
        pl.DataFrame({"id": ids, "n": counts}).sort(["n", "id"], descending=[True, False])
    )
    start = OOV + 1 if oov else OOV
    return df.with_columns(
        (pl.int_range(pl.len(), dtype=pl.Int32) + start).alias("index")
    ).select("id", "index", *(["n"] if counts is not None else []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    # 文件名带 protocol：否则跑一次 aux 就会覆盖主协议词表，而且不会有任何提示。
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    args = ap.parse_args()
    cfg = load_config(args.config)

    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    catalog = pl.read_parquet(proc / f"catalog_{args.protocol}.parquet")
    vf = pl.read_parquet(proc / "video_features.parquet").select(
        "video_id", "author_id", "tag"
    )
    side = catalog.select("video_id").join(vf, on="video_id", how="left")

    src = require(cfg, "vocab", "video", "source")
    if src != "catalog_union_train_positive":
        raise NotImplementedError(f"vocab.video.source 目前只支持 catalog_union_train_positive，收到 {src!r}")

    # --- video：候选库 ∪ train 段正向视频；候选库排在前面，保证打分可切连续区间 ---
    v_oov = bool(require(cfg, "vocab", "video", "oov"))
    cat_ids = catalog.get_column("video_id")
    positive = pl.any_horizontal(
        [pl.col(c) == 1 for c in require(cfg, "eval", "positive_signal")]
    )
    tr_pos = (
        pl.scan_parquet(proc / "logs_split.parquet")
        .filter(pl.col("split").is_in(require(cfg, "vocab", "video", "history_splits")) & positive)
        .select("video_id").unique().collect().get_column("video_id")
    )
    extra = (
        pl.DataFrame({"id": tr_pos})
        .join(pl.DataFrame({"id": cat_ids}), on="id", how="anti")
        .sort("id").get_column("id")
    )
    v = build_table(pl.concat([cat_ids.cast(extra.dtype), extra]), None, v_oov)
    n_catalog = len(cat_ids)
    log.info("video 词表 = 候选库 %s ∪ 仅历史 %s = %s",
             f"{n_catalog:,}", f"{len(extra):,}", f"{len(v):,}")

    # --- author：按候选库内视频数卡门槛，其余并入 OOV ---
    a_min = require(cfg, "vocab", "author", "min_videos")
    a_cnt = (
        side.filter(pl.col("author_id").is_not_null())
        .group_by("author_id").agg(pl.len().alias("n"))
        .filter(pl.col("n") >= a_min)
    )
    a_oov = bool(require(cfg, "vocab", "author", "oov"))
    a = build_table(a_cnt.get_column("author_id"), a_cnt.get_column("n"), a_oov)

    # --- tag：逗号分隔多值，拆开后统计 ---
    t_min = require(cfg, "vocab", "tag", "min_videos")
    t_cnt = (
        side.filter(pl.col("tag").is_not_null())
        .with_columns(pl.col("tag").str.split(","))
        .explode("tag", empty_as_null=True)
        .with_columns(pl.col("tag").str.strip_chars())
        .filter(pl.col("tag") != "")
        .group_by("tag").agg(pl.len().alias("n"))
        .filter(pl.col("n") >= t_min)
    )
    t_oov = bool(require(cfg, "vocab", "tag", "oov"))
    t = build_table(t_cnt.get_column("tag"), t_cnt.get_column("n"), t_oov)

    tables = {"video": (v, v_oov), "author": (a, a_oov), "tag": (t, t_oov)}
    meta: dict[str, object] = {
        "protocol": args.protocol, "pad_index": PAD, "oov_index": OOV,
        # 物品塔打分时切 [catalog_index_min : catalog_index_max+1]；
        # 当前为 [2 : 194312]，召回范围不随历史词表扩大而改变。
        "video_catalog_index_min": int(v.get_column("index").min()),
        "video_catalog_index_max": int(v.get_column("index").min()) + n_catalog - 1,
        "vocabs": {},
    }

    print()
    print(f"{'词表':<8}{'实体数':>10}{'index 范围':>14}{'embedding 行数':>16}{'覆盖候选视频':>14}")
    for name, (df, oov) in tables.items():
        lo, hi = int(df["index"].min()), int(df["index"].max())
        size = hi + 1                       # 0..hi，含 PAD（与 OOV，若有）
        if name == "video":
            cov = 1.0   # 候选库是它的子集，按定义全覆盖
        else:
            col = "author_id" if name == "author" else "tag"
            kept = set(df["id"].to_list())
            if name == "author":
                cov = side.get_column(col).is_in(list(kept)).mean()
            else:
                cov = (
                    side.filter(pl.col(col).is_not_null())
                    .with_columns(pl.col(col).str.split(","))
                    .explode(col, empty_as_null=True)
                    .with_columns(pl.col(col).str.strip_chars())
                    .get_column(col).is_in(list(kept))
                    .mean()
                ) * len(side.filter(pl.col(col).is_not_null())) / len(side)
        meta["vocabs"][name] = {
            "n_entities": len(df), "index_min": lo, "index_max": hi,
            "embedding_rows": size, "has_oov": oov,
            "oov_index": OOV if oov else None,
        }
        print(f"{name:<8}{len(df):>10,}{f'{lo}~{hi}':>14}{size:>16,}{cov:>13.1%}")
        df.write_parquet(proc / f"vocab_{name}_{args.protocol}.parquet", compression="zstd")
    print()

    (proc / f"vocab_meta_{args.protocol}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("已写出 vocab_{video,author,tag}_%s.parquet + vocab_meta_%s.json",
             args.protocol, args.protocol)
    log.info("embedding 行数含 index 0 (PAD)%s —— 建 nn.Embedding 时用这个数",
             "，index 1 (OOV)" if any(o for _, o in tables.values()) else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
