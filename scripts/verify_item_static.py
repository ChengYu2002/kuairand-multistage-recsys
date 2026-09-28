"""独立验算物品静态属性表（阶段 0 的 ② / ③）。

这里的错误全是静默的：age 算错只会让特征失去意义、tag 编错只会让多热向量指向别的类别，
都不抛异常。因此每一项都用独立路径重算或手算。

检查分五层：
  A. age_days 的定义：YYYYMMDD 是十进制拼接，**直接相减不是天数**。这里用手算的跨月、
     跨年、闰年三组边界钉死正确行为，并证明「整数直接相减」确实会给出不同答案
     （否则这条检查毫无意义）。
  B. 独立重算，分两级：
     B1 抽样逐行 —— 用 Python 字典 + 逐行循环（与生产代码的 polars join 完全不同的路径）
        重算若干视频，逐值比对。抓的是「规则写错了」。
     B2 全表 —— 覆盖全部 897,503 行。**必要性是实测出来的**：只做 B1 时，把某一行的
        tag_indices 改错，28 项验算全部通过（抽 400 行命中该行的概率仅 0.045%）。
        抽样只能抓系统性错误，抓不到少量行的错值。
  C. 不变量：行数与词表一致、index 落在各自词表值域内、tag 列表非空且已排序去重、
     has_* 标志与实际缺失一致。
  D. 数据完整性：不存在「曝光日早于上传日」的记录（age < 0 意味着数据有问题或口径错了）。
  E. 回归锁：缺失计数等与 config 冻结值逐项比对。

用法：
    python scripts/verify_item_static.py --config configs/retrieval.yaml --require-golden
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.build_vocab import OOV
from src.features.video_age import age_days
from src.utils.config import load_config, project_path, require


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--require-golden", action="store_true")
    ap.add_argument("--n-sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))

    failures: list[str] = []
    checks = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  {'OK  ' if ok else 'FAIL'} {msg}")
        if not ok:
            failures.append(msg)

    it = pl.read_parquet(proc / f"item_static_{args.protocol}.parquet")
    vocab = pl.read_parquet(proc / f"vocab_video_{args.protocol}.parquet")
    a_vocab = pl.read_parquet(proc / f"vocab_author_{args.protocol}.parquet")
    t_vocab = pl.read_parquet(proc / f"vocab_tag_{args.protocol}.parquet")
    vf = pl.read_parquet(proc / "video_features.parquet").select(
        "video_id", "author_id", "upload_dt", "tag", "video_duration"
    )

    # ---------- A. age_days 定义 ----------
    print("\n=== A. age_days 的定义（YYYYMMDD 不能直接相减）===")
    cases = [
        (20220501, 20220430, 1, "跨月"),
        (20220101, 20211231, 1, "跨年"),
        (20200301, 20200228, 2, "闰年 2/29 要算进去"),
        (20210301, 20210228, 1, "平年没有 2/29"),
        (20220508, 20220408, 30, "同月内 30 天"),
    ]
    df = pl.DataFrame({"req": [c[0] for c in cases], "up": [c[1] for c in cases]})
    got = df.with_columns(age_days(pl.col("req"), pl.col("up")).alias("age"))["age"].to_list()
    for (req, up, want, label), g in zip(cases, got):
        check(g == want, f"{label}: {req} − {up} = {want} 天（得到 {g}）")
    naive = [req - up for req, up, _, _ in cases]
    check(naive != [c[2] for c in cases],
          f"整数直接相减确实不同（{naive} vs {[c[2] for c in cases]}）—— 该陷阱真实存在")
    nulls = pl.DataFrame({"req": [20220501], "up": [None]}, schema={"req": pl.Int32, "up": pl.Int32})
    check(nulls.with_columns(age_days(pl.col("req"), pl.col("up")).alias("a"))["a"].null_count() == 1,
          "upload_date 为 null 时 age 也为 null（不静默填 0）")

    # ---------- B. 独立重算 ----------
    print("\n=== B. 独立重算（抽样比对）===")
    sample = it.sample(n=min(args.n_sample, len(it)), seed=args.seed).select("video_id")
    ref = sample.join(vf, on="video_id", how="left")
    a_map = dict(zip(a_vocab["id"].to_list(), a_vocab["index"].to_list()))
    t_map = dict(zip(t_vocab["id"].to_list(), t_vocab["index"].to_list()))
    want_rows = []
    for r in ref.iter_rows(named=True):
        tg = r["tag"]
        if tg is None:
            tags = [OOV]
        else:
            parts = [x.strip() for x in tg.split(",") if x.strip()]
            tags = sorted({t_map.get(x, OOV) for x in parts}) or [OOV]
        up = r["upload_dt"]
        want_rows.append({
            "video_id": r["video_id"],
            "author_index": a_map.get(r["author_id"], OOV),
            "tag_indices": tags,
            "upload_date": None if up is None else int(up.replace("-", "")),
            "duration_ms": None if r["video_duration"] is None else int(r["video_duration"]),
        })
    want = pl.DataFrame(want_rows).sort("video_id")
    got_df = (
        it.join(sample, on="video_id", how="semi")
        .select("video_id", "author_index", "tag_indices", "upload_date", "duration_ms")
        .sort("video_id")
    )
    for col in ("author_index", "upload_date", "duration_ms"):
        check(got_df[col].to_list() == want[col].to_list(), f"{col} 与独立重算一致（{len(want)} 个样本）")
    check([list(x) for x in got_df["tag_indices"].to_list()]
          == [list(x) for x in want["tag_indices"].to_list()],
          f"tag_indices 与独立重算一致（{len(want)} 个样本）")

    # ---------- B2. 全表比对 ----------
    print("\n=== B2. 全表比对（覆盖全部 897,503 行）===")
    full = it.select("video_id", "author_index", "upload_date", "duration_ms").join(
        vf, on="video_id", how="left"
    )
    # 必须用 ne_missing 而不是 !=：polars 里 null != 5 的结果是 **null**，而 filter 会把
    # null 行丢掉 —— 于是「该有值却是 null」和「该是 null 却有值」这两类错误全部漏过。
    # 实测：把两行的 upload_date 改成 null，用 != 抓到 0 行，用 ne_missing 抓到 2 行。
    # 冻结的缺失计数只能发现「缺失总数变了」，两个视频的缺失位置对调时数量不变，照样漏。
    check(
        full.filter(
            pl.col("upload_date").ne_missing(
                pl.col("upload_dt").str.to_date("%Y-%m-%d").dt.strftime("%Y%m%d").cast(pl.Int32)
            )
        ).height == 0,
        "全表 upload_date 与原表一致（含 null 位置）",
    )
    check(
        full.filter(
            pl.col("duration_ms").ne_missing(pl.col("video_duration").cast(pl.Int32))
        ).height == 0,
        "全表 duration_ms 与原表一致（含 null 位置）",
    )
    a_lookup = a_vocab.rename({"id": "author_id", "index": "want"})
    check(
        full.join(a_lookup, on="author_id", how="left")
        .with_columns(pl.col("want").fill_null(OOV))
        .filter(pl.col("author_index").ne_missing(pl.col("want"))).height == 0,
        "全表 author_index 与原表 + 词表一致",
    )
    # tag：比对 (video_id, tag_index) 的完整集合，逐对而非仅计数
    got_pairs = (
        it.select(pl.col("video_id").cast(pl.Int32), "tag_indices")
        .explode("tag_indices", empty_as_null=True)
        .rename({"tag_indices": "tag_index"})
        .with_columns(pl.col("tag_index").cast(pl.Int32))
        .sort(["video_id", "tag_index"])
    )
    want_pairs = (
        vf.join(it.select("video_id"), on="video_id", how="semi")
        .select("video_id", "tag")
        .with_columns(pl.col("tag").str.split(","))
        .explode("tag", empty_as_null=True)
        .with_columns(pl.col("tag").str.strip_chars())
        .filter(pl.col("tag").is_not_null() & (pl.col("tag") != ""))
        .join(t_vocab.rename({"id": "tag", "index": "tag_index"}), on="tag", how="left")
        .with_columns(pl.col("tag_index").fill_null(OOV))
        .select(pl.col("video_id").cast(pl.Int32), pl.col("tag_index").cast(pl.Int32))
        .unique()
    )
    # 完全没有 tag 的视频在生产表里是 [OOV]，参考侧要补上
    no_tag = it.filter(pl.col("has_tag") == 0).select(
        pl.col("video_id").cast(pl.Int32)
    ).with_columns(pl.lit(OOV, dtype=pl.Int32).alias("tag_index"))
    want_pairs = pl.concat([want_pairs, no_tag]).sort(["video_id", "tag_index"])
    check(got_pairs.height == want_pairs.height,
          f"全表 (video, tag) 对数一致（{got_pairs.height:,}）")
    check(got_pairs.equals(want_pairs), "全表 (video, tag) 对逐行一致")

    # ---------- C. 不变量 ----------
    print("\n=== C. 不变量 ===")
    check(len(it) == len(vocab), f"行数 = 词表大小（{len(it):,}）")
    check(it.sort("video_index")["video_index"].to_list()
          == vocab.sort("index")["index"].to_list(), "video_index 与词表逐行一致")
    a_max, t_max = int(a_vocab["index"].max()), int(t_vocab["index"].max())
    check(bool(it["author_index"].is_between(OOV, a_max).all()),
          f"author_index 全部落在 [{OOV}, {a_max:,}]")
    flat = it["tag_indices"].explode(empty_as_null=True)
    check(bool(flat.is_between(OOV, t_max).all()), f"tag_index 全部落在 [{OOV}, {t_max}]")
    check(bool((it["n_tags"] >= 1).all()), "每个视频至少 1 个 tag index（无 tag 者为 [OOV]）")
    bad = it.filter(
        (pl.col("tag_indices").list.unique().list.len() != pl.col("tag_indices").list.len())
        | (pl.col("tag_indices").list.sort() != pl.col("tag_indices"))
    )
    check(bad.height == 0, "tag_indices 已去重且升序（保证可复现）")
    for flag, col in (("has_upload_date", "upload_date"), ("has_duration", "duration_ms")):
        mis = it.filter((pl.col(flag) == 0) != pl.col(col).is_null()).height
        check(mis == 0, f"{flag} 与 {col} 的缺失完全对应")

    # ---------- D. 数据完整性 ----------
    print("\n=== D. 数据完整性：曝光不得早于上传 ===")
    logs = pl.scan_parquet(proc / "logs_split.parquet").select("video_id", "date").collect()
    joined = (
        logs.join(it.select("video_id", "upload_date"), on="video_id", how="inner")
        .filter(pl.col("upload_date").is_not_null())
        .with_columns(age_days(pl.col("date"), pl.col("upload_date")).alias("age"))
    )
    n_neg = joined.filter(pl.col("age") < 0).height
    check(n_neg == 0, f"无 age < 0 的曝光记录（检查了 {len(joined):,} 条）")
    check(int(joined["age"].min()) == 0, "最小 age = 0（当天上传当天曝光）")

    # ---------- E. 回归锁 ----------
    print("\n=== E. 口径回归锁 ===")
    golden = ((cfg.get("item_static") or {}).get("expected"))
    if not golden:
        if args.require_golden:
            check(False, "未配置 item_static.expected，但要求 --require-golden")
        else:
            print("  WARN 未配置 item_static.expected —— 本次运行没有口径回归保护")
    else:
        actual = {
            "rows": len(it),
            "missing_upload_date": it.filter(pl.col("has_upload_date") == 0).height,
            "missing_tag": it.filter(pl.col("has_tag") == 0).height,
            "missing_duration": it.filter(pl.col("has_duration") == 0).height,
            "author_oov": it.filter(pl.col("author_index") == OOV).height,
            "max_tags_per_video": int(it["n_tags"].max()),
            "max_age_days": int(joined["age"].max()),
        }
        for k, wv in golden.items():
            check(actual.get(k) == wv, f"{k} = {actual.get(k):,}（冻结值 {wv:,}）")

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("物品静态属性验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("物品静态属性验算通过（age 定义 / 抽样+全表重算 / 不变量 / 完整性 / 回归锁）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
