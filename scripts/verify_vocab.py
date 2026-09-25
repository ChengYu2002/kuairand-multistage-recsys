"""独立验算 ID 词表（plan §13）。

词表错了是**静默**的：embedding 里「第 N 行」学到的东西会对应到另一个实体，
不抛异常、不崩，只是指标莫名其妙变差。而且这类错误会在 checkpoint 里固化下来，
等训练完再发现就全废了。因此在任何模型开训之前必须先把它钉死。

检查分四层：
  A. 往返一致：id -> index -> id 必须回到原值；index 连续、无重复、无空洞。
  B. 与候选库的关系：候选库必须**恰好占据 index 2~N+1 这一段连续前缀**，且顺序逐行一致 ——
     物品塔打分靠切这个区间，切错等于静默改了召回范围。词表其余部分必须恰好是
     train 段正向视频里候选库之外的那些。
  C. 保留位未被占用：没有实体映射到 index 0 (PAD)；声明了 OOV 的词表，也没有实体占用 index 1。
  C2. author / tag 成员与顺序：从 video_features 用独立路径重算，逐行比对 ——
      只验数量和连续性的话，「数量相同但映射错位」会整批漏过。
  D. 回归锁：与 config 中冻结的实测值逐项比对。

用法：
    python scripts/verify_vocab.py --config configs/retrieval.yaml --require-golden
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.build_vocab import OOV, PAD
from src.utils.config import load_config, project_path, require


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--require-golden", action="store_true")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
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

    meta = json.loads((proc / f"vocab_meta_{args.protocol}.json").read_text(encoding="utf-8"))
    tables = {
        n: pl.read_parquet(proc / f"vocab_{n}_{args.protocol}.parquet")
        for n in ("video", "author", "tag")
    }

    # ---------- A. 往返一致 ----------
    print("\n=== A. 往返一致 / index 结构 ===")
    for name, df in tables.items():
        idx = df.get_column("index")
        check(df.get_column("id").n_unique() == len(df), f"{name}: id 无重复（{len(df):,} 个）")
        check(idx.n_unique() == len(df), f"{name}: index 无重复")
        lo, hi = int(idx.min()), int(idx.max())
        check(hi - lo + 1 == len(df), f"{name}: index 连续无空洞（{lo}~{hi}）")
        # id -> index -> id 往返
        fwd = dict(zip(df.get_column("id").to_list(), idx.to_list()))
        bwd = dict(zip(idx.to_list(), df.get_column("id").to_list()))
        check(all(bwd[fwd[i]] == i for i in list(fwd)[:5000]),
              f"{name}: id -> index -> id 往返一致（抽查 5,000）")

    # ---------- B. 与候选库的关系 ----------
    print("\n=== B. 候选库占据连续前缀 ===")
    catalog = pl.read_parquet(proc / f"catalog_{meta['protocol']}.parquet")
    v = tables["video"]
    n_cat = len(catalog)
    lo_c, hi_c = meta["video_catalog_index_min"], meta["video_catalog_index_max"]
    check(hi_c - lo_c + 1 == n_cat, f"meta 记录的候选区间 [{lo_c}, {hi_c}] 长度 = {n_cat:,}")
    check(lo_c == int(v.get_column("index").min()),
          f"候选区间起点 {lo_c} 就是词表最小 index（候选库排在最前）")
    head = v.filter(pl.col("index") <= hi_c).sort("index")
    check(head.get_column("id").to_list() == catalog.get_column("video_id").to_list(),
          f"index {lo_c}~{hi_c:,} 与候选库逐行一致（物品塔按此切片打分）")

    tail = v.filter(pl.col("index") > hi_c)
    check(tail.join(catalog.select("video_id"), left_on="id", right_on="video_id",
                    how="semi").height == 0,
          f"index >{hi_c:,} 的 {len(tail):,} 个 item 与候选库不重叠")

    positive = pl.any_horizontal(
        [pl.col(c) == 1 for c in require(cfg, "eval", "positive_signal")]
    )
    tr_pos = (
        pl.scan_parquet(proc / "logs_split.parquet")
        .filter(pl.col("split").is_in(require(cfg, "vocab", "video", "history_splits")) & positive)
        .select("video_id").unique().collect()
    )
    want_tail = (
        tr_pos.join(catalog.select("video_id"), on="video_id", how="anti")
        .sort("video_id").get_column("video_id").to_list()
    )
    check(tail.sort("index").get_column("id").to_list() == want_tail,
          f"尾部成员与顺序 = train 段正向视频 \\ 候选库，按 id 升序（{len(want_tail):,} 个）")

    # ---------- C. 保留位 ----------
    print("\n=== C. PAD / OOV 保留位 ===")
    for name, df in tables.items():
        spec = meta["vocabs"][name]
        check(int(df.get_column("index").min()) > PAD, f"{name}: 无实体占用 index {PAD} (PAD)")
        if spec["has_oov"]:
            check(int(df.get_column("index").min()) > OOV,
                  f"{name}: 无实体占用 index {OOV} (OOV)")
        check(spec["embedding_rows"] == int(df.get_column("index").max()) + 1,
              f"{name}: embedding_rows = max(index)+1 = {spec['embedding_rows']:,}")

    # ---------- C2. author / tag 独立重算 ----------
    print("\n=== C2. author / tag 成员与顺序（独立重算）===")
    vf = pl.read_parquet(proc / "video_features.parquet").select("video_id", "author_id", "tag")
    side = catalog.select("video_id").join(vf, on="video_id", how="left")

    a_min = require(cfg, "vocab", "author", "min_videos")
    want_a = (
        side.filter(pl.col("author_id").is_not_null())
        .group_by("author_id").agg(pl.len().alias("n"))
        .filter(pl.col("n") >= a_min)
        .sort(["n", "author_id"], descending=[True, False])
        .get_column("author_id").to_list()
    )
    check(tables["author"].sort("index").get_column("id").to_list() == want_a,
          f"author 成员与顺序逐行一致（min_videos={a_min}，{len(want_a):,} 个）")

    t_min = require(cfg, "vocab", "tag", "min_videos")
    want_t = (
        side.filter(pl.col("tag").is_not_null())
        .with_columns(pl.col("tag").str.split(","))
        .explode("tag", empty_as_null=True)
        .with_columns(pl.col("tag").str.strip_chars())
        .filter(pl.col("tag").is_not_null() & (pl.col("tag") != ""))
        .group_by("tag").agg(pl.len().alias("n"))
        .filter(pl.col("n") >= t_min)
        .sort(["n", "tag"], descending=[True, False])
        .get_column("tag").to_list()
    )
    check(tables["tag"].sort("index").get_column("id").to_list() == want_t,
          f"tag 成员与顺序逐行一致（{len(want_t)} 个）")

    # ---------- D. 回归锁 ----------
    print("\n=== D. 口径回归锁 ===")
    golden = (cfg.get("vocab") or {}).get("expected")
    if not golden:
        if args.require_golden:
            check(False, "未配置 vocab.expected，但本次运行要求 --require-golden")
        else:
            print("  WARN 未配置 vocab.expected —— 本次运行没有口径回归保护")
    else:
        actual = {k: meta[k] for k in ("video_catalog_index_min", "video_catalog_index_max")}
        for name, df in tables.items():
            actual[f"{name}_entities"] = len(df)
            actual[f"{name}_rows"] = int(df.get_column("index").max()) + 1
        for key, want in golden.items():
            got = actual.get(key)
            check(got == want, f"{key} = {got:,}（冻结值 {want:,}）")

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("词表验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("词表验算通过（往返一致 / 候选库前缀 / 保留位 / 独立重算 / 回归锁）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
