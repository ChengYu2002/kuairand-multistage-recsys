"""独立验算排序侧物品元数据（src/features/rank_item_static.py）。

这张表错了不会报错。作者行号错位只会让模型学到一堆无意义的向量，AUC 略低一点；
而四个排序模型共用这一份元数据，会一起错到同一个方向，对比表看上去完全正常。
所以这里全部**独立重算**，不复用 builder 的任何代码路径，也不看数字好不好看。

六层检查：
  A. 词表只由 train 构建：门槛独立重算；只在 valid/test 出现的作者必须一个都不在词表里
     （这是「只用 train」的可证伪检验，不是看注释）。
  A2. video 词表**与标签无关**（P0）：召回词表 =「候选库 ∪ train 正向视频」，成员身份
     由训练标签决定 —— train 段 OOV 样本的 is_click 正样本率**恰好为 0**。这里断言
     排序 video 词表没有这个性质，并同时断言召回词表**确实有**（可证伪：证明这条检查
     不是装饰）。
  B. 静态表逐字段对账：随机抽样回 basic 元数据手工核对；缺失标志与数值必须一致。
  C. 归一化统计只用 train：fill / mean / std 独立重算；再用 valid+test 重算一遍，
     必须得到**不同**的值 —— 否则说明统计口径根本没限定在 train。
  D. 覆盖率：spec 里写的每个数字独立复算。
  E. 防泄漏：statistic 表从未被读；曝光后字段不在表里；video_age 没有被预先算成固定值。
  F. 召回侧零改动：比对改动前记录的文件哈希；双塔 checkpoint 仍能对上召回词表；
     并断言排序词表与召回词表的 index 含义确实不同（混用会静默出错，所以要把它钉住）。

用法：
    python scripts/verify_rank_item_static.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, project_path, require

PAD, OOV = 0, 1
SPLITS = ("train", "valid", "test")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    proc = project_path(require(cfg, "dataset", "processed_dir"))
    p = a.protocol
    rng = np.random.default_rng(a.seed)

    failures: list[str] = []
    checks = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  {'OK  ' if ok else 'FAIL'} {msg}")
        if not ok:
            failures.append(msg)

    spec = json.loads((proc / f"rank_item_spec_{p}.json").read_text("utf-8"))
    static = pl.read_parquet(proc / f"item_static_rank_{p}.parquet")
    av = pl.read_parquet(proc / f"vocab_author_rank_{p}.parquet")
    tv = pl.read_parquet(proc / f"vocab_tag_rank_{p}.parquet")
    vv = pl.read_parquet(proc / f"vocab_video_rank_{p}.parquet")
    vf = pl.read_parquet(proc / "video_features.parquet").select(
        "video_id", "author_id", "tag", "video_duration", "upload_dt")
    samples = {s: pl.read_parquet(proc / f"samples_{s}_{p}.parquet",
                                  columns=["video_id", "target_idx"]) for s in SPLITS}
    tr_meta = samples["train"].select("video_id").join(vf, on="video_id", how="left")

    # ---------- A. 词表只由 train 构建 ----------
    print("\n=== A. 词表只由 train 构建 ===")
    a_min = int(require(cfg, "rank_item", "author_min_count"))
    t_min = int(require(cfg, "rank_item", "tag_min_count"))

    # 独立重算：train 曝光次数 >= 门槛的作者集合，必须与词表完全相等
    a_cnt = tr_meta.filter(pl.col("author_id").is_not_null()).group_by("author_id").len()
    want_a = set(a_cnt.filter(pl.col("len") >= a_min)["author_id"].to_list())
    got_a = set(av["id"].to_list())
    check(want_a == got_a,
          f"author 词表 == train 曝光 >= {a_min} 的作者集合"
          f"（独立重算 {len(want_a):,} / 表内 {len(got_a):,}）")

    tr_tags = (tr_meta.filter(pl.col("tag").is_not_null())
               .with_columns(pl.col("tag").str.split(","))
               .explode("tag", empty_as_null=True)
               .filter(pl.col("tag").is_not_null() & (pl.col("tag") != ""))
               .group_by("tag").len())
    want_t = set(tr_tags.filter(pl.col("len") >= t_min)["tag"].to_list())
    check(want_t == set(tv["id"].to_list()),
          f"tag 词表 == train 曝光 >= {t_min} 的 tag 集合（{len(want_t)} 个）")

    # 可证伪：只在 valid/test 出现过的作者，一个都不许进词表
    tr_authors = set(tr_meta["author_id"].drop_nulls().to_list())
    later = (pl.concat([samples["valid"], samples["test"]]).select("video_id")
             .join(vf, on="video_id", how="left")["author_id"].drop_nulls().to_list())
    unseen = set(later) - tr_authors
    leaked = unseen & got_a
    check(not leaked,
          f"只在 valid/test 出现的作者 {len(unseen):,} 个，进词表的 {len(leaked)} 个（必须为 0）")

    idx = av["index"].to_numpy()
    check(idx.min() == OOV + 1 and len(np.unique(idx)) == len(idx)
          and idx.max() - idx.min() + 1 == len(idx),
          f"author index 从 {OOV + 1} 起、连续、无重复（{idx.min()}..{idx.max()}，{len(idx):,} 行）")
    # 排序键必须是全序：(n desc, id asc) 重排一次结果相同
    re_sorted = av.sort(["n", "id"], descending=[True, False])["id"].to_list()
    check(re_sorted == av["id"].to_list(), "author 词表顺序由 (n desc, id asc) 唯一确定（可复现）")

    # ---------- A2. video 词表与标签无关（P0） ----------
    print("\n=== A2. video 词表与标签无关（P0 修复） ===")
    v_min = int(require(cfg, "rank_item", "video_min_count"))
    v_all = pl.read_parquet(proc / f"samples_train_{p}.parquet", columns=["video_id"])
    v_cnt = v_all.group_by("video_id").len().filter(pl.col("len") >= v_min)
    want_v = set(v_cnt["video_id"].to_list())
    check(want_v == set(vv["id"].to_list()),
          f"video 词表 == train 曝光 >= {v_min} 的视频集合"
          f"（独立重算 {len(want_v):,}）—— 只用曝光次数，不碰任何标签列")
    check(spec["vocabs"]["video"]["built_from"] == "train_exposures_only_no_labels",
          "spec 记录了构建口径")
    # 集合对了但顺序不稳定 -> 所有集合类检查照样通过，而重建后 checkpoint 的第 N 行
    # 会换成另一个视频：不报错，只是指标莫名变差。所以顺序必须单独验（与 author 同口径）。
    vidx = vv["index"].to_numpy()
    check(vidx.min() == OOV + 1 and len(np.unique(vidx)) == len(vidx)
          and vidx.max() - vidx.min() + 1 == len(vidx),
          f"video index 从 {OOV + 1} 起、连续、无重复（{vidx.min()}..{vidx.max()}，"
          f"{len(vidx):,} 行）")
    check(vv.sort(["n", "id"], descending=[True, False])["id"].to_list() == vv["id"].to_list(),
          "video 词表顺序由 (train 曝光数 desc, video_id asc) 唯一确定（全序，可复现）")
    # n 列必须真的是 train 曝光次数，不是别的东西
    want_n = (v_cnt.rename({"len": "n_want"})
              .join(vv, left_on="video_id", right_on="id", how="inner").sort("index"))
    check(np.array_equal(want_n["n_want"].to_numpy(), want_n["n"].to_numpy()),
          "词表里的 n 列 == 独立重算的 train 曝光次数")
    # 交叉验证：门槛 5 应恰好等于候选库（两边独立算出同一集合）
    cat = set(pl.read_parquet(proc / f"catalog_{p}.parquet")["video_id"].to_list())
    check(want_v == cat if v_min == 5 else True,
          f"门槛 {v_min} 下与候选库 {len(cat):,} 个"
          f"{'逐个相同（独立交叉验证）' if want_v == cat else '不同（门槛非 5 时正常）'}")

    # 核心：两种词表的成员标志在 train 上对标签的行为
    labels_all = require(cfg, "labels", "tasks")
    tr_s = pl.read_parquet(proc / f"samples_train_{p}.parquet",
                           columns=["video_id", "target_idx", *labels_all])
    rank_known = tr_s["video_id"].is_in(list(want_v)).to_numpy()
    retr_known = (tr_s["target_idx"] != OOV).to_numpy()
    bad_rank, bad_retr = [], []
    for lab in labels_all:
        y = tr_s[lab].to_numpy()
        if (~rank_known).any() and y[~rank_known].mean() == 0.0:
            bad_rank.append(lab)
        if (~retr_known).any() and y[~retr_known].mean() == 0.0:
            bad_retr.append(lab)
    check(not bad_rank,
          f"排序词表：train 段 OOV 组在五个标签上正样本率都不为 0（命中 {bad_rank}）")
    check(set(bad_retr) == {"is_click", "long_view"},
          f"可证伪：召回词表的 OOV 组在 {bad_retr} 上正样本率恰好为 0 "
          "-> 这条检查真的能抓到 P0，而召回词表确实有这个性质")

    # ---------- B. 静态表逐字段对账 ----------
    print("\n=== B. 静态表逐字段对账 ===")
    check(len(static) == len(vf) == spec["n_videos"],
          f"行数 == basic 表视频数 {len(vf):,}")
    check(static["video_id"].n_unique() == len(static), "video_id 唯一")

    a_map = dict(zip(av["id"].to_list(), av["index"].to_list(), strict=True))
    t_map = dict(zip(tv["id"].to_list(), tv["index"].to_list(), strict=True))
    pick = rng.choice(len(vf), 300, replace=False)
    src = vf[pick].to_dicts()
    got = static.join(pl.DataFrame({"video_id": [r["video_id"] for r in src]}),
                      on="video_id", how="inner").sort("video_id")
    src_sorted = sorted(src, key=lambda r: r["video_id"])
    bad_a = bad_t = bad_d = bad_u = 0
    for s_row, g_row in zip(src_sorted, got.to_dicts(), strict=True):
        if g_row["author_rank_idx"] != a_map.get(s_row["author_id"], OOV):
            bad_a += 1
        want_tags = [t_map.get(t, OOV) for t in (s_row["tag"] or "").split(",") if t]
        if list(g_row["tag_rank_idx"] or []) != want_tags:
            bad_t += 1
        sd, gd = s_row["video_duration"], g_row["duration_ms"]
        if not ((sd is None and gd is None) or (sd is not None and gd is not None
                                                and abs(float(sd) - float(gd)) <= 1.0)):
            bad_d += 1
        want_u = int((s_row["upload_dt"] or "0").replace("-", "") or 0)
        if g_row["upload_date"] != want_u:
            bad_u += 1
    check(bad_a == 0, f"随机 300 个视频：author 行号手工核对（错 {bad_a}）")
    check(bad_t == 0, f"随机 300 个视频：tag 行号手工核对（错 {bad_t}）")
    check(bad_d == 0, f"随机 300 个视频：时长原值手工核对（错 {bad_d}）")
    check(bad_u == 0, f"随机 300 个视频：upload_dt -> YYYYMMDD 手工核对（错 {bad_u}）")

    tags_flat = static["tag_rank_idx"].explode()
    check(int(tags_flat.drop_nulls().min()) >= OOV,
          f"tag 行号不含 PAD（最小 {int(tags_flat.drop_nulls().min())} >= {OOV}）")
    check((static["tag_rank_idx"].list.len() == static["n_tags"]).all()
          and int(static["n_tags"].max()) == spec["max_tags"]
          and static["n_tags"].null_count() == 0
          and static["tag_rank_idx"].null_count() == 0,
          f"n_tags == 列表长度、无 null、最大 {spec['max_tags']}"
          f"（无 tag 的 {int((static['n_tags'] == 0).sum()):,} 个视频是空列表而非 null）")

    # 缺失标志必须与数值一致，否则模型分不清「真的是 0」和「不知道」
    check((static["author_known"] == (static["author_rank_idx"] > OOV)).all(),
          "author_known == (行号 > OOV)")
    v_map = dict(zip(vv["id"].to_list(), vv["index"].to_list(), strict=True))
    want_vi = np.array([v_map.get(v, OOV) for v in static["video_id"].to_numpy()], np.int32)
    check(np.array_equal(static["video_rank_idx"].to_numpy(), want_vi),
          "video_rank_idx 独立重算逐行相同（未进词表的落 OOV）")
    check((static["video_id_known"] == (static["video_rank_idx"] > OOV)).all(),
          "video_id_known == (video 行号 > OOV)")
    check((static["duration_known"] == static["duration_ms"].is_not_null()).all(),
          "duration_known == 时长非空")
    check((static["upload_date_known"] == (static["upload_date"] > 0)).all(),
          "upload_date_known == 上传日期 > 0")
    want_tk = static["tag_rank_idx"].list.eval(pl.element() > OOV).list.any().fill_null(False)
    check((static["tag_known"] == want_tk).all(), "tag_known == 至少一个 tag 进了词表")

    # ---------- C. 归一化统计只用 train ----------
    print("\n=== C. 归一化统计只用 train ===")
    d = spec["columns"]["duration_ms"]
    tr_d = tr_meta["video_duration"]
    fill = float(tr_d.drop_nulls().median())
    lg = np.log1p(tr_d.fill_null(fill).to_numpy().astype(np.float64))
    check(abs(d["fill"] - fill) < 1e-6, f"fill == train 曝光非空时长的中位数 {fill:.0f}")
    check(abs(d["mean"] - lg.mean()) < 1e-9 and abs(d["std"] - lg.std()) < 1e-9,
          f"mean/std == log1p(填充后 train 序列) 的均值/标准差（{lg.mean():.4f} / {lg.std():.4f}）")

    # 可证伪：换成 valid+test 重算必须得到不同的值
    ot = (pl.concat([samples["valid"], samples["test"]]).select("video_id")
          .join(vf, on="video_id", how="left")["video_duration"])
    of = float(ot.drop_nulls().median())
    olg = np.log1p(ot.fill_null(of).to_numpy().astype(np.float64))
    check(abs(of - fill) > 1e-6 or abs(olg.mean() - lg.mean()) > 1e-6,
          f"用 valid+test 重算会得到不同统计（中位 {of:.0f} vs {fill:.0f}）-> 口径确实限定在 train")
    check(spec["stats_from"] == {"split": "train", "basis": "train_samples"},
          "spec 记录的 stats_from 口径")

    # ---------- D. 覆盖率独立复算 ----------
    print("\n=== D. 覆盖率独立复算 ===")
    flags = ("video_id_known", "author_known", "tag_known", "duration_known",
             "upload_date_known")
    for sp in SPLITS:
        j = samples[sp].join(static, on="video_id", how="left")
        c = spec["coverage"][sp]
        ok = c["n"] == len(j) and all(abs(c[f] - float(j[f].mean())) < 1e-9 for f in flags)
        # 旧口径只作记录（修复前后差多少），不进特征
        vk = float((j["target_idx"] != OOV).mean())
        ok = ok and abs(c["video_id_known_retrieval_vocab"] - vk) < 1e-9
        check(ok, f"{sp}: video {c['video_id_known']:.4f}（旧口径 "
                  f"{c['video_id_known_retrieval_vocab']:.4f}）author {c['author_known']:.4f} "
                  f"tag {c['tag_known']:.4f} duration {c['duration_known']:.4f}")
    check(spec["coverage"]["test"]["video_id_known"] < 0.2,
          f"test 段 video_id 覆盖率 {spec['coverage']['test']['video_id_known']:.4f} "
          "确实很低 —— 这是建这张表的理由，不是缺陷")

    # ---------- E. 防泄漏 ----------
    print("\n=== E. 防泄漏 ===")
    banned = require(cfg, "leakage", "excluded_files")
    watch = [ROOT / "src" / "features" / "rank_item_static.py",
             ROOT / "scripts" / "verify_rank_item_static.py",
             ROOT / "configs" / "data.yaml"]
    # 只查「以字符串字面量拼出文件名去读」的写法。禁用表的名字出现在注释或 config 的
    # excluded_files 清单里是应该的，不算命中。
    reads = [(f.name, b) for f in watch for b in banned
             if f'"{b}' in f.read_text("utf-8") or f"'{b}" in f.read_text("utf-8")]
    check(not reads, f"排序侧源码没有以文件名读取禁用表 {banned}（命中 {reads}）")
    check(spec["source"] == "video_features.parquet"
          and all(b not in spec["source"] for b in banned),
          f"spec 记录的来源 = {spec['source']}（basic 元数据）")

    stat_cols = {"show_cnt", "play_cnt", "like_cnt", "comment_cnt", "follow_cnt",
                 "play_duration", "valid_play_cnt", "show_user_num"}
    post = set(require(cfg, "leakage", "forbidden_fields"))
    check(not (set(static.columns) & (stat_cols | post)),
          f"表里没有任何统计表列名或曝光后字段（列：{len(static.columns)} 个）")
    check(not any("age" in c for c in static.columns),
          "表里没有 age 列 —— video_age 必须在取样时按请求日现算，不能预先固定")
    check(spec["columns"]["video_age_days"]["transform"] == "log1p",
          "spec 把 video_age 标注为取样时现算")

    # ---------- F. 召回侧零改动 ----------
    print("\n=== F. 召回侧零改动 ===")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "freeze_manifest.py"), "--check"],
                       capture_output=True, text=True, check=False)
    check(r.returncode == 0, "freeze_manifest --check：基线里的旧产物一个都没被改动")
    if r.returncode != 0:
        print("    " + "\n    ".join(r.stdout.strip().splitlines()[-6:]))

    o_vv = pl.read_parquet(proc / f"vocab_video_{p}.parquet")
    check(len(vv) != len(o_vv) and set(vv["id"].to_list()) != set(o_vv["id"].to_list()),
          f"排序 video 词表 {len(vv):,} 行 != 召回 {len(o_vv):,} 行（两份独立，口径不同）")
    r_av = pl.read_parquet(proc / f"vocab_author_rank_{p}.parquet")
    o_av = pl.read_parquet(proc / f"vocab_author_{p}.parquet")
    check(len(r_av) != len(o_av), f"排序 author 词表 {len(r_av):,} 行 != 召回 {len(o_av):,} 行（两份独立）")
    # 同一个 index 在两份词表里指向不同作者 —— 把「不可混用」钉成断言
    common = min(len(r_av), len(o_av))
    same = (r_av.sort("index")["id"].to_numpy()[:common]
            == o_av.sort("index")["id"].to_numpy()[:common])
    check(same.mean() < 0.5,
          f"同一 index 在两份词表里指向同一作者的比例仅 {same.mean():.4f} "
          "-> 混用会静默错位，必须按文件名区分")

    import torch
    meta = json.loads((proc / f"vocab_meta_{p}.json").read_text("utf-8"))
    ck_dir = ROOT / "experiments"
    cks = sorted(ck_dir.glob("two_tower_*_main_seed*.pt"))
    ok_ck = bool(cks)
    for ck_path in cks:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        rows = ck["state_dict"]["item_emb.weight"].shape[0]
        lo_hi = ck.get("catalog_range")
        ok_ck &= (rows == meta["vocabs"]["video"]["embedding_rows"]
                  and lo_hi == [meta["video_catalog_index_min"], meta["video_catalog_index_max"]])
    check(ok_ck, f"{len(cks)} 个双塔 checkpoint 的 item_emb 行数与候选库区间仍对得上召回词表")

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("排序物品元数据验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("验算通过（词表只用 train / 逐字段对账 / 统计只用 train / 覆盖率 / 防泄漏 / 召回零改动）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
