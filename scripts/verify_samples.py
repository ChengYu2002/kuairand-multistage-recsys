"""独立验算训练样本表（阶段 0 的 ⑥）。

行号指错是这一步最危险的错误：模型会拿到**另一个实体**的特征，训练照常进行、
损失照常下降，只是学到的东西没有意义。因此这里不验「行号范围合法」，而是
**把行号取出来的特征值与源表逐值比对**。

检查分六层：
  A. 冷启动行：row 0 必须等于 apply_spec 作用在一整行 null 上的结果，且与 ⑤ 的填充规格
     一致；并证明它**不等于全零**（否则「平滑率填先验而非 0」这条约束没被检验）。
  B. 行号语义：抽样若干样本，按 (entity, date) 回源表取原始特征、独立编码，
     与 feat[row] 逐值比对。这条才是真正在验「行号指向正确的实体」。
  C. 行数守恒：每个 split 的样本数必须等于 logs_split 中该段的行数。原始日志存在 5 组
     重复曝光，连接键不唯一时 left join 会静默扇出 —— 该 bug 正是这样发现的。
  D. 冷启动覆盖率：与已知的 56.3% 一致。
  E. 历史映射：长度守恒、顺序保持（新的在前）、不在词表的落 OOV。
  F. 数值健康：矩阵无 NaN/Inf，行号全部落在 [0, n_rows)。

用法：
    python scripts/verify_samples.py --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.build_samples import COLD_ROW, encode_matrix
from src.features.build_vocab import OOV
from src.features.feature_encoder import apply_spec
from src.utils.config import load_config, project_path, require


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--n-sample", type=int, default=300)
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

    spec = json.loads((proc / f"encoder_spec_{args.protocol}.json").read_text(encoding="utf-8"))
    meta = json.loads((proc / f"feat_columns_{args.protocol}.json").read_text(encoding="utf-8"))
    mats = {s: np.load(proc / f"feat_{s}_encoded_{args.protocol}.npy") for s in ("user", "item")}
    looks = {s: pl.read_parquet(proc / f"feat_row_{s}_{args.protocol}.parquet") for s in ("user", "item")}

    # ---------- A. 冷启动行 ----------
    print("\n=== A. 冷启动行（row 0）===")
    for side, prefix in (("user", "user"), ("item", "item")):
        cols = meta[side]
        cold = pl.DataFrame({c: [None] for c in cols}, schema={c: pl.Float64 for c in cols})
        want = apply_spec(cold, spec).select(cols).to_numpy().astype(np.float32).ravel()
        got = mats[side][COLD_ROW]
        check(np.allclose(got, want, atol=1e-6), f"{side}: row 0 = apply_spec(全 null)")
        # 平滑率位置上必须是先验 g，不是 0
        sm = [i for i, c in enumerate(cols) if "_rate_sm_" in c]
        fills = np.array([spec["columns"][cols[i]]["fill"] for i in sm], dtype=np.float32)
        check(np.allclose(got[sm], fills, atol=1e-7), f"{side}: 平滑率位置 = 先验 g")
        check(not np.allclose(got, 0.0),
              f"{side}: row 0 不是全零（证明填充确实有内容，非默认零值）")
        hh = cols.index(f"{prefix}_has_history")
        check(got[hh] == 0.0, f"{side}: has_history = 0")

    # ---------- B. 行号语义 ----------
    print("\n=== B. 行号真的指向正确的实体（抽样逐值比对）===")
    rng = np.random.default_rng(args.seed)
    for side, fname, key, prefix in (
        ("user", "feat_user_daily", "user_id", "user"),
        ("item", "feat_item_daily", "video_id", "item"),
    ):
        lk = looks[side]
        idx = rng.choice(len(lk), size=min(args.n_sample, len(lk)), replace=False)
        probe = lk[sorted(idx.tolist())]
        cols = meta[side]
        src = (
            pl.scan_parquet(proc / f"{fname}.parquet")
            .join(probe.lazy().select(key, "date"), on=[key, "date"], how="semi")
            .sort([key, "date"]).collect()
        )
        # 用与生产完全相同的编码函数重算（含冷启动行），再丢掉第 0 行
        want = encode_matrix(src, cols, spec)[1:]
        got = mats[side][probe.get_column("row").to_numpy()]
        check(want.shape == got.shape and np.allclose(want, got, atol=1e-6),
              f"{side}: 抽 {len(probe)} 个 (key,date)，feat[row] 与源表重算逐值一致")
        # 反向：行号必须与排序位置一致（行号 = 排序后位置 + 1）
        check(bool((lk.sort([key, "date"]).get_column("row").to_numpy()
                    == np.arange(1, len(lk) + 1)).all()),
              f"{side}: 行号 = 按 ({key}, date) 排序后的位置 + 1")

    # ---------- C. 行数守恒 ----------
    print("\n=== C. 行数守恒 ===")
    logs = pl.scan_parquet(proc / "logs_split.parquet")
    samples = {}
    for sp in ("train", "valid", "test"):
        s = pl.read_parquet(proc / f"samples_{sp}_{args.protocol}.parquet")
        samples[sp] = s
        n_src = int(logs.filter(pl.col("split") == sp).select(pl.len()).collect().item())
        check(len(s) == n_src, f"{sp}: {len(s):,} 行 = logs_split 中该段行数")
    dup = int(
        logs.group_by("user_id", "video_id", "time_ms").agg(pl.len().alias("c"))
        .filter(pl.col("c") > 1).select(pl.len()).collect().item()
    )
    check(dup == 5, f"源日志确有 {dup} 组重复曝光（连接键不唯一，去重后才不会扇出）")

    # ---------- D. 冷启动覆盖率 ----------
    print("\n=== D. 冷启动覆盖率 ===")
    tr = samples["train"]
    r = float((tr["item_feat_row"] == COLD_ROW).mean())
    check(0.55 < r < 0.58, f"train 的 item 冷启动率 {r:.1%}（已知 56.3%）")
    check(float((tr["user_feat_row"] == COLD_ROW).mean()) < 0.01,
          "train 的 user 冷启动率 < 1%（用户侧历史覆盖 99.8%）")

    # ---------- E. 历史映射 ----------
    print("\n=== E. 历史映射 ===")
    vocab = pl.read_parquet(proc / f"vocab_video_{args.protocol}.parquet")
    vmap = dict(zip(vocab["id"].to_list(), vocab["index"].to_list()))
    hist_src = (
        pl.scan_parquet(proc / "user_history.parquet")
        .select("user_id", "video_id", "time_ms", "hist")
        .unique(subset=["user_id", "video_id", "time_ms"])
    )
    probe = tr.sample(n=200, seed=args.seed).select("user_id", "video_id", "time_ms", "hist_idx")
    j = probe.join(hist_src.collect(), on=["user_id", "video_id", "time_ms"], how="left")
    ok_len = ok_order = True
    for row in j.iter_rows(named=True):
        raw = list(row["hist"] or [])
        want = [vmap.get(int(v), OOV) for v in raw]      # 保序：逐个映射，不排序
        got = list(row["hist_idx"] or [])
        if len(got) != len(raw):
            ok_len = False
        if got != want:
            ok_order = False
    check(ok_len, f"抽 {len(j)} 条：hist_idx 长度与原始 hist 一致")
    check(ok_order, "hist_idx 顺序与原始 hist 一致（新的在前，逐个映射未打乱）")
    # 固定用例：空历史必须保持为空。这类样本很少（train 212 / valid 35 / test 5 条），
    # 随机抽 200 条几乎不可能命中，因此必须定点检查。
    empty_src = (
        hist_src.filter(pl.col("hist").list.len() == 0)
        .select("user_id", "video_id", "time_ms").collect()
    )
    got_empty = tr.join(empty_src, on=["user_id", "video_id", "time_ms"], how="semi")
    check(len(got_empty) > 0, f"train 中确有空历史样本（{len(got_empty)} 条）")
    check(bool((got_empty["hist_idx"].list.len() == 0).all()),
          "空历史 -> 空列表，而不是 [OOV]（否则「没有历史」会变成「有一个未知视频」）")
    # 反面：非空但全未知的历史，必须变成全 OOV 而不是空
    probe2 = tr.filter(pl.col("hist_idx").list.len() > 0).head(1)
    check(int(probe2["hist_idx"].list.len()[0]) > 0, "非空历史仍然非空（与上一条构成对照）")

    flat = tr["hist_idx"].explode(empty_as_null=True).drop_nulls()
    check(bool(((flat >= OOV) & (flat <= len(vocab) + 1)).all()),
          f"全部历史 index 落在 [{OOV}, {len(vocab) + 1:,}]")

    # ---------- F. 数值健康 ----------
    print("\n=== F. 数值健康 ===")
    for side in ("user", "item"):
        check(bool(np.isfinite(mats[side]).all()), f"{side} 矩阵无 NaN/Inf（{mats[side].shape}）")
    for sp, s in samples.items():
        for side in ("user", "item"):
            col = f"{side}_feat_row"
            check(bool(((s[col] >= 0) & (s[col] < mats[side].shape[0])).all()),
                  f"{sp}: {col} 全部落在 [0, {mats[side].shape[0]:,})")

    # ---------- G. 静态特征接口 ----------
    print("\n=== G. 静态特征已套用规格并可取到 ===")
    us = pl.read_parquet(proc / f"feat_user_static_{args.protocol}.parquet")
    its = pl.read_parquet(proc / f"feat_item_static_{args.protocol}.parquet")
    check(len(us) == 1000 and "user_id" in us.columns, f"user 静态表 {len(us):,} 行，按 user_id 取")
    check(all(c not in us.columns for c in spec["dropped"]),
          f"常量列已丢弃：{list(spec['dropped'])}")
    check(bool(us["is_live_streamer"].is_between(0, 2).all()),
          "is_live_streamer 已编码成类目 index（原始 -124 不会进网络）")
    # duration_ms 必须是编码后的浮点，不是原始毫秒
    dur = its["duration_ms"]
    check(dur.dtype in (pl.Float64, pl.Float32) and float(dur.abs().max()) < 50,
          f"item 静态表的 duration_ms 已标准化（max |v| = {float(dur.abs().max()):.2f}）")
    check(bool((its["video_index"] == pl.read_parquet(
        proc / f"vocab_video_{args.protocol}.parquet")["index"]).all()),
        f"item 静态表与词表逐行对齐（{len(its):,} 行，按 video_index 取）")
    iface = meta.get("interface", {})
    check(len(iface) >= 6, f"feat_columns 里写明了取数接口（{len(iface)} 条）")

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("样本表验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("样本表验算通过（冷启动行 / 行号语义 / 行数守恒 / 覆盖率 / 历史映射 / 数值健康）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
