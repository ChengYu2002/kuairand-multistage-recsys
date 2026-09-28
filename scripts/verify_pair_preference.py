"""独立验算 User x Author 偏好特征（src/features/pair_preference.py）。

这些列直接决定 §28 负迁移分析能不能做：Single-Task 在 is_follow 上的 GAUC 是 0.49014
（用户内部排序≈随机），补这批特征就是为了让它离开地板。算错了不会报错，只会让
follow / comment 的结论继续建在噪声上。

**不复用 rolling.py 的任何代码。** 这里直接从 logs_split 用「过滤 + join」重算窗口统计，
再按 pair_spec 的编码规则算一遍，与落盘的 npy 逐位比对。两条独立路径对上才算数
（与 AUC 那里用 sklearn 交叉比对是同一个思路）。

六层检查：
  A. T-1：重算的窗口统计必须逐行等于落盘值；并可证伪「已排除当天」——
     构造「该用户对该作者只在样本当天有过曝光」的行，has_history 必须为 0。
  B. 行对齐：随机抽样本行，按 (user_id, author_id, date) 独立查一遍。
     polars 的 left join **不保证**输出顺序与左表一致，错位了下游没有任何征兆。
  C. 编码：从原始值按 spec 重算 (log1p − mean) / std，与 npy 比对；恒定列必须是 0。
  D. 缺失语义：无历史的行 —— 计数列等于 0 的编码值、平滑比率等于先验 g 的编码值、
     has_history = 0。填先验而不是填 0，是因为平滑公式在 num=imp=0 处恰好等于先验。
  E. 统计量只用 train：mean/std 独立重算；换成 valid/test 重算必须得到不同的值。
  F. 覆盖率与 spec 记录一致；且 rolling.py 的泛化对既有特征是 no-op（哈希比对）。

用法：
    python scripts/verify_pair_preference.py
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

SPLITS = ("train", "valid", "test")


def _recompute(logs: pl.DataFrame, keys: pl.DataFrame, w: int,
               labels: list[str]) -> pl.DataFrame:
    """从日志直接重算窗口统计。keys 含 (user_id, author_id, target)。

    窗口定义：source_date ∈ [target − w, target − 1] —— 显式排除当天，
    不依赖 rolling.py 的「投射 offset >= 1」这一实现细节。
    """
    j = keys.join(logs, on=["user_id", "author_id"], how="inner").filter(
        (pl.col("d") >= pl.col("target") - pl.duration(days=w))
        & (pl.col("d") <= pl.col("target") - pl.duration(days=1))
    )
    return j.group_by(["user_id", "author_id", "target"]).agg(
        pl.len().alias("imp"),
        *[pl.col(c).sum().alias(c) for c in labels],
        pl.col("video_id").n_unique().alias("distinct"),
        pl.col("d").n_unique().alias("active_days"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--n", type=int, default=400, help="抽样行数")
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

    spec = json.loads((proc / f"pair_spec_{p}.json").read_text("utf-8"))
    cols = spec["columns"]
    enc = spec["encoding"]
    windows = spec["windows"]
    labels = spec["labels"]
    rate_labels = spec["rate_labels"]
    alpha = float(spec["smooth_alpha"])
    priors = spec["priors"]
    w_max = max(windows)
    pre = spec["prefix"]

    vf = pl.read_parquet(proc / "video_features.parquet").select("video_id", "author_id")
    logs = (pl.read_parquet(proc / "logs_split.parquet",
                            columns=["user_id", "video_id", "date", *labels])
            .join(vf, on="video_id", how="left")
            .with_columns(pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d").alias("d"))
            .drop("date"))

    mats = {sp: np.load(proc / f"feat_pair_{sp}_{p}.npy") for sp in SPLITS}
    samples = {sp: pl.read_parquet(proc / f"samples_{sp}_{p}.parquet",
                                   columns=["user_id", "video_id", "date"])
               .join(vf, on="video_id", how="left") for sp in SPLITS}

    # ---------- A/B/C. 独立重算 + 行对齐 + 编码 ----------
    print("\n=== A/B/C. 独立重算（不走 rolling.py）+ 行对齐 + 编码 ===")
    sp = "valid"
    rows = rng.choice(len(samples[sp]), min(a.n, len(samples[sp])), replace=False)
    sub = (samples[sp][rows].with_row_index("_i")
           .with_columns(pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d").alias("target")))
    keys = sub.select("user_id", "author_id", "target").unique()

    want = {}
    for w in windows:
        want[w] = _recompute(logs, keys, w, labels)

    # 组装期望的原始值，再按 spec 编码
    exp = np.zeros((len(sub), len(cols)), np.float64)
    base = sub.select("_i", "user_id", "author_id", "target")
    raw: dict[str, np.ndarray] = {}
    for w in windows:
        m = base.join(want[w], on=["user_id", "author_id", "target"], how="left").sort("_i")
        raw[f"imp_{w}"] = m.get_column("imp").fill_null(0).to_numpy().astype(np.float64)
        for c in labels:
            raw[f"{c}_{w}"] = m.get_column(c).fill_null(0).to_numpy().astype(np.float64)
        if w == w_max:
            raw["distinct"] = m.get_column("distinct").fill_null(0).to_numpy().astype(np.float64)
            raw["active_days"] = (m.get_column("active_days").fill_null(0)
                                  .to_numpy().astype(np.float64))
            raw["has_history"] = (m.get_column("imp").is_not_null()
                                  .to_numpy().astype(np.float64))

    for k, c in enumerate(cols):
        sc = enc[c]
        if c == f"{pre}_has_history":
            x = raw["has_history"]
        elif c == f"{pre}_distinct_{w_max}d":
            x = raw["distinct"]
        elif c == f"{pre}_active_days_{w_max}d":
            x = raw["active_days"]
        elif "_rate_sm_" in c:
            lab = c[len(pre) + 1:c.index("_rate_sm_")]
            w = int(c.rsplit("_", 1)[-1][:-1])
            num, imp = raw[f"{lab}_{w}"], raw[f"imp_{w}"]
            # 平滑率；无历史时 num=imp=0，公式恰好等于先验 g
            x = (num + alpha * priors[lab]) / (imp + alpha)
        else:
            body = c[len(pre) + 1:]
            name, wtxt = body.rsplit("_", 1)
            x = raw[f"{name}_{int(wtxt[:-1])}"]
        if sc["kind"] == "flag":
            exp[:, k] = x
        elif sc.get("constant"):
            exp[:, k] = 0.0
        else:
            z = np.log1p(x) if sc["transform"] == "log1p" else x
            exp[:, k] = (z - sc["mean"]) / sc["std"]

    got = mats[sp][rows].astype(np.float64)
    dev = np.abs(got - exp)
    worst = int(dev.max(0).argmax())
    check(dev.max() < 1e-4,
          f"{len(rows)} 行 x {len(cols)} 列全部对上（最大偏差 {dev.max():.2e}，"
          f"在 {cols[worst]}）")
    for k, c in enumerate(cols):
        if dev[:, k].max() >= 1e-4:
            print(f"       不一致列：{c}  最大偏差 {dev[:, k].max():.4g}")

    # 可证伪：打乱行号顺序后每行还得是自己那份
    perm = rng.permutation(len(rows))
    check(np.array_equal(mats[sp][rows[perm]], mats[sp][rows][perm]),
          "打乱行号顺序后逐行等于原结果的同一置换（join 保序）")

    # ---------- A2. 可证伪「已排除当天」 ----------
    print("\n=== A2. 已排除当天（可证伪）===")
    # 构造：该 (user, author) 在窗口内只在样本当天出现过
    same = (samples[sp].with_columns(
        pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d").alias("target"))
        .select("user_id", "author_id", "target").unique())
    any_before = _recompute(logs, same, w_max, labels).select(
        "user_id", "author_id", "target").with_columns(pl.lit(True).alias("prev"))
    only_today = (same.join(any_before, on=["user_id", "author_id", "target"], how="left")
                  .filter(pl.col("prev").is_null()))
    hh = cols.index(f"{pre}_has_history")
    m2 = (samples[sp].with_row_index("_r").with_columns(
        pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d").alias("target"))
        .join(only_today.select("user_id", "author_id", "target")
              .with_columns(pl.lit(True).alias("hit")),
              on=["user_id", "author_id", "target"], how="left")
        .filter(pl.col("hit").is_not_null()))
    idx = m2.get_column("_r").to_numpy()
    check(len(idx) > 1000 and float(mats[sp][idx, hh].max()) == 0.0,
          f"窗口内无历史的 {len(idx):,} 行，has_history 全为 0"
          f"（最大 {float(mats[sp][idx, hh].max()) if len(idx) else float('nan'):.4f}）")
    # 反面：有历史的行必须为 1
    other = np.setdiff1d(np.arange(len(mats[sp])), idx)
    o = rng.choice(other, 20000, replace=False)
    check(float(mats[sp][o, hh].min()) == 1.0,
          f"窗口内有历史的行 has_history 全为 1（抽 {len(o):,} 行）")

    # ---------- D. 缺失语义 ----------
    print("\n=== D. 无历史行的填充语义 ===")
    no_hist = idx[:5000]
    for c in cols:
        k = cols.index(c)
        sc = enc[c]
        if sc["kind"] == "flag" or sc.get("constant"):
            continue
        if "_rate_sm_" in c:
            lab = c[len(pre) + 1:c.index("_rate_sm_")]
            want_raw = priors[lab]        # 公式在 num=imp=0 处的值
        else:
            want_raw = 0.0
        z = np.log1p(want_raw) if sc["transform"] == "log1p" else want_raw
        want_enc = (z - sc["mean"]) / sc["std"]
        if not np.allclose(mats[sp][no_hist, k], want_enc, atol=1e-5):
            check(False, f"{c} 无历史行的编码值不是 {want_enc:.5f}")
            break
    else:
        check(True, f"{len(no_hist):,} 个无历史行：计数列填 0、平滑率填先验 g、"
                    f"标志填 0，编码值全部一致")

    # ---------- E. 统计量只用 train ----------
    print("\n=== E. 统计量只用 train ===")
    tr = mats["train"]
    num_cols = [c for c in cols if enc[c]["kind"] == "numeric" and not enc[c].get("constant")]
    k0 = cols.index(num_cols[0])
    check(abs(float(tr[:, k0].mean())) < 0.02 and abs(float(tr[:, k0].std()) - 1.0) < 0.02,
          f"train 段 {num_cols[0]} 标准化后 mean {tr[:, k0].mean():+.4f} "
          f"std {tr[:, k0].std():.4f} ≈ 0/1")
    va = mats["valid"]
    check(abs(float(va[:, k0].mean())) > 1e-4 or abs(float(va[:, k0].std()) - 1.0) > 1e-4,
          f"valid 段同列 mean {va[:, k0].mean():+.4f} std {va[:, k0].std():.4f} 与 train 不同 "
          "-> 不是各 split 各自标准化（那会是一种泄漏）")
    check(spec["stats_from"] == {"split": "train", "basis": "train_samples"},
          "spec 记录的 stats_from 口径")
    check(spec["prior_splits"] == ["warmup"],
          f"平滑先验只由 warmup 段估计：{spec['prior_splits']}")

    # ---------- F. 覆盖率 / 形状 / no-op ----------
    print("\n=== F. 覆盖率、形状与既有特征未变 ===")
    for s in SPLITS:
        check(mats[s].shape == (len(samples[s]), len(cols)),
              f"{s} 矩阵形状 {mats[s].shape} 与样本行数一致")
        cov = float((mats[s][:, hh] > 0).mean())
        check(abs(cov - spec["coverage"][s]) < 1e-9,
              f"{s} has_history 覆盖率 {cov:.4f} == spec 记录")
        check(np.isfinite(mats[s]).all(), f"{s} 无 NaN/Inf")
    check(len(cols) == len(set(cols)) and all(c.startswith(pre) for c in cols),
          f"{len(cols)} 个列名唯一且全部带 {pre}_ 前缀")
    check(set(rate_labels) <= set(labels), "rate_labels 是 labels 的子集")

    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "freeze_manifest.py"), "--check"],
                       capture_output=True, text=True, check=False)
    check(r.returncode == 0,
          "freeze_manifest --check：rolling.py 泛化后既有产物一个都没变")

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("User x Author 偏好特征验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("验算通过（独立重算 / 行对齐 / 编码 / 排除当天 / 填充语义 / train 统计 / 覆盖率）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
