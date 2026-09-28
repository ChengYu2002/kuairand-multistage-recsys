"""独立验算排序指标（plan §27）。

排序指标算错不会报错，只会让 ΔAUC 整体偏移 —— 而 Single-Task / MMoE / PLE /
Selective Sharing 共用这一个入口，四个模型会一起错到同一个方向，对比表看上去正常。
因此这里全部用**手算**或**第二条独立实现**来验，不看数字好不好看。

检查分五层：
  A. AUC 手算：小例子逐个对着定义算；再与 sklearn.roc_auc_score 全量比对
     （项目自己的实现基于平均秩，sklearn 走另一条路径，两边对上才算可信）。
  B. 可证伪：随机预测 AUC ≈ 0.5；预测取反后 AUC = 1 − 原值；完美预测 = 1.0；
     预测全相同 = 0.5（并列必须用平均秩，用普通名次会得到 0 或 1）。
  C. GAUC：两用户小例子手算；标签恒定的用户必须被排除且计数正确；
     加权方式（按曝光数）可验证。
  D. PCOC 与校准：预测恒等于全局均值时 PCOC = 1；完美校准的合成数据每桶
     预测 ≈ 实际。
  E. 输入校验：非法输入必须报错，而不是静默算出一个**不可能的指标**。
     实测未加校验时：label=2 会得到 AUC = −1.0；weight 拼错会悄悄换成正样本加权；
     n_buckets=0 会产出 bucket = −1。这三类都不报错、不产生 NaN，
     只会让 ΔAUC 整体偏移，而四个模型共用这一个入口，会一起错到同一个方向。

用法：
    python scripts/verify_ranking_metrics.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.ranking_metrics import (
    auc,
    calibration,
    evaluate_ranking,
    gauc,
    pcoc,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    failures: list[str] = []
    checks = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  {'OK  ' if ok else 'FAIL'} {msg}")
        if not ok:
            failures.append(msg)

    # ---------- A. AUC 手算 + sklearn 交叉 ----------
    print("\n=== A. AUC ===")
    # 2 正 2 负，预测 [0.9, 0.4] vs [0.8, 0.1]：正负对共 4 组，正 > 负 的有 3 组
    p = np.array([0.9, 0.4, 0.8, 0.1]); y = np.array([1, 1, 0, 0])
    check(abs(auc(p, y) - 0.75) < 1e-12, f"手算：4 个正负对里 3 组正确 -> 0.75（得 {auc(p, y):.4f}）")
    # 并列：正负各一个且预测相同 -> 算半分
    check(abs(auc(np.array([0.5, 0.5]), np.array([1, 0])) - 0.5) < 1e-12,
          "一正一负且预测相同 -> 0.5（并列算半分）")

    from sklearn.metrics import roc_auc_score
    worst = 0.0
    for n, rate in ((5000, 0.5), (20000, 0.05), (3000, 0.002)):
        yy = (rng.random(n) < rate).astype(int)
        if yy.sum() == 0 or yy.sum() == n:
            continue
        pp = np.clip(0.1 + 0.4 * yy + rng.normal(0, 0.3, n), 0, 1)
        worst = max(worst, abs(auc(pp, yy) - roc_auc_score(yy, pp)))
    check(worst < 1e-10, f"与 sklearn.roc_auc_score 全量比对，最大偏差 {worst:.2e}")
    # 大量并列时两边仍须一致（平均秩的关键场景）
    yy = (rng.random(20000) < 0.1).astype(int)
    pp = np.round(rng.random(20000), 2)          # 只有 101 种取值 -> 大量并列
    check(abs(auc(pp, yy) - roc_auc_score(yy, pp)) < 1e-10,
          f"预测只有 {len(np.unique(pp))} 种取值（大量并列）时仍与 sklearn 一致")

    # ---------- B. 可证伪 ----------
    print("\n=== B. 可证伪 ===")
    yy = (rng.random(50000) < 0.2).astype(int)
    pp = rng.random(50000)
    a0 = auc(pp, yy)
    check(abs(a0 - 0.5) < 0.02, f"随机预测 AUC {a0:.4f} ≈ 0.5")
    check(abs(auc(-pp, yy) - (1 - a0)) < 1e-10, f"预测取反 AUC = 1 − {a0:.4f}")
    check(auc(yy.astype(float), yy) == 1.0, "完美预测 AUC = 1.0")
    check(abs(auc(np.full(len(yy), 0.3), yy) - 0.5) < 1e-12,
          "预测全相同 AUC = 0.5（若用普通名次而非平均秩会得到 0 或 1）")
    check(np.isnan(auc(pp, np.ones(len(pp)))), "标签全为 1 时 AUC = nan（无定义，不是 0.5）")

    # ---------- C. GAUC ----------
    print("\n=== C. GAUC ===")
    # 用户 A：4 条，AUC = 1.0；用户 B：2 条，AUC = 0.0；用户 C：2 条标签恒定 -> 排除
    u = np.array(["A"] * 4 + ["B"] * 2 + ["C"] * 2)
    yg = np.array([1, 1, 0, 0, 1, 0, 1, 1])
    pg = np.array([0.9, 0.8, 0.2, 0.1, 0.1, 0.9, 0.5, 0.5])
    g, used, skipped = gauc(pg, yg, u)
    want = (1.0 * 4 + 0.0 * 2) / 6
    check(abs(g - want) < 1e-12, f"按曝光数加权 (1.0x4 + 0.0x2)/6 = {want:.4f}（得 {g:.4f}）")
    check(used == 2 and skipped == 1, f"参与 2 个用户，排除 1 个（得 {used} / {skipped}）")
    g2, _, _ = gauc(pg, yg, u, weight="positives")
    check(abs(g2 - (1.0 * 2 + 0.0 * 1) / 3) < 1e-12, "按正样本数加权也正确")

    # ---------- D. PCOC 与校准 ----------
    print("\n=== D. PCOC 与校准 ===")
    yy = (rng.random(50000) < 0.13).astype(float)
    check(abs(pcoc(np.full(len(yy), yy.mean()), yy) - 1.0) < 1e-12,
          "预测恒等于全局均值 -> PCOC = 1.0")
    check(abs(pcoc(np.full(len(yy), 2 * yy.mean()), yy) - 2.0) < 1e-12,
          "预测放大一倍 -> PCOC = 2.0")
    # 完美校准：先造概率，再按该概率采样标签
    prob = rng.random(200000) * 0.4
    lab = (rng.random(200000) < prob).astype(float)
    cal = calibration(prob, lab, 10)
    dev = float((cal["pred"] - cal["actual"]).abs().max())
    check(dev < 0.02, f"完美校准的合成数据，每桶 |预测 − 实际| 最大 {dev:.4f} < 0.02")
    check(len(cal) == 10 and cal["n"].sum() == len(prob),
          f"分成 {len(cal)} 桶，样本数守恒 {cal['n'].sum():,}")

    # ---------- E. 输入校验 ----------
    print("\n=== E. 输入校验 ===")
    ok_p, ok_y, ok_u = np.array([0.1, 0.9]), np.array([0, 1]), np.array([1, 1])
    bad = [
        ("预测含 NaN", (np.array([0.1, np.nan]), ok_y, ok_u)),
        ("长度不一致", (np.array([0.1]), ok_y, ok_u)),
        ("不是概率（logit）", (np.array([-2.0, 3.0]), ok_y, ok_u)),
        ("概率 > 1", (np.array([0.1, 1.5]), ok_y, ok_u)),
        ("label = 2", (ok_p, np.array([0, 2]), ok_u)),
        ("label = 0.5", (ok_p, np.array([0.0, 0.5]), ok_u)),
        ("label 含 NaN", (ok_p, np.array([0.0, np.nan]), ok_u)),
        ("空数组", (np.array([]), np.array([]), np.array([]))),
        ("二维 label", (ok_p, np.array([[0], [1]]), ok_u)),
        ("二维 pred", (np.array([[0.1], [0.9]]), ok_y, ok_u)),
    ]
    for label, argv in bad:
        try:
            evaluate_ranking(*argv)
            check(False, f"{label}：未被拦下")
        except ValueError:
            check(True, f"{label}：已拦下")

    # 单独的入口也要挡住，不能只在 evaluate_ranking 里挡
    for label, fn in (
        ("auc(label=2)", lambda: auc(ok_p, np.array([0, 2]))),
        ("gauc(weight 拼错)", lambda: gauc(ok_p, ok_y, ok_u, weight="typo")),
        ("calibration(n_buckets=0)", lambda: calibration(ok_p, ok_y, 0)),
        ("calibration(n_buckets=-3)", lambda: calibration(ok_p, ok_y, -3)),
        ("pcoc(label 含 NaN)", lambda: pcoc(ok_p, np.array([0.0, np.nan]))),
    ):
        try:
            fn()
            check(False, f"{label}：未被拦下")
        except ValueError:
            check(True, f"{label}：已拦下")

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("排序指标验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("排序指标验算通过（AUC 手算+sklearn / 可证伪 / GAUC / PCOC+校准 / 输入校验）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
