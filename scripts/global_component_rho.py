"""ρ(ū·v_i, train_freq)：模型学到的「全局热度先验」有多强，正负朝哪边。

ū·v_i 是得分里**与用户无关**的那一份 —— 把用户向量在所有评估查询时刻上取平均，
再和每个候选 item 的向量做点积。它和 train_freq 的 Spearman ρ 直接回答一个问题：
这套负采样把模型推向了「按热度排」还是「按冷门排」。

为什么需要它：R@100 只说谁高谁低，说不出**为什么**。§7.2 的频次分层发现 in-batch
把 warm 层打到 7.6x（比它自己的 tail 层 10.6x 还低），当时的解释有两种，方向相反：
  (a) 学到了全局热度排序，所以热门反复被召回、挤掉长尾；
  (b) 热门 item 被**过度惩罚**，所以它在 warm 层反而最差。
这个 ρ 是能把两者分开的量 —— (a) 预测 ρ 升高，(b) 预测 ρ 降低。实测是降低。

口径（与 Recall 严格同一套，否则两个数对不上）：
  * 查询时刻来自 two_tower.eval_moments，与 request_topk 用的是同一份，不是另抄一遍；
  * ū 对 51,746 个唯一查询时刻的用户向量取算术平均；
  * v_i 来自 two_tower.item_matrix，覆盖整个候选库（catalog 行 j 即 item 下标 lo+j，
    这正是 request_topk 用 cat_ids[topk] 映射回 video_id 的依据）；
  * Spearman ρ = 两列各自转秩后的 Pearson r；并列取平均秩（scipy.stats.rankdata 默认）。

这个脚本是**补写**的：README §7.2 早就写着 ρ=+0.549/+0.202/+0.466/−0.231，但那是
seed 42 单点、脚本未留存。本脚本按上面的口径在 3 个 seed 上重算，seed 42 的四个值与
那组旧数字相差 0.001~0.003 —— 同号、同量级、结论不变，但**原口径已不可复原**，所以
不假装能复现它：--check 校的是本脚本自己的冻结基准，旧数字只作并列参考打印出来。

为什么敢用紧阈值：实测同机两次运行的 ρ 到小数第 12 位逐位相同（前向、rankdata、
corrcoef 全是确定性运算）。1e-3 是「ρ 只报到 3 位」这个事实给的余量，不是贴着某次
观测值拍的。

用法：
    python scripts/global_component_rho.py                      # 全部策略 x 全部 seed
    python scripts/global_component_rho.py --check              # 另外做回归校验
    python scripts/global_component_rho.py --strategies inbatch_logq --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
import torch
from scipy.stats import rankdata

from src.retrieval.dataset import RetrievalData
from src.retrieval.two_tower import (
    TwoTower,
    eval_moments,
    item_matrix,
    user_matrix,
)
from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)

STRATEGY_ORDER = ("random", "inbatch", "inbatch_logq", "hybrid", "exposure")
# 冻结基准：本脚本 2026-10-29 实测的 3-seed 均值。后续重构若动了口径，--check 会炸。
BASELINE_3SEED = {
    "random": 0.519,
    "inbatch": 0.200,
    "inbatch_logq": 0.379,
    "hybrid": 0.469,
    "exposure": -0.236,
}
CHECK_TOL = 1e-3
# README §7.2 早期写下的 ρ：seed 42 单点，原脚本未留存。只作并列参考，不作判据。
AD_HOC_SEED42 = {"random": 0.549, "inbatch": 0.202, "hybrid": 0.466, "exposure": -0.231}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--tower", default="id_only", choices=("id_only", "id_side"))
    ap.add_argument("--strategies", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--results", default="results")
    ap.add_argument(
        "--check",
        action="store_true",
        help="与冻结基准对账；对不上则非零退出（口径回归保护）",
    )
    ap.add_argument(
        "--force", action="store_true", help="允许范围更小的一次运行覆盖现有结果"
    )
    a = ap.parse_args()

    if a.tower != "id_only":
        raise SystemExit(
            "id_side 的 item 向量随评估日变化，ū·v_i 不是单一向量 —— 这个脚本只覆盖 "
            "id_only。要扩到 id_side 得先定义「按哪一天」，那是另一个决定。"
        )

    cfg = load_config(a.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    experiments = project_path("experiments")
    meta = json.loads((proc / f"vocab_meta_{a.protocol}.json").read_text("utf-8"))
    lo, hi = meta["video_catalog_index_min"], meta["video_catalog_index_max"]

    catalog = pl.read_parquet(proc / f"catalog_{a.protocol}.parquet")
    freq = catalog.get_column("train_freq").to_numpy().astype(np.float64)
    if len(freq) != hi - lo + 1:
        raise AssertionError(
            f"catalog 行数 {len(freq)} 与候选库下标区间 [{lo}, {hi}] 的长度不一致，"
            "「catalog 行 j <-> item 下标 lo+j」这个对应就不成立了。"
        )

    data = RetrievalData(proc, a.protocol, "test")  # 只读，各 checkpoint 共用
    _, mom, _ = eval_moments(proc, a.protocol)
    srows = mom["srow"].to_numpy()

    out: dict = {
        "protocol": a.protocol,
        "item_tower_config": a.tower,
        "n_moments": len(srows),
        "n_catalog": len(freq),
        "definition": "rho = Spearman(ubar . v_i, train_freq)，ubar 为全部查询时刻的用户向量均值",
        "models": {},
    }

    print(f"\n口径：{len(srows):,} 个查询时刻取 ū；候选库 {len(freq):,} 个 item")
    print(f"\n{'策略':<14}{'每 seed 的 ρ':>34}{'mean±std':>20}")
    for strat in a.strategies or list(STRATEGY_ORDER):
        rhos, used = [], []
        for seed in a.seeds:
            ck = experiments / f"two_tower_{strat}_{a.tower}_{a.protocol}_seed{seed}.pt"
            if not ck.exists():
                log.warning("缺 %s，跳过", ck.name)
                continue
            st = torch.load(ck, weights_only=False)
            model = TwoTower(data, cfg)
            model.load_state_dict(st["state_dict"])
            model.eval()
            ubar = user_matrix(model, data, srows).mean(axis=0)
            iv = item_matrix(model, data, lo, hi, None)
            rhos.append(spearman(iv @ ubar, freq))
            del iv
            used.append(seed)
        if not rhos:
            continue
        r = np.array(rhos)
        sd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
        out["models"][strat] = {
            "seeds": used,
            "values": [float(x) for x in r],
            "mean": float(r.mean()),
            "std": sd,
        }
        print(
            f"{strat:<14}"
            + "".join(f"{x:>+11.3f}" for x in r)
            + f"{r.mean():>+13.3f}±{sd:.3f}"
        )

    dst = project_path(a.results) / f"global_component_rho_{a.protocol}_{a.tower}.json"
    # 文件名不含 --strategies/--seeds 的范围：一次局部运行会静默吃掉完整那份。
    # 与 cold_start_analysis / negative_sampling_analysis 同一护栏（都是实测踩过的）。
    if dst.exists() and not a.force:
        try:
            old_m = json.loads(dst.read_text("utf-8")).get("models", {})
        except (ValueError, OSError):
            old_m = {}
        lost = {
            s: sorted(
                set(v.get("seeds", [])) - set(out["models"].get(s, {}).get("seeds", []))
            )
            for s, v in old_m.items()
        }
        lost = {s: v for s, v in lost.items() if v or s not in out["models"]}
        if lost:
            raise SystemExit(
                f"拒绝覆盖 {dst.name}：现有文件覆盖的范围更大，本次会丢掉 {lost}。"
                "局部运行请加 --force。"
            )
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("已写出 %s", dst.name)

    if not a.check:
        return 0
    print(f"\n=== 与冻结基准对账（3-seed 均值，容差 {CHECK_TOL:.0e}）===")
    bad = 0
    for strat, want in BASELINE_3SEED.items():
        if strat not in out["models"]:
            print(f"  {strat:<14} 本次没跑，跳过")
            continue
        got = out["models"][strat]["mean"]
        ok = abs(got - want) <= CHECK_TOL
        bad += not ok
        print(
            f"  {strat:<14} 基准 {want:+.3f}   本次 {got:+.3f}   "
            f"差 {got - want:+.4f}   {'OK' if ok else '**不符**'}"
        )
    print("\n--- 并列参考：早期单 seed 诊断（原脚本未留存，不作判据）---")
    for strat, old in AD_HOC_SEED42.items():
        m = out["models"].get(strat)
        if not m or 42 not in m["seeds"]:
            continue
        now = m["values"][m["seeds"].index(42)]
        print(
            f"  {strat:<14} 旧（seed 42）{old:+.3f}   本脚本（seed 42）{now:+.3f}   差 {now - old:+.3f}"
        )
    if bad:
        print(
            f"\n{bad} 项对不上冻结基准。口径漂了或 checkpoint 变了 —— 查清之前不要用"
            "本次的新数字去解释结论。"
        )
        return 1
    print("\n全部相符。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
