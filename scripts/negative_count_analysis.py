"""负样本**数量**扫描的汇总：n_neg ∈ {5, 10, 30} 对比（§7.3 的后半段）。

为什么是独立脚本而不是复用现成的两个：

  negative_sampling_analysis.py  回答「负样本从哪儿来」，固定 n_neg=10。它里面有一处
    显式 `continue` 把带 `_neg{n}` 的产物挡在外面 —— 不挡的话，`_neg30` 的 strategy
    仍是 inbatch_logq、seed 仍是 42，会按 (strategy, seed) 直接覆盖掉基线那一份。
  negative_scale_analysis.py     回答「10 vs 完整 in-batch(1023)」，两臂的 strategy 不同
    （inbatch_logq vs inbatch_full_logq），靠 strategy 名就能分开。

本脚本的两臂 **strategy 完全相同**，只有 `negatives_per_positive` 不同，产物靠 tag 上的
`_neg{n}` 后缀区分（见 two_tower.artifact_tag）。这是第三种口径，所以是第三个脚本。

它比前两个多做三件事，都是这次扫描踩出来的：

  1. **逐 seed 的配对差值一律打印出来**，不只打 mean±std。这次 seed 42 单点给出 3.6 倍、
     三个 seed 给出 0.18 倍，只看汇总值看不出发生了什么。
  2. **跨 seed 离散度的比值**。n_neg=5 的 std 是 n_neg=10 的 2.6~3.3 倍 —— 负样本少则
     梯度估计糙、跨 seed 抖得厉害。不量这个，就解释不了单 seed 筛选为什么会给出假信号。
  3. **读 results/negative_count_prediction.json 做对账**，包括证伪条件是否触发。
     预测写在跑之前，对账必须由脚本做，否则「事先预测」这件事无法复核。

单 seed 的臂（本次的 n_neg=30）单独列，**不参与配对统计** —— 一个点算不出离散度，
而这次的教训正是单点会骗人。

用法：
    python scripts/negative_count_analysis.py --config configs/retrieval.yaml
    python scripts/negative_count_analysis.py --neg 5 30 --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval.two_tower import artifact_tag
from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)

STRATEGY = "inbatch_logq"
KS = ("recall@50", "recall@100", "recall@500")
# 判据线。不是本脚本定的：§7.1 把 1.9 判为「3 个 seed 撑不住」，§7.2 把 1.15 降级为
# 「只作记录」。写死在这里，免得每次看着数字重新选一条线。
BAR = 2.0


def load(
    results: Path, tower: str, protocol: str, n_neg: int | None, seed: int
) -> dict:
    tag = artifact_tag(STRATEGY, tower, protocol, seed, n_neg_override=n_neg)
    f = results / f"two_tower_{tag}.json"
    if not f.exists():
        raise SystemExit(
            f"缺 {f.name} —— 配对差值要求两臂共用同一组 seed。\n"
            f"先跑：.venv/bin/python -m src.retrieval.two_tower --negative {STRATEGY} "
            f"--item-tower-config {tower} --seed {seed}"
            + (f" --negatives-per-positive {n_neg}" if n_neg else "")
        )
    return json.loads(f.read_text("utf-8"))


def positions(run: dict, cfg_n_neg: int) -> tuple[int, bool]:
    """这次跑实际用了多少个负样本位置，以及这个数是不是实测落盘的。

    基线那三份 JSON 产出于 negative_positions_per_positive 这个字段加进 two_tower.py
    之前，只能回落到 config —— 而 config 可以在跑完之后被改。§7.3 已披露这一层缺口，
    这里把它变成一个返回值，报告里就不会漏写。
    """
    v = run.get("negative_positions_per_positive")
    return (cfg_n_neg, True) if v is None else (int(v), False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--tower", default="id_side", choices=("id_only", "id_side"))
    ap.add_argument(
        "--neg", nargs="*", type=int, default=[5, 30], help="要对比的 n_neg"
    )
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--force", action="store_true", help="允许覆盖覆盖面更大的汇总")
    a = ap.parse_args()

    cfg = load_config(a.config)
    results = project_path("results")
    cfg_n_neg = int(require(cfg, "two_tower", "negatives_per_positive"))

    # 基线（n_neg 取自 config，产物名不带后缀）必须三个 seed 都在：配对差值的分母靠它。
    base = {s: load(results, a.tower, a.protocol, None, s) for s in a.seeds}
    n_base, base_from_cfg = positions(base[a.seeds[0]], cfg_n_neg)
    if base_from_cfg:
        log.warning(
            "基线的 n_neg=%d **来自 config，非实测** —— 它跑完的时候 two_tower.py 还没有"
            "negative_positions_per_positive 这个字段。config 可以在跑完之后被改，"
            "这不是等价的证据，报告中必须一并披露。",
            n_base,
        )

    # 实验臂：每个 n_neg 各自有多少 seed 是跑出来的，不强求齐 —— 但缺的要说清楚。
    arms: dict[int, dict[int, dict]] = {}
    for n in a.neg:
        got = {}
        for s in a.seeds:
            tag = artifact_tag(STRATEGY, a.tower, a.protocol, s, n_neg_override=n)
            f = results / f"two_tower_{tag}.json"
            if f.exists():
                got[s] = json.loads(f.read_text("utf-8"))
        if not got:
            raise SystemExit(f"n_neg={n} 一个 seed 都没有结果，无从对比。")
        for s, r in got.items():
            p, from_cfg = positions(r, cfg_n_neg)
            if from_cfg or p != n:
                raise SystemExit(
                    f"n_neg={n} 的 seed {s} 落盘的位置数是 {p}（实测={not from_cfg}）"
                    f"，与产物名声称的 {n} 不一致 —— 自变量对不上，不出表。"
                )
        arms[n] = got

    # 同一口径：protocol / 物品塔 / 步数必须一致，否则比的不是负样本数量。
    ref = base[a.seeds[0]]
    for label, runs in [("baseline", base)] + [(f"neg{n}", v) for n, v in arms.items()]:
        for s, r in runs.items():
            for key in ("protocol", "item_tower_config", "steps"):
                if r.get(key) != ref.get(key):
                    raise SystemExit(
                        f"{label} seed {s} 的 {key}={r.get(key)!r} 与基线的 "
                        f"{ref.get(key)!r} 不同 —— 除 n_neg 外必须全同。"
                    )
    log.info(
        "口径：protocol=%s  item_tower=%s  steps=%s  基线 n_neg=%d%s",
        ref["protocol"],
        ref["item_tower_config"],
        ref["steps"],
        n_base,
        "（来自 config）" if base_from_cfg else "（实测）",
    )

    def vals(runs: dict[int, dict], k: str, seeds) -> np.ndarray:
        return np.array([runs[s]["metrics"][k]["per_request"] for s in seeds], float)

    def norm_loss(runs: dict[int, dict], n: int, seeds) -> np.ndarray:
        c = n + 1
        return np.array([runs[s]["loss_tail"] / math.log(c) for s in seeds], float)

    multi = [n for n in a.neg if len(arms[n]) >= 2]
    single = [n for n in a.neg if len(arms[n]) < 2]

    print(f"\n=== Recall（per_request，mean ± std，{len(a.seeds)} seeds）===")
    print(f"{'实验臂':<22}" + "".join(f"{k:>22}" for k in KS) + f"{'归一化 loss':>16}")
    rows = [(f"n_neg={n_base}（基线）", base, n_base, a.seeds)]
    rows += [(f"n_neg={n}", arms[n], n, sorted(arms[n])) for n in multi]
    for label, runs, n, seeds in rows:
        cells = "".join(
            f"{vals(runs, k, seeds).mean():>13.5f}±{vals(runs, k, seeds).std(ddof=1):.5f}"
            for k in KS
        )
        nl = norm_loss(runs, n, seeds)
        print(f"{label:<22}{cells}{nl.mean():>9.4f}±{nl.std(ddof=1):.4f}")
    for n in single:
        seeds = sorted(arms[n])
        cells = "".join(
            f"{arms[n][seeds[0]]['metrics'][k]['per_request']:>22.5f}" for k in KS
        )
        nl = arms[n][seeds[0]]["loss_tail"] / math.log(n + 1)
        print(f"{f'n_neg={n}（单 seed）':<22}{cells}{nl:>16.4f}")
    if single:
        print(
            f"  单 seed 的臂（n_neg={single}）**不参与下面的配对统计** —— 一个点算不出"
            "离散度。本次的教训正是单点会骗人：n_neg=5 在 seed 42 上给出 3.6 倍，"
            "三个 seed 给出 0.18 倍。"
        )

    out: dict = {
        "protocol": ref["protocol"],
        "item_tower_config": ref["item_tower_config"],
        "strategy": STRATEGY,
        "baseline_n_neg": n_base,
        "baseline_n_neg_from_config": base_from_cfg,
        "bar_abs_mean_over_std": BAR,
        "arms": {},
        "paired": {},
        "dispersion_ratio": {},
        "single_seed_arms": single,
    }
    for label, runs, n, seeds in rows:
        out["arms"][str(n)] = {
            "seeds": list(seeds),
            "metrics": {k: vals(runs, k, seeds).tolist() for k in KS},
            "loss_tail_normalized": norm_loss(runs, n, seeds).tolist(),
        }
    for n in single:
        s0 = min(arms[n])
        out["arms"][str(n)] = {
            "seeds": [s0],
            "metrics": {k: [arms[n][s0]["metrics"][k]["per_request"]] for k in KS},
            "loss_tail_normalized": [arms[n][s0]["loss_tail"] / math.log(n + 1)],
        }

    for n in multi:
        seeds = sorted(arms[n])
        print(f"\n=== 配对差值：n_neg={n} − n_neg={n_base}（同 seed 相减）===")
        print(
            f"{'指标':<14}"
            + "".join(f"{f'seed{s}':>13}" for s in seeds)
            + f"{'mean±std':>24}{'同号':>7}{'|mean|/std':>12}{'过线':>7}"
        )
        for k in KS:
            d = vals(arms[n], k, seeds) - vals(base, k, seeds)
            mu, sd = float(d.mean()), float(d.std(ddof=1))
            same = max(int((d > 0).sum()), int((d < 0).sum()))
            ratio = abs(mu) / sd if sd else float("inf")
            print(
                f"{k:<14}"
                + "".join(f"{x:>+13.6f}" for x in d)
                + f"{mu:>+15.6f}±{sd:.6f}{same:>5}/{len(seeds)}"
                + f"{ratio:>12.2f}{'是' if ratio >= BAR else '否':>7}"
            )
            out["paired"].setdefault(str(n), {})[k] = {
                "per_seed": d.tolist(),
                "mean": mu,
                "std": sd,
                "n_same_sign": same,
                "abs_mean_over_std": ratio,
                "clears_bar": bool(ratio >= BAR),
            }
        print(
            f"  判据线 |mean|/std >= {BAR}。**不做显著性判定** —— {len(seeds)} 个 seed "
            "估出的 std 本身不可靠（§26）。逐 seed 的差值一律打印：只看 mean±std "
            "看不出一个 seed 能把结论翻过去。"
        )

        print(f"\n=== 跨 seed 离散度：n_neg={n} vs n_neg={n_base} ===")
        print(
            f"{'指标':<14}{f'n_neg={n_base} std':>18}{f'n_neg={n} std':>16}{'比值':>8}"
        )
        for k in KS:
            sb = float(vals(base, k, seeds).std(ddof=1))
            st_ = float(vals(arms[n], k, seeds).std(ddof=1))
            print(f"{k:<14}{sb:>18.5f}{st_:>16.5f}{st_ / sb:>8.1f}x")
            out["dispersion_ratio"].setdefault(str(n), {})[k] = {
                "baseline_std": sb,
                "arm_std": st_,
                "ratio": st_ / sb,
            }
        print(
            "  负样本越少，梯度估计越糙，跨 seed 抖得越厉害。这解释了单 seed 筛选为什么"
            f"会给出假信号：用 n_neg={n_base} 那组较小的离散度当尺子，去量一个方差大几倍"
            "的臂。**这个比值本身不作结论** —— 3 对 3 个 seed 估方差比，自由度 (2,2)，"
            "极不可靠。"
        )

    # 对账。预测写在跑之前，核对必须由脚本做，否则无法复核。
    pf = results / "negative_count_prediction.json"
    if pf.exists() and multi:
        pred = json.loads(pf.read_text("utf-8"))
        n0 = multi[0]
        print(f"\n=== 事先预测对账（{pf.name}，写于 {pred.get('written_at')}）===")
        print(
            f"{'指标':<14}{'预测 |m|/std':>14}{'预测过线':>10}{'实测 |m|/std':>14}{'实测过线':>10}{'结果':>8}"
        )
        hit = 0
        for k in KS:
            p = pred["predictions"][k]
            g = out["paired"][str(n0)][k]
            ok = p["clears_bar_2.0"] == g["clears_bar"]
            hit += ok
            print(
                f"{k:<14}{p['predicted_abs_mean_over_std']:>14.1f}"
                f"{'是' if p['clears_bar_2.0'] else '否':>10}"
                f"{g['abs_mean_over_std']:>14.2f}{'是' if g['clears_bar'] else '否':>10}"
                f"{'对' if ok else '错':>8}"
            )
        print(f"  过线/不过线的判断：{hit}/{len(KS)} 条正确。")
        # 证伪条件：R@500 的配对差值变号
        d500 = out["paired"][str(n0)]["recall@500"]["per_seed"]
        flipped = not (all(x > 0 for x in d500) or all(x < 0 for x in d500))
        print(
            f"  证伪条件（R@500 配对差值变号）：{'**已触发**' if flipped else '未触发'}"
        )
        if flipped:
            print(f"    逐 seed: {[round(x, 6) for x in d500]}")
            print(f"    {pred.get('falsification', '')}")
        out["prediction_accounting"] = {
            "file": pf.name,
            "written_at": pred.get("written_at"),
            "clears_bar_calls_correct": hit,
            "calls_total": len(KS),
            "falsification_triggered": flipped,
        }
    elif multi:
        log.warning(
            "没有 %s —— 这组数字就没有「事先预测」的资格，只能当事后观察。", pf.name
        )

    dst = results / f"negative_count_summary_{a.protocol}_{a.tower}.json"
    if dst.exists() and not a.force:
        old = json.loads(dst.read_text("utf-8"))
        lost = sorted(set(old.get("arms", {})) - set(out["arms"]))
        if lost:
            raise SystemExit(
                f"{dst.name} 里已有 n_neg={lost} 的结果，本次没覆盖到。"
                "局部运行不得覆盖更完整的汇总 —— 补齐 --neg 后重跑，或加 --force。"
            )
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n已写 {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
