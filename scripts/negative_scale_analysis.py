"""负样本规模对比汇总表（§16.5 之外的独立一项）。

## 它要回答的唯一问题

> 固定「最佳配置」不动，只把每条正样本的负样本数从 10 提到 B-1=1023，
> Recall 的变化是否超过 seed 方差？

为什么**不能**并进 negative_sampling_analysis：那张表的受控变量里包含
negatives_per_positive（§16.6），五个臂一律 n_neg=10，它回答「负样本从哪儿来」。
本脚本故意动了那个受控变量，回答「负样本有多少个」。两个问题混在一张表里，
「来源」与「数量」的贡献就不可归因了。negative_sampling_analysis 里那处显式
`continue` 是同一个决定的另一半。

## §16.6 的可比性：这里能查什么、查不了什么

    可查（结果 JSON）  protocol、item_tower_config、steps、seed 集合
    可查（checkpoint） 全部参数张量的名字与形状、catalog_range —— 架构编码在权重里
    **查不了**         lr、batch_size、temperature、optimizer

还有一个本脚本特有的缺口，必须点名而不是沉默：**基线的 n_neg 没有随结果落盘**。
negative_mode / negative_positions_per_positive 这两个字段是在 inbatch_logq 跑完
之后才加进 two_tower.py 的，所以基线那三份 JSON 里没有它们。本脚本会回落到
config 的 two_tower.negatives_per_positive，并在输出里标成「来自 config，非实测」——
config 可以在跑完之后被改，这不是等价的证据。实验臂的那一份是实测的。

用法：
    python scripts/negative_scale_analysis.py --tower id_side
    python scripts/negative_scale_analysis.py --tower id_side --no-check-arch
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)

BASELINE = "inbatch_logq"
TREATMENT = "inbatch_full_logq"
UNITS = ("per_request", "per_user")


def arch_signature(ckpt: Path) -> tuple:
    """参数张量的 (名字, 形状) + 候选库区间。架构编码在权重里，是事后可复原的证据。"""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sig = tuple(sorted((k, tuple(v.shape)) for k, v in ck["state_dict"].items()))
    return sig, (tuple(ck["catalog_range"]), int(ck["steps"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--tower", default="id_side", choices=("id_only", "id_side"))
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--results", default="results")
    ap.add_argument(
        "--no-check-arch",
        action="store_true",
        help="跳过逐个读 checkpoint 的架构检查（省约 20 秒，但保护也少一层）",
    )
    ap.add_argument(
        "--force", action="store_true", help="允许局部运行覆盖范围更大的现有汇总"
    )
    a = ap.parse_args()

    cfg = load_config(a.config)
    results, experiments = project_path(a.results), project_path("experiments")
    pop = require(cfg, "eval", "expected_popularity_baseline")
    cfg_n_neg = int(require(cfg, "two_tower", "negatives_per_positive"))

    runs: dict[str, dict[int, dict]] = {}
    for strat in (BASELINE, TREATMENT):
        runs[strat] = {}
        for seed in a.seeds:
            tag = f"{strat}_{a.tower}_{a.protocol}_seed{seed}"
            p = results / f"two_tower_{tag}.json"
            if not p.exists():
                raise SystemExit(
                    f"缺 {p.name} —— 配对差值要求两臂共用同一组 seed。"
                    f"先把 {strat} 的 seed {seed} 跑完，或用 --seeds 缩小范围。"
                )
            runs[strat][seed] = json.loads(p.read_text("utf-8"))

    # ---- 可比性：除负样本数量外，记录下来的一切都必须相同 ----
    for key in ("protocol", "item_tower_config", "steps"):
        got = {(s, sd): r.get(key) for s, d in runs.items() for sd, r in d.items()}
        if len(set(got.values())) != 1:
            raise SystemExit(f"{key} 在各次运行间不一致，拒绝出表：{got}")

    # 自变量必须**确实**不同，且是预期的那两个值 —— 否则可能跑错了臂却照样出表。
    n_base = runs[BASELINE][a.seeds[0]].get("negative_positions_per_positive")
    base_from_config = n_base is None
    if base_from_config:
        n_base = cfg_n_neg
    n_treat = runs[TREATMENT][a.seeds[0]].get("negative_positions_per_positive")
    if n_treat is None:
        raise SystemExit(
            f"{TREATMENT} 的结果里没有 negative_positions_per_positive —— "
            "这一臂必须由带该字段的 two_tower.py 产出，否则自变量无法事后确认。"
        )
    if n_treat <= n_base:
        raise SystemExit(
            f"自变量方向不对：实验臂 {n_treat} 个候选位置 <= 基线 {n_base} 个。"
        )
    for s, d in runs.items():
        want = n_base if s == BASELINE else n_treat
        for sd, r in d.items():
            got = r.get(
                "negative_positions_per_positive", want if s == BASELINE else None
            )
            if got != want:
                raise SystemExit(f"{s} seed {sd} 的候选位置数 {got} != {want}")

    if not a.no_check_arch:
        sigs, extras = {}, {}
        for s, d in runs.items():
            for sd, r in d.items():
                p = experiments / f"two_tower_{r['tag']}.pt"
                if not p.exists():
                    raise SystemExit(f"缺 checkpoint {p.name}，无法确认架构一致")
                sig, extra = arch_signature(p)
                sigs[(s, sd)], extras[(s, sd)] = hash(sig), extra
        if len(set(sigs.values())) != 1:
            raise SystemExit(f"各 checkpoint 的参数张量形状不同，架构不一致：{sigs}")
        if len(set(extras.values())) != 1:
            raise SystemExit(f"catalog_range 或 steps 不一致：{extras}")
        log.info("架构指纹一致（%d 份 checkpoint 逐个读过）", len(sigs))

    ks = [
        k
        for k in sorted(pop, key=lambda x: int(x.split("@")[1]))
        if k.startswith("recall@")
    ]
    ref = ks[len(ks) // 2]

    def vals(s: str, k: str, unit: str) -> np.ndarray:
        return np.array([runs[s][sd]["metrics"][k][unit] for sd in a.seeds], float)

    label = {BASELINE: f"n_neg={n_base}", TREATMENT: f"完整 in-batch({n_treat})"}
    log.info(
        "口径：protocol=%s  item_tower=%s  steps=%s  seeds=%s",
        a.protocol,
        a.tower,
        runs[TREATMENT][a.seeds[0]]["steps"],
        a.seeds,
    )
    if base_from_config:
        log.warning(
            "基线的 n_neg=%d **来自 config，非实测** —— 它跑完的时候 two_tower.py "
            "还没有 negative_positions_per_positive 这个字段。config 可以在跑完"
            "之后被改，这不是等价的证据，报告中必须一并披露。",
            n_base,
        )

    out: dict = {
        "protocol": a.protocol,
        "item_tower_config": a.tower,
        "baseline": BASELINE,
        "treatment": TREATMENT,
        "meta": {
            "seeds": a.seeds,
            "steps": runs[TREATMENT][a.seeds[0]]["steps"],
            "negative_positions": {BASELINE: n_base, TREATMENT: n_treat},
            "baseline_n_neg_from_config": base_from_config,
            "unverifiable": ["lr", "batch_size", "temperature", "optimizer"],
        },
        "popularity_baseline": {k: pop[k] for k in ks},
        "metrics": {},
        "loss": {},
        f"delta_vs_{BASELINE}_paired": {},
    }

    for unit in UNITS:
        print(f"\n=== Recall（{unit}，mean ± std，{len(a.seeds)} seeds）===")
        print(
            f"{'实验臂':<22}"
            + "".join(f"{k:>22}" for k in ks)
            + f"{'倍热度基线(@' + ref.split('@')[1] + ')':>18}"
        )
        for s in (BASELINE, TREATMENT):
            row = f"{label[s]:<22}"
            for k in ks:
                v = vals(s, k, unit)
                row += (
                    f"{v.mean():>14.5f}±{v.std(ddof=1):.5f}"
                    if len(v) > 1
                    else f"{v.mean():>15.5f}   n/a"
                )
            row += f"{vals(s, ref, unit).mean() / pop[ref]:>17.2f}x"
            print(row)
        print(
            f"{'热度基线':<20}"
            + "".join(f"{pop[k]:>22.5f}" for k in ks)
            + f"{1.0:>17.2f}x"
        )

    print(f"\n=== 配对差值 vs {BASELINE}（同 seed 相减）===")
    print(f"{'指标':<14}{'per_request mean±std':>24}  同号  |mean|/std  相对基准")
    for k in ks:
        v, b = vals(TREATMENT, k, "per_request"), vals(BASELINE, k, "per_request")
        d = v - b
        sd = d.std(ddof=1) if len(d) > 1 else float("nan")
        same = bool((d > 0).all() or (d < 0).all())
        print(
            f"{k:<14}{d.mean():>+15.5f}±{sd:.5f}  "
            f"{'同号' if same else '异号'}  {abs(d.mean()) / sd:>9.1f}  "
            f"{v.mean() / b.mean():>9.3f}x"
        )
        out[f"delta_vs_{BASELINE}_paired"][k] = {
            "mean": float(d.mean()),
            "std": float(sd),
            "values": [float(x) for x in d],
            "same_sign": same,
            "abs_mean_over_std": float(abs(d.mean()) / sd),
            "ratio": float(v.mean() / b.mean()),
        }
    print(
        f"  说明：先算每个 seed 的配对差值再求 mean ± std（seed 是受控的配对因子）。"
        f"「同号」是 {len(a.seeds)} 个 seed 的方向一致性，**不做显著性判定** —— "
        f"{len(a.seeds)} 个 seed 估出的 std 本身不可靠。"
    )

    # loss 必须带着分类数一起读，否则一定被误读成「完整 in-batch 训练更差」。
    print("\n=== 训练末端 loss（**不可直接跨臂比较**）===")
    print(f"{'实验臂':<22}{'loss_tail':>12}{'分类数':>8}{'随机猜':>10}{'归一化':>10}")
    for s in (BASELINE, TREATMENT):
        n = n_base if s == BASELINE else n_treat
        chance = math.log(n + 1)
        lt = np.array([runs[s][sd]["loss_tail"] for sd in a.seeds], float)
        print(
            f"{label[s]:<22}{lt.mean():>12.4f}{n + 1:>8}{chance:>10.3f}"
            f"{lt.mean() / chance:>10.3f}"
        )
        out["loss"][s] = {
            "loss_tail_mean": float(lt.mean()),
            "n_classes": n + 1,
            "chance": chance,
            "normalized": float(lt.mean() / chance),
        }
    print(
        "  说明：n_neg=10 是 11 分类、完整 in-batch 是 B 分类，题的难度不同，"
        "loss_tail 大一个量级不代表训练更差。归一化量也**不是 Recall 的代理** —— "
        "判据只有上面那两张表。"
    )

    for s in (BASELINE, TREATMENT):
        out["metrics"][s] = {
            k: {
                u: {
                    "mean": float(vals(s, k, u).mean()),
                    "std": float(vals(s, k, u).std(ddof=1)),
                    "values": [float(x) for x in vals(s, k, u)],
                }
                for u in UNITS
            }
            for k in ks
        }

    dst = results / f"negative_scale_summary_{a.protocol}_{a.tower}.json"
    # 与 negative_sampling_analysis / cold_start_analysis 同一个护栏：范围更小的
    # 一次运行不得静默覆盖范围更大的现有汇总。
    if dst.exists() and not a.force:
        try:
            old = json.loads(dst.read_text("utf-8"))
        except (ValueError, OSError):
            old = {}
        lost = sorted(set((old.get("meta") or {}).get("seeds") or []) - set(a.seeds))
        if lost:
            raise SystemExit(
                f"拒绝覆盖 {dst.name}：现有文件覆盖的 seed 更多，本次会丢掉 {lost}。"
                "确实要覆盖请加 --force。"
            )
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n已写 {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
