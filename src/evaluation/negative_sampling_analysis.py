"""负采样策略对比汇总表（plan §16 / §16.6）。

## 它要回答的唯一问题

> 四种负样本构造策略在**全库检索**下的 Recall 差异，是否超过 seed 方差？

与排序侧 `multitask_analysis.py` 同一个模式：先算每个 seed 的配对差值，再报这组差值的
mean ± std 与方向一致性。seed 是受控的配对因子（四种策略用同一组 seed、同一份数据、
同一个候选库），所以配对才是正确的尺度。

## 必须对照热度基线

Recall 的绝对值很小（@100 约 0.009），脱离分母无法判断「好」还是「坏」。
`configs/retrieval.yaml` 里 `eval.expected_popularity_baseline` 是**冻结的实测值**：
Protocol A 下给所有 request 推同一份榜单能拿到多少。它是与模型无关、可独立重算的定点。

实测 exposure 只有热度基线的 0.09 倍 —— 没有这个分母，0.00046 这个数读不出任何意思。
因此本脚本的每张表都附倍数，且倍数的分母必须来自 config，不能是写死的数字。

## §16.6 的可比性：能查什么、查不了什么

§16.6 要求固定 architecture / embedding dim / optimizer / lr / batch size /
negatives per positive / loss form / training steps / user subset / candidate catalog / seeds。
而**结果 JSON 只记录了其中一部分**，所以这里把检查分成三层，并把第三层明说出来：

    可查（结果 JSON）  protocol、item_tower_config、steps、seed 集合
    可查（checkpoint） 全部参数张量的名字与形状、catalog_range
                       —— 架构编码在权重里，这覆盖 embedding_dim、两座塔的层宽、
                          share_item_embedding，是真正的事后证据
    **查不了**         lr、batch_size、negatives_per_positive、temperature、optimizer
                       —— 这些只存在于 config，未逐次落盘

查不了的那一层是**空的保护**，不是「大概没问题」。排序侧有 input_fingerprint 挡着，
召回侧没有。已知的具体缺口：`configs/retrieval.yaml` 的最后写入时间（2026-09-25 23:51）
晚于 random / inbatch 的 seed 42 跑完的时间（21:07 / 21:52），那两次用的配置
**无法从 git 复原** —— 同一个提交里既加了新键又改了注释，提交内的时间线不存在。
报告中必须披露，不能因为「多半只是注释」就当作已验证。

用法：
    python -m src.evaluation.negative_sampling_analysis
    python -m src.evaluation.negative_sampling_analysis --baseline inbatch --no-check-arch
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)

BASELINE = "random"
STRATEGY_ORDER = ("random", "inbatch", "inbatch_logq", "hybrid", "exposure")
UNITS = ("per_request", "per_user")
# 结果 JSON 里确实记录了、因而可以断言的字段
RECORDED_KEYS = ("protocol", "item_tower_config", "steps")
# §16.6 要求固定但**未逐次落盘**的字段。空的保护必须点名，不能沉默。
UNVERIFIABLE = (
    "lr",
    "batch_size",
    "negatives_per_positive",
    "temperature",
    "optimizer",
    "embedding_dim(仅能由权重形状间接确认)",
)


def load_runs(results: Path, protocol: str, tower: str) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for f in sorted(results.glob("two_tower_*.json")):
        try:
            r = json.loads(f.read_text("utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(r, dict) or "strategy" not in r or "metrics" not in r:
            continue
        if r.get("protocol") != protocol or r.get("item_tower_config") != tower:
            continue
        # 这张表回答固定 n_neg=10 的五臂负样本来源实验。完整 in-batch 是独立的
        # 负样本规模实验；若混进来，不仅 seed 集合暂时不齐，还会把 §16.6 的
        # 「除来源外一切固定」改成同时比较来源与数量。
        if r["strategy"] not in STRATEGY_ORDER:
            continue
        if "smoke" in str(r.get("tag", "")):
            continue
        # `_neg{n}` 是负样本**数量**扫描的产物。它的 strategy 仍是 inbatch_logq、
        # seed 仍是 42，所以不挡的话它会按 (strategy, seed) 覆盖掉 n_neg=10 的那一份
        # —— 而 glob 的字典序让 `_seed42_neg30` 排在 `_seed42` 之后，它会赢。
        # 这与上面挡掉完整 in-batch 是同一个理由：这张表的自变量只有「来源」一个。
        if re.search(r"_neg\d+(_|$)", str(r.get("tag", ""))):
            continue
        runs[r["strategy"]][int(r["seed"])] = r
    return runs


def arch_signature(ckpt: Path) -> tuple:
    """checkpoint 的架构指纹：全部参数张量的 (名字, 形状)，以及候选库区间。

    架构编码在权重里，所以这是**事后可复原**的证据 —— 比读当前 config 强，
    因为 config 可能在跑完之后被改过。
    """
    import torch

    ck = torch.load(ckpt, weights_only=False)
    sig = tuple(sorted((k, tuple(v.shape)) for k, v in ck["state_dict"].items()))
    extra = (tuple(ck.get("catalog_range") or ()), int(ck.get("steps") or 0))
    sampler = ck.get("sampler")
    del ck
    return sig, extra, sampler


def assert_comparable(
    runs: dict[str, dict[int, dict]], experiments: Path, check_arch: bool
) -> dict:
    seeds = {s: sorted(d) for s, d in runs.items()}
    if len({tuple(v) for v in seeds.values()}) != 1:
        raise AssertionError(
            f"各策略的 seed 集合不一致：{seeds}。配对差值会变成不同 seed 相减，"
            "§16.6 要求 same seeds —— 补齐后再出表。"
        )
    for k in RECORDED_KEYS:
        vals = {(s, sd): r.get(k) for s, d in runs.items() for sd, r in d.items()}
        if len(set(vals.values())) != 1:
            raise AssertionError(f"{k} 不一致，拒绝出表：{vals}")

    meta: dict = {
        k: next(iter(next(iter(runs.values())).values())).get(k) for k in RECORDED_KEYS
    }
    meta["seeds"] = seeds
    meta["unverifiable_by_design"] = list(UNVERIFIABLE)
    meta["arch_checked"] = bool(check_arch)
    if not check_arch:
        log.warning(
            "跳过架构检查（--no-check-arch）。此时「可比」只建立在结果 JSON "
            "记录的 %s 之上。",
            list(RECORDED_KEYS),
        )
        return meta

    sigs, extras, samplers = {}, {}, {}
    for s, d in runs.items():
        for sd, r in d.items():
            p = experiments / f"two_tower_{r['tag']}.pt"
            if not p.exists():
                raise AssertionError(f"缺 checkpoint {p.name}，无法确认架构一致")
            sig, extra, sampler = arch_signature(p)
            sigs[(s, sd)] = hash(sig)
            extras[(s, sd)] = extra
            if sampler:
                samplers[f"{s}_seed{sd}"] = sampler
    if len(set(sigs.values())) != 1:
        raise AssertionError(
            f"各 checkpoint 的参数张量形状不同，架构不一致，拒绝出表：{sigs}"
        )
    if len(set(extras.values())) != 1:
        raise AssertionError(f"catalog_range 或 steps 不一致：{extras}")
    meta["arch_param_shapes_identical"] = True
    meta["catalog_range"], meta["steps_from_ckpt"] = next(iter(extras.values()))
    meta["sampler_stats"] = samplers
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--tower", default="id_only", choices=("id_only", "id_side"))
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--results", default="results")
    ap.add_argument(
        "--no-check-arch",
        action="store_true",
        help="跳过逐个读 checkpoint 的架构检查（省约 30 秒，但保护也少一层）",
    )
    ap.add_argument(
        "--force", action="store_true", help="允许覆盖基准不同或范围更大的现有汇总"
    )
    a = ap.parse_args()

    cfg = load_config(a.config)
    results = project_path(a.results)
    runs = load_runs(results, a.protocol, a.tower)
    if not runs:
        raise SystemExit(
            f"results/ 下没有 protocol={a.protocol} / "
            f"item_tower_config={a.tower} 的双塔结果。"
        )
    if a.baseline not in runs:
        raise SystemExit(
            f"基准策略 {a.baseline!r} 没有结果，配对差值的分母缺失。"
            f"现有：{sorted(runs)}"
        )
    meta = assert_comparable(runs, project_path("experiments"), not a.no_check_arch)
    pop = require(cfg, "eval", "expected_popularity_baseline")
    ks = [
        k
        for k in sorted(pop, key=lambda x: int(x.split("@")[1]))
        if k.startswith("recall@")
    ]
    seeds = meta["seeds"][a.baseline]
    order = [s for s in STRATEGY_ORDER if s in runs] + [
        s for s in sorted(runs) if s not in STRATEGY_ORDER
    ]

    log.info("策略与 seed：%s", meta["seeds"])
    log.info(
        "口径：protocol=%s  item_tower=%s  steps=%s  候选库 %s",
        meta["protocol"],
        meta["item_tower_config"],
        meta["steps"],
        meta.get("catalog_range", "（未查）"),
    )

    out: dict = {
        "protocol": a.protocol,
        "item_tower_config": a.tower,
        "baseline": a.baseline,
        "meta": meta,
        "popularity_baseline": {k: pop[k] for k in ks},
        "metrics": {},
        f"delta_vs_{a.baseline}_paired": {},
    }

    def vals(s: str, k: str, unit: str) -> np.ndarray:
        return np.array([runs[s][sd]["metrics"][k][unit] for sd in seeds], float)

    for unit in UNITS:
        print(f"\n=== Recall（{unit}，mean ± std，{len(seeds)} seeds）===")
        print(
            f"{'策略':<12}"
            + "".join(f"{k:>22}" for k in ks)
            + f"{'倍热度基线(@' + ks[len(ks) // 2].split('@')[1] + ')':>18}"
        )
        ref = ks[len(ks) // 2]
        for s in order:
            row = f"{s:<12}"
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
            f"{'热度基线':<10}"
            + "".join(f"{pop[k]:>22.5f}" for k in ks)
            + f"{1.0:>17.2f}x"
        )

    print(f"\n=== 配对差值 vs {a.baseline}（同 seed 相减，§16.6 同 seed 集合）===")
    print(
        f"{'策略':<12}{'指标':<14}{'per_request mean±std':>24}  同号  |mean|/std  相对基准"
    )
    for s in order:
        if s == a.baseline:
            continue
        for k in ks:
            v, b = vals(s, k, "per_request"), vals(a.baseline, k, "per_request")
            d = v - b
            same = bool((d > 0).all() or (d < 0).all())
            sd = d.std(ddof=1) if len(d) > 1 else float("nan")
            ratio = v.mean() / b.mean()
            print(
                f"{s:<12}{k:<14}{d.mean():>+15.5f}±{sd:.5f}  "
                f"{'同号' if same else '异号'}  {abs(d.mean()) / sd:>9.1f}  {ratio:>9.3f}x"
            )
        print()
    print(
        "  说明：先算每个 seed 的配对差值再求 mean ± std（seed 是受控的配对因子）。"
        f"「同号」是 {len(seeds)} 个 seed 的方向一致性，**不做显著性判定** —— "
        f"{len(seeds)} 个 seed 估出的 std 本身不可靠。"
    )

    for s in order:
        out["metrics"][s] = {
            k: {
                u: {
                    "mean": float(vals(s, k, u).mean()),
                    "std": float(vals(s, k, u).std(ddof=1)) if len(seeds) > 1 else None,
                    "values": vals(s, k, u).tolist(),
                    "x_popularity": float(vals(s, k, u).mean() / pop[k]),
                }
                for u in UNITS
            }
            for k in ks
        }
        out["metrics"][s]["loss_tail"] = [runs[s][sd].get("loss_tail") for sd in seeds]
        if s == a.baseline:
            continue
        out[f"delta_vs_{a.baseline}_paired"][s] = {}
        for k in ks:
            e = {}
            for u in UNITS:
                d = vals(s, k, u) - vals(a.baseline, k, u)
                e[u] = {
                    "mean": float(d.mean()),
                    "std": float(d.std(ddof=1)) if len(d) > 1 else None,
                    "per_seed": d.tolist(),
                    "all_same_sign": bool((d > 0).all() or (d < 0).all()),
                    "ratio_to_baseline": float(
                        vals(s, k, u).mean() / vals(a.baseline, k, u).mean()
                    ),
                }
            out[f"delta_vs_{a.baseline}_paired"][s][k] = e

    print("\n=== loss_tail（**跨策略不可比**）===")
    for s in order:
        v = np.array([runs[s][sd].get("loss_tail", np.nan) for sd in seeds], float)
        print(f"  {s:<12}{v.mean():.4f}±{v.std(ddof=1):.4f}")
    print(
        "  负样本来自不同分布时，loss 衡量的是不同难度的题。exposure 的 loss 最低而"
        "Recall 最差 —— 它只能用来看同一策略内是否收敛，不能跨策略比。"
    )

    print("\n=== §16.6 可比性：查不了的部分 ===")
    for k in UNVERIFIABLE:
        print(f"  未逐次落盘，无法断言：{k}")
    print(
        "  已知缺口：configs/retrieval.yaml 最后写入 2026-09-25 23:51，晚于 "
        f"{a.baseline} / inbatch 的 seed 42 跑完时间（21:07 / 21:52）。"
        "那两次用的配置无法从 git 复原（同一提交内既加键又改注释）。"
    )
    if meta.get("arch_param_shapes_identical"):
        n_ck = sum(len(v) for v in meta["seeds"].values())
        print(
            f"  已确认（由权重形状）：{n_ck} 个 checkpoint 的参数张量名字与形状"
            "逐一相同，catalog_range 与 steps 一致。"
        )

    dst = results / f"negative_sampling_summary_{a.protocol}_{a.tower}.json"
    # 文件名不含 baseline，内容却含（delta_vs_<baseline>_paired）：换基准重跑会把
    # 原有那份静默覆盖，范围更小的一次运行同样会。与 cold_start_analysis 同一护栏。
    if dst.exists() and not a.force:
        try:
            old = json.loads(dst.read_text("utf-8"))
        except (ValueError, OSError):
            old = {}
        if old.get("baseline") not in (None, a.baseline):
            raise SystemExit(
                f"拒绝覆盖 {dst.name}：现有文件的基准是 {old['baseline']!r}，"
                f"本次是 {a.baseline!r}，配对差值的含义不同。"
                "要换基准请加 --force，或先把现有文件挪开。"
            )
        old_seeds = (old.get("meta") or {}).get("seeds") or {}
        lost = {
            m: sorted(set(v) - set(meta["seeds"].get(m, [])))
            for m, v in old_seeds.items()
        }
        lost = {m: s for m, s in lost.items() if s or m not in meta["seeds"]}
        if lost:
            raise SystemExit(
                f"拒绝覆盖 {dst.name}：现有文件覆盖的范围更大，本次会丢掉 {lost}。"
                "确实要覆盖请加 --force。"
            )
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("汇总已写出 %s", dst.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
