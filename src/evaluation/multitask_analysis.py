"""多任务结果汇总：mean ± std 与 ΔAUC / ΔGAUC（plan §26 / §27.1 / §28）。

## 它要回答的唯一问题

> 某个 MTL 模型相对 Single-Task 的差异，**是否超过 seed 方差**？

§28 的原话是「损失是否超过 seed variance」。文献里的效应量只有 +0.0016~+0.0045 AUC
（PLE Table 1、MMoE Table 1），不带 std 的 ΔAUC 是没有意义的数字。

## 差值必须**配对**计算

四个模型用的是同一组 seed、同一份数据、同一个行序（epoch_order 只依赖
(seed, epoch, n)），所以 seed 是**受控的配对因子**：正确的做法是先算每个 seed 的
`MMoE_s − ST_s`，再报这组配对差值的 mean ± std。

均值两种算法相同，但**不确定性尺度差很多**。实测（3 seeds）：

    is_like    配对 std 0.00060   非配对合并 0.00179   差 3 倍
    is_follow  配对 std 0.00543   非配对合并 0.01693   差 3 倍

用非配对尺度会把 is_follow 误判成「读不出」，而配对后它是 3/3 个 seed 同号、
|mean|/std = 2.2。反过来 is_click 的比值会从 1.6 掉到 1.2。**结论会被算法本身改变。**

## 不做显著性判定

3 个 seed 估出来的 std 本身就极不可靠，这里只报配对 mean ± std 与**方向一致性**
（几个 seed 同号），不打任何「显著 / 不显著」的标记。之前用 `+/-` 标记是把一个启发式
判断说成了统计结论。

## 口径

- **逐任务报，不报聚合指标。** 三种加权（等权 / 用户数 / 正样本数）会给出**相反**的结论，
  而没有客观标准去选。MMoE / PLE / STEM 三篇论文也都逐任务报。聚合值只在 README 里
  作记录，不作判据。
- **ΔAUC 的分母是 Single-Task**（§27.1）。Single-Task 是参照点、不是候选方案：工业上
  部署的是一套权重同时服务多目标，五个独立模型不可部署（MMoE 论文 §6.4：分开训练每个
  任务需要数十亿参数）。但「负迁移」只有相对不共享才有定义，所以它必须留在表里。
- **判定用 seed std**，不用单次训练内部的波动 —— 后者只是代用品，它量的是优化抖动，
  不含初始化与数据顺序的差异。

## 一致性前提

汇总前断言所有结果的 input_fingerprint 与训练协议一致。不一致时**拒绝出表**而不是
打印警告：一张混了两种口径的对比表，看上去和正确的表完全一样。

用法：
    python -m src.evaluation.multitask_analysis
    python -m src.evaluation.multitask_analysis --protocol main --split valid
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils.config import project_path
from src.utils.logger import get_logger

log = get_logger(__name__)

BASELINE = "single_task"
MODEL_ORDER = ("single_task", "mmoe", "ple", "selective_sharing")
METRICS = ("auc", "gauc", "pcoc")
# 汇总前必须一致的字段。不一致 = 两张表混了口径，而混出来的表看不出异常。
PROTOCOL_KEYS = ("epochs", "batch_size", "lr", "weight_decay", "dropout",
                 "eval_every_steps", "checkpoint_rule")


def load_runs(results: Path, protocol: str) -> dict[str, dict[int, dict]]:
    """results/ 下的正式结果，按 {model: {seed: 结果}} 组织。跳过烟测。"""
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for f in sorted(results.glob("*.json")):
        try:
            r = json.loads(f.read_text("utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(r, dict) or r.get("model") not in MODEL_ORDER:
            continue
        if r.get("smoke") or r.get("protocol") != protocol:
            continue
        runs[r["model"]][int(r["seed"])] = r
    return runs


def assert_comparable(runs: dict[str, dict[int, dict]]) -> dict:
    """所有 run 的输入指纹与训练协议必须一致，否则拒绝出表。"""
    fps = {(m, s): r.get("input_fingerprint", {}).get("sha")
           for m, d in runs.items() for s, r in d.items()}
    if len(set(fps.values())) != 1:
        raise AssertionError(
            f"输入指纹不一致，拒绝出表：{fps}。"
            "不同指纹意味着模型看到的不是同一个问题，ΔAUC 不能归因于共享机制。")
    protos = {(m, s): tuple(r.get(k) for k in PROTOCOL_KEYS)
              for m, d in runs.items() for s, r in d.items()}
    if len(set(protos.values())) != 1:
        raise AssertionError(f"训练协议不一致，拒绝出表：{protos}")
    one = next(iter(next(iter(runs.values())).values()))
    return {"input_fingerprint": next(iter(fps.values())),
            **{k: one.get(k) for k in PROTOCOL_KEYS}}


def collect(runs: dict[str, dict[int, dict]], split: str,
            tasks: list[str]) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """{model: {task: {metric: 各 seed 的值}}}。"""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for m, by_seed in runs.items():
        for t in tasks:
            for k in METRICS:
                out[m][t][k] = np.array(
                    [by_seed[s][split][t][k] for s in sorted(by_seed)], float)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="main")
    ap.add_argument("--split", default="valid", choices=("valid", "test"))
    ap.add_argument("--results", default="results")
    a = ap.parse_args()

    results = project_path(a.results)
    runs = load_runs(results, a.protocol)
    if BASELINE not in runs:
        raise SystemExit(
            f"results/ 下没有 {BASELINE} 的正式结果（protocol={a.protocol}）。"
            "ΔAUC 的分母缺失，无法出表。")
    meta = assert_comparable(runs)
    tasks = list(next(iter(runs[BASELINE].values()))["tasks"])
    seeds = {m: sorted(d) for m, d in runs.items()}
    log.info("模型与 seed：%s", {m: seeds[m] for m in MODEL_ORDER if m in seeds})
    log.info("指纹 %s | %s", meta["input_fingerprint"],
             "  ".join(f"{k}={meta[k]}" for k in PROTOCOL_KEYS))
    n_seed = len(seeds[BASELINE])
    # 至少要 2 个 seed 才谈得上 std，§26 要求 3 个。不足时只列数字、不做判定。
    judgeable = n_seed >= 3 and all(len(v) >= 3 for v in seeds.values())
    if len({tuple(sorted(v)) for v in seeds.values()}) > 1:
        log.warning("各模型的 seed 集合不一致：%s —— ΔAUC 会混进 seed 差异，"
                    "补齐后再读。", {m: sorted(v) for m, v in seeds.items()})
    if not judgeable:
        log.warning("seed 数不足（%s）。§26 要求 3 seeds —— 在此之前 ΔAUC 只能作为"
                    "假设，本表不做显著性判定。",
                    {m: len(v) for m, v in seeds.items()})

    vals = collect(runs, a.split, tasks)
    base = vals[BASELINE]

    # 配对差值：只在 seed 集合完全相同时才算，否则相减的是不同 seed 的结果
    paired: dict = defaultdict(lambda: defaultdict(dict))
    base_seeds = sorted(runs[BASELINE])
    for m in runs:
        if m == BASELINE:
            continue
        ok = sorted(runs[m]) == base_seeds
        for t in tasks:
            for k in METRICS:
                paired[m][t][k] = None if not ok else np.array(
                    [runs[m][sd][a.split][t][k] - runs[BASELINE][sd][a.split][t][k]
                     for sd in base_seeds], float)

    for k in ("auc", "gauc"):
        # 表头不写统一的 seed 数：各模型可能不同（跑挂过、补跑中），写一个数会让人读错。
        print(f"\n=== {k.upper()}（{a.split}，mean ± std；seed 数见每行括号）===")
        head = f"{'model':<18}" + "".join(f"{t:>22}" for t in tasks)
        print(head)
        for m in MODEL_ORDER:
            if m not in vals:
                continue
            row = f"{m + f'({len(seeds[m])})':<18}"
            for t in tasks:
                v = vals[m][t][k]
                sd = f"{v.std(ddof=1):.5f}" if len(v) > 1 else "  n/a"
                row += f"{v.mean():>15.5f}±{sd}"
            print(row)
        print(f"\n--- Δ{k.upper()} 相对 {BASELINE}（配对差值，§27.1）---")
        print(f"{'model':<18}" + "".join(f"{t:>26}" for t in tasks))
        for m in MODEL_ORDER:
            if m == BASELINE or m not in vals:
                continue
            row = f"{m:<18}"
            for t in tasks:
                d = paired[m][t][k]
                if d is None:
                    row += f"{'seed 集合不同':>26}"
                    continue
                mu = float(d.mean())
                if len(d) < 2:
                    row += f"{mu:>+16.5f}±  n/a  "
                    continue
                sd = float(d.std(ddof=1))
                same = "同号" if (d > 0).all() or (d < 0).all() else "异号"
                row += f"{mu:>+13.5f}±{sd:.5f} {same}"
            print(row)
        print("  说明：先算每个 seed 的配对差值再求 mean ± std（seed 是受控的配对因子）。"
              f"「同号/异号」是 {n_seed} 个 seed 的方向一致性。"
              "**不做显著性判定** —— 3 个 seed 估出的 std 本身极不可靠。")

    print(f"\n=== PCOC（{a.split}，mean；1.0 为校准良好）===")
    print(f"{'model':<18}" + "".join(f"{t:>16}" for t in tasks))
    for m in MODEL_ORDER:
        if m not in vals:
            continue
        print(f"{m + f'({len(seeds[m])})':<18}"
              + "".join(f"{vals[m][t]['pcoc'].mean():>16.4f}" for t in tasks))

    out = {
        "protocol": a.protocol, "split": a.split, "tasks": tasks,
        "seeds": {m: seeds[m] for m in seeds}, "meta": meta,
        "metrics": {m: {t: {k: {"mean": float(vals[m][t][k].mean()),
                                "std": float(vals[m][t][k].std(ddof=1))
                                if len(vals[m][t][k]) > 1 else None,
                                "values": vals[m][t][k].tolist()}
                            for k in METRICS} for t in tasks} for m in vals},
        "n_seeds": n_seed, "significance_judgeable": judgeable,
        "delta_vs_single_task_paired": {
            m: {t: {k: (None if paired[m][t][k] is None else {
                "mean": float(paired[m][t][k].mean()),
                "std": float(paired[m][t][k].std(ddof=1)) if len(paired[m][t][k]) > 1 else None,
                "per_seed": paired[m][t][k].tolist(),
                "all_same_sign": bool((paired[m][t][k] > 0).all()
                                      or (paired[m][t][k] < 0).all()),
            }) for k in METRICS} for t in tasks} for m in vals if m != BASELINE},
    }
    dst = results / f"multitask_summary_{a.protocol}_{a.split}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("汇总已写出 %s", dst.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
