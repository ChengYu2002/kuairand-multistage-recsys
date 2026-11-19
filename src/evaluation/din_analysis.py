"""DIN-softmax history pooling vs masked mean 的预注册配对汇总。

主指标与门槛不在代码里临时决定，而是读取训练前落盘的
``results/din_prediction.json``：逐任务先算同 seed 的 ``DIN - baseline``，再报告
mean ± sample std；方向性结论要求 AUC 同时满足 ``|mean|/std >= 2.0`` 与 3/3 同号。
这只是项目内的事前启发式门槛，不标记统计显著性。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.utils.config import project_path
from src.utils.logger import get_logger

log = get_logger(__name__)

SEEDS = (42, 43, 44)
METRICS = ("auc", "gauc", "pcoc")
PROTOCOL_KEYS = (
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    # 分组 weight decay 不改变任何维度或指纹，光比 weight_decay 分辨不出两次运行
    # 用的是不同正则协议。旧产物没有这个键，三臂同为旧产物时仍然相等。
    "weight_decay_spec",
    "dropout",
    "eval_every_steps",
    "checkpoint_rule",
)
FP_SHARED_KEYS = (
    "channels",
    "out_dim",
    "vocabs",
    "cat_sizes",
    "flags",
    "data_params",
    "pair_cols",
    "forbidden_in_x",
    "id_dropout",
    "item_id_channel",
)


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少正式结果：{path}")
    return json.loads(path.read_text("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="main")
    ap.add_argument("--split", default="valid", choices=("valid", "test"))
    ap.add_argument("--results", default="results")
    a = ap.parse_args()

    results = project_path(a.results)
    pred = _read(results / "din_prediction.json")
    if a.split != "valid":
        log.warning("预注册的开发判据只针对 valid；test 只能在架构冻结后统一评一次。")

    baseline = {
        s: _read(results / f"single_task_{a.protocol}_seed{s}.json") for s in SEEDS
    }
    din = {
        s: _read(results / f"din_single_task_{a.protocol}_seed{s}.json") for s in SEEDS
    }
    for s in SEEDS:
        if baseline[s].get("model") != "single_task":
            raise AssertionError(
                f"seed {s} baseline model 字段错误：{baseline[s].get('model')}"
            )
        if din[s].get("model") != "din_single_task":
            raise AssertionError(f"seed {s} DIN model 字段错误：{din[s].get('model')}")
        if baseline[s].get("seed") != s or din[s].get("seed") != s:
            raise AssertionError(f"seed {s} 的文件名与结果内 seed 不一致")
        if baseline[s].get("tasks") != din[s].get("tasks"):
            raise AssertionError(f"seed {s} 两臂任务集合/顺序不同")
        pdiff = {
            k: (baseline[s].get(k), din[s].get(k))
            for k in PROTOCOL_KEYS
            if baseline[s].get(k) != din[s].get(k)
        }
        if pdiff:
            raise AssertionError(f"seed {s} 训练协议不同，拒绝出表：{pdiff}")
        bfp = baseline[s]["input_fingerprint"]
        dfp = din[s]["input_fingerprint"]
        fdiff = {
            k: (bfp.get(k), dfp.get(k))
            for k in FP_SHARED_KEYS
            if bfp.get(k) != dfp.get(k)
        }
        if fdiff:
            raise AssertionError(
                f"seed {s} 除 attention 外输入口径不同，拒绝出表：{fdiff}"
            )
        if dfp.get("history_pooling") != "din" or "din" not in dfp:
            raise AssertionError(f"seed {s} 的处理臂指纹没有记录 DIN 结构")

    tasks = list(baseline[SEEDS[0]]["tasks"])
    prereg_tasks = pred["contrast"]["tasks"]
    if tasks != prereg_tasks:
        raise AssertionError(f"实际任务 {tasks} 与预注册 {prereg_tasks} 不同")
    if list(SEEDS) != pred["contrast"]["seeds"]:
        raise AssertionError("实际 seed 与预注册不同")

    summary: dict = {
        "protocol": a.protocol,
        "split": a.split,
        "seeds": list(SEEDS),
        "tasks": tasks,
        "reference": "masked_mean",
        "treatment": "din_softmax",
        "treatment_artifact_prefix": "din_single_task",
        "primary_metric": "auc",
        "rule": "|mean(delta)|/std(delta) >= 2.0 and 3/3 same sign",
        "statistical_significance_claimed": False,
        "per_task": {},
    }

    print(f"\n=== DIN-softmax vs masked mean（{a.split}，同 seed 配对）===")
    print(
        f"{'task':<13}{'metric':<7}{'baseline':>12}{'DIN':>12}{'delta mean±std':>24}"
        f"{'|m|/sd':>10}{'方向':>9}{'过线':>8}"
    )
    for task in tasks:
        summary["per_task"][task] = {}
        for metric in METRICS:
            b = np.array([baseline[s][a.split][task][metric] for s in SEEDS], float)
            d = np.array([din[s][a.split][task][metric] for s in SEEDS], float)
            delta = d - b
            mu = float(delta.mean())
            sd = float(delta.std(ddof=1))
            ratio = float(abs(mu) / sd) if sd > 0 else float("inf")
            same_sign = bool((delta > 0).all() or (delta < 0).all())
            clears = bool(metric == "auc" and ratio >= 2.0 and same_sign)
            summary["per_task"][task][metric] = {
                "baseline_values": b.tolist(),
                "din_values": d.tolist(),
                "baseline_mean": float(b.mean()),
                "din_mean": float(d.mean()),
                "delta_per_seed": delta.tolist(),
                "delta_mean": mu,
                "delta_std": sd,
                "abs_mean_over_std": ratio,
                "all_same_sign": same_sign,
                "clears_preregistered_bar": clears,
            }
            print(
                f"{task:<13}{metric:<7}{b.mean():>12.5f}{d.mean():>12.5f}"
                f"{mu:>+12.5f}±{sd:<9.5f}{ratio:>10.2f}"
                f"{('3/3' if same_sign else '异号'):>9}"
                f"{('是' if clears else '-'):>8}"
            )

    dense = ("is_click", "long_view")
    dense_supported = [
        t
        for t in dense
        if summary["per_task"][t]["auc"]["clears_preregistered_bar"]
        and summary["per_task"][t]["auc"]["delta_mean"] > 0
    ]
    summary["preregistered_outcome"] = {
        "dense_hypothesis_supported": bool(dense_supported),
        "dense_tasks_clearing_positive_bar": dense_supported,
        "interpretation": (
            "至少一个稠密任务通过事前门槛，DIN 假设获得方向性支持。"
            if dense_supported
            else "is_click / long_view 均未通过事前门槛，不宣称 DIN 有效。"
        ),
        "sparse_task_caveat": (
            "is_like / is_follow / is_comment 事前标注为低测量能力；完整报告，但不从单次 "
            "GAUC 正负号推导机制结论。"
        ),
    }

    dst = results / f"din_summary_{a.protocol}_{a.split}.json"
    dst.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("DIN 汇总已写出 %s", dst.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
