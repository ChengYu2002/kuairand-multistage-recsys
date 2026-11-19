"""DIN-raw vs DIN-softmax 的预注册三 seed 配对汇总。

主对比是 ``DIN-raw - DIN-softmax``；同时报告相对 masked mean 的次要对比。只有主对比
的 valid AUC 使用事前固定的 ``|mean|/std >= 2.0 + 3/3 同号`` 门槛。GAUC、PCOC
和 valid 曲线只作并列报告/诊断，不追加事后判据。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from src.utils.config import project_path
from src.utils.logger import get_logger

log = get_logger(__name__)

SEEDS = (42, 43, 44)
TASKS = ("is_click", "long_view", "is_like", "is_follow", "is_comment")
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


def _read_raw_attention_gate(results: Path, protocol: str) -> dict:
    """读取逐任务存活门；失败任务不影响其他通过任务。"""
    path = results / f"din_attention_diagnostics_{protocol}_valid.json"
    if not path.is_file():
        raise RuntimeError(
            "缺少 DIN-raw attention 存活诊断；禁止生成性能结论。请先运行 "
            f"scripts/diagnose_din_attention.py --protocol {protocol}。"
        )
    diagnostics = json.loads(path.read_text("utf-8"))
    if diagnostics.get("protocol") != protocol or diagnostics.get("split") != "valid":
        raise RuntimeError(f"attention 诊断口径错误：{path}")
    raw = diagnostics.get("variants", {}).get("din_raw")
    if raw is None:
        raise RuntimeError("attention 诊断不含 din_raw；禁止生成性能结论。")
    if not isinstance(raw, dict):
        raise TypeError("attention 诊断中的 din_raw 结构非法。")

    per_task: dict[str, dict] = {}
    for task in TASKS:
        blocked_seeds: list[dict] = []
        for seed in SEEDS:
            seed_rows = raw.get(str(seed))
            row = seed_rows.get(task) if isinstance(seed_rows, dict) else None
            if not isinstance(row, dict):
                blocked_seeds.append({"seed": seed, "reasons": ["diagnostic_missing"]})
                continue
            if not row.get("module_survival_ok", False):
                blocked_seeds.append(
                    {
                        "seed": seed,
                        "reasons": row.get("failure_reasons")
                        or ["module_survival_failed"],
                        "history_embedding_norm_ratio_to_initial": row.get(
                            "history_embedding_norm_ratio_to_initial"
                        ),
                        "minimum_attention_parameter_norm_ratio": row.get(
                            "minimum_parameter_norm_ratio"
                        ),
                        "pool_scale_ok": row.get("pool_scale_ok"),
                    }
                )
        per_task[task] = {
            "interpretation_allowed": not blocked_seeds,
            "blocked_seeds": blocked_seeds,
        }
    return {
        "path": path.name,
        "split": "valid",
        "scope": "per_task_all_three_seeds_must_pass",
        "n_checked": len(SEEDS) * 5,
        "eligible_tasks": [
            task for task in TASKS if per_task[task]["interpretation_allowed"]
        ],
        "blocked_tasks": [
            task for task in TASKS if not per_task[task]["interpretation_allowed"]
        ],
        "per_task": per_task,
    }


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少正式结果：{path}")
    return json.loads(path.read_text("utf-8"))


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    """返回 a-b 的同 seed 配对统计。"""
    delta = a - b
    mu = float(delta.mean())
    sd = float(delta.std(ddof=1))
    ratio = float(abs(mu) / sd) if sd > 0 else float("inf")
    same_sign = bool((delta > 0).all() or (delta < 0).all())
    return {
        "values": delta.tolist(),
        "mean": mu,
        "std": sd,
        "abs_mean_over_std": ratio,
        "all_same_sign": same_sign,
    }


def _curve(run: dict, task: str) -> dict:
    points: list[dict] = []
    for row in run["per_task"][task].get("history", []):
        valid = row.get("valid", {})
        metric = valid.get(task, {}) if isinstance(valid, dict) else {}
        if "auc" in metric:
            points.append({"step": int(row["step"]), "auc": float(metric["auc"])})
    if not points:
        return {"points": [], "peak": None, "final": None, "final_minus_peak": None}
    peak = max(points, key=lambda x: x["auc"])
    final = points[-1]
    return {
        "points": points,
        "peak": peak,
        "final": final,
        "final_minus_peak": float(final["auc"] - peak["auc"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="main")
    ap.add_argument("--split", default="valid", choices=("valid", "test"))
    ap.add_argument("--results", default="results")
    a = ap.parse_args()

    results = project_path(a.results)
    pred = _read(results / "din_raw_prediction.json")
    attention_gate = _read_raw_attention_gate(results, a.protocol)
    if a.split != "valid":
        log.warning("预注册判据只针对 valid；本轮开发不应评 test。")

    base = {s: _read(results / f"single_task_{a.protocol}_seed{s}.json") for s in SEEDS}
    soft = {
        s: _read(results / f"din_single_task_{a.protocol}_seed{s}.json") for s in SEEDS
    }
    raw = {
        s: _read(results / f"din_raw_single_task_{a.protocol}_seed{s}.json")
        for s in SEEDS
    }

    for s in SEEDS:
        expected = (
            (base[s], "single_task"),
            (soft[s], "din_single_task"),
            (raw[s], "din_raw_single_task"),
        )
        for run, model in expected:
            if run.get("model") != model or run.get("seed") != s:
                raise AssertionError(
                    f"seed {s} 结果身份错误：期望 {model}，收到 "
                    f"{run.get('model')} / seed {run.get('seed')}"
                )
        if not (base[s]["tasks"] == soft[s]["tasks"] == raw[s]["tasks"]):
            raise AssertionError(f"seed {s} 三臂任务集合/顺序不同")
        for other_name, other in (("DIN-softmax", soft[s]), ("DIN-raw", raw[s])):
            pdiff = {
                k: (base[s].get(k), other.get(k))
                for k in PROTOCOL_KEYS
                if base[s].get(k) != other.get(k)
            }
            if pdiff:
                raise AssertionError(
                    f"seed {s} baseline 与 {other_name} 训练协议不同：{pdiff}"
                )
            fdiff = {
                k: (
                    base[s]["input_fingerprint"].get(k),
                    other["input_fingerprint"].get(k),
                )
                for k in FP_SHARED_KEYS
                if base[s]["input_fingerprint"].get(k)
                != other["input_fingerprint"].get(k)
            }
            if fdiff:
                raise AssertionError(
                    f"seed {s} baseline 与 {other_name} 共享输入口径不同：{fdiff}"
                )

        sfp = copy.deepcopy(soft[s]["input_fingerprint"])
        rfp = copy.deepcopy(raw[s]["input_fingerprint"])
        sfp.pop("sha", None)
        rfp.pop("sha", None)
        if sfp.get("din", {}).get("normalization") != "masked_softmax":
            raise AssertionError(f"seed {s} 既有 DIN 结果不是 masked_softmax")
        if rfp.get("din", {}).get("normalization") != "raw":
            raise AssertionError(f"seed {s} 新 DIN 结果不是 raw")
        sfp["din"]["normalization"] = "raw"
        if sfp != rfp:
            raise AssertionError(f"seed {s} 两个 DIN 变体不只差 normalization")

    tasks = list(base[SEEDS[0]]["tasks"])
    if tasks != pred["contrast"]["tasks"] or list(SEEDS) != pred["contrast"]["seeds"]:
        raise AssertionError("实际任务或 seeds 与 DIN-raw 预注册不一致")

    summary: dict = {
        "protocol": a.protocol,
        "split": a.split,
        "seeds": list(SEEDS),
        "tasks": tasks,
        "primary_contrast": "din_raw_minus_din_softmax",
        "primary_metric": "auc",
        "rule": "|mean(delta)|/std(delta) >= 2.0 and 3/3 same sign",
        "statistical_significance_claimed": False,
        "attention_survival_gate": attention_gate,
        "per_task": {},
        "curve_diagnostics": {},
    }

    print(f"\n=== DIN-raw vs DIN-softmax（{a.split}，同 seed 配对）===")
    print(
        f"{'task':<13}{'metric':<7}{'softmax':>11}{'raw':>11}"
        f"{'raw-soft mean±std':>23}{'|m|/sd':>10}{'方向':>9}{'过线':>8}"
    )
    for task in tasks:
        interpretation_allowed = attention_gate["per_task"][task][
            "interpretation_allowed"
        ]
        summary["per_task"][task] = {}
        summary["curve_diagnostics"][task] = {}
        for metric in METRICS:
            b = np.array([base[s][a.split][task][metric] for s in SEEDS], float)
            so = np.array([soft[s][a.split][task][metric] for s in SEEDS], float)
            ra = np.array([raw[s][a.split][task][metric] for s in SEEDS], float)
            primary = _paired(ra, so)
            secondary = _paired(ra, b)
            clears = (
                bool(
                    metric == "auc"
                    and primary["abs_mean_over_std"] >= 2.0
                    and primary["all_same_sign"]
                )
                if interpretation_allowed
                else None
            )
            summary["per_task"][task][metric] = {
                "baseline_values": b.tolist(),
                "din_softmax_values": so.tolist(),
                "din_raw_values": ra.tolist(),
                "din_softmax_mean": float(so.mean()),
                "din_raw_mean": float(ra.mean()),
                "raw_minus_softmax": {
                    **primary,
                    "performance_interpretation_allowed": interpretation_allowed,
                    "clears_preregistered_bar": clears,
                },
                "raw_minus_baseline": secondary,
            }
            print(
                f"{task:<13}{metric:<7}{so.mean():>11.5f}{ra.mean():>11.5f}"
                f"{primary['mean']:>+11.5f}±{primary['std']:<8.5f}"
                f"{primary['abs_mean_over_std']:>10.2f}"
                f"{('3/3' if primary['all_same_sign'] else '异号'):>9}"
                f"{('屏蔽' if clears is None else ('是' if clears else '-')):>8}"
            )
        for s in SEEDS:
            summary["curve_diagnostics"][task][str(s)] = {
                "din_softmax": _curve(soft[s], task),
                "din_raw": _curve(raw[s], task),
            }

    dense_supported = [
        task
        for task in ("is_click", "long_view")
        if summary["per_task"][task]["auc"]["raw_minus_softmax"][
            "clears_preregistered_bar"
        ]
        and summary["per_task"][task]["auc"]["raw_minus_softmax"]["mean"] > 0
    ]
    eligible_dense = [
        task
        for task in ("is_click", "long_view")
        if attention_gate["per_task"][task]["interpretation_allowed"]
    ]
    if dense_supported:
        hypothesis_supported: bool | None = True
        interpretation = (
            "至少一个通过存活门的稠密任务通过事前性能门槛，"
            "raw 激活强度假设获得方向性支持。"
        )
    elif len(eligible_dense) == 2:
        hypothesis_supported = False
        interpretation = (
            "两个稠密任务均通过存活门，但均未通过事前性能门槛，"
            "不宣称取消 softmax 有效。"
        )
    else:
        hypothesis_supported = None
        interpretation = (
            "至少一个稠密任务未通过模块存活门，主假设证据不完整；"
            "仅报告通过任务，不给出整体支持/否定结论。"
        )
    summary["preregistered_outcome"] = {
        "raw_intensity_hypothesis_supported": hypothesis_supported,
        "dense_tasks_clearing_positive_bar": dense_supported,
        "eligible_dense_tasks": eligible_dense,
        "interpretation": interpretation,
    }

    dst = results / f"din_raw_summary_{a.protocol}_{a.split}.json"
    dst.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("DIN-raw 汇总已写出 %s", dst.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
