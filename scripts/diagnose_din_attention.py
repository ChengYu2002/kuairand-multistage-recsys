"""训练后诊断 DIN-softmax / DIN-raw 的 attention 数值形态。

只读取 fixed valid 行与正式 checkpoint，不读取标签、不参与模型选择。它回答三个机械问题：

1. softmax 权重是否仍严格和为 1；
2. raw 权重是否真的保留了非归一化强度，而非静默退化成 softmax；
3. raw 权重和与历史池化向量是否出现 NaN/Inf 或数值爆炸迹象。

结果写入 ``results/din_attention_diagnostics_<protocol>_valid.json``。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ranking.dataset import RankingData, RankItemStatic
from src.ranking.single_task import SingleTaskDNN
from src.utils.config import load_config, project_path, require
from src.utils.seed import set_seed

SEEDS = (42, 43, 44)
TASKS = ("is_click", "long_view", "is_like", "is_follow", "is_comment")
VARIANTS = {
    "din_softmax": ("configs/din.yaml", "din_single_task"),
    "din_raw": ("configs/din_raw.yaml", "din_raw_single_task"),
}
MIN_PARAMETER_NORM_RATIO = 1e-3
MIN_HISTORY_EMBEDDING_NORM_RATIO = 1e-3
MIN_SPREAD_RATIO = 1e-2
MIN_RAW_ABS_WEIGHT_RATIO = 1e-3
MAX_RAW_POOL_L2_P95 = 5.401233956217766
TRACKED_WEIGHTS = (
    "query_proj.weight",
    "local_activation.0.weight",
    "local_activation.2.weight",
    "local_activation.4.weight",
)


def _quantiles(x: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "p05": float(torch.quantile(x, 0.05)),
        "p50": float(torch.quantile(x, 0.50)),
        "p95": float(torch.quantile(x, 0.95)),
        "max": float(x.max()),
    }


def _row_spreads(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = [
        weights[i][mask[i]].max() - weights[i][mask[i]].min()
        for i in range(len(weights))
        if bool(mask[i].any())
    ]
    return torch.stack(values)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument(
        "--variants",
        default="din_softmax,din_raw",
        help="逗号分隔的诊断变体；训练前可只传 din_softmax 复核已知塌缩",
    )
    a = ap.parse_args()
    if a.rows <= 0:
        raise ValueError("--rows 必须为正")

    raw_cfg = load_config("configs/din_raw.yaml")
    proc = project_path(require(raw_cfg, "data", "dataset", "processed_dir"))
    ic = require(raw_cfg, "input")
    static = RankItemStatic(proc, a.protocol)
    data = RankingData(
        proc,
        a.protocol,
        "valid",
        static,
        max_hist=int(ic["max_hist"]),
        mask_oov=bool(ic["mask_oov_in_history"]),
        pair_features=bool(ic["pair_features"]),
    )
    rng = np.random.default_rng(20261116)
    rows = np.sort(rng.choice(len(data), min(a.rows, len(data)), replace=False))
    batch = data.batch(rows)
    mask = batch["hist_mask"]
    nonempty = mask.any(dim=1)

    out: dict = {
        "protocol": a.protocol,
        "split": "valid",
        "row_selection_seed": 20261116,
        "n_rows": len(rows),
        "label_free": True,
        "variants": {},
    }
    task_gate_failures: list[str] = []
    observations: list[str] = []

    selected = [v.strip() for v in a.variants.split(",") if v.strip()]
    unknown = [v for v in selected if v not in VARIANTS]
    if not selected or unknown:
        raise ValueError(f"--variants 非法：{selected}；未知 {unknown}")

    for variant in selected:
        config_path, prefix = VARIANTS[variant]
        cfg = load_config(config_path)
        out["variants"][variant] = {}
        print(f"\n=== {variant} ===")
        for seed in SEEDS:
            out["variants"][variant][str(seed)] = {}
            for task in TASKS:
                path = project_path("experiments") / (
                    f"{prefix}_{task}_{a.protocol}_seed{seed}.pt"
                )
                if not path.is_file():
                    raise FileNotFoundError(f"缺少正式 checkpoint：{path}")
                saved = torch.load(path, map_location="cpu", weights_only=False)
                # Single-Task 每个任务建模前都会重新 set_seed(seed)，因此这里能逐位重建
                # 对应 checkpoint 的初始化，用相对初始值判断“变弱”还是“近零死亡”。
                set_seed(seed)
                model = SingleTaskDNN(data, cfg, task)
                expected_fp = model.encoder.fingerprint()
                if saved.get("input_fingerprint", {}).get("sha") != expected_fp["sha"]:
                    raise AssertionError(f"{path.name} 的输入指纹与当前 config 不一致")
                model.eval()
                assert model.encoder.din is not None
                initial_norms = {
                    name: float(param.detach().norm())
                    for name, param in model.encoder.din.named_parameters()
                    if name in TRACKED_WEIGHTS
                }
                initial_history_embedding_norm = float(
                    model.encoder.hist_emb.weight.detach().norm()
                )
                with torch.no_grad():
                    initial_weights = model.encoder.attention_weights(batch)
                initial_spreads = _row_spreads(initial_weights, mask)
                initial_abs_mean = float(initial_weights[mask].abs().mean())

                model.load_state_dict(saved["state_dict"])
                model.eval()
                trained_history_embedding = model.encoder.hist_emb.weight.detach()
                trained_history_embedding_norm = float(trained_history_embedding.norm())
                history_embedding_norm_ratio = (
                    trained_history_embedding_norm / initial_history_embedding_norm
                    if initial_history_embedding_norm > 0
                    else 0.0
                )
                history_embedding_survival_ok = (
                    history_embedding_norm_ratio >= MIN_HISTORY_EMBEDDING_NORM_RATIO
                )
                with torch.no_grad():
                    weights = model.encoder.attention_weights(batch)
                    history = model.encoder.hist_emb(batch["hist"])
                    pooled = torch.bmm(weights.unsqueeze(1), history).squeeze(1)

                finite = bool(
                    torch.isfinite(weights).all() and torch.isfinite(pooled).all()
                )
                masked_zero = float(weights[~mask].abs().max()) == 0.0
                sums = weights.sum(dim=1)[nonempty]
                valid_abs = weights[mask].abs()
                pool_norm = pooled[nonempty].norm(dim=1)
                spreads = _row_spreads(weights, mask)
                is_softmax = variant == "din_softmax"
                normalization_shape_ok = (
                    bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-6))
                    if is_softmax
                    else float((sums - 1.0).abs().mean()) > 1e-3
                )
                trained_norms = {
                    name: float(param.detach().norm())
                    for name, param in model.encoder.din.named_parameters()
                    if name in TRACKED_WEIGHTS
                }
                norm_ratios = {
                    name: trained_norms[name] / initial_norms[name]
                    for name in TRACKED_WEIGHTS
                }
                min_norm_ratio = min(norm_ratios.values())
                initial_spread_p95 = float(torch.quantile(initial_spreads, 0.95))
                trained_spread_p95 = float(torch.quantile(spreads, 0.95))
                spread_ratio = (
                    trained_spread_p95 / initial_spread_p95
                    if initial_spread_p95 > 0
                    else 0.0
                )
                raw_abs_ratio = float(valid_abs.mean()) / initial_abs_mean
                pool_l2_p95 = float(torch.quantile(pool_norm, 0.95))
                pool_scale_ok = is_softmax or pool_l2_p95 <= MAX_RAW_POOL_L2_P95
                attention_parameter_survival_ok = (
                    min_norm_ratio >= MIN_PARAMETER_NORM_RATIO
                )
                attention_weight_behavior_ok = spread_ratio >= MIN_SPREAD_RATIO and (
                    is_softmax or raw_abs_ratio >= MIN_RAW_ABS_WEIGHT_RATIO
                )
                structural_ok = finite and masked_zero and normalization_shape_ok
                survival_ok = (
                    structural_ok
                    and history_embedding_survival_ok
                    and attention_parameter_survival_ok
                    and attention_weight_behavior_ok
                    and pool_scale_ok
                )
                failure_reasons = []
                if not structural_ok:
                    failure_reasons.append("structural_or_nonfinite_failure")
                if not history_embedding_survival_ok:
                    failure_reasons.append("upstream_history_embedding_collapse")
                if not attention_parameter_survival_ok:
                    failure_reasons.append("attention_parameter_collapse")
                if not attention_weight_behavior_ok:
                    failure_reasons.append("attention_weight_degenerate")
                if not pool_scale_ok:
                    failure_reasons.append("finite_pool_scale_explosion")
                if not structural_ok:
                    task_gate_failures.append(
                        f"{path.name}: finite={finite}, masked_zero={masked_zero}, "
                        f"normalization_shape_ok={normalization_shape_ok}"
                    )
                collapse_msg = (
                    f"{path.name}: min_param_ratio={min_norm_ratio:.3e}, "
                    f"spread_ratio={spread_ratio:.3e}, abs_weight_ratio={raw_abs_ratio:.3e}, "
                    f"pool_l2_p95={pool_l2_p95:.3e}"
                )
                if not survival_ok:
                    if is_softmax:
                        observations.append("已知 DIN-softmax 塌缩：" + collapse_msg)
                    else:
                        task_gate_failures.append(
                            "DIN-raw task blocked："
                            + collapse_msg
                            + f", reasons={failure_reasons}"
                        )
                stats = {
                    "finite": finite,
                    "masked_weights_zero": masked_zero,
                    "normalization_shape_ok": normalization_shape_ok,
                    "module_survival_ok": survival_ok,
                    "failure_reasons": failure_reasons,
                    "history_embedding_survival_ok": history_embedding_survival_ok,
                    "attention_parameter_survival_ok": attention_parameter_survival_ok,
                    "attention_weight_behavior_ok": attention_weight_behavior_ok,
                    "pool_scale_ok": pool_scale_ok,
                    "thresholds": {
                        "minimum_parameter_norm_ratio": MIN_PARAMETER_NORM_RATIO,
                        "minimum_history_embedding_norm_ratio": (
                            MIN_HISTORY_EMBEDDING_NORM_RATIO
                        ),
                        "minimum_spread_ratio": MIN_SPREAD_RATIO,
                        "minimum_raw_abs_weight_ratio": MIN_RAW_ABS_WEIGHT_RATIO,
                        "maximum_raw_pool_l2_p95": MAX_RAW_POOL_L2_P95,
                    },
                    "parameter_norm_initial": initial_norms,
                    "parameter_norm_trained": trained_norms,
                    "parameter_norm_ratio": norm_ratios,
                    "minimum_parameter_norm_ratio": min_norm_ratio,
                    "history_embedding_norm_initial": initial_history_embedding_norm,
                    "history_embedding_norm_trained": trained_history_embedding_norm,
                    "history_embedding_norm_ratio_to_initial": (
                        history_embedding_norm_ratio
                    ),
                    "history_embedding_max_abs_trained": float(
                        trained_history_embedding.abs().max()
                    ),
                    "within_row_spread_initial": _quantiles(initial_spreads),
                    "within_row_spread_trained": _quantiles(spreads),
                    "p95_spread_ratio_to_initial": spread_ratio,
                    "mean_abs_valid_weight_ratio_to_initial": raw_abs_ratio,
                    "weight_sum_nonempty": _quantiles(sums),
                    "weight_abs_valid": _quantiles(valid_abs),
                    "pooled_l2_nonempty": _quantiles(pool_norm),
                }
                out["variants"][variant][str(seed)][task] = stats
                print(
                    f"seed {seed} {task:<11} sum p50/p95 "
                    f"{stats['weight_sum_nonempty']['p50']:+.3f}/"
                    f"{stats['weight_sum_nonempty']['p95']:+.3f}  "
                    f"norm-ratio {min_norm_ratio:.2e}  spread-ratio {spread_ratio:.2e}  "
                    f"pool-norm p95 {stats['pooled_l2_nonempty']['p95']:.3f}"
                )
                del model

    out["task_gate_failures"] = task_gate_failures
    out["observations"] = observations
    dst = project_path("results") / (
        f"din_attention_diagnostics_{a.protocol}_valid.json"
    )
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\n诊断已写出 {dst.name}；任务级阻断 {len(task_gate_failures)} 项；"
        f"非阻断观察 {len(observations)} 项。"
    )
    if observations:
        print("已知的 DIN-softmax 塌缩只作历史观察，不替 DIN-raw 放行：")
        for observation in observations:
            print(f"  - {observation}")
    if task_gate_failures:
        print("以下失败只阻断对应任务的性能解释，不阻断其他通过任务：")
        for failure in task_gate_failures:
            print(f"  - {failure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
