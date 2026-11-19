"""DIN attention 塌缩的仪表化短跑（诊断，不产出正式模型指标）。

复用正式训练的数据顺序、batch、BCE、Adam 与 weight decay，但只跑一个任务/seed，且不评
valid AUC、不保存 checkpoint。它在固定步数记录：

- attention 参数范数；
- BCE 任务梯度范数与 ``weight_decay * parameter`` 范数（仅作量级诊断）；
- query / score / pooled RMS；
- 历史 embedding 整表的参数范数、任务梯度与 weight-decay 项；
- attention 权重和与行内位置分化。

默认只跑到 step 1000。该脚本本身是训练，必须在代码 review 后再执行。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ranking.dataset import RankingData, RankItemStatic
from src.ranking.single_task import SingleTaskDNN
from src.ranking.trainer import epoch_order, resolve_tasks
from src.utils.config import load_config, project_path, require
from src.utils.seed import set_seed

DEFAULT_STEPS = (0, 1, 5, 10, 25, 50, 100, 200, 500, 1000)


def _rms(x: torch.Tensor) -> float:
    return float(x.detach().square().mean().sqrt())


def _snapshot(
    model: SingleTaskDNN,
    batch: dict[str, torch.Tensor],
    *,
    step: int,
    loss: float | None,
    task_grad_norms: dict[str, float] | None,
    wd_term_norms: dict[str, float] | None,
) -> dict:
    enc = model.encoder
    din = enc.din
    if din is None:
        raise AssertionError("诊断要求 history_pooling=din")
    with torch.no_grad():
        flags, item_vec, author_vec, tag_vec = enc._target_embeddings(batch)
        query_parts = []
        if item_vec is not None:
            query_parts.append(item_vec)
        query_parts.extend(
            [
                author_vec,
                tag_vec,
                batch["duration"].unsqueeze(-1),
                batch["age"].unsqueeze(-1),
                torch.stack([flags[k] for k in enc.din_query_flag_keys], dim=-1),
            ]
        )
        query = torch.cat(query_parts, dim=-1)
        history = enc.hist_emb(batch["hist"])
        q = din.query_proj(query).unsqueeze(1).expand(-1, history.shape[1], -1)
        interaction = torch.cat([history, q, history - q, history * q], dim=-1)
        score = din.local_activation(interaction).squeeze(-1)
        weights = enc.attention_weights(batch)
        pooled = torch.bmm(weights.unsqueeze(1), history).squeeze(1)

    mask = batch["hist_mask"]
    nonempty = mask.any(dim=1)
    spreads = torch.stack(
        [
            weights[i][mask[i]].max() - weights[i][mask[i]].min()
            for i in range(len(weights))
            if bool(mask[i].any())
        ]
    )
    sums = weights.sum(dim=1)[nonempty]
    param_norms = {
        name: float(param.detach().norm()) for name, param in din.named_parameters()
    }
    history_embedding_parameter = enc.hist_emb.weight.detach()
    wd_over_grad = None
    if task_grad_norms is not None and wd_term_norms is not None:
        wd_over_grad = {
            name: (
                wd_term_norms[name] / task_grad_norms[name]
                if task_grad_norms[name] > 0
                else float("inf")
            )
            for name in task_grad_norms
        }
    return {
        "step": step,
        "loss": loss,
        "parameter_norms": param_norms,
        "history_embedding_parameter_norm": float(history_embedding_parameter.norm()),
        "history_embedding_parameter_rms": _rms(history_embedding_parameter),
        "history_embedding_parameter_max_abs": float(
            history_embedding_parameter.abs().max()
        ),
        "task_gradient_norms_before_step": task_grad_norms,
        "weight_decay_term_norms_before_step": wd_term_norms,
        "wd_term_over_task_gradient": wd_over_grad,
        "query_projection_rms": _rms(q),
        "attention_score_rms": _rms(score[mask]),
        "history_embedding_rms": _rms(history[mask]),
        "history_pool_rms": _rms(pooled[nonempty]),
        "weight_abs_mean": float(weights[mask].abs().mean()),
        "weight_sum_mean": float(sums.mean()),
        "weight_sum_std": float(sums.std(unbiased=False)),
        "within_row_spread_p50": float(torch.quantile(spreads, 0.50)),
        "within_row_spread_p95": float(torch.quantile(spreads, 0.95)),
        "finite": bool(
            torch.isfinite(query).all()
            and torch.isfinite(score).all()
            and torch.isfinite(weights).all()
            and torch.isfinite(pooled).all()
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/din.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", default="click", help="单个任务简称")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--probe-rows", type=int, default=512)
    ap.add_argument(
        "--exclude-din-history-from-weight-decay",
        action="store_true",
        help="仅作因果诊断：attention 与历史 embedding 不施加 weight decay",
    )
    a = ap.parse_args()
    if a.max_steps <= 0 or a.probe_rows <= 0:
        raise ValueError("--max-steps / --probe-rows 必须为正")

    cfg = load_config(a.config)
    ic = require(cfg, "input")
    if ic.get("history_pooling") != "din":
        raise ValueError("诊断只支持 history_pooling=din")
    normalization = str(require(ic, "din").get("normalization"))
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    label_cols = require(cfg, "data", "labels", "tasks")
    task = resolve_tasks((a.task,), label_cols)[0]
    static = RankItemStatic(proc, a.protocol)
    kwargs = {
        "max_hist": int(ic["max_hist"]),
        "mask_oov": bool(ic["mask_oov_in_history"]),
        "pair_features": bool(ic["pair_features"]),
    }
    train = RankingData(proc, a.protocol, "train", static, **kwargs)
    valid = RankingData(proc, a.protocol, "valid", static, **kwargs)
    probe_rows = np.arange(min(a.probe_rows, len(valid)))
    probe = valid.batch(probe_rows)

    set_seed(a.seed)
    model = SingleTaskDNN(train, cfg, task)
    model.train()
    lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 0.0))
    batch_size = int(cfg["batch_size"])
    if a.exclude_din_history_from_weight_decay:
        no_decay_parameters = list(model.encoder.din.parameters()) + list(
            model.encoder.hist_emb.parameters()
        )
        no_decay_ids = {id(param) for param in no_decay_parameters}
        decayed_parameters = [
            param for param in model.parameters() if id(param) not in no_decay_ids
        ]
        if len(no_decay_ids) != len(no_decay_parameters):
            raise AssertionError("attention / history 参数组存在重复参数")
        if len(no_decay_parameters) + len(decayed_parameters) != len(
            list(model.parameters())
        ):
            raise AssertionError("optimizer 参数分组没有完整覆盖模型")
        optimizer = torch.optim.Adam(
            [
                {"params": decayed_parameters, "weight_decay": wd},
                {"params": no_decay_parameters, "weight_decay": 0.0},
            ],
            lr=lr,
        )
        diagnostic_mode = "din_history_no_wd"
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        diagnostic_mode = "flat_wd"
    order = epoch_order(len(train), a.seed, 1)
    steps = min(a.max_steps, -(-len(train) // batch_size))
    schedule = sorted({s for s in DEFAULT_STEPS if s <= steps} | {steps})
    tracked_parameters = {
        **{
            f"attention.{name}": param
            for name, param in model.encoder.din.named_parameters()
        },
        "history_embedding.weight": model.encoder.hist_emb.weight,
    }
    tracked_weight_decay = {
        name: (0.0 if a.exclude_din_history_from_weight_decay else wd)
        for name in tracked_parameters
    }

    trace = [
        _snapshot(
            model,
            probe,
            step=0,
            loss=None,
            task_grad_norms=None,
            wd_term_norms=None,
        )
    ]
    for i in range(steps):
        rows = order[i * batch_size : (i + 1) * batch_size]
        batch = train.batch(rows)
        logits = model(batch)[task]
        loss = F.binary_cross_entropy_with_logits(logits, batch[f"label_{task}"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        step = i + 1
        grad_norms = None
        wd_norms = None
        if step in schedule:
            grad_norms = {
                name: (
                    float(param.grad.detach().norm()) if param.grad is not None else 0.0
                )
                for name, param in tracked_parameters.items()
            }
            wd_norms = {
                name: tracked_weight_decay[name] * float(param.detach().norm())
                for name, param in tracked_parameters.items()
            }
        optimizer.step()
        if step in schedule:
            row = _snapshot(
                model,
                probe,
                step=step,
                loss=float(loss.detach()),
                task_grad_norms=grad_norms,
                wd_term_norms=wd_norms,
            )
            trace.append(row)
            print(
                f"step {step:4d} loss {row['loss']:.5f}  "
                f"q_rms {row['query_projection_rms']:.3e}  "
                f"score_rms {row['attention_score_rms']:.3e}  "
                f"spread_p95 {row['within_row_spread_p95']:.3e}"
            )

    result = {
        "kind": "diagnostic_short_run_not_formal_result",
        "config": a.config,
        "protocol": a.protocol,
        "seed": a.seed,
        "task": task,
        "normalization": normalization,
        "max_steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": wd,
        "diagnostic_mode": diagnostic_mode,
        "attention_and_history_weight_decay": (
            0.0 if a.exclude_din_history_from_weight_decay else wd
        ),
        "all_other_parameters_weight_decay": wd,
        "probe_split": "valid_features_without_labels",
        "probe_rows": len(probe_rows),
        "schedule": schedule,
        "caveat": (
            "wd_term/task_gradient 只比较进入 Adam 前的量级；Adam 的一二阶矩缩放意味着它"
            "不是参数更新贡献的精确分解。"
        ),
        "trace": trace,
    }
    tag = (
        f"din_collapse_trace_{normalization}_{diagnostic_mode}_{task}_seed{a.seed}.json"
    )
    dst = project_path("results") / tag
    dst.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"诊断轨迹已写出 {dst.name}；未保存 checkpoint，未评 valid/test 指标。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
