"""分组 weight decay 的协议决策运行（不是正式实验臂）。

为什么需要它：results/din_collapse_trace_*.json 已经定位了两个失效模式 ——
扁平 ``weight_decay=1e-5`` 让 attention 在 is_click / is_like 上都塌缩到 1e-38，并让
稀疏任务的 hist_emb 整张表死掉；解除衰减后 hist_emb 在 1000 步内长到初始值的
4.01x / 7.31x。但那四条轨迹**没有评 AUC**，所以"长 4 倍算不算过头"无法判断。

拿范数定正则是在对代理指标调参。这个脚本改成拿 valid AUC 定：同一任务、同一 seed、
同一数据顺序，跑满正式步数，只改 weight decay 分组，直接比曲线。

它刻意**不保存 checkpoint、不评 test、不写 single_task / din_* 前缀的产物**，因此
不可能污染任何正式实验臂；输出带 ``kind`` 标记，分析脚本也不会把它当成实验结果。

用法：
    python scripts/decide_wd_grouping.py                       # 三臂 x is_click/is_like
    python scripts/decide_wd_grouping.py --history-wd 0 1e-6 3e-6
    python scripts/decide_wd_grouping.py --tasks click --max-steps 200   # 冒烟
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ranking.dataset import RankingData, RankItemStatic
from src.ranking.single_task import SingleTaskDNN
from src.ranking.trainer import Head, resolve_tasks, train
from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)

PROBE_SEED = 20261116  # 与 diagnose_din_attention.py 同一批探针行，数值可直接对照
PROBE_ROWS = 4096


def _probe(model: SingleTaskDNN, batch: dict) -> dict:
    """历史通道与 attention 的模块状态。只读 valid 特征，不读标签。"""
    enc = model.encoder
    mask = batch["hist_mask"]
    nonempty = mask.any(dim=1)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        hist = enc.hist_emb(batch["hist"])
        pooled_rows = None
        spread_p95 = None
        if enc.din is not None:
            weights = enc.attention_weights(batch)
            pooled_rows = torch.bmm(weights.unsqueeze(1), hist).squeeze(1)
            spreads = torch.stack([
                weights[i][mask[i]].max() - weights[i][mask[i]].min()
                for i in range(len(weights))
                if bool(mask[i].any())
            ])
            spread_p95 = float(torch.quantile(spreads, 0.95))
        else:
            m = mask.unsqueeze(-1).to(hist.dtype)
            pooled_rows = (hist * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    if was_training:
        model.train()
    pool_norm = pooled_rows[nonempty].norm(dim=1)
    out = {
        "history_embedding_norm": float(enc.hist_emb.weight.detach().norm()),
        "history_embedding_rms": float(
            enc.hist_emb.weight.detach().square().mean().sqrt()
        ),
        "history_pool_l2_p50": float(torch.quantile(pool_norm, 0.50)),
        "history_pool_l2_p95": float(torch.quantile(pool_norm, 0.95)),
        "within_row_spread_p95": spread_p95,
    }
    if enc.din is not None:
        out["attention_parameter_norms"] = {
            name: float(param.detach().norm())
            for name, param in enc.din.named_parameters()
            if name.endswith(".weight")
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/din.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tasks", default="click,like", help="逗号分隔的任务简称")
    ap.add_argument(
        "--history-wd", nargs="+", type=float, default=[0.0, 1e-6],
        help="历史 embedding 组要比较的 weight decay；attention 组在这些臂里固定为 0",
    )
    ap.add_argument(
        "--no-flat-reference", action="store_true",
        help="不跑扁平 weight decay 参照臂（默认跑，用来锚定现有协议）",
    )
    ap.add_argument("--max-steps", type=int, default=None, help="仅供冒烟，截断训练行数")
    ap.add_argument("--limit-eval", type=int, default=None, help="仅供冒烟，截断 valid 行数")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    label_cols = require(cfg, "data", "labels", "tasks")
    shorts = [s.strip() for s in a.tasks.split(",") if s.strip()]
    tasks = resolve_tasks(shorts, label_cols)
    ic = require(cfg, "input")
    kwargs = {
        "max_hist": int(ic["max_hist"]),
        "mask_oov": bool(ic["mask_oov_in_history"]),
        "pair_features": bool(ic["pair_features"]),
    }
    static = RankItemStatic(proc, a.protocol)
    tr_full = RankingData(proc, a.protocol, "train", static, **kwargs)
    va = RankingData(proc, a.protocol, "valid", static, **kwargs)

    smoke = a.max_steps is not None or a.limit_eval is not None
    n = None
    if a.max_steps is not None:
        if a.max_steps <= 0:
            raise ValueError("--max-steps 必须为正")
        n = min(len(tr_full), a.max_steps * int(cfg["batch_size"]))
    if a.limit_eval is not None:
        if a.limit_eval <= 0:
            raise ValueError("--limit-eval 必须为正")
        va = Head(va, a.limit_eval)
    if smoke:
        log.warning(
            "冒烟模式：train 行数 %s / valid 行数 %s，结果不得解释",
            n if n is not None else "全量", len(va),
        )

    rng = np.random.default_rng(PROBE_SEED)
    probe_rows = np.sort(
        rng.choice(len(va), min(PROBE_ROWS, len(va)), replace=False)
    )
    probe_batch = va.batch(probe_rows)

    arms: list[tuple[str, dict | None]] = []
    if not a.no_flat_reference:
        arms.append(("flat_wd", None))
    for hw in a.history_wd:
        if hw < 0.0:
            raise ValueError(f"--history-wd 必须非负，收到 {hw}")
        label = "attn0_hist0" if hw == 0.0 else f"attn0_hist{hw:g}"
        arms.append((label, {"attention": 0.0, "history_embedding": hw}))
    if len({name for name, _ in arms}) != len(arms):
        raise ValueError("臂名重复；--history-wd 里有重复取值")

    out: dict = {
        "kind": "protocol_decision_run_not_formal_arm",
        "purpose": (
            "用 valid AUC 而不是参数范数，决定 attention / hist_emb 的分组 weight decay。"
            "本文件不是实验臂，不得用于任何 masked_mean / DIN 的性能结论。"
        ),
        "config": a.config,
        "protocol": a.protocol,
        "split": "valid",
        "seed": a.seed,
        "tasks": tasks,
        "smoke": smoke,
        "probe": {"row_selection_seed": PROBE_SEED, "n_rows": len(probe_rows)},
        "checkpoints_saved": False,
        "test_evaluated": False,
        "arms": {},
    }

    for arm_name, groups in arms:
        out["arms"][arm_name] = {}
        for task in tasks:
            cfg_arm = copy.deepcopy(cfg)
            if groups is not None:
                cfg_arm["weight_decay_groups"] = dict(groups)
            # 与正式 Single-Task 逐字一致：每个任务建模前重新 set_seed，
            # 因此同一 seed 下所有臂的初始化和数据顺序逐位相同，差异只来自正则。
            set_seed(a.seed)
            model = SingleTaskDNN(tr_full, cfg_arm, task)
            fp = model.encoder.fingerprint()
            before = _probe(model, probe_batch)
            tr = Head(tr_full, n) if n is not None else tr_full
            t0 = time.perf_counter()
            res = train(
                model, (task,), tr, va, None, cfg_arm, a.seed,
                f"wd_decision_{arm_name}_{task}_seed{a.seed}", fp, eval_test=False,
            )
            after = _probe(model, probe_batch)
            curve = [
                {
                    "step": h["step"],
                    "train_loss": h["train_loss"],
                    "auc": h["valid"][task]["auc"],
                    "gauc": h["valid"][task]["gauc"],
                    "pcoc": h["valid"][task]["pcoc"],
                }
                for h in res["history"]
            ]
            peak = max(curve, key=lambda r: r["auc"])
            last = curve[-1]
            out["arms"][arm_name][task] = {
                "weight_decay_spec": res["weight_decay_spec"],
                "steps_per_epoch": res["steps_per_epoch"],
                "train_seconds": time.perf_counter() - t0,
                "valid_curve": curve,
                "valid_auc_last": last["auc"],
                "valid_auc_peak": peak["auc"],
                "valid_auc_peak_step": peak["step"],
                "valid_auc_peak_to_last": last["auc"] - peak["auc"],
                "module_before": before,
                "module_after": after,
                "history_embedding_norm_ratio": (
                    after["history_embedding_norm"] / before["history_embedding_norm"]
                ),
            }
            log.info(
                "%s | %s | AUC 末 %.5f 峰 %.5f@%d (Δ%+.5f) | hist_emb %.2fx | spread_p95 %s",
                arm_name, task, last["auc"], peak["auc"], peak["step"],
                last["auc"] - peak["auc"],
                out["arms"][arm_name][task]["history_embedding_norm_ratio"],
                f"{after['within_row_spread_p95']:.3e}"
                if after["within_row_spread_p95"] is not None else "n/a",
            )
            del model

    name = a.out or f"wd_grouping_decision_{a.protocol}_valid"
    if smoke:
        name += "_smoke"
    dst = project_path("results") / f"{name}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'arm':<16}{'task':<11}{'AUC末':>10}{'AUC峰':>10}{'峰后':>10}"
          f"{'hist_emb':>10}{'spread_p95':>13}")
    for arm_name, _ in arms:
        for task in tasks:
            r = out["arms"][arm_name][task]
            sp = r["module_after"]["within_row_spread_p95"]
            print(f"{arm_name:<16}{task:<11}{r['valid_auc_last']:>10.5f}"
                  f"{r['valid_auc_peak']:>10.5f}{r['valid_auc_peak_to_last']:>+10.5f}"
                  f"{r['history_embedding_norm_ratio']:>9.2f}x"
                  f"{(f'{sp:.3e}' if sp is not None else 'n/a'):>13}")
    print(f"\n已写出 {dst.name}；未保存 checkpoint，未评 test。"
          f"{' 冒烟结果不得解释。' if smoke else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
