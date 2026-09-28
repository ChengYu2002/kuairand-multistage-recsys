"""Single-Task DNN 基线（plan §22）。每个任务**独立**训练一个模型。

## 为什么必须独立到 embedding

五个模型不共享任何参数，包括 embedding 表。这才是 §27.1 里 ΔAUC 的合法分母 ——
它要回答"完全不共享时每个任务能做到多好"。一旦五个任务共享 embedding，那已经是一种
（最底层的）共享机制，ΔAUC 就不再是"有共享 vs 无共享"的对比。

代价：总参数量约等于单模型的 5 倍（embedding 占大头）。§25.1 的 parameter budget 里
Single-Task 是**参照点**，不是"预算内的一档"；MMoE / PLE / Selective 之间才比预算。

## 结构

    batch -> RankingEncoder（四个模型共用，见 features.py）-> x (308 维)
          -> MLP tower_hidden -> 1 个 logit

共享机制作用在 x 之上，这里的"机制"就是"没有机制"：一个塔，一个任务。

## 任务名映射

config 里写的是简称 `[click, long_view, like, follow, comment]`，实际列名是
`[is_click, long_view, is_like, is_comment, is_follow]`。这里显式映射并在歧义时报错 ——
猜错列名会训练出一个"预测另一个任务"的模型，而它照样收敛、照样出 AUC。

用法：
    python -m src.ranking.single_task --config configs/single_task.yaml --seed 42
"""

from __future__ import annotations

import argparse
import gc

import torch
from torch import nn

from src.ranking.dataset import RankingData, RankItemStatic
from src.ranking.features import RankingEncoder, mlp
from src.ranking.trainer import (
    format_table,
    save_checkpoint,
    train,
    write_results,
)
from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)


def resolve_tasks(short_names: list[str], label_cols: list[str]) -> list[str]:
    """简称 -> 真实列名。歧义或缺失一律报错，不猜。"""
    out = []
    for s in short_names:
        cands = [c for c in label_cols if c == s or c == f"is_{s}"]
        if len(cands) != 1:
            raise ValueError(
                f"任务简称 {s!r} 在标签列 {label_cols} 里匹配到 {cands}，"
                "必须恰好一个。猜错列名会训练出一个预测别的任务的模型，而它照样收敛。"
            )
        out.append(cands[0])
    return out


class SingleTaskDNN(nn.Module):
    def __init__(self, data: RankingData, cfg: dict, task: str) -> None:
        super().__init__()
        self.task = task
        self.encoder = RankingEncoder(data, cfg)
        self.tower = mlp(self.encoder.out_dim, list(cfg["tower_hidden"]), out_dim=1,
                         dropout=float(cfg.get("dropout", 0.0)))

    def forward(self, b: dict) -> dict[str, torch.Tensor]:
        return {self.task: self.tower(self.encoder(b)).squeeze(-1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/single_task.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tasks", default=None, help="逗号分隔的简称子集，覆盖 config")
    ap.add_argument("--eval-test", action="store_true",
                    help="在 test 上评估。默认不评 —— 开发期反复看 test 会把它变成第二个 "
                         "validation，而 test 只能用一次。架构与超参冻结后再统一开。")
    # 下面三个是烟测专用。带上任意一个都会改 tag，避免覆盖正式结果（同 two_tower 的做法）。
    ap.add_argument("--epochs", type=int, default=None, help="烟测：覆盖轮数")
    ap.add_argument("--limit-train", type=int, default=None, help="烟测：只用前 N 行训练")
    ap.add_argument("--limit-eval", type=int, default=None, help="烟测：valid/test 只取前 N 行")
    a = ap.parse_args()

    cfg = load_config(a.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    label_cols = require(cfg, "data", "labels", "tasks")
    shorts = [s.strip() for s in a.tasks.split(",")] if a.tasks else list(require(cfg, "tasks"))
    tasks = resolve_tasks(shorts, label_cols)
    if a.epochs:
        cfg["epochs"] = a.epochs

    smoke = any(v is not None for v in (a.epochs, a.limit_train, a.limit_eval))
    tag = f"single_task_{a.protocol}_seed{a.seed}" + ("_smoke" if smoke else "")
    log.info("任务映射 %s -> %s%s", shorts, tasks, "（烟测）" if smoke else "")

    set_seed(a.seed)
    # 装载器口径必须来自 config（否则 YAML 改了不生效，而且指纹察觉不到）
    ic = require(cfg, "input")
    st = RankItemStatic(proc, a.protocol)
    dl = {"max_hist": int(ic["max_hist"]), "mask_oov": bool(ic["mask_oov_in_history"])}
    tr = RankingData(proc, a.protocol, "train", st, **dl)
    va = RankingData(proc, a.protocol, "valid", st, **dl)
    te = RankingData(proc, a.protocol, "test", st, **dl) if a.eval_test else None
    if a.limit_train or a.limit_eval:
        tr = _Head(tr, a.limit_train)
        va = _Head(va, a.limit_eval)
        te = _Head(te, a.limit_eval) if te is not None else None

    per_task, fp = {}, None
    for t in tasks:
        # 每个任务从头初始化：同一个 seed 下五个模型的初值相同，但彼此不共享参数
        set_seed(a.seed)
        model = SingleTaskDNN(tr, cfg, t)
        fp = model.encoder.fingerprint()
        res = train(model, (t,), tr, va, te, cfg, a.seed, f"{tag}/{t}", fp,
                    eval_test=a.eval_test)
        per_task[t] = res
        save_checkpoint(model, {"task": t, "seed": a.seed, "protocol": a.protocol,
                                "checkpoint_rule": res["checkpoint_rule"],
                                "input_fingerprint": fp},
                        f"single_task_{t}_{a.protocol}_seed{a.seed}"
                        + ("_smoke" if smoke else ""))
        # 每个任务一落盘就写出：跑到第 4 个任务挂掉时，前 3 个的指标不能只留在内存里，
        # 否则只剩 checkpoint 而没有指标，被迫重跑。
        write_results(res, f"{tag}__{t}")
        del model
        gc.collect()

    combined = {
        "tag": tag, "model": "single_task", "protocol": a.protocol, "seed": a.seed,
        "tasks": tasks, "smoke": smoke,
        "checkpoint_rule": "fixed_epochs_last",
        # 训练协议要在**汇总**里也记全 —— 报告引的是这个文件，缺一项就等于没记录。
        # verify_single_task 的 J 段会比对各模型结果之间这些字段是否一致。
        "epochs": int(cfg["epochs"]), "batch_size": int(cfg["batch_size"]),
        "lr": float(cfg["lr"]), "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "dropout": float(cfg.get("dropout", 0.0)),
        "eval_every_steps": int(cfg.get("eval_every_steps", 0)),
        "input_fingerprint": fp,
        # 5 个独立模型之和：§25.1 里 Single-Task 是参照点，不是预算内的一档
        "params_total": {k: sum(per_task[t]["params"][k] for t in tasks)
                         for k in ("embedding", "dense", "total")},
        "per_task": per_task,
        "eval_test": a.eval_test,
        "valid": {t: per_task[t]["valid"][t] for t in tasks},
    }
    if a.eval_test:
        combined["test"] = {t: per_task[t]["test"][t] for t in tasks}
    write_results(combined, tag)
    print()
    print(format_table(combined, "test" if a.eval_test else "valid"))
    print()
    return 0


class _Head:
    """烟测用：把一个 RankingData 截成前 n 行。正式跑不会用到。"""

    def __init__(self, data: RankingData, n: int | None) -> None:
        self._d = data
        self._n = len(data) if n is None else min(n, len(data))

    def __len__(self) -> int:
        return self._n

    def __getattr__(self, k):
        return getattr(self._d, k)

    @property
    def labels(self):
        return {k: v[: self._n] for k, v in self._d.labels.items()}

    @property
    def user_id(self):
        return self._d.user_id[: self._n]

    def batch(self, rows):
        return self._d.batch(rows)


if __name__ == "__main__":
    raise SystemExit(main())
