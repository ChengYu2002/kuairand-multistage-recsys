"""Selective Sharing：逐任务选择共享表示，同时保留任务私有表示（plan §25）。

## 结构

    x ── shared expert ── s(x), 96 维 ──┬─ gate_click(x) ⊙ s(x) ─┐
    │                                    ├─ gate_like(x)  ⊙ s(x)  │
    │                                    └─ ...                    │
    └─ task expert t ── p_t(x), 32 维 ────────────────────────────┤
                                                                  ▼
                                      concat[选择后的共享, 私有] -> task tower

每个任务的 gate 是一个 96 维 sigmoid 向量，每个值独立落在 [0, 1]：接近 1 表示该任务
保留这一维共享信息，接近 0 表示屏蔽。它与 MMoE / PLE 的 softmax 门控不同：这里不是在
多个专家之间分配总和为 1 的概率，而是对共享表示做逐维开关，所以各维之和**不要求为 1**。

## 为什么 shared / private 是并行分支

`shared expert` 和 `task expert` 都直接读取 RankingEncoder 输出的同一个 `x`，不是
`x -> shared -> private` 的先后关系。这样设计是为了避免共享网络成为信息瓶颈：稠密任务的
强梯度可能让共享表示优先保留 click / long-view 信号，并提前丢掉只对 follow / comment
有用的弱信号；私有分支直接读取完整的 `x`，仍有机会把这些任务特有信号提取出来。最后再把
`gate_t(x) ⊙ s(x)` 与 `p_t(x)` 拼接，分别承担「可迁移的公共规律」和「任务独有规律」。

## 与纯门控式 Selective Sharing 的边界

“Selective Sharing”不是像 MMoE / PLE 那样只有一种公认网络拓扑。若把常见的纯门控式版本
定义为 `x -> shared -> task gate -> task tower`，它只有被逐维筛选的共享表示，没有独立的
任务私有分支。本文件实现的是更明确的 **gated shared + task-private** 变体：每个任务同时
获得 96 维门控共享表示和 32 维私有表示。删除 `task_experts` 后才退化为纯门控式版本；把
所有 gate 固定为 1，则退化为普通的 shared + private 网络。因此报告中应写清这个结构，
不能只写“传统 Selective Sharing”，也不能把它称为 STEM：embedding 与 RankingEncoder
在这里仍由五个任务共享。

私有分支始终只服务自己的任务。这样模型既能复用对多个任务都有效的模式，也能把容易产生
负迁移的信号留在任务内部。若所有 gate 恒为 1，它退化成 shared + private 的普通多任务
网络；若 gate 恒为 0，任务只剩私有分支。因此门控是否随任务、样本变化必须单独验算。

这里选择的是 plan §25 写明的 **task-specific representation**，不是 STEM 的多套
task-specific embedding：RankingEncoder（含约 3,747 万 embedding 参数）仍由五个任务共享。
若复制五套 embedding，总参数会接近五个 Single-Task，既破坏与 MMoE/PLE 的参数预算，
也把「表示层选择」和「embedding 隔离」混成两个变量。后者若要研究，应单列 ME 消融，
不能把本模型的结果表述成 embedding-level negative transfer 已被解决。

## 公平性与参数预算

输入编码器、训练循环、loss、指标和训练协议全部复用公共实现。当前 96 维共享表示与 32 维
私有表示拼成 128 维，和 MMoE 的专家输出维度相同。按当前配置 dense 参数约为 MMoE 的
94%，位于 §25.1 约定的 ±20% 范围内；`match_parameter_budget` 开启时会在构造阶段计算并
强制检查，不让 YAML 里的预算承诺沦为注释。

但**不能声称共享机制是唯一自变量**：MMoE 是 8 个共享专家之间的 softmax 选择；本模型
是 1 条共享分支 + 5 条私有分支，并对 96 维共享表示做 sigmoid 开关。两者同时改变了专家
数量、容量分配和门控语义。这里控制的是输入、训练协议与 dense 参数量级，而不是构造一组
只差一个开关的同构网络。最终共享/私有投影后都接 ReLU，与 MMoE/PLE 的专家输出口径一致。

plan §25 的三方比较由三个模型家族共同承担，而不是本文件的三个运行模式：MMoE 代表
fully-shared expert pool，CGC/PLE 代表 task-specific expert capacity，本文件只负责
selective task-specific representation。

用法：
    python -m src.ranking.selective_sharing --config configs/selective.yaml --seed 42
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from src.ranking.dataset import RankingData, RankItemStatic
from src.ranking.features import RankingEncoder, count_params_grouped, mlp
from src.ranking.trainer import (
    Head,
    format_table,
    resolve_tasks,
    save_checkpoint,
    train,
    write_results,
)
from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)


def _mlp_params(in_dim: int, hidden: list[int], out_dim: int | None = None) -> int:
    """只按 Linear 层独立计算参数量，供预算检查使用。"""
    total = 0
    d = in_dim
    for h in hidden:
        total += d * h + h
        d = h
    if out_dim is not None:
        total += d * out_dim + out_dim
    return total


def _mmoe_reference_dense(d_in: int, n_tasks: int, ref: dict) -> int:
    """由配置独立计算 MMoE dense 参数，不实例化第二份巨大 embedding。"""
    n_experts = int(ref["n_experts"])
    expert_hidden = [int(v) for v in ref["expert_hidden"]]
    tower_hidden = [int(v) for v in ref["tower_hidden"]]
    if n_experts < 1 or not expert_hidden:
        raise ValueError("budget_reference 的 n_experts / expert_hidden 无效")
    experts = n_experts * _mlp_params(d_in, expert_hidden)
    gates = n_tasks * (d_in * n_experts + n_experts)
    towers = n_tasks * _mlp_params(expert_hidden[-1], tower_hidden, out_dim=1)
    return experts + gates + towers


def _representation_mlp(
    in_dim: int, hidden: list[int], out_dim: int, dropout: float
) -> nn.Sequential:
    """分支骨架 + 投影 + ReLU；最终表示与 MMoE/PLE 专家一样为激活后输出。"""
    return nn.Sequential(*mlp(in_dim, hidden, out_dim, dropout).children(), nn.ReLU())


class SelectiveSharing(nn.Module):
    """一条共享分支 + 每任务私有分支 + 每任务逐维共享门控。"""

    def __init__(self, data: RankingData, cfg: dict, tasks: list[str]) -> None:
        super().__init__()
        self.tasks = list(tasks)
        mode = str(cfg.get("sharing_mode", "selective"))
        if mode != "selective":
            raise ValueError(
                f"selective_sharing.py 只实现 sharing_mode='selective'，收到 {mode!r}。"
                "不能让配置写 full_shared / ple_style，代码却静默继续跑 Selective。"
            )
        if len(self.tasks) < 2:
            raise ValueError(
                f"Selective Sharing 至少需要 2 个任务，收到 {len(self.tasks)}。"
                "单任务下不存在要选择的共享关系。"
            )
        self.encoder = RankingEncoder(data, cfg)
        d_in = self.encoder.out_dim
        self.shared_dim = int(cfg["shared_dim"])
        self.task_specific_dim = int(cfg["task_specific_dim"])
        if self.shared_dim < 1 or self.task_specific_dim < 1:
            raise ValueError(
                "shared_dim / task_specific_dim 都必须 >= 1，收到 "
                f"{self.shared_dim} / {self.task_specific_dim}。任一为 0 都会删除"
                "共享或私有路径，使模型退化。"
            )
        hidden = [int(v) for v in cfg["expert_hidden"]]
        if not hidden or any(v < 1 for v in hidden):
            raise ValueError(f"expert_hidden 必须是正整数列表，收到 {hidden}")
        drop = float(cfg.get("dropout", 0.0))

        # 名称沿用 *_experts / gates / towers，使公共参数分段器能正确归类。
        # 虽然名字是复数 shared_experts，目前实际只有一个。 它完成 323 → 256 → 128 → 96
        self.shared_experts = nn.ModuleList(
            [_representation_mlp(d_in, hidden, self.shared_dim, drop)]
        )
        self.task_experts = nn.ModuleList(
            [
                _representation_mlp(d_in, hidden, self.task_specific_dim, drop)
                for _ in self.tasks
            ]
        )
        # sigmoid 是逐维保留率，不是专家概率分布，不能在任务或维度上 softmax。
        self.gates = nn.ModuleList(
            [nn.Linear(d_in, self.shared_dim) for _ in self.tasks]
        )
        tower_in = self.shared_dim + self.task_specific_dim
        self.towers = nn.ModuleList(
            [
                mlp(
                    tower_in,
                    list(cfg["tower_hidden"]),
                    out_dim=1,
                    dropout=drop,
                )
                for _ in self.tasks
            ]
        )

        # 预算约束是模型定义的一部分。用解析公式算参照，避免为了计数再创建一套 embedding。
        groups = count_params_grouped(self)
        self.dense_params = groups["expert"] + groups["gate"] + groups["tower"]
        self.reference_dense_params: int | None = None
        self.budget_ratio: float | None = None
        self.budget_enforced = False
        if bool(cfg.get("match_parameter_budget", False)):
            ref_path = str(require(cfg, "budget_reference_config"))
            ref_cfg = load_config(ref_path)
            ref = {
                "n_experts": require(ref_cfg, "n_experts"),
                "expert_hidden": require(ref_cfg, "expert_hidden"),
                "tower_hidden": require(ref_cfg, "tower_hidden"),
            }
            tolerance = float(cfg.get("budget_tolerance", 0.2))
            if not 0 <= tolerance < 1:
                raise ValueError(f"budget_tolerance 必须在 [0,1) 内，收到 {tolerance}")
            self.reference_dense_params = _mmoe_reference_dense(
                d_in, len(self.tasks), ref
            )
            self.budget_ratio = self.dense_params / self.reference_dense_params
            # 参数预算只定义在正式五任务结构上。--tasks 做子集诊断时，shared 分支只留
            # 一份而 private 分支变少，比例天然会下降；这时仍记录 ratio，但不误报失败。
            self.budget_enforced = len(self.tasks) == len(cfg.get("tasks", []))
            if self.budget_enforced and not (
                1 - tolerance <= self.budget_ratio <= 1 + tolerance
            ):
                raise ValueError(
                    "Selective dense 参数未匹配 MMoE 预算："
                    f"{self.dense_params:,} / {self.reference_dense_params:,} = "
                    f"{self.budget_ratio:.3f}，要求落在 "
                    f"[{1 - tolerance:.2f}, {1 + tolerance:.2f}]。"
                )

    def gate_weights(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """返回逐任务、逐样本、逐共享维度的保留率，形状均为 (B, shared_dim)。"""
        return {
            task: torch.sigmoid(self.gates[i](x)) for i, task in enumerate(self.tasks)
        }

    def representations(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """返回共享表示、私有表示和门控，供 forward 与结构诊断共用。"""
        shared = self.shared_experts[0](x)
        private = {task: self.task_experts[i](x) for i, task in enumerate(self.tasks)}
        gates = self.gate_weights(x)
        return shared, private, gates

    def forward(self, b: dict) -> dict[str, torch.Tensor]:
        x = self.encoder(b)
        shared, private, gates = self.representations(x)
        out = {}
        for i, task in enumerate(self.tasks):
            selected_shared = gates[task] * shared
            h = torch.cat([selected_shared, private[task]], dim=-1)
            out[task] = self.towers[i](h).squeeze(-1)
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/selective.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tasks", default=None, help="逗号分隔的简称子集，覆盖 config")
    ap.add_argument(
        "--eval-test",
        action="store_true",
        help="在 test 上评估。默认不评；架构与超参冻结后再统一开。",
    )
    ap.add_argument("--epochs", type=int, default=None, help="覆盖轮数（诊断曲线用）")
    ap.add_argument(
        "--limit-train", type=int, default=None, help="烟测：只用前 N 行训练"
    )
    ap.add_argument(
        "--limit-eval", type=int, default=None, help="烟测：valid/test 只取前 N 行"
    )
    a = ap.parse_args()

    cfg = load_config(a.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    label_cols = require(cfg, "data", "labels", "tasks")
    shorts = (
        [s.strip() for s in a.tasks.split(",")]
        if a.tasks
        else list(require(cfg, "tasks"))
    )
    tasks = resolve_tasks(shorts, label_cols)
    if a.epochs is not None:
        if a.epochs < 1:
            raise ValueError(f"epochs 必须 >= 1，收到 {a.epochs}")
        cfg["epochs"] = a.epochs

    smoke = any(v is not None for v in (a.epochs, a.limit_train, a.limit_eval))
    tag = f"selective_sharing_{a.protocol}_seed{a.seed}" + ("_smoke" if smoke else "")
    log.info("任务映射 %s -> %s%s", shorts, tasks, "（烟测/诊断）" if smoke else "")

    set_seed(a.seed)
    ic = require(cfg, "input")
    st = RankItemStatic(proc, a.protocol)
    dl = {
        "max_hist": int(ic["max_hist"]),
        "mask_oov": bool(ic["mask_oov_in_history"]),
        "pair_features": bool(ic["pair_features"]),
    }
    tr = RankingData(proc, a.protocol, "train", st, **dl)
    va = RankingData(proc, a.protocol, "valid", st, **dl)
    te = RankingData(proc, a.protocol, "test", st, **dl) if a.eval_test else None
    if a.limit_train or a.limit_eval:
        tr = Head(tr, a.limit_train)
        va = Head(va, a.limit_eval)
        te = Head(te, a.limit_eval) if te is not None else None

    model = SelectiveSharing(tr, cfg, tasks)
    fp = model.encoder.fingerprint()
    res = train(
        model, tuple(tasks), tr, va, te, cfg, a.seed, tag, fp, eval_test=a.eval_test
    )
    res.update(
        {
            "model": "selective_sharing",
            "protocol": a.protocol,
            "smoke": smoke,
            "sharing_mode": "selective_feature_gate",
            "representation_activation": "relu",
            "shared_dim": model.shared_dim,
            "task_specific_dim": model.task_specific_dim,
            "expert_hidden": list(cfg["expert_hidden"]),
            "tower_hidden": list(cfg["tower_hidden"]),
            "params_grouped": count_params_grouped(model),
            "budget_reference_dense": model.reference_dense_params,
            "budget_ratio_vs_mmoe": model.budget_ratio,
            "budget_enforced": model.budget_enforced,
            "budget_reference_config": str(cfg.get("budget_reference_config")),
            "budget_tolerance": float(cfg.get("budget_tolerance", 0.2)),
        }
    )
    save_checkpoint(
        model,
        {
            "model": "selective_sharing",
            "tasks": tasks,
            "seed": a.seed,
            "protocol": a.protocol,
            "shared_dim": model.shared_dim,
            "task_specific_dim": model.task_specific_dim,
            "checkpoint_rule": res["checkpoint_rule"],
            "input_fingerprint": fp,
        },
        tag,
    )
    write_results(res, tag)
    print()
    print(format_table(res, "test" if a.eval_test else "valid"))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
