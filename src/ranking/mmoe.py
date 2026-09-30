"""MMoE：共享专家 + 每任务门控（plan §23）。

## 与 Single-Task 唯一的差别

    Single-Task:  x -> 一个塔 [256,128,64] -> 1 个 logit        （五个独立模型）
    MMoE:         x -> K 个共享专家 -> 每任务门控 -> 每任务小塔 -> 5 个 logit（一个模型）

核心就是一个加权和：

    h_t = Σ_k g_tk(x) · E_k(x)

专家 E_k 五个任务共用，门控 g_t 每个任务一套。门控决定「这个任务要多听哪几个专家」——
共享与专属的权衡由它学出来，而不是人手设计。

**输入层、训练循环、指标实现全部原样复用**（features.py / trainer.py /
ranking_metrics.py 一行不改）。§27.1 的 ΔAUC 要求除共享机制外无任何差异，所以
「共享机制」必须是四个模型之间**唯一**的不同点。输入指纹会被写进结果，
verify_single_task 的 J 段比对四份是否一致。

## loss 等权，是有意的

trainer 取各任务 BCE 的**均值**。共享专家上的更新确实由稠密任务主导 —— 但依据是
**梯度范数**，不是 loss 值。实测（seed 42 的 checkpoint，8,192 行 batch，对
experts.* 求逐任务梯度的 L2 范数）：

    task          梯度范数    倍数        轮末 loss   倍数
    long_view     0.068560   35.6x       0.53113    80.1x
    is_click      0.042889   22.3x       0.57976    87.4x
    is_like       0.004734    2.5x       0.05594     8.4x
    is_comment    0.002121    1.1x       0.01762     2.7x
    is_follow     0.001924    1.0x       0.00663     1.0x

**loss 不是梯度的好代理**：loss 最大的是 is_click（87×），梯度最大的却是 long_view
（35.6×），连顺序都不一样，倍数也差得远（87× vs 22×）。早先版本用「loss 差 87 倍」直接
推出「click 主导共享层」，方向碰巧对但证据链不成立。结论应以梯度范数为准。

**这正是「稀疏任务被负迁移」的主要机制，是被研究的现象本身，不是要修掉的 bug。**
加任务权重是另一个研究问题（怎么调权重能救稀疏任务），混进来会让「共享机制的影响」
不可归因。三个 MTL 变体必须用同一套（等权）。

## 参数预算（§25.1）

    专家  K × MLP(323 -> 256 -> 128)     8 × 115,840 = 926,720
    门控  T × Linear(323 -> K)           5 ×   2,592 =  12,960
    塔    T × MLP(128 -> 64 -> 1)        5 ×   8,321 =  41,605
    dense 合计                                          981,285
    embedding                                        37,477,196（一份）

对照 Single-Task：dense 124,161 × 5 = 620,805，embedding 37,477,196 × **5**。
ST 的 embedding 是五份，所以总参数差约 5 倍 —— 这是「共享 vs 不共享」的固有形态，
§25.1 里 ST 是**参照点**，预算可比性只在 MMoE / PLE / Selective 三者之间要求。
count_params_grouped 把四段分开记，供预算表直接引用。

用法：
    python -m src.ranking.mmoe --config configs/mmoe.yaml --seed 42
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


class MMoE(nn.Module):
    def __init__(self, data: RankingData, cfg: dict, tasks: list[str]) -> None:
        super().__init__()
        self.tasks = list(tasks)
        self.encoder = RankingEncoder(data, cfg)
        # RankingEncoder 最终输出的特征维度
        d_in = self.encoder.out_dim
        # 专家数量
        self.n_experts = int(cfg["n_experts"])
        if self.n_experts < 2:
            raise ValueError(
                f"n_experts 必须 >= 2，收到 {self.n_experts}。只有 1 个专家时门控恒为 1，"
                "MMoE 退化成一个共享底座 + 每任务小塔，那是 Shared-Bottom 而不是 MMoE。"
            )
        # “专家网络每层有多少个神经元”
        eh = list(cfg["expert_hidden"])
        drop = float(cfg.get("dropout", 0.0))
        # 专家：五个任务共用这一组
        self.experts = nn.ModuleList(
            [mlp(d_in, eh, dropout=drop) for _ in range(self.n_experts)]
        )
        # 门控：每个任务一套，softmax over 专家
        self.gates = nn.ModuleList(
            [nn.Linear(d_in, self.n_experts) for _ in self.tasks]
        )
        # 塔：每个任务一套
        self.towers = nn.ModuleList(
            # 每个tower128 → 64 → 1
            [
                mlp(eh[-1], list(cfg["tower_hidden"]), out_dim=1, dropout=drop)
                for _ in self.tasks
            ]
        )

    def gate_weights(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """每个任务对各专家的权重，(B, K)，非负且按行和为 1。验算与门控分析都用它。"""
        return {
            t: torch.softmax(self.gates[i](x), dim=-1) for i, t in enumerate(self.tasks)
        }

    # MMoE最核心的完整前向传播
    def forward(self, b: dict) -> dict[str, torch.Tensor]:
        # 把batch编码成统一特征
        x = self.encoder(b)
        # (B, K, d_expert)：专家只前向一次，被所有任务共用 —— 这就是「共享」的实处
        e = torch.stack([E(x) for E in self.experts], dim=1)
        # 等价于 expert_outputs = []， 每个专家都是 323 → 256 → 128
        # for E in self.experts:
        #     expert_output = E(x)
        #     expert_outputs.append(expert_output)
        #  stack再把每个专家拼接
        out = {}
        for i, t in enumerate(self.tasks):
            g = torch.softmax(self.gates[i](x), dim=-1)  # (B, K)
            h = torch.einsum("bk,bkd->bd", g, e)  # h_t = Σ_k g_tk · E_k
            # 以一条样本为例：
            # h =
            # 0.10 × 专家1的128维输出
            # + 0.20 × 专家2的128维输出
            # + ...
            # + 0.08 × 专家8的128维输出
            out[t] = self.towers[i](h).squeeze(-1)  # 删除最后那个大小为1的维度
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mmoe.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tasks", default=None, help="逗号分隔的简称子集，覆盖 config")
    ap.add_argument(
        "--eval-test",
        action="store_true",
        help="在 test 上评估。默认不评 —— 开发期反复看 test 会把它变成第二个 "
        "validation，而 test 只能用一次。架构与超参冻结后再统一开。",
    )
    # 下面三个是烟测/诊断专用。带上任意一个都会改 tag，避免覆盖正式结果。
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
    if a.epochs:
        cfg["epochs"] = a.epochs

    smoke = any(v is not None for v in (a.epochs, a.limit_train, a.limit_eval))
    tag = f"mmoe_{a.protocol}_seed{a.seed}" + ("_smoke" if smoke else "")
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
    #     tr：训练数据
    #     va：验证数据
    #     te：测试数据
    if a.limit_train or a.limit_eval:
        tr = Head(tr, a.limit_train)
        va = Head(va, a.limit_eval)
        te = Head(te, a.limit_eval) if te is not None else None

    # 一个模型带五个头 —— 没有 Single-Task 那个五次循环
    model = MMoE(tr, cfg, tasks)
    fp = model.encoder.fingerprint()
    # 调用公共训练器
    res = train(
        model, tuple(tasks), tr, va, te, cfg, a.seed, tag, fp, eval_test=a.eval_test
    )
    res.update(
        {
            "model": "mmoe",
            "protocol": a.protocol,
            "smoke": smoke,
            "n_experts": model.n_experts,
            "expert_hidden": list(cfg["expert_hidden"]),
            "tower_hidden": list(cfg["tower_hidden"]),
            # §25.1 的 parameter budget 要分段报
            "params_grouped": count_params_grouped(model),
        }
    )
    save_checkpoint(
        model,
        {
            "model": "mmoe",
            "tasks": tasks,
            "seed": a.seed,
            "protocol": a.protocol,
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
