"""CGC / PLE：共享专家 + 任务专属专家（plan §24；Tang et al., RecSys'20）。

## 与 MMoE 的结构差别：门控的**选择集**

    MMoE:  任务 t 的门控在**全部** K 个专家上做 softmax —— 专家不分工，
           所有任务共用同一个池子。
    CGC:   专家分成「共享组」+「每任务专属组」；任务 t 的门控只在
           「共享专家 ∪ 任务 t 的专属专家」上做 softmax，**看不到别的任务的专属专家**。

**但这不是与 MMoE 之间唯一变化的量。** 为了让总参数预算可比（§25.1），本模型的
expert_hidden 是 [128, 96]，而 MMoE 是 [256, 128]。所以准确的表述是：

    相同：输入（同一份数据、同一个 323 维编码器、同一个输入指纹）、训练协议
          （轮数 / batch / lr / weight_decay / dropout / eval_every_steps）、
          总参数量（CGC 38.27M、MMoE 38.46M、PLE 38.63M，相差 < 0.5%）
    不同：共享结构（选择集）**以及**专家宽度与表示维度

这是个无法两全的取舍：每层专家数 14（CGC）/ 28（PLE 两层）对 MMoE 的 8，若保持同样的
专家宽度，参数量就会差 1.7~2.65 倍。**不能声称「共享机制是唯一自变量」** —— 报告时要
把 dense 参数与训练时间一并列出，让读者看到容量分配的差异。

顺带：总参数接近不等于**计算量**接近。MMoE 是 8 个宽专家，PLE 两层合计 28 个窄专家，
FLOPs 与访存模式都不同，所以训练时间必须实测报告，不能由参数量推断。

PLE 论文的主张是：这种显式分离能减少任务间的有害参数干扰（negative transfer）。
本项目在 MMoE 上已经观测到干净的负迁移证据（3 seeds 配对差值，is_like −0.00400±0.00060、
is_comment −0.00716±0.00110、is_follow −0.01180±0.00543，三个 seed 全负），CGC/PLE 正是
对这个主张的直接检验。

## n_levels：一个文件覆盖两个模型

    n_levels = 1  ->  CGC（论文 Figure 4，Table 1 里单独作为一个模型报）
    n_levels >= 2 ->  PLE（论文 Figure 5，progressive separation）

多层时每层多一个**共享门控**：它在该层**全部**专家（含所有任务的专属专家）上做 softmax，
产出的向量喂给下一层的共享专家。这就是「渐进分离」——低层还在联合抽取，高层才逐步分开。

论文 Eq. 6 的一个细节容易漏：**高层门控的权重由上一层该任务的输出算出，不是由原始输入
算出**（`g^{k,j}(x) = w^{k,j}(g^{k,j-1}(x)) · S^{k,j}(x)`）。这里照此实现。

## 一个结构性后果，验算会钉住它

CGC（单层）下，任务 A 的专属专家**只影响任务 A**。
PLE（多层）下不成立：任务 A 的专属专家会经由该层的共享门控进入下一层的共享专家，
从而影响所有任务。这不是缺陷，正是「progressive separation」的定义 —— 但它意味着
「专属专家 = 完全隔离」的直觉只在 CGC 上成立，报告时不能混着说。

## 参数预算（§25.1）

每层专家数 = n_shared + n_task × T = 4 + 2×5 = **14**，而 MMoE 是 8。若沿用 MMoE 的
expert_hidden=[256,128]，PLE 会有 2.65 倍的 dense 参数，那样「PLE 赢了」说不清是机制还是
参数。因此 expert_hidden 压到 [128, 96]，实测 CGC 0.81x / PLE 1.17x MMoE。
取值依据写在 configs/ple.yaml 的注释里。

## 其余一律不变

编码器、训练循环、指标、汇总脚本原样复用。§27.1 的 ΔAUC 要求除共享机制外无任何差异，
所以「共享机制」必须是四个模型之间唯一的不同点；输入指纹会被写进结果，
verify_single_task 的 J 段比对各模型是否一致。

用法：
    python -m src.ranking.ple --config configs/ple.yaml --seed 42            # PLE
    python -m src.ranking.ple --config configs/ple.yaml --seed 42 --levels 1 # CGC
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


class PLE(nn.Module):
    def __init__(self, data: RankingData, cfg: dict, tasks: list[str]) -> None:
        super().__init__()
        self.tasks = list(tasks)
        T = len(self.tasks)
        self.encoder = RankingEncoder(data, cfg)
        # 有几层“专家＋门控”
        self.n_levels = int(cfg["n_levels"])
        # 多少个共享
        self.n_shared = int(cfg["n_shared_experts"])
        self.n_task = int(cfg["n_task_experts"])
        if self.n_levels < 1:
            raise ValueError(f"n_levels 必须 >= 1（1=CGC），收到 {self.n_levels}")
        if self.n_shared < 1 or self.n_task < 1:
            raise ValueError(
                f"n_shared_experts / n_task_experts 都必须 >= 1，"
                f"收到 {self.n_shared} / {self.n_task}。任一为 0 时就不再是"
                "「共享 + 专属」的结构：n_task=0 退化成 MMoE，n_shared=0 退化成"
                "五个互不相干的单任务模型。"
            )
        # 专家的隐藏结构
        eh = list(cfg["expert_hidden"])
        drop = float(cfg.get("dropout", 0.0))

        # 命名要能被 features.count_params_grouped 归类：
        # *_experts. -> expert 段，*gates. -> gate 段，towers. -> tower 段
        self.shared_experts = nn.ModuleList()  # [level][i]
        self.task_experts = nn.ModuleList()  # [level][task][i]
        self.gates = nn.ModuleList()  # [level][task]
        self.shared_gates = nn.ModuleList()  # [level]，只有非最后一层需要
        d = self.encoder.out_dim
        for lv in range(self.n_levels):
            self.shared_experts.append(
                nn.ModuleList([mlp(d, eh, dropout=drop) for _ in range(self.n_shared)])
            )
            self.task_experts.append(
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [mlp(d, eh, dropout=drop) for _ in range(self.n_task)]
                        )
                        for _ in self.tasks
                    ]
                )
            )
            # 任务门控的选择集 = 自己的专属 + 共享（**不含别的任务的专属**）
            self.gates.append(
                nn.ModuleList(
                    [nn.Linear(d, self.n_task + self.n_shared) for _ in self.tasks]
                )
            )
            if lv < self.n_levels - 1:
                # 共享门控的选择集 = 该层**全部**专家
                self.shared_gates.append(nn.Linear(d, self.n_shared + self.n_task * T))
            # 这一层的输出维度是下一层的输入维度
            d = eh[-1]
        self.towers = nn.ModuleList(
            [
                mlp(d, list(cfg["tower_hidden"]), out_dim=1, dropout=drop)
                for _ in self.tasks
            ]
        )

    # forward 会 for循环逐层调用
    def _level(
        self, lv: int, cur: dict[str, torch.Tensor], cur_shared: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """跑一层，返回（各任务的新表示，共享模块的新表示，各任务门控权重）。"""
        # 计算共享专家
        sh = [E(cur_shared) for E in self.shared_experts[lv]]
        # 计算每个任务的专属专家
        tk = {
            t: [E(cur[t]) for E in self.task_experts[lv][i]]
            for i, t in enumerate(self.tasks)
        }
        new, weights = {}, {}

        # 依次遍历每个任务
        for i, t in enumerate(self.tasks):
            # 把当前任务的专属专家输出和共享专家输出合并起来，组成该任务可以选择的专家集合
            sel = torch.stack(tk[t] + sh, dim=1)  # (B, n_task+n_shared, h)
            # 门控权重由**上一层该任务的输出**算出（论文 Eq. 6），不是原始输入
            w = torch.softmax(self.gates[lv][i](cur[t]), dim=-1)
            weights[t] = w
            new[t] = torch.einsum("bk,bkd->bd", w, sel)
        if lv < self.n_levels - 1:
            # 共享模块：在该层全部专家上加权。顺序固定为「按 tasks 顺序的专属 + 共享」
            all_exp = torch.stack([e for t in self.tasks for e in tk[t]] + sh, dim=1)
            ws = torch.softmax(self.shared_gates[lv](cur_shared), dim=-1)
            cur_shared = torch.einsum("bk,bkd->bd", ws, all_exp)
        return new, cur_shared, weights

    def gate_weights(self, b: dict) -> dict[str, torch.Tensor]:
        """**最后一层**各任务的门控权重，(B, n_task+n_shared)。"""
        return {
            k[2]: v
            for k, v in self.all_gate_weights(b).items()
            if k[0] == "task" and k[1] == self.n_levels - 1
        }

    def all_gate_weights(self, b: dict) -> dict[tuple, torch.Tensor]:
        """**所有层**的门控权重：{("task", level, task): w, ("shared", level): w}。

        只看最后一层不够：PLE 相对 CGC 的全部增量在**共享门控**上，若它失效，多层就等于
        一个更深的 CGC，而任务门控看上去可能完全正常。诊断必须覆盖每一层。
        """
        x = self.encoder(b)
        cur = dict.fromkeys(self.tasks, x)
        cur_shared = x
        out: dict[tuple, torch.Tensor] = {}
        for lv in range(self.n_levels):
            sh = [E(cur_shared) for E in self.shared_experts[lv]]
            tk = {
                t: [E(cur[t]) for E in self.task_experts[lv][i]]
                for i, t in enumerate(self.tasks)
            }
            new = {}
            for i, t in enumerate(self.tasks):
                w = torch.softmax(self.gates[lv][i](cur[t]), dim=-1)
                out[("task", lv, t)] = w
                new[t] = torch.einsum("bk,bkd->bd", w, torch.stack(tk[t] + sh, dim=1))
            if lv < self.n_levels - 1:
                ws = torch.softmax(self.shared_gates[lv](cur_shared), dim=-1)
                out[("shared", lv)] = ws
                cur_shared = torch.einsum(
                    "bk,bkd->bd",
                    ws,
                    torch.stack([e for t in self.tasks for e in tk[t]] + sh, dim=1),
                )
            cur = new
        return out

    def forward(self, b: dict) -> dict[str, torch.Tensor]:
        x = self.encoder(b)
        # 所有任务从相同输入开始
        cur = dict.fromkeys(self.tasks, x)
        cur_shared = x

        for lv in range(self.n_levels):
            cur, cur_shared, _ = self._level(lv, cur, cur_shared)
        return {t: self.towers[i](cur[t]).squeeze(-1) for i, t in enumerate(self.tasks)}


# gate_weights()和forward()都会把数据跑完整个多层PLE，区别在最后想拿到什么：
# gate_weights()返回最后一层门控权重；
# forward()返回五个任务的最终预测logit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ple.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--seed", type=int, default=42)
    # choices 限死 1/2：产物名只有 cgc / ple 两种，而 n_levels>=2 一律叫 ple ——
    # --levels 3 会静默覆盖 2 层 PLE 的正式结果。另外 `if a.levels` 会漏掉 0，
    # 用 is not None 才能让 --levels 0 走到模型里的报错。
    ap.add_argument(
        "--levels",
        type=int,
        default=None,
        choices=(1, 2),
        help="覆盖 n_levels：1=CGC，2=PLE。产物名随之变，两者不会互相覆盖",
    )
    ap.add_argument("--tasks", default=None, help="逗号分隔的简称子集，覆盖 config")
    ap.add_argument(
        "--eval-test",
        action="store_true",
        help="在 test 上评估。默认不评 —— 开发期反复看 test 会把它变成第二个 "
        "validation，而 test 只能用一次。",
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
    if a.levels is not None:
        cfg["n_levels"] = a.levels
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

    # n_levels=1 是 CGC，论文里是独立的一个模型，产物名必须区分，否则两组结果互相覆盖
    # 1层是CGC，2层是PLE
    # 这里很重要很重要很重要
    name = "cgc" if int(cfg["n_levels"]) == 1 else "ple"
    smoke = any(v is not None for v in (a.epochs, a.limit_train, a.limit_eval))
    tag = f"{name}_{a.protocol}_seed{a.seed}" + ("_smoke" if smoke else "")
    log.info(
        "%s（n_levels=%d）| 任务映射 %s -> %s%s",
        name.upper(),
        cfg["n_levels"],
        shorts,
        tasks,
        "（烟测/诊断）" if smoke else "",
    )

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

    model = PLE(tr, cfg, tasks)
    fp = model.encoder.fingerprint()
    res = train(
        model, tuple(tasks), tr, va, te, cfg, a.seed, tag, fp, eval_test=a.eval_test
    )
    res.update(
        {
            "model": name,
            "protocol": a.protocol,
            "smoke": smoke,
            "n_levels": int(cfg["n_levels"]),
            "n_shared_experts": model.n_shared,
            "n_task_experts": model.n_task,
            "expert_hidden": list(cfg["expert_hidden"]),
            "tower_hidden": list(cfg["tower_hidden"]),
            "params_grouped": count_params_grouped(model),
        }
    )
    save_checkpoint(
        model,
        {
            "model": name,
            "tasks": tasks,
            "seed": a.seed,
            "protocol": a.protocol,
            "n_levels": int(cfg["n_levels"]),
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
