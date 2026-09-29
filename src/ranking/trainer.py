"""排序侧公共训练循环（plan §26/§27）。Single-Task / MMoE / PLE / Selective 共用。

## 固定轮数，不挑 checkpoint

四个模型一律训固定轮数、报**最后一轮**，不按 valid AUC 挑最佳 checkpoint。

理由是公平性，不是省事。Single-Task 是 5 个独立模型，可以各自挑自己 valid AUC 最好的
那一刻停下来 —— **挑 5 次**；MMoE 是一个模型带 5 个头，只能挑一个时刻，五个任务共享
这一次选择。让 ST 挑 5 次而 MTL 挑 1 次，ΔAUC 会系统性偏向 ST，凭空造出负迁移。
而多任务模型也没有天然唯一的"最佳指标"：按平均 AUC 挑会牺牲 follow / comment，
按某个稀疏任务挑又偏向那个任务。固定轮数让四个模型用同一条规则。

每轮仍然记录 train loss 与逐任务 valid AUC，只是**不据此选择**。若曲线显示轮末
仍未收敛或已经崩掉，要统一改所有模型的轮数，不能只调某一个。

## 曲线按**步**记，不是按轮

config 的 eval_every_steps 决定记录密度。理由是这次的问题只有轮内评估才看得见：
按轮看，wd=0 的 is_click 是"第 1 轮 0.7287 然后下滑"；按 250 步看才发现峰值在
step 250（第 0.11 轮），后面整轮是平的 —— 两种读法会导出完全不同的处理。
epochs=1 时按轮记只有一个点，等于没有曲线。

## 训练预算是一次性在 valid 上定的

"固定轮数报最后一轮"只在**轮末恰好落在平台上**时才既公平又不浪费信号。最初的
epochs=5 / weight_decay=0 不满足这一点：valid AUC 从第 1 轮起单调下滑，轮末比峰值低
0.06 AUC。而 Single-Task 是 §27.1 里 ΔAUC 的**分母** —— 分母被压低会让四个 MTL 模型
集体显得有正迁移，这是方向性的偏差，不是噪声。

所以预算（轮数 + weight_decay）经过一次诊断后选定，全部在 **valid** 上做、test 一次未碰，
四个模型共用同一套值。取值与完整实测依据写在 configs/*.yaml 的 epochs / weight_decay
注释里（那里是唯一来源，避免两处漂移）。要点：过拟合在历史 embedding，不在目标 ID；
只有 weight decay 有效，dropout / id_dropout / 降 lr 都无效。

## 数据顺序与模型无关

每轮的行序由 (seed, epoch) 唯一决定，与模型、与任务无关 —— 所以 Single-Task 的第 3 个
任务和 MMoE 看到的是同一个顺序。否则"同样 5 轮"里还藏着一个变量。

## 不用 pos_weight

follow 训练正样本只有 4,923 条（0.11%），但这里仍用朴素 BCE。加 pos_weight 会把预测
概率整体抬高，PCOC 与校准曲线就失去意义 —— 而 §27 要靠它们抓"共享让稀疏任务失准
而非失序"。稀疏任务学不动本身是要报告的结果，不是要掩盖的问题。

## 模型接口

    model(batch) -> {task_name: logit(B,)}     logit，不是概率

Single-Task 返回 1 个键，MMoE / PLE / Selective 返回 5 个。loss 是各任务 BCE 的**均值**
（ST 只有一个任务，等同单任务 BCE）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from src.evaluation.ranking_metrics import evaluate_ranking
from src.ranking.dataset import RankingData
from src.ranking.features import count_params
from src.utils.config import project_path
from src.utils.logger import get_logger

log = get_logger(__name__)

EVAL_BATCH = 8192


def resolve_tasks(short_names: list[str], label_cols: list[str]) -> list[str]:
    """简称 -> 真实列名。歧义或缺失一律报错，不猜。

    config 里写的是 [click, long_view, like, follow, comment]，实际列名是
    [is_click, long_view, is_like, is_comment, is_follow]。猜错列名会训练出一个预测
    别的任务的模型，而它照样收敛、照样出 AUC。四个模型共用这一个映射。
    """
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


class Head:
    """烟测用：把一个 RankingData 截成前 n 行。正式跑不会用到。"""

    def __init__(self, data, n: int | None) -> None:
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


def epoch_order(n: int, seed: int, epoch: int) -> np.ndarray:
    """第 epoch 轮的行序。只依赖 (seed, epoch, n)，与模型和任务无关。"""
    return np.random.default_rng([seed, epoch, n]).permutation(n)


@torch.no_grad()
def predict(model: nn.Module, data: RankingData, tasks: tuple[str, ...]) -> dict[str, np.ndarray]:
    """全量预测，返回**概率**（evaluate_ranking 要求 sigmoid 之后的值）。"""
    model.eval()
    out = {t: np.empty(len(data), np.float32) for t in tasks}
    for s in range(0, len(data), EVAL_BATCH):
        rows = np.arange(s, min(s + EVAL_BATCH, len(data)))
        logits = model(data.batch(rows))
        for t in tasks:
            out[t][rows] = torch.sigmoid(logits[t]).numpy()
    model.train()
    return out


def evaluate_split(model: nn.Module, data: RankingData, tasks: tuple[str, ...]) -> dict:
    """逐任务过统一的 evaluate_ranking。calibration 是 polars 表，转成行以便进 JSON。"""
    pred = predict(model, data, tasks)
    res = {}
    for t in tasks:
        r = evaluate_ranking(pred[t], data.labels[t], data.user_id)
        cal = r.pop("calibration")
        r["calibration"] = cal.to_dicts() if isinstance(cal, pl.DataFrame) else cal
        res[t] = r
    return res


def train(model: nn.Module, tasks: tuple[str, ...], tr: RankingData, va: RankingData,
          te: RankingData | None, cfg: dict, seed: int, tag: str,
          fingerprint: dict, eval_test: bool = False) -> dict:
    """训练固定轮数并返回结果。不做 checkpoint 选择，报最后一轮。

    eval_test 默认 **False**：开发 MMoE / PLE 时若每次都看到 test 指标，会不知不觉
    把 test 当第二个 validation 用，而 test 只能用一次。架构与超参全部冻结后，
    再显式 --eval-test 统一开一次。
    """
    epochs = int(cfg["epochs"])
    bs = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 0.0))
    eval_every = int(cfg.get("eval_every_steps", 0))
    if cfg.get("optimizer", "adam") != "adam":
        raise NotImplementedError(f"optimizer 目前只支持 adam，收到 {cfg.get('optimizer')!r}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    params = count_params(model)
    steps = -(-len(tr) // bs)
    log.info("%s | 任务 %s | %d 轮 x %s 批（bs=%d）| 参数 embedding %s + dense %s",
             tag, ",".join(tasks), epochs, f"{steps:,}", bs,
             f"{params['embedding']:,}", f"{params['dense']:,}")

    history: list[dict] = []
    last_vres: dict = {}
    t0 = time.perf_counter()
    gstep = 0
    for ep in range(1, epochs + 1):
        model.train()
        order = epoch_order(len(tr), seed, ep)
        losses = []
        since_eval = []
        for i in range(steps):
            rows = order[i * bs : (i + 1) * bs]
            if len(rows) == 0:
                # 当前 steps = ceil(n/bs) 的口径下不会发生（实测 2,196 批、最小 946 行、
                # 合计恰好 n）。但空 batch 的 BCE 是 NaN，而它**不会崩** —— 只会让报出去的
                # train_loss 变成 NaN，而 valid 曲线看着完全正常。
                raise AssertionError(
                    f"第 {i} 批为空（bs={bs}, len(order)={len(order)}）—— "
                    "行序与步数的口径不一致，空 batch 会静默污染 train_loss")
            b = tr.batch(rows)
            logits = model(b)
            loss = torch.stack(
                [F.binary_cross_entropy_with_logits(logits[t], b[f"label_{t}"]) for t in tasks]
            ).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.detach().item())
            since_eval.append(losses[-1])
            gstep += 1
            # 轮内按步记曲线（只记录，不据此挑 checkpoint）。最后一步一定评，
            # 所以 epochs=1 也不会只剩一个点。
            last_of_epoch = i + 1 == steps
            if last_of_epoch or (eval_every > 0 and gstep % eval_every == 0):
                vres = last_vres = evaluate_split(model, va, tasks)
                history.append({
                    "epoch": ep, "step": gstep,
                    # 全局轮位置（第 2 轮末 = 2.0），不是轮内比例。名字写清楚：
                    # 叫 epoch_frac 时第一次读自己的曲线就会误读成 0~1。
                    "epoch_pos": round(gstep / steps, 4),
                    "train_loss": float(np.mean(since_eval)),
                    "valid": {t: {k: vres[t][k] for k in ("auc", "gauc", "pcoc",
                                                          "gauc_users", "gauc_skipped")}
                              for t in tasks},
                })
                log.info("  ep%d %5d/%d  loss %.5f | %s  (%.0fs)", ep, i + 1, steps,
                         float(np.mean(since_eval)),
                         "  ".join(f"{t} AUC {vres[t]['auc']:.5f}" for t in tasks),
                         time.perf_counter() - t0)
                since_eval = []

    out = {
        "tag": tag, "seed": seed, "tasks": list(tasks),
        "epochs": epochs, "batch_size": bs, "lr": lr, "weight_decay": wd,
        "dropout": float(cfg.get("dropout", 0.0)), "eval_every_steps": eval_every,
        "steps_per_epoch": steps,
        "checkpoint_rule": "fixed_epochs_last",   # 不挑最佳轮：理由见模块 docstring
        "params": params,
        "input_fingerprint": fingerprint,
        "train_seconds": time.perf_counter() - t0,
        "history": history,
        # history 里只留指标子集（曲线用）；这里存最后一轮的**完整**结果，
        # 含 pos_rate / n / 校准分桶，供报告表使用
        "valid": last_vres,
    }
    out["eval_test"] = bool(eval_test)
    if eval_test:
        if te is None:
            raise ValueError("eval_test=True 但没有传 test 数据")
        out["test"] = evaluate_split(model, te, tasks)
        log.info("%s | test | %s", tag,
                 "  ".join(f"{t} AUC {out['test'][t]['auc']:.5f}" for t in tasks))
    return out


def write_results(res: dict, name: str) -> Path:
    dst = project_path("results") / f"{name}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("结果已写出 %s", dst.name)
    return dst


def save_checkpoint(model: nn.Module, meta: dict, name: str) -> Path:
    dst = project_path("experiments") / f"{name}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, dst)
    log.info("checkpoint 已保存 %s", dst.name)
    return dst


def format_table(res: dict, split: str = "test") -> str:
    """一行一个任务：AUC / GAUC（含参与用户数）/ PCOC。"""
    header = (f"  {'task':<12}{'AUC':>9}{'GAUC':>9}{'PCOC':>8}"
              f"{'正样本率':>11}{'GAUC用户':>10}")
    lines = [f"{res['tag']}  seed {res['seed']}  {split}", header]
    for t in res["tasks"]:
        r = res[split][t]
        lines.append(f"  {t:<12}{r['auc']:>9.5f}{r['gauc']:>9.5f}{r['pcoc']:>8.3f}"
                     f"{r['pos_rate']:>10.4%}{r['gauc_users']:>7}/"
                     f"{r['gauc_users'] + r['gauc_skipped']:<4}")
    return "\n".join(lines)
