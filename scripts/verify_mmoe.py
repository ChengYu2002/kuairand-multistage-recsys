"""独立验算 MMoE（src/ranking/mmoe.py）。

MMoE 与 Single-Task 的差值就是 §27.1 的 ΔAUC。差值只有在「除共享机制外一切相同」时才
可归因，而这里大部分失败模式**不会报错**：门控塌缩成只用一个专家、五个任务的门控其实
一模一样（退化成 Shared-Bottom）、某个任务的塔意外影响别的任务 —— 都照样收敛、照样出
AUC，只是结论变成假的。

八层检查：
  A. 门控是合法的概率分布：非负、按行和为 1。
  A2. **训练后**的 checkpoint 没有专家塌缩。A 段查的是随机初始化的模型 —— 而随机初始化的
     softmax 必然接近均匀，那条检查在初始化时永远会过，没有验证力。真正会出问题的是
     训练之后：若某个任务的门控塌缩到单个专家，MMoE 就退化成「那一个专家 + 一个塔」，
     而它照样收敛、照样出 AUC。所以必须对落盘的 checkpoint 查。
  B. 五个任务的门控**真的不一样**。若相同，MMoE 退化成一个共享底座 + 五个小塔，
     那是 Shared-Bottom，不是 MMoE —— 而 ΔAUC 会被当成「共享专家的效果」报出去。
  C. 共享与隔离都成立（用扰动实测，不看结构）：
       改一个专家的权重      -> 五个任务的 logit **全部**变化（专家确实共享）
       改任务 A 的塔         -> 只有 A 变（头部隔离）
       改任务 A 的门控       -> 只有 A 变
  D. 与 Single-Task 输入完全可比：两个模型的输入指纹必须逐字相同。
  E. 退化防护：n_experts < 2 必须报错（只有 1 个专家时门控恒为 1）。
  F. 学习能力（可证伪）：小批量过拟合，五个任务的训练集 AUC 都要逼近 1。
  G. loss 是各任务 BCE 的**均值**：独立重算比对。量级对比读的是 Single-Task **实际训练
     的轮末 loss**（results/*.json），不是探针模型的临时值 —— 刚在 512 行上过拟合过的
     模型在别的 batch 上的 loss 没有意义，拿它算「差多少倍」会因为错误的原因通过。
  H. 确定性与参数分段：同 seed 两次训练逐位相同；分段计数独立重算。

用法：
    python scripts/verify_mmoe.py
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.ranking_metrics import auc
from src.ranking.dataset import RankingData, RankItemStatic
from src.ranking.features import count_params_grouped
from src.ranking.mmoe import MMoE
from src.ranking.single_task import SingleTaskDNN
from src.ranking.trainer import resolve_tasks
from src.utils.config import load_config, project_path, require
from src.utils.seed import set_seed


def _fit(model, data, rows, tasks, steps, lr=3e-3, seed=0):
    """在给定行上跑 steps 步（loss = 各任务 BCE 均值，与 trainer 同口径）。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    bs = min(512, len(rows))
    first = last = None
    for i in range(steps):
        b = data.batch(rows[rng.integers(0, len(rows), bs)])
        logits = model(b)
        loss = torch.stack([F.binary_cross_entropy_with_logits(
            logits[t], b[f"label_{t}"]) for t in tasks]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.detach().item()
        last = loss.detach().item()
    return first, last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mmoe.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = load_config(a.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))

    failures: list[str] = []
    checks = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  {'OK  ' if ok else 'FAIL'} {msg}")
        if not ok:
            failures.append(msg)

    ic = require(cfg, "input")
    st = RankItemStatic(proc, a.protocol)
    data = RankingData(proc, a.protocol, "train", st,
                       max_hist=int(ic["max_hist"]),
                       mask_oov=bool(ic["mask_oov_in_history"]),
                       pair_features=bool(ic["pair_features"]))
    tasks = resolve_tasks(list(require(cfg, "tasks")),
                          require(cfg, "data", "labels", "tasks"))
    set_seed(a.seed)
    m = MMoE(data, cfg, tasks)
    rows = np.arange(4096)
    b = data.batch(rows)
    m.eval()

    # ---------- A. 门控是合法的概率分布 ----------
    print("\n=== A. 门控是合法概率分布 ===")
    with torch.no_grad():
        x = m.encoder(b)
        g = m.gate_weights(x)
    K = m.n_experts
    check(all(v.shape == (len(rows), K) for v in g.values()),
          f"每个任务的门控形状 (B, K) = ({len(rows)}, {K})")
    check(all(float(v.min()) >= 0 for v in g.values()), "全部非负")
    check(all(abs(float(v.sum(-1).mean()) - 1.0) < 1e-5
              and float((v.sum(-1) - 1.0).abs().max()) < 1e-5 for v in g.values()),
          "每行和为 1（softmax 之后）")
    # 初始化时的塌缩检测只是形式：随机 softmax 必然接近均匀，真正的检查在 A2 段
    maxw = {t: float(v.max(-1).values.mean()) for t, v in g.items()}
    check(max(maxw.values()) < 0.9,
          f"初始化时未塌缩（最大平均权重 {max(maxw.values()):.4f}，均匀为 {1 / K:.4f}）"
          " —— 这条几乎必然通过，有验证力的是 A2 段")

    # ---------- A2. 训练后的 checkpoint ----------
    print("\n=== A2. 训练后 checkpoint 的门控（真正有验证力的一条）===")
    ckpts = sorted((ROOT / "experiments").glob(f"mmoe_{a.protocol}_seed*.pt"))
    ckpts = [c for c in ckpts if "smoke" not in c.name]
    if not ckpts:
        check(True, "experiments/ 下暂无 MMoE checkpoint（跑过之后这条会开始比对）")
    unif = float(np.log(K))
    for c in ckpts:
        mm = MMoE(data, cfg, tasks)
        mm.load_state_dict(torch.load(c, map_location="cpu",
                                      weights_only=False)["state_dict"])
        mm.eval()
        with torch.no_grad():
            gg = mm.gate_weights(mm.encoder(b))
        ent = {t: float((-(w * torch.log(w + 1e-12)).sum(-1)).mean()) / unif
               for t, w in gg.items()}
        mx = {t: float(w.max(-1).values.mean()) for t, w in gg.items()}
        dmin = min(float((gg[t1] - gg[t2]).abs().mean())
                   for i, t1 in enumerate(tasks) for t2 in tasks[i + 1:])
        check(max(mx.values()) < 0.7 and min(ent.values()) > 0.5 and dmin > 1e-3,
              f"{c.name}：最大平均权重 {max(mx.values()):.3f}（<0.7）、"
              f"最小归一化熵 {min(ent.values()):.3f}（>0.5）、"
              f"任务间最小差异 {dmin:.4f}（>1e-3）")
        del mm

    # ---------- B. 五个任务的门控真的不同 ----------
    print("\n=== B. 各任务门控互不相同 ===")
    pairs = [(t1, t2) for i, t1 in enumerate(tasks) for t2 in tasks[i + 1:]]
    diffs = {(t1, t2): float((g[t1] - g[t2]).abs().mean()) for t1, t2 in pairs}
    check(min(diffs.values()) > 1e-4,
          f"任意两个任务的门控平均绝对差 >= {min(diffs.values()):.5f} > 1e-4 —— "
          "若相同则退化成 Shared-Bottom，而 ΔAUC 会被误报成共享专家的效果")
    # 同一任务对不同样本也应有差异（门控是输入的函数，不是常数）
    check(all(float(v.std(0).mean()) > 1e-4 for v in g.values()),
          f"门控随样本变化（每列标准差均值最小 "
          f"{min(float(v.std(0).mean()) for v in g.values()):.5f}）—— "
          "否则它是个常数偏置，不是「门控」")

    # ---------- C. 共享与隔离（扰动实测） ----------
    print("\n=== C. 专家共享、头部隔离（扰动实测）===")
    with torch.no_grad():
        base = {t: m(b)[t].clone() for t in tasks}

    def changed(after: dict) -> set[str]:
        return {t for t in tasks if not torch.equal(after[t], base[t])}

    with torch.no_grad():
        saved = m.experts[0][0].weight.clone()
        m.experts[0][0].weight.add_(1.0)
        got = changed({t: m(b)[t] for t in tasks})
        m.experts[0][0].weight.copy_(saved)
    check(got == set(tasks),
          f"改 1 号专家的权重 -> 全部 {len(tasks)} 个任务都变（受影响：{len(got)} 个）"
          " —— 专家确实被共享")

    t0 = tasks[0]
    with torch.no_grad():
        saved = m.towers[0][0].weight.clone()
        m.towers[0][0].weight.add_(1.0)
        got = changed({t: m(b)[t] for t in tasks})
        m.towers[0][0].weight.copy_(saved)
    check(got == {t0}, f"改 {t0} 的塔 -> 只有它变（受影响：{sorted(got)}）")

    with torch.no_grad():
        saved = m.gates[0].weight.clone()
        m.gates[0].weight.add_(1.0)
        got = changed({t: m(b)[t] for t in tasks})
        m.gates[0].weight.copy_(saved)
    check(got == {t0}, f"改 {t0} 的门控 -> 只有它变（受影响：{sorted(got)}）")
    check(len({tuple(base[t][:8].tolist()) for t in tasks}) == len(tasks),
          f"一次前向得到 {len(tasks)} 个**互不相同**的 logit")

    # ---------- D. 与 Single-Task 输入完全可比 ----------
    print("\n=== D. 与 Single-Task 的输入指纹一致 ===")
    st_cfg = load_config("configs/single_task.yaml")
    set_seed(a.seed)
    st_model = SingleTaskDNN(data, st_cfg, tasks[0])
    fp_mm, fp_st = m.encoder.fingerprint(), st_model.encoder.fingerprint()
    check(fp_mm["sha"] == fp_st["sha"],
          f"两个模型的输入指纹相同：{fp_mm['sha']}")
    check(fp_mm["out_dim"] == fp_st["out_dim"] == m.encoder.out_dim,
          f"输入维度相同：{fp_mm['out_dim']}")
    for k in ("epochs", "batch_size", "lr", "weight_decay", "dropout", "eval_every_steps"):
        check(cfg.get(k) == st_cfg.get(k), f"训练协议 {k} 相同：{cfg.get(k)!r}")

    # ---------- E. 退化防护 ----------
    print("\n=== E. 退化防护 ===")
    bad = copy.deepcopy(cfg)
    bad["n_experts"] = 1
    try:
        MMoE(data, bad, tasks)
        check(False, "n_experts=1：未被拦下")
    except ValueError:
        check(True, "n_experts=1：已拦下（门控恒为 1，那是 Shared-Bottom 不是 MMoE）")

    # ---------- F. 学习能力 ----------
    print("\n=== F. 小批量过拟合（可证伪）===")
    tiny = np.arange(512)
    set_seed(a.seed)
    m2 = MMoE(data, cfg, tasks)
    f0, f1 = _fit(m2, data, tiny, tasks, steps=300)
    check(f1 < f0 * 0.5, f"512 行 300 步：loss {f0:.4f} -> {f1:.4f}（降幅过半）")
    m2.eval()
    with torch.no_grad():
        lo = m2(data.batch(tiny))
    aucs = {t: auc(torch.sigmoid(lo[t]).numpy(),
                   data.labels[t][tiny].astype(np.int64)) for t in tasks}
    ok = {t: (np.isnan(v) or v > 0.95) for t, v in aucs.items()}
    check(all(ok.values()),
          "五个任务的训练集 AUC 都 > 0.95 或无定义（正样本为 0）："
          + "  ".join(f"{t} {aucs[t]:.4f}" for t in tasks))

    # ---------- G. loss 是各任务均值 ----------
    print("\n=== G. loss 口径与各任务量级 ===")
    with torch.no_grad():
        lo = m2(b)
        per = {t: float(F.binary_cross_entropy_with_logits(lo[t], b[f"label_{t}"]))
               for t in tasks}
        want = float(np.mean(list(per.values())))
        got_loss = float(torch.stack([F.binary_cross_entropy_with_logits(
            lo[t], b[f"label_{t}"]) for t in tasks]).mean())
    check(abs(want - got_loss) < 1e-6,
          f"总 loss == 各任务 BCE 的算术均值（{got_loss:.6f}）")

    # 量级对比必须用**实际训练**的轮末 loss。探针模型（刚在 512 行上过拟合 300 步）
    # 在别的 batch 上的 loss 是无意义的大数，用它算倍数会因为错误的原因通过。
    import json as _json
    real: dict[str, float] = {}
    for t in tasks:
        f = ROOT / "results" / f"single_task_{a.protocol}_seed{a.seed}__{t}.json"
        if f.is_file():
            real[t] = float(_json.loads(f.read_text("utf-8"))["history"][-1]["train_loss"])
    if len(real) == len(tasks):
        lo_min = min(real.values())
        print("    Single-Task 实际训练的轮末 BCE（等权取均值 -> 大 loss 的任务主导底座）：")
        for t in tasks:
            print(f"      {t:<12} {real[t]:.5f}   ({real[t] / lo_min:>5.1f}x)")
        check(max(real.values()) / lo_min > 5,
              f"量级差 {max(real.values()) / lo_min:.0f} 倍 —— 这是稀疏任务被负迁移的"
              "主要通道，必须在 README 披露；等权是有意选择")
    else:
        check(True, f"跳过量级对比：results/ 下只有 {len(real)}/{len(tasks)} 个 "
                    "Single-Task 逐任务结果（跑过之后这条会开始比对）")

    # ---------- H. 确定性与参数分段 ----------
    print("\n=== H. 确定性与参数分段 ===")
    outs = []
    for sd in (a.seed, a.seed, a.seed + 1):
        set_seed(sd)
        outs.append(_fit(MMoE(data, cfg, tasks), data, tiny, tasks, steps=20)[1])
    check(outs[0] == outs[1] and outs[0] != outs[2],
          f"同 seed 两次末 loss 逐位相同、换 seed 不同（{outs[0]!r}）")

    grp = count_params_grouped(m)
    d_in = m.encoder.out_dim
    eh = list(cfg["expert_hidden"])
    th = list(cfg["tower_hidden"])
    w = d_in
    exp = 0
    for h in eh:
        exp += w * h + h
        w = h
    exp *= m.n_experts
    gate = (d_in * m.n_experts + m.n_experts) * len(tasks)
    w = eh[-1]
    tw = 0
    for h in th:
        tw += w * h + h
        w = h
    tw = (tw + w + 1) * len(tasks)
    check(grp["expert"] == exp and grp["gate"] == gate and grp["tower"] == tw,
          f"分段计数独立重算一致：专家 {exp:,} / 门控 {gate:,} / 塔 {tw:,}")
    check(grp["other"] == 0,
          f"没有参数落进 other 段（{grp['other']:,}）—— 否则预算表会漏统计")
    check(grp["embedding"] + grp["expert"] + grp["gate"] + grp["tower"] == grp["total"],
          f"五段之和 == 总数 {grp['total']:,}")

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("MMoE 验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("验算通过（门控分布 / 任务间差异 / 专家共享与头部隔离 / 输入可比 / "
          "退化防护 / 学习能力 / loss 口径 / 确定性）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
