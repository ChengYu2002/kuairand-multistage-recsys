"""独立验算 CGC / PLE（src/ranking/ple.py）。

CGC/PLE 与 MMoE 的差别只在**门控的选择集**：任务 t 只能看见「共享专家 ∪ 自己的专属专家」。
这一处写错不会报错 —— 若门控误在全部专家上做 softmax，模型就变回 MMoE，而它照样收敛、
照样出 AUC，对比表看上去完全正常，结论却变成「MMoE vs MMoE」。

九层检查：
  A. 门控是合法概率分布，且**选择集宽度正确**（= n_task + n_shared，不是全部专家数）。
  B. **任务 t 的门控看不到别的任务的专属专家**（扰动实测，不看结构）：
       改任务 B 的专属专家 -> 单层(CGC) 下任务 A 的 logit **必须逐位不变**。
     这是 CGC 相对 MMoE 的全部实质差别，也是最容易写错的一处。
  C. 多层的结构性后果：PLE(n_levels>=2) 下 B 的专属专家**会**影响 A —— 经由该层的共享
     门控进入下一层的共享专家。这不是缺陷，正是 progressive separation 的定义。
     若 PLE 下也完全隔离，说明共享门控没接上（多层退化成多个独立的 CGC）。
  D. 论文 Eq. 6：高层门控的权重由**上一层该任务的输出**算出，不是原始输入。
     扰动上一层的输出应改变高层门控权重。
  E. 退化防护：n_task_experts=0 会退化成 MMoE，n_shared_experts=0 会退化成五个独立模型，
     n_levels=0 无意义 —— 三者都必须报错。
  F. 与 Single-Task / MMoE 输入完全可比：指纹逐字相同，训练协议逐键相同。
  G. 学习能力（可证伪）：小批量过拟合，五个任务的训练集 AUC 都要逼近 1。
  H. 参数分段独立重算；other 段必须为 0（否则 §25.1 的预算表会漏统计）。
  I. 确定性：同 seed 两次训练逐位相同。
  J. 训练后 checkpoint 的**每一层**门控诊断。判据区分两类情况：

     **结构性错误（判失败）**
       - 门控不随输入变化（逐样本 std ≈ 0）-> 它退化成了常数偏置，不是门控
       - 共享门控失效 -> PLE 相对 CGC 的全部增量就在它上面，它没了多层就等于更深的 CGC
       - 各任务门控雷同 -> 退化成 Shared-Bottom

     **学习结果（只记录，不判失败）**
       - 某个任务在某一层集中到少数专家。实测 PLE 第二层有 1~2 个任务的归一化熵低到
         0.07~0.29，但**逐样本 std 仍在 0.009~0.063**，即仍随输入变化 —— 那是学出来的
         专门化，不是坏掉。早先版本用「最大权重 > 0.8 即失败」一刀切，会把正常的专门化
         判成故障。

用法：
    python scripts/verify_ple.py              # 默认按 config 的 n_levels
    python scripts/verify_ple.py --levels 1   # 只查 CGC
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
from src.ranking.ple import PLE
from src.ranking.trainer import resolve_tasks
from src.utils.config import load_config, project_path, require
from src.utils.seed import set_seed


def _fit(model, data, rows, tasks, steps, lr=3e-3, seed=0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    bs = min(512, len(rows))
    first = last = None
    for i in range(steps):
        b = data.batch(rows[rng.integers(0, len(rows), bs)])
        logits = model(b)
        loss = torch.stack(
            [
                F.binary_cross_entropy_with_logits(logits[t], b[f"label_{t}"])
                for t in tasks
            ]
        ).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.detach().item()
        last = loss.detach().item()
    return first, last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ple.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--levels", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.levels:
        cfg["n_levels"] = a.levels
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
    data = RankingData(
        proc,
        a.protocol,
        "train",
        st,
        max_hist=int(ic["max_hist"]),
        mask_oov=bool(ic["mask_oov_in_history"]),
        pair_features=bool(ic["pair_features"]),
    )
    tasks = resolve_tasks(
        list(require(cfg, "tasks")), require(cfg, "data", "labels", "tasks")
    )
    rows = np.arange(4096)
    b = data.batch(rows)

    def build(levels: int):
        c = copy.deepcopy(cfg)
        c["n_levels"] = levels
        set_seed(a.seed)
        m = PLE(data, c, tasks)
        m.eval()
        return m, c

    # ---------- A. 门控分布与选择集宽度 ----------
    print("\n=== A. 门控分布与选择集宽度 ===")
    m, _ = build(int(cfg["n_levels"]))
    with torch.no_grad():
        g = m.gate_weights(b)
    want = m.n_task + m.n_shared
    n_all = m.n_shared + m.n_task * len(tasks)
    check(
        all(v.shape == (len(rows), want) for v in g.values()),
        f"任务门控宽度 = n_task + n_shared = {want}（**不是**全部专家数 {n_all}）"
        " —— 若等于后者说明退化成了 MMoE",
    )
    check(
        all(float(v.min()) >= 0 for v in g.values())
        and all(float((v.sum(-1) - 1).abs().max()) < 1e-5 for v in g.values()),
        "非负且按行和为 1",
    )
    pair = min(
        float((g[t1] - g[t2]).abs().mean())
        for i, t1 in enumerate(tasks)
        for t2 in tasks[i + 1 :]
    )
    check(pair > 1e-4, f"任意两任务的门控不同（最小平均绝对差 {pair:.5f}）")

    # ---------- B/C. 专属专家的隔离性：CGC 严格，PLE 不严格 ----------
    print("\n=== B/C. 专属专家的隔离性（扰动实测）===")
    for levels, strict in ((1, True), (2, False)):
        mm, _ = build(levels)
        with torch.no_grad():
            base = {t: mm(b)[t].clone() for t in tasks}
            saved = mm.task_experts[0][1][0][0].weight.clone()  # 任务 1 的专属专家
            mm.task_experts[0][1][0][0].weight.add_(1.0)
            after = {t: mm(b)[t] for t in tasks}
            mm.task_experts[0][1][0][0].weight.copy_(saved)
        changed = {t for t in tasks if not torch.equal(after[t], base[t])}
        name = "CGC" if levels == 1 else "PLE"
        if strict:
            check(
                changed == {tasks[1]},
                f"{name}(n_levels=1)：改任务 B 的专属专家 -> 只有 B 变"
                f"（受影响 {sorted(changed)}）—— 这是 CGC 相对 MMoE 的全部实质差别",
            )
        else:
            check(
                changed == set(tasks),
                f"{name}(n_levels=2)：改任务 B 的专属专家 -> **全部**任务都变"
                f"（受影响 {len(changed)} 个）—— 经由共享门控传到下一层，"
                "这是 progressive separation 的定义，不是缺陷",
            )
        # 共享专家在任何层数下都影响全部任务
        with torch.no_grad():
            saved = mm.shared_experts[0][0][0].weight.clone()
            mm.shared_experts[0][0][0].weight.add_(1.0)
            ch2 = {t for t in tasks if not torch.equal(mm(b)[t], base[t])}
            mm.shared_experts[0][0][0].weight.copy_(saved)
        check(ch2 == set(tasks), f"{name}：改共享专家 -> 全部任务都变（{len(ch2)} 个）")
        del mm

    # ---------- D. 高层门控的输入是上一层的输出（论文 Eq. 6）----------
    print("\n=== D. 高层门控由上一层输出驱动（论文 Eq. 6）===")
    m2, _ = build(2)
    with torch.no_grad():
        g_before = {t: v.clone() for t, v in m2.gate_weights(b).items()}
        # 改第 0 层任务 0 的专属专家 -> 第 1 层任务 0 的门控权重应随之改变
        saved = m2.task_experts[0][0][0][0].weight.clone()
        m2.task_experts[0][0][0][0].weight.add_(1.0)
        g_after = m2.gate_weights(b)
        moved = float((g_after[tasks[0]] - g_before[tasks[0]]).abs().mean())
        m2.task_experts[0][0][0][0].weight.copy_(saved)
    check(
        moved > 1e-6,
        f"改第 0 层的专属专家后，第 1 层该任务的门控权重变化 {moved:.6f} > 0"
        " —— 若为 0 说明高层门控用的是原始输入，与论文 Eq. 6 不符",
    )
    del m2

    # ---------- E. 退化防护 ----------
    print("\n=== E. 退化防护 ===")
    for key, val, why in (
        ("n_task_experts", 0, "退化成 MMoE"),
        ("n_shared_experts", 0, "退化成五个独立单任务模型"),
        ("n_levels", 0, "无意义"),
    ):
        bad = copy.deepcopy(cfg)
        bad[key] = val
        try:
            PLE(data, bad, tasks)
            check(False, f"{key}={val}（{why}）：未被拦下")
        except ValueError:
            check(True, f"{key}={val}（{why}）：已拦下")

    # ---------- F. 与 ST / MMoE 输入完全可比 ----------
    print("\n=== F. 与 Single-Task / MMoE 可比 ===")
    st_cfg = load_config("configs/single_task.yaml")
    mm_cfg = load_config("configs/mmoe.yaml")
    fp = m.encoder.fingerprint()
    set_seed(a.seed)
    from src.ranking.single_task import SingleTaskDNN

    fp_st = SingleTaskDNN(data, st_cfg, tasks[0]).encoder.fingerprint()
    check(fp["sha"] == fp_st["sha"], f"输入指纹与 Single-Task 相同：{fp['sha']}")
    for k in (
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "dropout",
        "eval_every_steps",
    ):
        check(
            cfg.get(k) == st_cfg.get(k) == mm_cfg.get(k),
            f"训练协议 {k} 与 ST / MMoE 三方相同：{cfg.get(k)!r}",
        )

    # ---------- G. 学习能力 ----------
    print("\n=== G. 小批量过拟合（可证伪）===")
    tiny = np.arange(512)
    set_seed(a.seed)
    m3 = PLE(data, cfg, tasks)
    f0, f1 = _fit(m3, data, tiny, tasks, steps=300)
    check(f1 < f0 * 0.5, f"512 行 300 步：loss {f0:.4f} -> {f1:.4f}（降幅过半）")
    m3.eval()
    with torch.no_grad():
        lo = m3(data.batch(tiny))
    aucs = {
        t: auc(torch.sigmoid(lo[t]).numpy(), data.labels[t][tiny].astype(np.int64))
        for t in tasks
    }
    check(
        all(np.isnan(v) or v > 0.95 for v in aucs.values()),
        "五个任务训练集 AUC 都 > 0.95 或无定义："
        + "  ".join(f"{t} {aucs[t]:.4f}" for t in tasks),
    )

    # ---------- H. 参数分段 ----------
    print("\n=== H. 参数分段（§25.1 预算表要用）===")
    for levels in (1, 2):
        mm, c = build(levels)
        grp = count_params_grouped(mm)
        d_in = mm.encoder.out_dim
        eh = list(c["expert_hidden"])
        T = len(tasks)

        def mlp_n(i, h):
            n = 0
            for x in h:
                n += i * x + x
                i = x
            return n

        exp = gate = 0
        d = d_in
        for lv in range(levels):
            exp += (mm.n_shared + mm.n_task * T) * mlp_n(d, eh)
            gate += T * (d * (mm.n_task + mm.n_shared) + mm.n_task + mm.n_shared)
            if lv < levels - 1:
                gate += d * (mm.n_shared + mm.n_task * T) + mm.n_shared + mm.n_task * T
            d = eh[-1]
        th = list(c["tower_hidden"])
        tw = T * (mlp_n(d, th) + th[-1] * 1 + 1)  # mlp(..., out_dim=1) 多一个线性头
        check(
            grp["expert"] == exp and grp["gate"] == gate and grp["tower"] == tw,
            f"n_levels={levels}：专家 {exp:,}、门控 {gate:,}、塔 {tw:,} 与独立重算一致",
        )
        check(
            grp["other"] == 0,
            f"n_levels={levels}：other 段为 0（{grp['other']:,}）—— 否则预算表会漏统计",
        )
        del mm

    # ---------- I. 确定性与训练后门控 ----------
    print("\n=== I. 确定性与训练后门控 ===")
    outs = []
    for sd in (a.seed, a.seed, a.seed + 1):
        set_seed(sd)
        outs.append(_fit(PLE(data, cfg, tasks), data, tiny, tasks, steps=20)[1])
    check(
        outs[0] == outs[1] and outs[0] != outs[2],
        f"同 seed 两次末 loss 逐位相同、换 seed 不同（{outs[0]!r}）",
    )

    # ---------- J. 训练后每一层门控 ----------
    print("\n=== J. 训练后 checkpoint 的逐层门控 ===")
    ckpts = [
        c
        for c in sorted((ROOT / "experiments").glob("*_main_seed*.pt"))
        if c.name.split("_")[0] in ("cgc", "ple") and "smoke" not in c.name
    ]
    if not ckpts:
        check(True, "experiments/ 下暂无 CGC/PLE checkpoint（跑过之后这条会开始比对）")
    for c in ckpts:
        sd = torch.load(c, map_location="cpu", weights_only=False)
        lv_n = int(sd.get("n_levels", cfg["n_levels"]))
        mm, _ = build(lv_n)
        mm.load_state_dict(sd["state_dict"])
        mm.eval()
        with torch.no_grad():
            gw = mm.all_gate_weights(b)

        def stat(w):
            k = w.shape[1]
            ent = float((-(w * torch.log(w + 1e-12)).sum(-1)).mean()) / float(np.log(k))
            return float(w.max(-1).values.mean()), ent, float(w.std(0).mean())

        # 结构性错误 1：任何门控退化成常数偏置（不随输入变化）
        flat = {kk: stat(v) for kk, v in gw.items()}
        worst = min(flat.items(), key=lambda kv: kv[1][2])
        check(
            worst[1][2] > 1e-3,
            f"{c.name}：所有 {len(flat)} 个门控都随输入变化"
            f"（最小逐样本 std {worst[1][2]:.5f}，在 {worst[0]}）—— "
            "若为 0 则它是常数偏置，不是门控",
        )

        # 结构性错误 2：共享门控失效（PLE 相对 CGC 的增量全在它上面）
        sh = {kk: v for kk, v in flat.items() if kk[0] == "shared"}
        if sh:
            k_sh = min(v[1] for v in sh.values())
            check(
                k_sh > 0.4,
                f"{c.name}：共享门控归一化熵 {k_sh:.3f} > 0.4 —— "
                "它失效则多层退化成更深的 CGC",
            )

        # 结构性错误 3：同一层各任务门控雷同
        for lv in range(lv_n):
            ws = [gw[("task", lv, t)] for t in tasks]
            dmin = min(
                float((ws[i] - ws[j]).abs().mean())
                for i in range(len(ws))
                for j in range(i + 1, len(ws))
            )
            check(
                dmin > 1e-4,
                f"{c.name} 第{lv}层：各任务门控互不相同（最小平均绝对差 {dmin:.5f}）",
            )

        # 学习结果：只打印，不判失败
        for lv in range(lv_n):
            line = "  ".join(
                f"{t.replace('is_', '')[:4]}:{flat[('task', lv, t)][0]:.2f}/"
                f"{flat[('task', lv, t)][1]:.2f}"
                for t in tasks
            )
            print(f"       {c.stem} 第{lv}层任务门控 max/熵  {line}")
        for kk, v in sh.items():
            print(f"       {c.stem} 第{kk[1]}层共享门控 max {v[0]:.3f} 熵 {v[1]:.3f}")
        del mm

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("CGC / PLE 验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        "验算通过（门控选择集 / CGC 严格隔离 / PLE 渐进分离 / Eq.6 / 退化防护 / "
        "输入可比 / 学习能力 / 参数分段 / 确定性 / 逐层门控）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
