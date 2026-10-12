"""独立验算 Selective Sharing（src/ranking/selective_sharing.py）。

这类模型最危险的错误不是崩溃，而是**静默退化**：

- 把 sigmoid 写成 softmax，会把逐维开关误写成“96 维争夺总和为 1 的概率”；
- 五个任务误用同一个 gate，会退化成普通 shared-bottom；
- 私有分支接线错误，会让任务 A 的梯度更新任务 B；
- gate 虽然存在但没乘到 shared 表示上，代码能训练，所谓 selective 却是假的；
- 参数预算只写在 YAML 注释里，模型扩容后结果会被误解成结构收益。

本脚本用输出扰动和反向梯度实测连接关系，不靠阅读 ModuleList 猜：

  A. 表示与门控形状正确；sigmoid 值逐维在 (0,1)，不要求行和为 1。
  B. gate 随任务和样本变化，五个任务没有共享同一套选择。
  C. 扰动 shared -> 全任务变化；扰动 private/gate/tower -> 只影响所属任务。
  D. 只用任务 A 的 loss 反传：shared 与 A 的私有路径有梯度，其他任务私有路径无梯度。
  E. 强制 gate≈0 / gate≈1 后，共享表示确实被屏蔽 / 放行。
  F. 退化防护：单任务、shared_dim=0、task_specific_dim=0 都必须报错。
  G. 输入指纹与训练协议和 ST/MMoE/PLE 一致。
  H. 参数分段独立重算；预算参照与正式 MMoE config 一致且在 ±20%。
  I. 小批量可过拟合；同 seed 确定性成立。
  J. 已训练 checkpoint 的 gate 仍随样本变化；饱和度只记录，不武断判坏。

用法：
    python scripts/verify_selective_sharing.py
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
from src.ranking.selective_sharing import SelectiveSharing
from src.ranking.single_task import SingleTaskDNN
from src.ranking.trainer import resolve_tasks
from src.utils.config import load_config, project_path, require
from src.utils.seed import set_seed


def _fit(model, data, rows, tasks, steps, lr=3e-3, seed=0):
    """与公共 trainer 相同：各任务朴素 BCE 的算术均值。"""
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


def _mlp_count(in_dim: int, hidden: list[int], out_dim: int | None = None) -> int:
    total = 0
    d = in_dim
    for h in hidden:
        total += d * h + h
        d = h
    if out_dim is not None:
        total += d * out_dim + out_dim
    return total


def _has_grad(module: torch.nn.Module) -> bool:
    return any(
        p.grad is not None and float(p.grad.detach().abs().sum()) > 0
        for p in module.parameters()
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/selective.yaml")
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
    set_seed(a.seed)
    m = SelectiveSharing(data, cfg, tasks)
    m.eval()

    # ---------- A. 表示与门控形状 ----------
    print("\n=== A. 表示与门控形状 ===")
    with torch.no_grad():
        x = m.encoder(b)
        shared, private, gates = m.representations(x)
        logits = m(b)
    check(
        shared.shape == (len(rows), m.shared_dim),
        f"共享表示形状 = ({len(rows)}, {m.shared_dim})",
    )
    check(
        all(v.shape == (len(rows), m.task_specific_dim) for v in private.values()),
        f"每个私有表示形状 = ({len(rows)}, {m.task_specific_dim})",
    )
    check(
        float(shared.min()) >= 0 and all(float(v.min()) >= 0 for v in private.values()),
        "共享/私有投影后均经过 ReLU（与 MMoE/PLE 专家输出口径一致）",
    )
    check(
        all(v.shape == (len(rows), m.shared_dim) for v in gates.values()),
        f"每个 gate 形状 = ({len(rows)}, {m.shared_dim})",
    )
    check(
        all(float(v.min()) > 0 and float(v.max()) < 1 for v in gates.values()),
        "sigmoid gate 全部严格位于 (0,1)",
    )
    mean_sum = float(torch.stack([v.sum(-1).mean() for v in gates.values()]).mean())
    check(
        abs(mean_sum - 1.0) > 1.0,
        f"gate 不是误写的 softmax（平均行和 {mean_sum:.2f}，不要求等于 1）",
    )
    check(
        all(
            v.shape == (len(rows),) and torch.isfinite(v).all() for v in logits.values()
        ),
        f"一次前向得到 {len(tasks)} 个有限 logit，形状均为 (B,)",
    )

    # ---------- B. 选择真的因任务、样本而异 ----------
    print("\n=== B. gate 因任务、样本而异 ===")
    pairs = [(a, z) for i, a in enumerate(tasks) for z in tasks[i + 1 :]]
    task_diffs = {(a, z): float((gates[a] - gates[z]).abs().mean()) for a, z in pairs}
    sample_std = {t: float(gates[t].std(0).mean()) for t in tasks}
    check(
        min(task_diffs.values()) > 1e-4,
        f"任意两任务 gate 平均绝对差至少 {min(task_diffs.values()):.5f}",
    )
    check(
        min(sample_std.values()) > 1e-4,
        f"gate 随样本变化（逐维 std 均值最小 {min(sample_std.values()):.5f}）",
    )

    # ---------- C. 输出扰动验证共享与隔离 ----------
    print("\n=== C. shared 共享，private/gate/tower 隔离（扰动实测）===")
    with torch.no_grad():
        base = {t: m(b)[t].clone() for t in tasks}

    def changed(after: dict[str, torch.Tensor]) -> set[str]:
        return {t for t in tasks if not torch.equal(after[t], base[t])}

    with torch.no_grad():
        saved = m.shared_experts[0][0].weight.clone()
        m.shared_experts[0][0].weight.add_(0.5)
        got = changed(m(b))
        m.shared_experts[0][0].weight.copy_(saved)
    check(got == set(tasks), f"改共享分支 -> 全任务变化（实际 {sorted(got)}）")

    t0 = tasks[0]
    with torch.no_grad():
        saved = m.task_experts[0][0].weight.clone()
        m.task_experts[0][0].weight.add_(0.5)
        got = changed(m(b))
        m.task_experts[0][0].weight.copy_(saved)
    check(got == {t0}, f"改 {t0} 私有分支 -> 只有它变化（实际 {sorted(got)}）")

    with torch.no_grad():
        saved = m.gates[0].bias.clone()
        m.gates[0].bias.add_(1.0)
        got = changed(m(b))
        m.gates[0].bias.copy_(saved)
    check(got == {t0}, f"改 {t0} gate -> 只有它变化（实际 {sorted(got)}）")

    with torch.no_grad():
        saved = m.towers[0][0].weight.clone()
        m.towers[0][0].weight.add_(0.5)
        got = changed(m(b))
        m.towers[0][0].weight.copy_(saved)
    check(got == {t0}, f"改 {t0} tower -> 只有它变化（实际 {sorted(got)}）")

    # ---------- D. 反向梯度验证 ----------
    print("\n=== D. 单任务 loss 的梯度边界 ===")
    m.zero_grad(set_to_none=True)
    lo = m(data.batch(np.arange(512)))
    lo[t0].sum().backward()
    check(_has_grad(m.shared_experts[0]), f"{t0} loss 会更新共享分支")
    check(_has_grad(m.task_experts[0]), f"{t0} loss 会更新自己的私有分支")
    check(_has_grad(m.gates[0]), f"{t0} loss 会更新自己的 gate")
    check(_has_grad(m.towers[0]), f"{t0} loss 会更新自己的 tower")
    check(
        all(not _has_grad(m.task_experts[i]) for i in range(1, len(tasks))),
        f"{t0} loss 不更新其他任务的私有分支",
    )
    check(
        all(not _has_grad(m.gates[i]) for i in range(1, len(tasks))),
        f"{t0} loss 不更新其他任务的 gate",
    )
    check(
        all(not _has_grad(m.towers[i]) for i in range(1, len(tasks))),
        f"{t0} loss 不更新其他任务的 tower",
    )
    m.zero_grad(set_to_none=True)

    # ---------- E. gate 确实屏蔽/放行共享表示 ----------
    print("\n=== E. gate 的语义可证伪 ===")
    xb = x[:128]
    with torch.no_grad():
        weight = m.gates[0].weight.clone()
        bias = m.gates[0].bias.clone()
        m.gates[0].weight.zero_()
        m.gates[0].bias.fill_(-30.0)
        g0 = m.gate_weights(xb)[t0]
        m.gates[0].bias.fill_(30.0)
        g1 = m.gate_weights(xb)[t0]
        m.gates[0].weight.copy_(weight)
        m.gates[0].bias.copy_(bias)
    check(
        float(g0.max()) < 1e-10, f"bias=-30 时共享保留率≈0（max {float(g0.max()):.2e}）"
    )
    check(
        float(g1.min()) > 1 - 1e-6,
        f"bias=+30 时共享保留率≈1（min {float(g1.min()):.6f}）",
    )
    with torch.no_grad():
        s = m.shared_experts[0](xb)
    check(
        float((g0 * s).abs().max()) < 1e-8 and torch.allclose(g1 * s, s, atol=1e-6),
        "gate≈0 屏蔽共享表示，gate≈1 原样放行（不是只计算了权重却没接入）",
    )

    # ---------- F. 退化防护 ----------
    print("\n=== F. 退化防护 ===")
    for key, value, label in (
        ("shared_dim", 0, "shared_dim=0 删除共享路径"),
        ("task_specific_dim", 0, "task_specific_dim=0 删除私有路径"),
    ):
        bad = copy.deepcopy(cfg)
        bad[key] = value
        try:
            SelectiveSharing(data, bad, tasks)
            check(False, f"{label}：未被拦下")
        except ValueError:
            check(True, f"{label}：已拦下")
    try:
        SelectiveSharing(data, cfg, tasks[:1])
        check(False, "单任务退化：未被拦下")
    except ValueError:
        check(True, "单任务退化：已拦下（不存在共享关系）")
    subset = SelectiveSharing(data, cfg, tasks[:2])
    check(
        subset.budget_ratio is not None and not subset.budget_enforced,
        "--tasks 子集诊断可构造；记录预算比例但不套用正式五任务阈值",
    )
    del subset
    bad = copy.deepcopy(cfg)
    bad["sharing_mode"] = "full_shared"
    try:
        SelectiveSharing(data, bad, tasks)
        check(False, "sharing_mode 与实现不符：未被拦下")
    except ValueError:
        check(True, "sharing_mode 与实现不符：已拦下（配置不会成为装饰）")

    # ---------- G. 输入与训练协议可比 ----------
    print("\n=== G. 与 ST/MMoE/PLE 的输入和训练协议一致 ===")
    protocol_keys = (
        "epochs",
        "batch_size",
        "lr",
        "optimizer",
        "weight_decay",
        "dropout",
        "eval_every_steps",
        "seeds",
        "tasks",
    )
    fps = {"selective": m.encoder.fingerprint()}
    for path in ("configs/single_task.yaml", "configs/mmoe.yaml", "configs/ple.yaml"):
        other = load_config(path)
        set_seed(a.seed)
        probe = SingleTaskDNN(data, other, tasks[0])
        fps[Path(path).stem] = probe.encoder.fingerprint()
        diff = {
            k: (cfg.get(k), other.get(k))
            for k in protocol_keys
            if cfg.get(k) != other.get(k)
        }
        check(not diff, f"{Path(path).name} 训练协议一致（差异 {diff}）")
        del probe
    check(
        len({v["sha"] for v in fps.values()}) == 1,
        f"四份输入指纹逐字相同：{ {k: v['sha'] for k, v in fps.items()} }",
    )

    # ---------- H. 参数分段与预算 ----------
    print("\n=== H. 参数分段与 MMoE 预算 ===")
    grp = count_params_grouped(m)
    d_in = m.encoder.out_dim
    hidden = [int(v) for v in cfg["expert_hidden"]]
    tower_hidden = [int(v) for v in cfg["tower_hidden"]]
    n_tasks = len(tasks)
    expert_want = _mlp_count(d_in, hidden, m.shared_dim) + n_tasks * _mlp_count(
        d_in, hidden, m.task_specific_dim
    )
    gate_want = n_tasks * (d_in * m.shared_dim + m.shared_dim)
    tower_want = n_tasks * _mlp_count(
        m.shared_dim + m.task_specific_dim, tower_hidden, 1
    )
    check(
        grp["expert"] == expert_want
        and grp["gate"] == gate_want
        and grp["tower"] == tower_want,
        f"独立重算一致：分支 {expert_want:,} / gate {gate_want:,} / tower {tower_want:,}",
    )
    check(grp["other"] == 0, f"other 段为 0（实际 {grp['other']:,}）")
    check(
        grp["embedding"] + grp["expert"] + grp["gate"] + grp["tower"] == grp["total"],
        f"参数分段之和 == 总数 {grp['total']:,}",
    )
    ref_path = str(require(cfg, "budget_reference_config"))
    mm_cfg = load_config(ref_path)
    check(
        ref_path == "configs/mmoe.yaml",
        f"预算直接读取正式 MMoE 配置：{ref_path}（没有手抄结构）",
    )
    mm_expert = int(mm_cfg["n_experts"]) * _mlp_count(
        d_in, list(mm_cfg["expert_hidden"])
    )
    mm_gate = n_tasks * (d_in * int(mm_cfg["n_experts"]) + int(mm_cfg["n_experts"]))
    mm_tower = n_tasks * _mlp_count(
        int(mm_cfg["expert_hidden"][-1]), list(mm_cfg["tower_hidden"]), 1
    )
    mm_dense = mm_expert + mm_gate + mm_tower
    ratio = (expert_want + gate_want + tower_want) / mm_dense
    check(
        abs(mm_dense - int(m.reference_dense_params or -1)) == 0,
        f"MMoE 参照 dense 独立重算一致：{mm_dense:,}",
    )
    check(
        0.8 <= ratio <= 1.2,
        f"Selective / MMoE dense = {ratio:.3f}（{expert_want + gate_want + tower_want:,} / {mm_dense:,}），在 ±20% 内",
    )

    # ---------- I. 学习能力与确定性 ----------
    print("\n=== I. 小批量过拟合与确定性 ===")
    tiny = np.arange(512)
    set_seed(a.seed)
    m2 = SelectiveSharing(data, cfg, tasks)
    first, last = _fit(m2, data, tiny, tasks, steps=250)
    check(
        last < first * 0.5, f"512 行 250 步：loss {first:.4f} -> {last:.4f}（降幅过半）"
    )
    m2.eval()
    with torch.no_grad():
        lo = m2(data.batch(tiny))
    aucs = {
        t: auc(torch.sigmoid(lo[t]).numpy(), data.labels[t][tiny].astype(np.int64))
        for t in tasks
    }
    check(
        all(np.isnan(v) or v > 0.95 for v in aucs.values()),
        "五任务训练集 AUC > 0.95 或因单类无定义："
        + "  ".join(f"{t} {v:.4f}" for t, v in aucs.items()),
    )
    del m2
    losses = []
    for sd in (a.seed, a.seed, a.seed + 1):
        set_seed(sd)
        probe = SelectiveSharing(data, cfg, tasks)
        losses.append(_fit(probe, data, tiny, tasks, steps=12)[1])
        del probe
    check(
        losses[0] == losses[1] and losses[0] != losses[2],
        f"同 seed 末 loss 逐位相同，换 seed 不同（{losses}）",
    )

    # ---------- J. 训练后 checkpoint ----------
    print("\n=== J. 训练后 checkpoint 的 gate ===")
    ckpts = sorted(
        (ROOT / "experiments").glob(f"selective_sharing_{a.protocol}_seed*.pt")
    )
    ckpts = [p for p in ckpts if "smoke" not in p.name]
    if not ckpts:
        check(True, "暂无正式 Selective checkpoint（跑过后本段会自动开始诊断）")
    for path in ckpts:
        set_seed(a.seed)
        probe = SelectiveSharing(data, cfg, tasks)
        probe.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        )
        probe.eval()
        with torch.no_grad():
            gg = probe.gate_weights(probe.encoder(b))
        std_min = min(float(v.std(0).mean()) for v in gg.values())
        diff_min = min(float((gg[a] - gg[z]).abs().mean()) for a, z in pairs)
        saturation = max(
            float(((v < 0.01) | (v > 0.99)).float().mean()) for v in gg.values()
        )
        check(
            std_min > 1e-4 and diff_min > 1e-4,
            f"{path.name}：gate 随样本变化 std={std_min:.5f}，任务差异={diff_min:.5f}；"
            f"最大饱和比例={saturation:.1%}（只记录，不武断判坏）",
        )
        del probe

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("Selective Sharing 验算未通过：")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        "验算通过（门控语义 / 共享与隔离 / 梯度边界 / 输入可比 / 参数预算 / "
        "学习能力 / 确定性）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
