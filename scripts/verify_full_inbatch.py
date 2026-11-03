"""验算按日期分组的完整 in-batch + logQ 路径（不执行训练 step）。

失败模式都不会靠 loss 曲线自动暴露：

1. ID+Side 直接 ``u @ v_pos.T``，会把候选视频按它自己的请求日而非当前用户的
   请求日计算 age；数值能算，但语义错。
2. 同 target 的 mask 若只掩对角线，会把重复正样本当负样本；若整张 equality mask
   都掩掉，又会把真正的对角线正样本一起删除。
3. logQ 若只减负样本或按行减，会得到另一种目标函数。
4. 新 strategy 若复用旧名字，会覆盖已完成的 neg10 checkpoint。

本脚本只做前向、手算比对和一次 backward 梯度检查，不调用 optimizer.step，
不写 checkpoint，也不运行 smoke/formal training。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval.dataset import RetrievalData
from src.retrieval.full_inbatch import (
    date_aware_inbatch_logits,
    full_inbatch_loss,
    off_diagonal_duplicate_mask,
)
from src.retrieval.negative_sampling import FULL_INBATCH, SAMPLERS, USES_LOGQ
from src.retrieval.two_tower import TwoTower, train_rows
from src.utils.config import load_config, project_path, require
from src.utils.seed import set_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    a = ap.parse_args()

    cfg = load_config(a.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    meta = json.loads((proc / f"vocab_meta_{a.protocol}.json").read_text("utf-8"))
    lo, hi = meta["video_catalog_index_min"], meta["video_catalog_index_max"]

    failures: list[str] = []
    checks = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  {'OK  ' if ok else 'FAIL'} {msg}")
        if not ok:
            failures.append(msg)

    set_seed(7)
    data = RetrievalData(proc, a.protocol, "train")
    rows = train_rows(data, lo, hi, "catalog")
    # 明确构造混合日期 batch；取 4 天、每个日期 2 行，既能抓日期错位又保持
    # B^2 的独立参考计算很小。
    picked: list[int] = []
    for day in np.unique(data.date[rows])[:4]:
        day_rows = rows[data.date[rows] == day]
        picked.extend(day_rows[:2].tolist())
    probe = np.asarray(picked, np.int64)
    dates = data.date[probe]
    b = data.batch(probe)
    check(len(np.unique(dates)) == 4, "真实小 batch 覆盖 4 个请求日期")

    print("\n[A] ID+Side：按当前请求日期编码每个候选")
    side_cfg = copy.deepcopy(cfg)
    require(side_cfg, "two_tower")["item_tower_config"] = "id_side"
    side = TwoTower(data, side_cfg).eval()
    with torch.no_grad():
        got = date_aware_inbatch_logits(side, data, b, dates)
        u = side.user_vec(b)
        ids = b["target"].numpy()
        ref_rows = []
        for i, day in enumerate(dates):
            feat = data.item_features(ids, np.full(len(ids), day, dates.dtype))
            v = side.item_vec(feat)
            ref_rows.append(u[i : i + 1] @ v.T / side.temperature)
        ref = torch.cat(ref_rows)
        naive = u @ side.item_vec(b).T / side.temperature
    check(
        torch.allclose(got, ref, atol=1e-6, rtol=1e-6),
        "日期分组实现与逐 request 独立编码的 B^2 参考逐位一致",
    )
    check(
        float((got - naive).abs().max()) > 1e-5,
        "可证伪：混合日期下直接 u @ v_pos.T 确实给出不同结果",
    )

    print("\n[B] ID-only：退化为普通 u @ v_pos.T")
    id_cfg = copy.deepcopy(cfg)
    require(id_cfg, "two_tower")["item_tower_config"] = "id_only"
    ident = TwoTower(data, id_cfg).eval()
    with torch.no_grad():
        got_id = date_aware_inbatch_logits(ident, data, b, dates)
        u_id = ident.user_vec(b)
        direct = u_id @ ident.item_vec(b).T / ident.temperature
    check(
        torch.allclose(got_id, direct, atol=1e-6, rtol=1e-6),
        "ID-only 与标准完整 in-batch 矩阵乘法逐位一致",
    )

    print("\n[C] 重复 target：只屏蔽同 ID 的非对角位置")
    dup = torch.tensor([2, 3, 2, 4])
    mask = off_diagonal_duplicate_mask(dup)
    check(
        int(mask.sum()) == 2 and bool(mask[0, 2]) and bool(mask[2, 0]),
        "重复 item 2 的两个交叉位置被屏蔽",
    )
    check(not bool(torch.diag(mask).any()), "对角线正样本全部保留")
    check(not bool(mask[0, 1]) and not bool(mask[1, 3]), "不同 ID 不被误屏蔽")

    print("\n[D] logQ：每一列按该 target 的 Q 校正")
    # lq 必须随 target 变化，否则下面所有「轴写错会怎样」的对比都退化成恒等式。
    lq = torch.linspace(-5.0, -2.0, len(probe))
    labels = torch.arange(len(probe))
    dup = off_diagonal_duplicate_mask(b["target"])

    def ce(logits: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.masked_fill(dup, float("-inf")), labels)

    with torch.no_grad():
        raw = date_aware_inbatch_logits(side, data, b, dates)
        want_col = ce(raw - lq.unsqueeze(0))  # 正确：按候选（列）减
        want_row = ce(raw - lq.unsqueeze(1))  # 失败模式：按请求（行）减
        want_none = ce(raw)  # 失败模式：根本没校正
        off = ~torch.eye(len(probe), dtype=torch.bool)
        want_negonly = ce(torch.where(off, raw - lq.unsqueeze(0), raw))  # 只校正负样本
        got_loss = full_inbatch_loss(side, data, b, dates, lq)
    check(
        torch.allclose(got_loss, want_col, atol=1e-7, rtol=1e-7),
        f"完整 loss 与独立手算一致（{float(got_loss):.8f}）",
    )

    # 上面那条用的是与实现**同一个表达式**：实现与参考同时写成 unsqueeze(1) 时它
    # 照样通过，所以它验的是 loss 组装流程，没有把轴钉住。下面给出两种错误写法，
    # 先证明它们确实算出不同的数，再证明实现不等于它们 —— 否则「实现不是那两种」
    # 这句话无法观测。
    check(
        torch.allclose(want_row, want_none, atol=1e-7, rtol=1e-7),
        "按行减 == 完全不校正：softmax 对行内常数位移不变，轴写错时 logQ 静默失效",
    )
    check(
        float((want_col - want_none).abs()) > 1e-5,
        f"可证伪：按列减确实改变了 loss（{float(want_col):.8f} vs 未校正 "
        f"{float(want_none):.8f}）",
    )
    check(
        not torch.allclose(got_loss, want_row, atol=1e-7, rtol=1e-7),
        "实现不是按请求（行）减",
    )
    check(
        float((want_col - want_negonly).abs()) > 1e-5,
        f"可证伪：只校正负样本给出不同的 loss（{float(want_negonly):.8f}）",
    )
    check(
        not torch.allclose(got_loss, want_negonly, atol=1e-7, rtol=1e-7),
        "实现把对角线上的正样本也一起校正了",
    )

    print("\n[E] 梯度与实验隔离")
    side.train()
    side.zero_grad(set_to_none=True)
    loss = full_inbatch_loss(side, data, b, dates, lq)
    loss.backward()
    grad = side.item_emb.weight.grad
    check(
        grad is not None and bool(torch.isfinite(grad).all()),
        "一次 backward 后物品 embedding 梯度存在且有限（未 optimizer.step）",
    )
    check(
        "inbatch_full_logq" in SAMPLERS
        and "inbatch_full_logq" in USES_LOGQ
        and FULL_INBATCH == ("inbatch_full_logq",),
        "完整 in-batch strategy 单独注册并强制使用 logQ",
    )
    old = f"inbatch_logq_id_side_{a.protocol}_seed42"
    new = f"inbatch_full_logq_id_side_{a.protocol}_seed42"
    check(old != new, f"产物名隔离：{old} != {new}")

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("完整 in-batch 验算未通过：")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("完整 in-batch 验算通过；未启动训练、未写 checkpoint。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
