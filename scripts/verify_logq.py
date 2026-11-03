"""独立验算 logQ 校正（src/retrieval/logq_correction.py，plan §16.5）。

logQ 的失败模式几乎都**不会报错**：

  - 符号搞反（加成减）-> 热门 item 被压得更狠，偏差加倍。loss 照样下降。
  - 只校正负样本不校正正样本 -> 正样本被相对位移一个与它自身热度相关的量，
    那是另一个偏差，而 Recall 只会"有点不对"。
  - Q 用成曝光次数或 catalog.train_freq -> 描述的是另一个 proposal 分布，
    校正的是一个本不存在的偏差。
  - Q=0 处悄悄加 epsilon -> 把口径错误变成一个静默的常数偏置。
  - 推理时也减了 log Q -> 线上打分里根本没有"采样"这回事，排序全错。

七层检查：
  A. 合成样例：手算 corrected logits 与 loss，与实现逐位比对（float32 容差 1e-6）。
     含"只给一侧 log_q 必须报错"与"两侧都不给必须与校正前逐位相同"。
  B. 符号方向可证伪：构造一个热门 item 与一个冷门 item，校正后热门的 logit
     必须**变小更多**。反了就抓住。
  C. Q 的口径：由 train_rows 独立重算，与 build_log_q 逐位一致；
     且 Q 之和为 1、与曝光次数/catalog.train_freq **不同**（否则就是用错了表）。
  D. Q 只由 train 构建：valid/test 的 target 不参与（换一份 rows 结果必须变）。
  E. Q=0 直接报错，不做 epsilon 掩盖；越界 id 也报错。
  F. 训练时每个进入 loss 的 item 都有 Q>0（in-batch 的负样本来自 batch 内正样本）。
  G. 产物隔离：inbatch_logq 与 inbatch 的 tag 不同，不会互相覆盖。

用法：
    python scripts/verify_logq.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval.dataset import RetrievalData
from src.retrieval.logq_correction import build_log_q
from src.retrieval.negative_sampling import SAMPLERS, USES_LOGQ
from src.retrieval.two_tower import TwoTower, artifact_tag, train_rows
from src.utils.config import load_config, project_path, require

# float32 前向 vs float64 手算的累加差异实测约 1e-6；真有符号错或漏校正正样本时
# loss 会差 1e-2（见 A 段"校正确实改变了 loss"那条），所以放宽四个数量级仍然抓得住。
# 之前设 1e-6 时实测余量只剩 6% —— 阈值贴着单次观测设是本项目栽过的老错。
TOL = 1e-5


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

    # ---------- A. 合成样例：手算 ----------
    print("\n[A] 合成样例：手算 corrected logits 与 loss")
    # 候选库 [2, 5]，4 个 item。训练正样本行的 target 如下（手工指定频次）：
    #   item 2 出现 5 次, 3 出现 3 次, 4 出现 2 次, 5 出现 0 次  -> 共 10 行
    tgt = np.array([2, 2, 2, 2, 2, 3, 3, 3, 4, 4], np.int64)
    tab = build_log_q(tgt, np.arange(10), 2, 5)
    want_q = np.array([0.5, 0.3, 0.2, np.nan])
    got_q = np.exp(tab.log_q.astype(np.float64))
    check(
        np.allclose(got_q[:3], want_q[:3], atol=1e-7) and np.isnan(tab.log_q[3]),
        f"Q 手算一致 [0.5, 0.3, 0.2, NaN]（实际 {np.round(got_q, 4).tolist()}）",
    )

    # 手工构造一轮 loss：2 条样本、2 个负样本
    T = 0.05
    u = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v_pos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v_neg = torch.tensor([[[0.6, 0.8], [0.8, 0.6]], [[0.6, 0.8], [0.8, 0.6]]])
    pos_ids = torch.tensor([2, 3])
    neg_ids = torch.tensor([[3, 4], [2, 4]])

    class _M:
        temperature = T

    m = _M()
    m.loss = TwoTower.loss.__get__(m, _M)

    lq_pos, lq_neg = tab.lookup(pos_ids), tab.lookup(neg_ids)
    got = float(m.loss(u, v_pos, v_neg, pos_ids, neg_ids, lq_pos, lq_neg))

    # 独立重算（不碰被测代码的任何一行）
    sp = (u * v_pos).sum(-1).numpy() / T
    sn = np.einsum("bd,bnd->bn", u.numpy(), v_neg.numpy()) / T
    lq = np.log(np.array([0.5, 0.3, 0.2]))  # item 2/3/4
    sp = sp - lq[pos_ids.numpy() - 2]
    sn = sn - lq[neg_ids.numpy() - 2]
    z = np.concatenate([sp[:, None], sn], 1)
    want = float(np.mean(-z[:, 0] + np.log(np.exp(z).sum(1))))
    check(
        abs(got - want) < TOL,
        f"校正后 loss 手算一致：实现 {got:.8f} vs 手算 {want:.8f}",
    )

    # 不给 log_q 时必须与校正前逐字相同
    plain = float(m.loss(u, v_pos, v_neg, pos_ids, neg_ids))
    sp0 = (u * v_pos).sum(-1).numpy() / T
    sn0 = np.einsum("bd,bnd->bn", u.numpy(), v_neg.numpy()) / T
    z0 = np.concatenate([sp0[:, None], sn0], 1)
    want0 = float(np.mean(-z0[:, 0] + np.log(np.exp(z0).sum(1))))
    check(
        abs(plain - want0) < TOL,
        f"不给 log_q 时 loss 与未校正手算一致：{plain:.8f} vs {want0:.8f}",
    )
    check(
        abs(plain - got) > 1e-3,
        f"校正确实改变了 loss（{plain:.5f} -> {got:.5f}）—— 否则这把尺子没量到东西",
    )

    for kw in ({"log_q_pos": lq_pos}, {"log_q_neg": lq_neg}):
        try:
            m.loss(u, v_pos, v_neg, pos_ids, neg_ids, **kw)
            ok = False
        except ValueError:
            ok = True
        check(ok, f"只给 {next(iter(kw))} 一侧时报错（只校正一侧是另一个偏差）")

    # ---------- B. 符号方向（可证伪）----------
    print("\n[B] 符号方向：热门 item 相对被压低（可证伪）")
    # 注意符号：Q < 1 所以 log Q < 0，减去它是让 logit **上升**。
    # 关键不是绝对升降，而是**谁升得少**：热门 item 的 |log Q| 小，升得少，
    # 于是相对冷门被压低 —— 这正好抵消它"被抽成负样本的次数多"。
    hot, cold = 2, 4  # Q = 0.5 vs 0.2
    d_hot = float(tab.lookup(torch.tensor([hot]))[0])
    d_cold = float(tab.lookup(torch.tensor([cold]))[0])
    check(
        d_hot > d_cold,
        f"log Q(热门 {hot}) = {d_hot:.4f} > log Q(冷门 {cold}) = {d_cold:.4f}",
    )
    check(
        (-d_hot) < (-d_cold),
        f"校正使热门 logit 上升 {-d_hot:.4f}、冷门上升 {-d_cold:.4f} —— "
        "热门升得**更少**，故相对被压低。符号写反时这条会失败",
    )

    # ---------- C/D/E. Q 的口径 ----------
    print("\n[C] 真实 Q：由 train_rows 独立重算")
    data = RetrievalData(proc, a.protocol, "train")
    scope = require(cfg, "two_tower", "train_target_scope")
    rows = train_rows(data, lo, hi, scope)
    tab2 = build_log_q(data.target, rows, lo, hi)
    cnt = np.bincount(data.target[rows] - lo, minlength=hi - lo + 1).astype(np.float64)
    q_ref = cnt / cnt.sum()
    with np.errstate(divide="ignore"):
        ref = np.where(q_ref > 0, np.log(q_ref), np.nan)
    check(
        np.allclose(
            np.nan_to_num(tab2.log_q, nan=0.0),
            np.nan_to_num(ref.astype(np.float32), nan=0.0),
            atol=1e-6,
        ),
        "log Q 与独立 bincount 重算逐位一致",
    )
    check(
        abs(float(np.exp(tab2.log_q[~np.isnan(tab2.log_q)]).sum()) - 1.0) < 1e-4,
        "Q 之和为 1",
    )
    check(tab2.n_rows == len(rows), f"Q 的分母 == train_rows 行数 {len(rows):,}")

    import polars as pl

    cat = pl.read_parquet(proc / f"catalog_{a.protocol}.parquet")
    tf = cat.get_column("train_freq").to_numpy().astype(np.float64)
    q_tf = tf / tf.sum()
    check(
        not np.allclose(q_ref, q_tf, atol=1e-6),
        "Q 与 catalog.train_freq 归一化后**不同** —— 相同就说明用错了表",
    )
    print(
        f"       （相关但不等：Spearman 约 {np.corrcoef(np.argsort(np.argsort(q_ref)), np.argsort(np.argsort(q_tf)))[0, 1]:.3f}）"
    )

    print("\n[D] Q 只由 train 构建")
    half = build_log_q(data.target, rows[: len(rows) // 2], lo, hi)
    check(
        not np.allclose(
            np.nan_to_num(half.log_q, nan=0.0),
            np.nan_to_num(tab2.log_q, nan=0.0),
            atol=1e-6,
        ),
        "换一份 rows 后 Q 改变 —— 说明它真的依赖传入的行，而非某张固定表",
    )

    print("\n[E] Q=0 与越界必须报错")
    try:
        tab.lookup(torch.tensor([5]))  # item 5 的 Q = 0
        ok = False
    except ValueError as e:
        ok = "Q=0" in str(e)
    check(ok, "Q=0 的 item 直接报错，不加 epsilon 掩盖")
    try:
        tab.lookup(torch.tensor([99]))
        ok = False
    except ValueError:
        ok = True
    check(ok, "越界 id 报错")

    print("\n[F] 训练时进入 loss 的 item 都有 Q>0")
    zero = int(np.isnan(tab2.log_q).sum())
    print(
        f"       候选库 {hi - lo + 1:,} 个 item 里 {zero:,} 个 Q=0（从未成为训练正样本）"
    )
    tr_items = np.unique(data.target[rows])
    check(
        not bool(np.isnan(tab2.log_q[tr_items - lo]).any()),
        f"全部 {len(tr_items):,} 个训练正样本 item 的 Q 都 > 0",
    )
    check(zero > 0, f"确实存在 Q=0 的 item（{zero:,} 个）—— 所以 E 段那条检查不是空跑")

    # ---------- G. 产物隔离 ----------
    print("\n[G] 产物隔离")
    check(
        "inbatch_logq" in SAMPLERS and "inbatch" in SAMPLERS,
        f"两个实验臂都注册：{sorted(SAMPLERS)}",
    )
    check(
        USES_LOGQ == ("inbatch_logq", "inbatch_full_logq"),
        f"只有两个明确命名的 logQ 实验臂走校正：{USES_LOGQ}",
    )
    check(
        SAMPLERS["inbatch_logq"] is SAMPLERS["inbatch"],
        "两者用同一个采样器类 —— 差异只在 loss，§16.6 要求除此之外全同",
    )
    check(
        SAMPLERS["inbatch_full_logq"] is SAMPLERS["inbatch"],
        "完整 in-batch 沿用同一批 target；差异由专用 BxB loss 明确实现",
    )
    tt = require(cfg, "two_tower")
    # 走 artifact_tag() 而不是在这里重抄一遍命名规则 —— 重抄的版本在规则改了之后
    # 照样通过，等于没校。命名契约本身在 verify_two_tower.py 的 H 段。
    t1, t2, t3 = (
        artifact_tag(s, tt["item_tower_config"], a.protocol, 42)
        for s in ("inbatch", "inbatch_logq", "inbatch_full_logq")
    )
    check(len({t1, t2, t3}) == 3, f"三种产物名互异：{t1} / {t2} / {t3}")
    check(
        not (project_path("experiments") / f"two_tower_{t2}.pt").exists()
        or (project_path("experiments") / f"two_tower_{t1}.pt").exists(),
        "已有的 inbatch checkpoint 未被覆盖",
    )

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("logQ 验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        "验算通过（手算比对 / 符号方向 / Q 口径 / 只用 train / Q=0 报错 / "
        "训练覆盖 / 产物隔离）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
