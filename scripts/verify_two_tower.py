"""独立验算双塔损失与负采样（阶段 3 第二段）。

损失写错不会报错，只会让曲线"看起来在降"。负采样接口若不统一，Week 3 插 exposure /
hybrid 时就得改别处代码，验收标准第 3 条当场失效。因此这里全部用**有闭式解**的构造。

检查分六层：
  A. 损失的闭式解：把正样本分数推到极高则 loss → 0。随机初始化时 loss ≈ ln(1 + n_neg)
     **只对 id_only 成立** —— id_side 下正负样本在初始状态就不可交换：实测正样本的
     duration 均值 0.162 而负样本 −0.063、has_duration 0.987 vs 0.953（被点击/长看的
     视频系统性更长）。随机投影会把这个差异变成系统性的分数差，初始 loss 因此偏离
     均匀猜（实测 2.99 > ln(11)=2.40）。这是真实信号不是 bug，但断言要分开写。
  B. accidental hit：负样本恰好等于正样本时，其 logit 被置 -inf，loss 必须**精确等于**
     少一个负样本时的 loss。并证明不屏蔽会得到不同的值 —— 否则这层处理没有内容。
  C. 采样器契约：两个采样器形状、取值域、可复现性一致，可互换。random 落在候选库区间
     且近似均匀；inbatch 全部来自本批目标。
  D. L2 归一化与 temperature：向量模长为 1；改 temperature 必须改变 loss。
  E. 梯度：PAD 行不吃梯度；参与的 item embedding 吃到梯度。
  G. 评估链路：这一段串起 物品向量 -> 用户向量 -> 打分器 -> 下标映射 -> video_id -> evaluate()，
     任何一环错位都只会得到"看起来合理"的 Recall。两条互补的检查：
       G1 随机初始化 -> Recall 必须 ≈ 随机猜（K/194310）。证明链路通，但**不证明映射对**
          —— 映射错了同样是随机结果。
       G2 把用户向量直接设成目标物品的向量 -> Top-1 必须就是那个物品。这条才验映射：
          下标 -> 候选库位置 -> video_id 任何一步错位都会当场失败。
  F. 训练样本范围与端到端确定性：train_rows 的目标全在候选库内；
     同 seed 跑两次真实训练，loss 必须逐位相同（验收标准第 5 条）。

用法：
    python scripts/verify_two_tower.py --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval.dataset import RetrievalData
from src.retrieval.inbatch_negative import InBatchNegative
from src.retrieval.random_negative import RandomNegative
from src.retrieval.two_tower import TwoTower, train_rows
from src.utils.config import load_config, project_path, require
from src.utils.seed import set_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    tt = require(cfg, "two_tower")
    n_neg = tt["negatives_per_positive"]
    meta = json.loads((proc / f"vocab_meta_{args.protocol}.json").read_text("utf-8"))
    lo, hi = meta["video_catalog_index_min"], meta["video_catalog_index_max"]

    failures: list[str] = []
    checks = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  {'OK  ' if ok else 'FAIL'} {msg}")
        if not ok:
            failures.append(msg)

    set_seed(0)
    data = RetrievalData(proc, args.protocol, "test")
    rows = train_rows(data, lo, hi, "catalog")[: args.n]
    b = data.batch(rows)
    model = TwoTower(data, cfg)
    model.eval()
    gen = torch.Generator().manual_seed(0)

    # ---------- A. 损失的闭式解 ----------
    print("\n=== A. 损失的闭式解 ===")
    with torch.no_grad():
        u, v_pos = model.user_vec(b), model.item_vec(b)
        neg_ids = RandomNegative(lo, hi).sample(b, n_neg, gen)
        neg_feat = data.item_features(neg_ids.numpy().ravel(),
                                      np.repeat(data.date[rows], n_neg))
        v_neg = model.item_vec(neg_feat).view(len(rows), n_neg, -1)
        loss0 = float(model.loss(u, v_pos, v_neg, b["target"], neg_ids))
    want = math.log(1 + n_neg)
    if tt["item_tower_config"] == "id_only":
        check(abs(loss0 - want) < 0.05,
              f"id_only 随机初始化 loss {loss0:.4f} ≈ ln(1+{n_neg}) = {want:.4f}（均匀猜）")
    else:
        check(math.isfinite(loss0) and 0.5 * want < loss0 < 3 * want,
              f"id_side 随机初始化 loss {loss0:.4f} 有限且在 ln(11)={want:.2f} 的量级内"
              "（正负样本因 duration 分布不同而不可交换，不应恰等于 ln(11)）")
    with torch.no_grad():
        # 正样本向量与用户向量完全对齐、负样本正交 -> loss 应趋近 0
        perfect = model.loss(u, u.clone(), torch.zeros_like(v_neg),
                            b["target"], torch.full_like(neg_ids, -1))
    check(float(perfect) < 0.01, f"完美对齐时 loss {float(perfect):.6f} → 0")

    # ---------- B. accidental hit ----------
    print("\n=== B. accidental hit 屏蔽 ===")
    with torch.no_grad():
        hit_ids = neg_ids.clone()
        hit_ids[:, 0] = b["target"]                      # 第 0 个负样本 = 正样本本身
        hit_feat = data.item_features(hit_ids.numpy().ravel(),
                                      np.repeat(data.date[rows], n_neg))
        v_hit = model.item_vec(hit_feat).view(len(rows), n_neg, -1)
        l_masked = float(model.loss(u, v_pos, v_hit, b["target"], hit_ids))
        # 参照：直接去掉那一个负样本
        l_drop = float(model.loss(u, v_pos, v_hit[:, 1:], b["target"], hit_ids[:, 1:]))
        # 反例：不屏蔽（伪造一组不相等的 id）
        l_unmasked = float(model.loss(u, v_pos, v_hit, b["target"],
                                      torch.full_like(hit_ids, -1)))
    check(abs(l_masked - l_drop) < 1e-6,
          f"屏蔽后 loss {l_masked:.6f} == 去掉该负样本 {l_drop:.6f}")
    check(abs(l_masked - l_unmasked) > 1e-4,
          f"不屏蔽会得到 {l_unmasked:.6f} —— 与屏蔽后不同，说明这层处理有内容")

    # ---------- C. 采样器契约 ----------
    print("\n=== C. 采样器接口契约 ===")
    for S in (RandomNegative, InBatchNegative):
        s = S(lo, hi)
        g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
        a, c = s.sample(b, n_neg, g1), s.sample(b, n_neg, g2)
        check(tuple(a.shape) == (len(rows), n_neg) and a.dtype == torch.long,
              f"{s.name}: 形状 {tuple(a.shape)}，dtype long")
        check(torch.equal(a, c), f"{s.name}: 同 generator 种子两次采样完全相同")
    r = RandomNegative(lo, hi).sample(b, 200, torch.Generator().manual_seed(1))
    check(bool(((r >= lo) & (r <= hi)).all()), f"random: 全部落在候选库区间 [{lo}, {hi}]")
    frac = float((r < lo + (hi - lo) // 2).float().mean())
    check(abs(frac - 0.5) < 0.02, f"random: 前半区间占比 {frac:.3f} ≈ 0.5（近似均匀）")
    ib = InBatchNegative(lo, hi).sample(b, n_neg, torch.Generator().manual_seed(1))
    check(bool(torch.isin(ib, b["target"]).all()), "inbatch: 全部来自本批的正样本目标")
    check(float(torch.isin(ib, b["target"].unsqueeze(1).expand(-1, n_neg)).float().mean()) == 1.0,
          "inbatch: 不会采到批外的 item")

    # ---------- D. 归一化与 temperature ----------
    print("\n=== D. L2 归一化与 temperature ===")
    with torch.no_grad():
        check(torch.allclose(u.norm(dim=-1), torch.ones(len(rows)), atol=1e-5),
              "用户向量模长 = 1")
        check(torch.allclose(v_pos.norm(dim=-1), torch.ones(len(rows)), atol=1e-5),
              "物品向量模长 = 1")
        t_old = model.temperature
        model.temperature = t_old * 4
        l_t = float(model.loss(u, v_pos, v_neg, b["target"], neg_ids))
        model.temperature = t_old
    check(abs(l_t - loss0) > 1e-6, f"temperature x4 后 loss 由 {loss0:.4f} 变为 {l_t:.4f}")

    # ---------- E. 梯度 ----------
    print("\n=== E. 梯度流向 ===")
    model.train()
    model.zero_grad(set_to_none=True)
    u2, v2 = model.user_vec(b), model.item_vec(b)
    v_neg2 = model.item_vec(neg_feat).view(len(rows), n_neg, -1)
    model.loss(u2, v2, v_neg2, b["target"], neg_ids).backward()
    g = model.item_emb.weight.grad
    check(g is not None and bool((g[0] == 0).all()), "PAD 行（index 0）梯度为零")
    touched = torch.unique(torch.cat([b["target"], neg_ids.ravel()]))
    check(bool((g[touched].abs().sum(1) > 0).any()), f"参与的 {len(touched):,} 个 item 吃到梯度")

    # ---------- F. 训练范围与端到端确定性 ----------
    print("\n=== F. 训练样本范围与确定性 ===")
    tr = RetrievalData(proc, args.protocol, "train")
    r_cat = train_rows(tr, lo, hi, "catalog")
    r_voc = train_rows(tr, lo, hi, "vocab")
    check(bool(((tr.target[r_cat] >= lo) & (tr.target[r_cat] <= hi)).all()),
          f"catalog 口径：{len(r_cat):,} 条，目标全在候选库内")
    check(len(r_voc) > len(r_cat),
          f"vocab 口径 {len(r_voc):,} 条 > catalog {len(r_cat):,} 条"
          f"（差 {len(r_voc) - len(r_cat):,}，即 42% 的正样本）")
    del tr

    # 显式传 --tag：确定性检查不该依赖产物命名规则，否则命名一改这里就静默挂掉
    # （而且若用 2>/dev/null 过滤 stderr，挂掉会伪装成"通过"）。
    runs = []
    for _ in range(2):
        subprocess.run(
            [sys.executable, "-m", "src.retrieval.two_tower", "--config", args.config,
             "--negative", "random", "--max-steps", "8", "--seed", "123",
             "--tag", "_determinism_probe", "--no-eval"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        ck = torch.load(ROOT / "experiments" / "two_tower__determinism_probe.pt",
                        weights_only=False)
        runs.append(ck["loss_tail"])
    check(runs[0] == runs[1],
          f"同 seed 跑两次真实训练，loss 逐位相同（{runs[0]!r}）")

    # ---------- G. 评估链路 ----------
    print("\n=== G. 评估链路（向量 -> Top-K -> video_id）===")
    import polars as pl

    from src.retrieval.ann_index import topk_scores
    from src.retrieval.two_tower import evaluate_checkpoint, item_matrix

    set_seed(0)
    m_rand = TwoTower(data, cfg)
    m_rand.eval()
    df = evaluate_checkpoint(m_rand, cfg, proc, args.protocol)
    for kk in require(cfg, "eval", "k_list"):
        got = df.filter((pl.col("metric") == "recall") & (pl.col("k") == kk))["per_request"].item()
        want = kk / (hi - lo + 1)
        check(0.6 * want < got < 1.6 * want,
              f"随机初始化 Recall@{kk} = {got:.6f} ≈ 随机猜 {want:.6f}")

    # G2：用户向量 = 目标物品向量，Top-1 必须命中
    # id_side 的物品向量依赖请求日（age = 请求日 − 上传日），因此必须挑定一天、
    # 只用那天的考题来比对。传 None 会让物品塔拿不到 age 而直接报错。
    reqs = pl.read_parquet(proc / f"eval_requests_{args.protocol}.parquet")
    catalog = pl.read_parquet(proc / f"catalog_{args.protocol}.parquet")
    cat_ids = catalog.get_column("video_id").to_numpy()
    pos = {int(v): i for i, v in enumerate(cat_ids)}
    per_day = tt["item_tower_config"] == "id_side"
    if per_day:
        day = int(reqs.get_column("date").mode().item())
        probe = reqs.filter(pl.col("date") == day).head(2000)
        print(f"  （id_side：物品向量按请求日 {day} 计算 age）")
    else:
        day, probe = None, reqs.head(2000)
    iv = item_matrix(m_rand, data, lo, hi, day)
    rowpos = np.array([pos[int(v)] for v in probe.get_column("video_id")], np.int64)
    idx, _ = topk_scores(iv[rowpos], iv, k=5)
    check(bool((idx[:, 0] == rowpos).all()),
          f"完美预测：抽 {len(rowpos):,} 条，Top-1 下标 == 目标在候选库中的位置")
    check(bool((cat_ids[idx[:, 0]] == probe.get_column("video_id").to_numpy()).all()),
          "下标映射回 video_id 后与考题目标一致（这一步错位会静默给出随机水平的 Recall）")
    del iv, m_rand

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("双塔（损失与负采样）验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("双塔验算通过（闭式解 / accidental hit / 采样契约 / 归一化 / 梯度 / 确定性 / 评估链路）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
