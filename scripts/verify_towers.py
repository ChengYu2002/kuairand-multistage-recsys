"""独立验算物品塔与用户塔（阶段 3 第一段）。

塔的输出是一串向量，"对不对"离开损失函数无从判断。但有几条**性质**可以直接钉死，
而且它们恰好是最容易出错、出错了又不报错的地方：

  A. 掩码不变性（最关键）：同一段历史，补到 50 与不补，必须得到**完全相同**的向量。
     torch 的 padding_idx=0 只保证 PAD 行是零向量，**不会**把它从 mean 的分母里去掉。
     这里同时证明"若用普通 mean"会给出不同结果 —— 否则这条检查没有内容。
     实测训练时平均 49.67/50 条可编码、评估时 12.05/50，写错会让两边尺度差约 4 倍。
  A0. 历史行号没有错位：hist 按 (user_id, time_ms) 去重存储，样本只记行号。这个映射靠
      一次 left join 得到，而 polars **不保证** left join 保持左表顺序 —— 一旦顺序变了，
      每个样本都会拿到别人的历史，且不会报错（下游用的正是这份已错位的映射）。
      因此必须回到 samples 表，拿原始 hist_idx 与 data.hist[data.hist_row[i]] 逐条比对。
  A2. 掩码是谁建的：上面那几条用的是测试自己构造的掩码，验不到 RetrievalData.batch()
      内部的建法。实测把 dataset 里的 OOV 掩码删掉，A 层全绿 —— 因此必须单独验
      真实 batch 里的 hist_mask 与 hist_len。
  B. 空历史：全 PAD 时历史通道必须是零向量，不能是 NaN，也不能除零。
  C. 顺序不变性：mean pooling 对历史顺序不敏感（这是当前设计的性质，不是缺陷 ——
     DIN 是 P2，不在本阶段）。写成对顺序敏感说明实现里混进了别的东西。
  D. 确定性：同 seed 两次构造 + 前向，输出逐位相同。
  E. Config A/B：id_only 与 id_side 必须给出不同结果，且 B 确实用到了侧信息
     （改变 author/tag/age 输入，输出必须变；对 A 则不变）。
  F. 共享 embedding：两塔用的是同一份权重（改一处，两处都变）。
  G. 真实数据：装载一批真实样本跑通，输出有限、形状正确。

用法：
    python scripts/verify_towers.py --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval.dataset import OOV, PAD, RetrievalData
from src.retrieval.item_tower import ItemTower
from src.retrieval.user_tower import UserTower
from src.utils.config import load_config, project_path, require
from src.utils.seed import set_seed


def build(data: RetrievalData, dim: int, config: str, hidden: list[int]):
    n_vocab = data.item_tags.shape[0]
    emb = nn.Embedding(n_vocab, dim, padding_idx=PAD)
    item = ItemTower(emb, config, n_authors=int(data.item_author.max()) + 2,
                     n_tags=int(data.item_tags.max()) + 2, hidden=hidden)
    user = UserTower(emb, n_t1=data.user_t1.shape[1],
                     n_static_num=data.user_static_num.shape[1],
                     static_cat_sizes=data.user_static_cat_sizes, hidden=hidden)
    return emb, item, user


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    dim = require(cfg, "two_tower", "embedding_dim")
    hidden = require(cfg, "two_tower", "user_tower_hidden")

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
    b = data.batch(np.arange(args.n))
    emb, item, user = build(data, dim, "id_side", hidden)
    item.eval(); user.eval()

    # ---------- A0. 历史行号 ----------
    print("\n=== A0. 历史行号未错位（回源表逐条比对）===")
    raw = pl.read_parquet(proc / f"samples_test_{args.protocol}.parquet",
                          columns=["hist_idx"])["hist_idx"]
    rng = np.random.default_rng(0)
    probe = np.sort(rng.choice(len(data), size=min(500, len(data)), replace=False))
    bad = 0
    for i in probe:
        v = raw[int(i)]
        want = [] if v is None else list(v)[: data.max_hist]
        # 存储是定宽补 0 的，所以只比对前 len(want) 个，再确认其后全是 PAD
        got = data.hist[data.hist_row[i]]
        if got[: len(want)].tolist() != want or bool((got[len(want):] != PAD).any()):
            bad += 1
    check(bad == 0, f"抽 {len(probe)} 条：hist[hist_row[i]] 与源表 hist_idx 逐元素一致")
    check(int(data.hist_row.max()) < data.hist.shape[0],
          f"行号全部落在 [0, {data.hist.shape[0]:,})")

    # ---------- A. 掩码不变性 ----------
    print("\n=== A. 掩码不变性（补位不得改变结果）===")
    with torch.no_grad():
        short = torch.tensor([[11, 12, 13]], dtype=torch.long)
        padded = torch.zeros((1, 50), dtype=torch.long)
        padded[0, :3] = short[0]
        m_s = torch.ones((1, 3), dtype=torch.bool)
        m_p = padded != PAD
        v_short = user.pool_history(short, m_s)
        v_pad = user.pool_history(padded, m_p)
        check(torch.allclose(v_short, v_pad, atol=1e-6),
              f"3 条历史 补到 50 与不补，池化结果相同（最大差 {(v_short - v_pad).abs().max():.2e}）")
        # 可证伪：若用普通 mean（把 PAD 算进分母），结果会差 50/3 倍
        naive = emb(padded).mean(dim=1)
        ratio = float(v_pad.abs().sum() / naive.abs().sum())
        check(abs(ratio - 50 / 3) < 1.0,
              f"普通 mean 会小 {ratio:.1f} 倍（理论 {50 / 3:.1f}）—— 说明掩码确实在起作用")
        # OOV 也必须被排除
        with_oov = padded.clone(); with_oov[0, 3] = OOV
        v_oov = user.pool_history(with_oov, (with_oov != PAD) & (with_oov != OOV))
        check(torch.allclose(v_short, v_oov, atol=1e-6), "插入一个 OOV 后结果不变（OOV 同样被 mask）")

    # ---------- A2. dataset 建出来的掩码 ----------
    print("\n=== A2. RetrievalData.batch() 建的掩码 ===")
    h, mk = b["hist"], b["hist_mask"]
    n_pad, n_oov = int((h == PAD).sum()), int((h == OOV).sum())
    check(n_oov > 0, f"本批历史中确有 OOV 条目（{n_oov:,} 个），这条检查才有意义")
    check(not bool(mk[h == PAD].any()), f"PAD 位置全部被 mask（{n_pad:,} 个）")
    check(not bool(mk[h == OOV].any()), f"OOV 位置全部被 mask（{n_oov:,} 个）")
    check(bool((mk[(h != PAD) & (h != OOV)]).all()), "其余位置全部保留")
    check(torch.equal(b["hist_len"], mk.sum(1).float()), "hist_len == 有效位数")

    # ---------- B. 空历史 ----------
    print("\n=== B. 空历史 ===")
    with torch.no_grad():
        empty = torch.zeros((2, 50), dtype=torch.long)
        v = user.pool_history(empty, empty != PAD)
        check(bool(torch.isfinite(v).all()), "无 NaN / Inf（没有除零）")
        check(bool((v == 0).all()), "历史通道输出零向量（而非 OOV 向量）")

    # ---------- C. 顺序不变性 ----------
    print("\n=== C. 顺序不变性 ===")
    with torch.no_grad():
        h = torch.tensor([[11, 12, 13, 14, 0, 0]], dtype=torch.long)
        rev = torch.tensor([[14, 13, 12, 11, 0, 0]], dtype=torch.long)
        check(torch.allclose(user.pool_history(h, h != PAD),
                             user.pool_history(rev, rev != PAD), atol=1e-6),
              "mean pooling 对历史顺序不敏感（当前设计如此；DIN 是 P2）")

    # ---------- D. 确定性 ----------
    print("\n=== D. 确定性 ===")
    set_seed(0); _, i1, u1 = build(data, dim, "id_side", hidden); i1.eval(); u1.eval()
    set_seed(0); _, i2, u2 = build(data, dim, "id_side", hidden); i2.eval(); u2.eval()
    with torch.no_grad():
        check(torch.equal(i1(b), i2(b)) and torch.equal(u1(b), u2(b)),
              "同 seed 两次构造 + 前向，输出逐位相同")

    # ---------- E. Config A / B ----------
    print("\n=== E. Config A / B ===")
    set_seed(1); _, item_a, _ = build(data, dim, "id_only", hidden); item_a.eval()
    with torch.no_grad():
        oa, ob = item_a(b), item(b)
        check(oa.shape == ob.shape == (args.n, item.out_dim),
              f"两个 Config 输出形状一致 {tuple(ob.shape)}（只有输入特征不同）")
        check(not torch.allclose(oa, ob), "id_only 与 id_side 结果不同")
        b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
        b2["author"] = torch.zeros_like(b2["author"])
        b2["tags"] = torch.zeros_like(b2["tags"])
        b2["age"] = torch.zeros_like(b2["age"])
        check(not torch.allclose(item(b), item(b2)), "id_side：改变 author/tag/age，输出随之改变")
        check(torch.allclose(item_a(b), item_a(b2)), "id_only：改变侧信息，输出**不**变（未误用）")

    # ---------- F. 共享 embedding ----------
    print("\n=== F. 两塔共享 embedding ===")
    check(item.item_emb is user.item_emb, "物品塔与用户塔引用同一个 nn.Embedding")
    with torch.no_grad():
        # 必须改这批历史里**真实出现过**的行。改 [2:10] 是没用的 —— 历史 index 动辄
        # 四五万，压根碰不到，那样测出来的"不变"是测试写错了而不是共享没生效。
        touched = torch.unique(b["hist"][b["hist_mask"]])[:16]
        before = user.pool_history(b["hist"], b["hist_mask"]).clone()
        item.item_emb.weight[touched] += 1.0
        after = user.pool_history(b["hist"], b["hist_mask"])
        check(not torch.allclose(before, after),
              f"改动物品塔 embedding 的 {len(touched)} 行，用户塔的池化随之改变")
        item.item_emb.weight[touched] -= 1.0
    check(bool((emb.weight[PAD] == 0).all()), f"index {PAD} (PAD) 行恒为零向量")

    # ---------- G. 真实数据 ----------
    print("\n=== G. 真实数据前向 ===")
    with torch.no_grad():
        vu, vi = user(b), item(b)
    check(vu.shape == (args.n, user.out_dim) and vi.shape == (args.n, item.out_dim),
          f"输出形状 user {tuple(vu.shape)} / item {tuple(vi.shape)}")
    check(bool(torch.isfinite(vu).all() and torch.isfinite(vi).all()), "输出无 NaN / Inf")
    n_empty = int((b["hist_len"] == 0).sum())
    print(f"  参考  本批 {args.n} 条中空历史 {n_empty} 条，"
          f"hist_len 中位 {b['hist_len'].median():.0f}（评估段整体中位 12）")

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("双塔（塔部分）验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("双塔（塔部分）验算通过（掩码 / 空历史 / 顺序 / 确定性 / Config / 共享 / 真实数据）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
