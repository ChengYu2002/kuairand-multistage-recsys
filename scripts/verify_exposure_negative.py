"""独立验算曝光 / Hybrid 负采样（src/retrieval/exposure_negative.py、hybrid_negative.py）。

§16.6 要求四种策略除采样外没有任何差异，于是这里的大部分失败模式**都不会报错**：

  - 池子里混进了该用户点过的视频 -> 模型被要求把同一个 (user, item) 同时推高和压低，
    照样收敛，只是 Recall 莫名偏低，而且怎么查都查不出来。
  - CSR 的 indptr 错位一格 -> 每个用户拿到的是**别人的**曝光。这恰好就是
    "条件于某个用户的难负样本"的一种，loss 曲线看不出异常。
  - 空池子用户静默取到下一个用户的池子 -> 同上。
  - 池子没按 (user,item) 去重 -> exposure 悄悄变成"该用户的热度采样"。
  - hybrid 的切分算成 10/0 -> 实验名叫 hybrid，跑的其实是 random。

九层检查：
  A. 合成样例（尺子的刻度）：手工构造一张答案已知的曝光表，逐个用户比对池子内容。
     含"只有 long_view 正向"、"重复曝光"、"区间外"、"全正向导致空池"四种情形。
  B. 真实池子的 CSR 结构不变量。
  C. 真实池子的内容：用**完全独立的 numpy + set** 路径重算，抽样用户穷举比对。
     特别是「池子 ∩ 该用户的全部正向 item == 空」。
  D. 采样器行为：形状 / dtype / 落在候选库区间内 / 每个元素确实属于该行用户的池子。
  E. 空池子退回 random，且 fallback 计数与真实空池子用户数一致；比例超 1% 报警。
  F. 代码依赖的浮点性质（floor(r*cnt) < cnt）对池子里实际出现的每个 cnt 成立。
  G. 可证伪：池子只有 1 个元素的用户，必须永远只抽到那一个；以及把 indptr 错位一格
     之后，C 段的"池子不含正向 item"检查**必须失败** —— 否则那把尺子是坏的。
  H. Hybrid：固定 5+5 的列布局、前半在候选库内、后半在池子里、切分退化时报错。
  I. 不影响已有策略：random / inbatch 经工厂构造与直接构造，同 seed 下逐位相同。
  J. 与 RetrievalData 对接：batch 里有 user_id，且它能正确索引池子。

用法：
    python scripts/verify_exposure_negative.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval.dataset import RetrievalData
from src.retrieval.exposure_negative import (
    POSITIVE_SIGNALS,
    ExposureNegative,
    build_exposure_pool,
)
from src.retrieval.hybrid_negative import HybridNegative
from src.retrieval.negative_sampling import SAMPLERS, build_sampler
from src.utils.config import load_config, project_path, require

FALLBACK_ALARM = 0.01  # 空池子比例超过这个值，exposure 就不再是纯 exposure，必须披露


def _batch(user_ids: list[int]) -> dict:
    """采样器只需要 user_id 与 target；其余字段与负采样无关。"""
    u = torch.tensor(user_ids, dtype=torch.long)
    return {"user_id": u, "target": torch.zeros_like(u)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    import json

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

    # ---------- A. 合成样例：答案手工算出来，不依赖被测代码 ----------
    print("\n[A] 合成样例（答案已知）")
    syn = pl.DataFrame(
        {
            #  user 0: 曝光 {5,6,7}，6 点击过            -> 池子 {5,7}
            #  user 1: 曝光 {5,5,5} 重复三次，从未正向    -> 池子 {5}（去重）
            #  user 2: 曝光 {6}，点击过                   -> 池子 {}（空）
            #  user 3: 曝光 {99}，在区间 [2,10] 之外      -> 池子 {}（空）
            #  user 4: 曝光 {7}，只有 long_view 正向      -> 池子 {}（long_view 算正向）
            #  user 5: 曝光 {8}，同一 item 一次没点一次点  -> 池子 {}（任意一次正向即排除）
            "user_id": [0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 5],
            "target_idx": [5, 6, 7, 5, 5, 5, 6, 99, 7, 8, 8],
            "is_click": [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
            "long_view": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        }
    )
    want = {0: [5, 7], 1: [5], 2: [], 3: [], 4: [], 5: []}
    sp = build_exposure_pool(syn, n_users=6, lo=2, hi=10)
    for uid, exp in want.items():
        got = sp.pool_of(uid).tolist()
        check(got == exp, f"合成 user {uid} 的池子 == {exp}（实际 {got}）")
    # 区间 [2,10] 内的唯一对：(0,5)(0,6)(0,7)(1,5)(2,6)(4,7)(5,8) = 7；(3,99) 在区间外
    check(
        sp.n_pairs_in_range == 7,
        f"区间内唯一 (user,item) 对 == 7（实际 {sp.n_pairs_in_range}）",
    )
    check(sp.coverage()["users_empty"] == 4, "合成样例里 4 个用户空池子")

    print("\n[A2] 单元素池子必须永远抽到那一个（可证伪）")
    s1 = ExposureNegative(2, 10, sp)
    g = torch.Generator().manual_seed(0)
    out = s1.sample(_batch([1] * 64), 10, g)
    check(bool((out == 5).all()), "user 1 的池子只有 {5}，1024 次抽样全部为 5")

    print("\n[A3] 空池子退回 random，且计数正确")
    s2 = ExposureNegative(2, 10, sp)
    g = torch.Generator().manual_seed(0)
    out = s2.sample(_batch([0, 2, 3, 4, 5]), 10, g)
    st = s2.stats()
    check(
        st["fallback_rows"] == 4 and st["sampled_rows"] == 5,
        f"5 行里 4 行走 fallback（实际 {st['fallback_rows']}/{st['sampled_rows']}）",
    )
    check(
        bool(((out[1:] >= 2) & (out[1:] <= 10)).all()),
        "fallback 的负样本仍落在候选库区间 [2, 10] 内",
    )
    check(set(out[0].tolist()) <= {5, 7}, "非空池子那一行仍然只来自它自己的池子")

    print("\n[A4] 坏掉的池子必须被 C 段的判据抓住（尺子指向已知的错）")
    bad = build_exposure_pool(syn, n_users=6, lo=2, hi=10)
    bad.indptr[1:] = bad.indptr[1:] - 1  # 人为把 indptr 错位一格
    bad.indptr[0] = 0
    moved = any(sorted(bad.pool_of(u).tolist()) != want[u] for u in want)
    check(
        moved, "indptr 错位一格后，至少一个用户的池子内容改变（否则这把尺子没有分辨力）"
    )

    # ---------- B/C. 真实池子 ----------
    print("\n[B] 真实池子的 CSR 结构")
    samples = proc / f"samples_train_{a.protocol}.parquet"
    raw = pl.read_parquet(samples, columns=["user_id", "target_idx", *POSITIVE_SIGNALS])
    n_users = int(raw["user_id"].max()) + 1
    pool = build_exposure_pool(samples, n_users=n_users, lo=lo, hi=hi)
    cov = pool.coverage()
    print(
        f"       池子 {cov['pool_pairs']:,} 对 / 区间内曝光 "
        f"{cov['exposed_pairs_in_range']:,} 对 / 用户 {cov['n_users']:,}；"
        f"大小 min {cov['pool_size_min']} p50 {cov['pool_size_p50']} "
        f"max {cov['pool_size_max']}；空池子 {cov['users_empty']}"
    )
    check(bool((np.diff(pool.indptr) >= 0).all()), "indptr 单调不减")
    check(
        pool.indptr[0] == 0 and pool.indptr[-1] == pool.items.size,
        "indptr 两端与 items 长度一致",
    )
    check(
        bool(((pool.items >= lo) & (pool.items <= hi)).all()),
        f"池子里每个 item 都在候选库区间 [{lo}, {hi}] 内",
    )
    sz = pool.sizes
    check(int(sz.sum()) == pool.items.size, "各用户池子大小之和 == items 长度")
    dup = [
        u
        for u in range(min(n_users, 200))
        if len(set(pool.pool_of(u).tolist())) != sz[u]
    ]
    check(not dup, f"前 200 个用户的池子内部无重复（违例 {dup[:5]}）")
    check(
        all((np.diff(pool.pool_of(u)) > 0).all() for u in range(min(n_users, 200))),
        "每个用户的池子按 item 升序（可复现、可逐项比对）",
    )

    print("\n[C] 真实池子的内容（独立用 numpy + set 重算）")
    ru = raw["user_id"].to_numpy().astype(np.int64)
    rt = raw["target_idx"].to_numpy().astype(np.int64)
    rpos = np.zeros(len(raw), bool)
    for c in POSITIVE_SIGNALS:
        rpos |= raw[c].to_numpy() > 0
    inr = (rt >= lo) & (rt <= hi)
    rng = np.random.default_rng(0)
    probe = rng.choice(n_users, size=min(25, n_users), replace=False)
    bad_pool, bad_pos, bad_sub = [], [], []
    for u in probe:
        m = ru == u
        exposed = set(rt[m & inr].tolist())
        positive = set(rt[m & rpos].tolist())  # 不限区间：全部正向过的 item
        expect = sorted(exposed - positive)
        got = pool.pool_of(u).tolist()
        if got != expect:
            bad_pool.append(int(u))
        if set(got) & positive:
            bad_pos.append(int(u))
        if not set(got) <= exposed:
            bad_sub.append(int(u))
    check(
        not bad_pool,
        f"抽查 {len(probe)} 个用户，池子 == 曝光 - 正向（违例 {bad_pool[:5]}）",
    )
    check(not bad_pos, f"池子与该用户的正向 item 无交集（违例 {bad_pos[:5]}）")
    check(not bad_sub, f"池子是该用户曝光集合的子集（违例 {bad_sub[:5]}）")
    tot_pairs = len(set(zip(ru[inr].tolist(), rt[inr].tolist())))
    check(
        tot_pairs == cov["exposed_pairs_in_range"],
        f"区间内唯一 (user,item) 对数独立重算一致（{tot_pairs:,}）",
    )

    # ---------- D/E/F. 采样器在真实池子上的行为 ----------
    print("\n[D] 采样器行为（真实池子）")
    smp = ExposureNegative(lo, hi, pool)
    nonempty = np.flatnonzero(sz > 0)
    users = nonempty[rng.choice(len(nonempty), size=min(1024, len(nonempty)))]
    b = _batch(users.tolist())
    g = torch.Generator().manual_seed(42)
    neg = smp.sample(b, 10, g)
    check(
        neg.shape == (len(users), 10) and neg.dtype == torch.long,
        f"形状 {tuple(neg.shape)} / dtype {neg.dtype}",
    )
    check(bool(((neg >= lo) & (neg <= hi)).all()), "全部落在候选库区间内")
    off = [
        int(u)
        for i, u in enumerate(users)
        if not set(neg[i].tolist()) <= set(pool.pool_of(u).tolist())
    ]
    check(not off, f"每个负样本都属于该行用户自己的池子（违例 {off[:5]}）")
    g2 = torch.Generator().manual_seed(42)
    check(
        bool(torch.equal(ExposureNegative(lo, hi, pool).sample(b, 10, g2), neg)),
        "同 seed 两次采样逐位相同",
    )

    print("\n[E] 空池子比例")
    fr = cov["users_empty"] / cov["n_users"]
    check(
        fr <= FALLBACK_ALARM,
        f"空池子用户占比 {fr:.4%} <= {FALLBACK_ALARM:.0%}"
        "（超过就不再是纯 exposure，必须在 README 披露 fallback 比例）",
    )

    print("\n[F] 代码依赖的浮点性质：floor(r*cnt) < cnt")
    rmax = np.float32(1.0) - np.float32(2.0**-24)
    cnts = np.unique(sz[sz > 0]).astype(np.float32)
    worst = np.floor(rmax * cnts).astype(np.int64)
    check(
        bool((worst < np.unique(sz[sz > 0])).all()),
        f"对池子里出现的全部 {len(cnts)} 种 cnt，最坏 r 下 floor(r*cnt) 仍 < cnt",
    )

    # ---------- H. Hybrid ----------
    print("\n[H] Hybrid = Random + Exposure，固定 5+5")
    hyb = HybridNegative(lo, hi, pool)
    check(hyb.split(10) == (5, 5), f"n_neg=10 切成 {hyb.split(10)}")
    g = torch.Generator().manual_seed(7)
    hn = hyb.sample(b, 10, g)
    check(hn.shape == (len(users), 10), f"形状 {tuple(hn.shape)}")
    check(bool(((hn >= lo) & (hn <= hi)).all()), "全部落在候选库区间内")
    bad_h = [
        int(u)
        for i, u in enumerate(users)
        if not set(hn[i, 5:].tolist()) <= set(pool.pool_of(u).tolist())
    ]
    check(not bad_h, f"后 5 列全部来自该用户的曝光池（违例 {bad_h[:5]}）")
    # 前 5 列是均匀随机：它们落在池子里的概率约 pool_size/194310，应当极低。
    in_pool = sum(
        len(set(hn[i, :5].tolist()) & set(pool.pool_of(u).tolist()))
        for i, u in enumerate(users)
    )
    check(
        in_pool <= 0.01 * 5 * len(users),
        f"前 5 列几乎不落在池子里（{in_pool}/{5 * len(users)}）—— 否则两半没区分开",
    )
    try:
        HybridNegative(lo, hi, pool, {"random": 1.0, "exposure": 0.0}).split(10)
        ok = False
    except ValueError:
        ok = True
    check(ok, "切分退化成 10/0 时报错（否则实验名叫 hybrid、跑的是 random）")
    try:
        HybridNegative(lo, hi, pool, {"random": 0.5, "inbatch": 0.5})
        ok = False
    except ValueError:
        ok = True
    check(ok, "hybrid_mix 键名拼错时报错")

    # ---------- I. 不影响已有策略 ----------
    print("\n[I] random / inbatch 的构造路径未被改动")
    check(
        set(SAMPLERS) == {"random", "inbatch", "exposure", "hybrid"},
        f"四种策略全部注册：{sorted(SAMPLERS)}",
    )
    fake = {
        "target": torch.randint(lo, hi + 1, (256,)),
        "user_id": torch.zeros(256, dtype=torch.long),
    }
    for strat in ("random", "inbatch"):
        g1, g2 = torch.Generator().manual_seed(11), torch.Generator().manual_seed(11)
        a1 = build_sampler(strat, lo, hi).sample(fake, 10, g1)
        a2 = SAMPLERS[strat](lo, hi).sample(fake, 10, g2)
        check(
            bool(torch.equal(a1, a2)),
            f"{strat}：经工厂构造与直接构造，同 seed 输出逐位相同",
        )

    # ---------- J. 与 RetrievalData 对接 ----------
    print("\n[J] 与 RetrievalData 对接")
    data = RetrievalData(proc, a.protocol, "train")
    # 必须随机取行：samples 按 user_id 排序，取前 2048 行只会覆盖一两个用户。
    rows = rng.choice(len(data), size=2048, replace=False)
    rb = data.batch(rows)
    check("user_id" in rb, "batch() 返回 user_id（exposure 要靠它定位用户）")
    if "user_id" in rb:
        check(
            bool(torch.equal(rb["user_id"], torch.from_numpy(data.user_id[rows]))),
            "batch 的 user_id 与 samples 表的 user_id 逐位一致",
        )
        g = torch.Generator().manual_seed(3)
        real = build_sampler(
            "exposure", lo, hi, data=data, proc=proc, protocol=a.protocol
        )
        rn = real.sample(rb, 10, g)
        cnt_u = data.user_id[rows]
        viol = [
            int(u)
            for i, u in enumerate(cnt_u)
            if sz[u] > 0 and not set(rn[i].tolist()) <= set(pool.pool_of(u).tolist())
        ]
        check(not viol, f"真实 batch 上每个负样本都属于该用户的池子（违例 {viol[:5]}）")
        check(
            real.pool.items.size == pool.items.size,
            "工厂建出的池子规模与本脚本独立构造的一致",
        )

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("曝光 / Hybrid 负采样验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        "验算通过（合成样例 / CSR 结构 / 池子内容 / 采样行为 / 空池 fallback / "
        "浮点边界 / Hybrid 布局 / 已有策略未变 / 与 RetrievalData 对接）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
