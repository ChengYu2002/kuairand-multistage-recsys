"""独立验算全库检索（plan §16）。

检索错了不会报错：Top-K 照样返回 500 个 id，指标照样算得出来，只是挑错了。
因此这里全部用「有已知正确答案」的构造来验，不看跑出来的数好不好看。

检查分五层：
  A. 已知答案：单位向量、正交向量等有闭式解的小例子。
  B. 与暴力全量比对：小规模上直接物化整个分数矩阵、按 (分数降序, 下标升序) 全排序，
     与分块实现逐元素比对。
  C. 分块不变性：block=1 / 7 / 全量 必须给出**完全相同**的结果。分块是纯工程手段，
     不允许影响结果。
  D. 并列处理：构造精确并列，验证取下标较小者；并证明「只用 argpartition」会给出
     不同答案 —— 否则这条处理没有被真正检验。
  E. 输入防线：NaN/Inf 必须被拦下。torch.topk 遇到 NaN 不报错，会静默返回一组看似
     正常的 Top-K —— 实测全 NaN 时返回 [[0,1,2], ...]，而候选库按热度降序排，
     这恰好酷似热度基线。不拦的话，一个训练发散的模型会产出"合理"的指标。
  F. 真实规模与内存：在**独立子进程**里跑 51,746 x 194,310，量它的内存增量。
     必须另起进程：本进程的 ru_maxrss 是高水位，受解释器、torch 导入、以及测试
     自己造的向量影响；两个高水位相减并不是真正的增量。子进程里也直接生成
     float32（不经 float64 再转），避免污染基线。

用法：
    python scripts/verify_ann_index.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval.ann_index import (
    DEFAULT_BLOCK,
    peak_block_bytes,
    peak_total_bytes,
    topk_scores,
)


def brute_force(u: np.ndarray, v: np.ndarray, k: int) -> np.ndarray:
    """独立路径：物化整个分数矩阵，按 (分数降序, 下标升序) 全排序后取前 k。"""
    s = (u.astype(np.float32) @ v.astype(np.float32).T)
    n, m = s.shape
    cols = np.tile(np.arange(m), (n, 1))
    return np.lexsort((cols, -s))[:, :k].astype(np.int32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-scale", action="store_true", help="跑一遍 51,746 x 194,310 的真实规模")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    failures: list[str] = []
    checks = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  {'OK  ' if ok else 'FAIL'} {msg}")
        if not ok:
            failures.append(msg)

    # ---------- A. 已知答案 ----------
    print("\n=== A. 有闭式解的小例子 ===")
    v = np.eye(5, dtype=np.float32)                     # 5 个互相正交的单位向量
    u = np.array([[0, 0, 1, 0, 0]], dtype=np.float32)   # 与第 2 号完全一致
    idx, sc = topk_scores(u, v, k=3, return_scores=True)
    check(idx[0, 0] == 2, f"与某物品向量完全相同的查询，该物品排第一（得到 {idx[0, 0]}）")
    check(abs(sc[0, 0] - 1.0) < 1e-6, f"其分数 = 1.0（得到 {sc[0, 0]:.6f}）")
    check(list(idx[0, 1:]) == [0, 1],
          f"其余全为 0 分且并列 -> 按下标升序取 0,1（得到 {list(idx[0, 1:])}）")

    u2 = np.array([[3.0, 4.0]], dtype=np.float32)
    v2 = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    idx2, sc2 = topk_scores(u2, v2, k=3, return_scores=True)
    check(list(idx2[0]) == [1, 0, 2] and np.allclose(sc2[0], [4.0, 3.0, -3.0]),
          f"点积排序正确：{list(idx2[0])} 分数 {list(np.round(sc2[0], 3))}")

    # ---------- B. 与暴力全量比对 ----------
    print("\n=== B. 与暴力全排序比对 ===")
    for n, m, k in ((50, 300, 20), (200, 1000, 100), (7, 40, 40)):
        u = rng.standard_normal((n, m := m)).astype(np.float32)[:, :16]
        vv = rng.standard_normal((m, 16)).astype(np.float32)
        got, _ = topk_scores(u, vv, k=k, block=13)
        want = brute_force(u, vv, k)
        check(np.array_equal(got, want), f"n={n} m={m} k={k}：与暴力全排序逐元素一致")

    # ---------- C. 分块不变性 ----------
    print("\n=== C. 分块不变性 ===")
    u = rng.standard_normal((97, 16)).astype(np.float32)
    vv = rng.standard_normal((500, 16)).astype(np.float32)
    ref, _ = topk_scores(u, vv, k=50, block=1)
    for b in (7, 32, 97, 1000):
        got, _ = topk_scores(u, vv, k=50, block=b)
        check(np.array_equal(got, ref), f"block={b} 与 block=1 结果完全相同")

    # ---------- D. 并列处理 ----------
    print("\n=== D. 并列必须取下标较小者 ===")
    # 20 个候选里有 10 个与查询的点积完全相同，取 k=5 必然切在并列组中间
    vv = np.zeros((20, 4), dtype=np.float32)
    vv[:10] = np.array([1.0, 0, 0, 0], dtype=np.float32)      # 前 10 个完全相同
    vv[10:] = np.array([0, 1.0, 0, 0], dtype=np.float32)
    u = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
    got, _ = topk_scores(u, vv, k=5)
    check(list(got[0]) == [0, 1, 2, 3, 4],
          f"10 个并列中取下标最小的 5 个（得到 {list(got[0])}）")
    # 可证伪：只用 argpartition 会挑出不同的一组
    s = (u @ vv.T)[0]
    naive = np.sort(np.argpartition(-s, 5)[:5])
    check(list(naive) != [0, 1, 2, 3, 4],
          f"只用 argpartition 得到 {list(naive)} —— 与规则不符，说明这层处理是必要的")

    # ---------- E. 输入防线 ----------
    print("\n=== E. NaN / Inf 必须被拦下 ===")
    vg = np.eye(6, dtype=np.float32)
    for label, bad in (("用户向量含 NaN", np.array([[np.nan, 0, 0, 0, 0, 0]], np.float32)),
                       ("用户向量含 Inf", np.array([[np.inf, 0, 0, 0, 0, 0]], np.float32))):
        try:
            topk_scores(bad, vg, k=3)
            check(False, f"{label}：未被拦下")
        except ValueError:
            check(True, f"{label}：已拦下")
    try:
        bad_v = vg.copy(); bad_v[3, 3] = np.nan
        topk_scores(np.array([[0, 0, 0, 1.0, 0, 0]], np.float32), bad_v, k=3)
        check(False, "物品向量含 NaN：未被拦下")
    except ValueError:
        check(True, "物品向量含 NaN：已拦下")
    # 可证伪：不拦的话返回的是什么 —— 证明这道防线不是形式主义
    s_nan = np.full((3, 6), np.nan, np.float32) @ vg.T
    naive = np.argsort(-s_nan, axis=1, kind="stable")[:, :3]
    check(naive.tolist() == [[0, 1, 2]] * 3,
          f"若不拦，全 NaN 会得到 {naive[0].tolist()} —— 即候选库最热门的三个，酷似热度基线")

    # ---------- F. 真实规模与内存 ----------
    print("\n=== F. 真实规模与内存（独立子进程测量）===")
    n_items, k, block = 194_310, 500, DEFAULT_BLOCK
    est = peak_block_bytes(n_items, block)
    check(est / 1024**3 < 2.0, f"估算峰值 {est / 1024**3:.2f} GB < 2 GB")
    if args.full_scale:
        n_q = 51_746
        code = f"""
import json, resource, sys, time
import numpy as np
sys.path.insert(0, {str(ROOT)!r})
from src.retrieval.ann_index import topk_scores
rng = np.random.default_rng(0)
v = rng.standard_normal(({n_items}, 64), dtype=np.float32)
u = rng.standard_normal(({n_q}, 64), dtype=np.float32)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t0 = time.perf_counter()
idx, _ = topk_scores(u, v, k={k}, block={block})
dt = time.perf_counter() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
dup = bool((np.diff(np.sort(idx, axis=1), axis=1) <= 0).any())
print(json.dumps({{"grew": peak - base, "dt": dt, "shape": list(idx.shape), "dup": dup}}))
"""
        r = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=False
            )
        if r.returncode != 0:
            check(False, f"子进程失败：{r.stderr.strip()[-200:]}")
        else:
            res = json.loads(r.stdout.strip().splitlines()[-1])
            unit = 1024**3 if sys.platform == "darwin" else 1024**2   # mac 字节 / linux KB
            grew = res["grew"] / unit
            check(res["shape"] == [n_q, k], f"{n_q:,} x {n_items:,} 跑通，耗时 {res['dt']:.1f}s")
            check(not res["dup"], "每行 Top-K 内无重复候选")
            ratio = grew / (est / 1024**3)
            # 容差 0.7~1.5：估算要能贴着机器内存选块大小，2.5 倍的余量等于不能用
            check(0.7 < ratio < 1.5,
                  f"子进程内存增量 {grew:.2f} GB / 估算 {est / 1024**3:.2f} GB = {ratio:.2f}x"
                  "（应在 0.7~1.5 之间）")
            tot = peak_total_bytes(n_q, n_items, 64, k, block) / 1024**3
            print(f"  参考  全过程总峰值估算 {tot:.2f} GB（含常驻的向量与输出）；"
                  f"27K 粗估需按 peak_total_bytes 重新规划块大小")
    else:
        print("  (跳过全量烟测；加 --full-scale 开启)")

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("全库检索验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("全库检索验算通过（闭式解 / 暴力比对 / 分块不变 / 并列 / NaN 防线 / 规模）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
