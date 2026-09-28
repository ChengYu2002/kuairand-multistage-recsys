"""独立验算排序装载器（src/ranking/dataset.py）。

装载器是纯搬运工，但搬错了**不会报错**：某一列错位只会让模型学到噪声，AUC 低一点，
而四个排序模型共用这一个装载器，会一起错到同一个方向，对比表看上去完全正常。
所以这里全部回到原始文件独立取值再比对，不复用装载器的任何代码路径。

七层检查：
  A. 行对齐：随机抽样，把每个通道回原始 parquet / npy 按 (user_id, video_id, date)
     独立查一遍；再打乱行号顺序，确认每行拿到的还是自己那份（polars 的 join 不保证
     保序，而错位了下游不会有任何征兆）。
  B. 召回侧物品静态键确实被换掉：在 item_idx == OOV 的行上，召回装载器给出的是一排零，
     排序装载器必须给出真实的作者 / 标签 / 时长 —— 这就是建排序表的全部目的，
     所以把它写成断言而不是靠注释。
  C. video_age 是现算的：独立用日期重算；同一个视频在不同请求日必须得到不同的 age。
  D. 时长标准化用 train 统计量：train 段标准化后均值≈0 标准差≈1，valid/test 段不是
     —— 可证伪，证明不是各 split 各自标准化。
  E. 掩码与标志自洽：hist_mask / tag_mask / *_known 与数值必须一一对应。
  F. 覆盖率与 rank_item_spec 记录一致。
  G. 确定性与无副作用：两次同样的 batch 逐位相同；召回侧文件一个字节没变。
  H. 捷径扫描（这一层是补的，因为前面七层都没抓到 P0）：先**逐个扰动** batch 里的每个
     键、看 x 会不会变，从而实测「哪些键真的进了模型」（不靠人工维护清单）；再只对这些
     键比较 train 与 test 上的单特征 AUC 强度。只在 train 上强、在 test 上弱的通道，
     就是模型学得到却用不上的捷径。

     标签置换检验查不出这一类：打乱标签后那条「只在 train 成立的关系」在训练数据里
     就被破坏了，模型学不到，留出行自然回到 0.5。它只能证明「输入里没有标签本身」。

T-1 特征**内容**（有没有只用 <= T-1 的日志）由 scripts/verify_t1_features.py 负责，
已单独验过。这里只负责「这条样本有没有拿到属于它自己那一天的那一行」。

用法：
    python scripts/verify_ranking_data.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.ranking_metrics import _avg_ranks
from src.ranking.dataset import KNOWN_FLAGS, RankingData, RankItemStatic
from src.ranking.features import RankingEncoder
from src.retrieval.dataset import OOV, PAD, RetrievalData
from src.utils.config import load_config, project_path, require

# 捷径判定阈值。强度定义为 |AUC − 0.5|，差距 = train 强度 − **valid** 强度。
#
# 为什么用 valid 而不是 test：这段检查每次跑 run_ranking.sh 都会执行，若比较 train/test
# 就等于反复读取 test 标签 —— 那会抵消 trainer "默认不评 test" 的整个用意（模型没评
# test，但开发过程一直在看 test 标签，阈值甚至是按它定的）。test 只在架构冻结后的
# 最终评估里用一次。
#
# 实测依据（train/valid，595 对，含 user_t1 / item_t1 / user_static_num 逐列展开）：
#   已修的标签泄漏通道 item_idx_retrieval    差距 +0.2144
#   合法通道最大值                          +0.09~0.12（video_id_known vs is_follow，热度漂移；
#                                            随抽样波动，默认 --seed 0 下为 +0.1174）
# 阈值 0.15 与合法最大值之间只有约 0.03 的余量。这是有意的：宁可偶尔误报去查一次，
# 也不要把 0.2 这一档漏掉。误报是一次可见的失败，漏报是一张看着正常的错表。
# 0.15 落在两者之间：能抓住标签类泄漏，不会误杀分布漂移。
# 0.05~0.15 之间的一律打印出来披露，不静默通过。
GAP_FAIL = 0.15
GAP_WARN = 0.05


def _fast_auc(x: np.ndarray, y: np.ndarray) -> float:
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _avg_ranks(x.astype(np.float64))
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

SPLITS = ("train", "valid", "test")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--split", default="test", choices=SPLITS)
    ap.add_argument("--n", type=int, default=512, help="抽样行数")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    proc = project_path(require(cfg, "dataset", "processed_dir"))
    p, sp = a.protocol, a.split
    rng = np.random.default_rng(a.seed)

    failures: list[str] = []
    checks = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  {'OK  ' if ok else 'FAIL'} {msg}")
        if not ok:
            failures.append(msg)

    st = RankItemStatic(proc, p)
    data = RankingData(proc, p, sp, st)
    spec = st.spec
    raw = pl.read_parquet(proc / f"samples_{sp}_{p}.parquet")
    static = pl.read_parquet(proc / f"item_static_rank_{p}.parquet")
    rows = rng.choice(len(data), min(a.n, len(data)), replace=False)
    b = data.batch(rows)
    sub = raw[rows]

    # ---------- A. 行对齐 ----------
    print(f"\n=== A. 行对齐（{sp} 段随机 {len(rows)} 行）===")
    labels = require(cfg, "labels", "tasks")
    bad = [t for t in labels
           if not np.array_equal(b[f"label_{t}"].numpy(),
                                 sub.get_column(t).to_numpy().astype(np.float32))]
    check(not bad, f"五个标签逐行对上原始 samples 表（不一致的：{bad}）")
    check(np.array_equal(b["item_idx_retrieval"].numpy(),
                         sub.get_column("target_idx").to_numpy()),
          "item_idx_retrieval == samples.target_idx（诊断键，不进模型）")
    want_rank = static.join(pl.DataFrame({"video_id": sub.get_column("video_id")})
                           .with_row_index("_i"), on="video_id", how="inner").sort("_i")
    check(np.array_equal(b["item_idx"].numpy(),
                         want_rank.get_column("video_rank_idx").to_numpy()),
          "item_idx == 排序词表行号（只由 train 曝光构建，与标签无关）")
    check(np.array_equal(b["tab"].numpy(), sub.get_column("tab").to_numpy()),
          "tab 对上原始表")

    # T-1 特征行：必须是这条样本自己那天的那一行（不是别人的，也不是未来那天的）
    fr_u = pl.read_parquet(proc / f"feat_row_user_{p}.parquet")
    want_u = (sub.select("user_id", "date").with_row_index("_i")
              .join(fr_u, on=["user_id", "date"], how="left").sort("_i")
              .get_column("row").fill_null(0).to_numpy())
    check(np.array_equal(sub.get_column("user_feat_row").to_numpy(), want_u),
          "user_feat_row == (user_id, date) 独立查表的结果")
    fr_i = pl.read_parquet(proc / f"feat_row_item_{p}.parquet")
    want_i = (sub.select("video_id", "date").with_row_index("_i")
              .join(fr_i, on=["video_id", "date"], how="left").sort("_i")
              .get_column("row").fill_null(0).to_numpy())
    check(np.array_equal(sub.get_column("item_feat_row").to_numpy(), want_i),
          f"item_feat_row == (video_id, date) 独立查表的结果"
          f"（冷启动行 0 占 {float((want_i == 0).mean()):.4f}）")
    u_npy = np.load(proc / f"feat_user_encoded_{p}.npy")
    i_npy = np.load(proc / f"feat_item_encoded_{p}.npy")
    check(np.array_equal(b["user_t1"].numpy(), u_npy[want_u]), "user_t1 == npy[独立算出的行号]")
    check(np.array_equal(b["item_t1"].numpy(), i_npy[want_i]), "item_t1 == npy[独立算出的行号]")

    # 打乱顺序后每行必须还是自己那份（polars 的 join 不保证保序）
    perm = rng.permutation(len(rows))
    b2 = data.batch(rows[perm])
    same = all(torch.equal(b2[k], b[k][perm]) for k in b if b[k].shape[0] == len(rows))
    check(same, "打乱行号顺序后逐通道等于原结果的同一置换（没有隐藏的顺序假设）")
    # 单行取值 == 批量取值的对应切片
    one = data.batch(rows[:1])
    check(all(torch.equal(one[k], b[k][:1]) for k in one),
          "单行 batch == 批量 batch 的第一行")

    # ---------- B. 召回侧物品静态键确实被换掉 ----------
    print("\n=== B. 物品静态属性换成了排序表 ===")
    from src.ranking.dataset import _RETRIEVAL_ITEM_KEYS
    check(not any(k in b for k in ("has_duration", "has_upload")),
          f"召回侧键已弹出（{list(_RETRIEVAL_ITEM_KEYS)} 中的标志键不在 batch 里）")
    # 按**召回词表**是否 OOV 选行：那正是原先物品侧全空的那批样本
    oov = np.flatnonzero(b["item_idx_retrieval"].numpy() == OOV)
    check(len(oov) > 0, f"{sp} 段抽样里有 {len(oov)} 行是召回词表 OOV（可比较的样本存在）")
    rd = RetrievalData(proc, p, sp)
    rb = rd.batch(rows[oov])
    r_zero = (rb["author"].numpy() == 0).all() and (rb["tags"].numpy() == 0).all()
    rank_live = float((b["author_known"].numpy()[oov] > 0).mean())
    check(r_zero, "同样这些行，召回装载器给出的 author / tags 全为 0（这是原来的问题）")
    check(rank_live > 0.3,
          f"排序装载器在这些行上 author_known 比例 {rank_live:.4f} > 0 —— 物品侧不再是空的")
    tk = float((b["tag_known"].numpy()[oov] > 0).mean())
    check(tk > 0.8, f"同样这些行 tag_known 比例 {tk:.4f}")

    # ---------- C. video_age 现算 ----------
    print("\n=== C. video_age 按请求日现算 ===")
    up = st.upload[sub.get_column("video_id").to_numpy()]
    d0 = np.datetime64("1970-01-01")

    def to_days(ymd: np.ndarray) -> np.ndarray:
        s = ymd.astype(np.int64)
        y, m, d = s // 10000, (s // 100) % 100, s % 100
        arr = np.array([f"{a:04d}-{b:02d}-{c:02d}" for a, b, c in zip(y, m, d, strict=True)],
                       dtype="datetime64[D]")
        return (arr - d0).astype(np.int64)

    ok_up = up > 0
    want_age = np.zeros(len(rows), np.float32)
    if ok_up.any():
        raw_age = (to_days(sub.get_column("date").to_numpy()[ok_up]) - to_days(up[ok_up]))
        want_age[ok_up] = np.clip(raw_age, 0, None)
    check(np.allclose(b["age"].numpy(), np.log1p(want_age), atol=1e-6),
          "age == log1p(请求日 − 上传日)，独立用日期重算")
    check(float(np.abs(b["age"].numpy()[~ok_up]).max() if (~ok_up).any() else 0.0) == 0.0,
          f"缺上传日的 {int((~ok_up).sum())} 行 age 记 0（由 upload_date_known 标注）")
    # 同一个视频在不同请求日必须得到不同 age
    vids = sub.get_column("video_id").to_numpy()
    probe = vids[ok_up][:1]
    if len(probe):
        dates = np.array([20220505, 20220508], np.int64)
        f2 = st.features(np.repeat(probe, 2), dates)
        check(float(f2["age"][0]) != float(f2["age"][1]),
              f"同一视频在 20220505 / 20220508 的 age 不同（{float(f2['age'][0]):.4f} vs "
              f"{float(f2['age'][1]):.4f}）-> 不是预先算好的固定值")

    # ---------- D. 时长标准化用 train 统计量 ----------
    print("\n=== D. 时长标准化用 train 统计量 ===")
    dc = spec["columns"]["duration_ms"]
    d_raw = (static.select("video_id", "duration_ms").with_columns(
        pl.col("duration_ms").fill_null(dc["fill"]))
        .join(pl.DataFrame({"video_id": vids}).with_row_index("_i"),
              on="video_id", how="inner").sort("_i").get_column("duration_ms").to_numpy())
    want_z = (np.log1p(d_raw.astype(np.float64)) - dc["mean"]) / dc["std"]
    check(np.allclose(b["duration"].numpy(), want_z, atol=1e-5),
          "duration == (log1p(填充后原值) − mean) / std，用 spec 的 train 统计量独立算")
    tr = RankingData(proc, p, "train", st)
    tz = tr.batch(rng.choice(len(tr), 20000, replace=False))["duration"].numpy()
    ez = data.batch(rng.choice(len(data), 20000, replace=False))["duration"].numpy()
    check(abs(tz.mean()) < 0.05 and abs(tz.std() - 1.0) < 0.05,
          f"train 段标准化后 mean {tz.mean():+.4f} std {tz.std():.4f} ≈ 0/1")
    check(abs(ez.mean() - tz.mean()) > 1e-4 or abs(ez.std() - tz.std()) > 1e-4,
          f"{sp} 段 mean {ez.mean():+.4f} std {ez.std():.4f} 与 train 不同 "
          "-> 不是各 split 各自标准化（那会是一种泄漏）")
    del tr

    # ---------- E. 掩码与标志自洽 ----------
    print("\n=== E. 掩码与标志自洽 ===")
    h, hm = b["hist"].numpy(), b["hist_mask"].numpy()
    check(np.array_equal(hm, (h != PAD) & (h != OOV)), "hist_mask == (非 PAD 且非 OOV)")
    check(np.array_equal(b["hist_len"].numpy(), hm.sum(1).astype(np.float32)),
          f"hist_len == mask 求和（{sp} 段均值 {hm.sum(1).mean():.2f}/50）")
    tg, tm = b["tags"].numpy(), b["tag_mask"].numpy()
    check(np.array_equal(tm, tg != PAD), "tag_mask == (tags != PAD)")
    check(np.array_equal(b["tag_len"].numpy(), tm.sum(1).astype(np.float32)), "tag_len == mask 求和")
    check(np.array_equal(b["author_known"].numpy(), (b["author"].numpy() > OOV).astype(np.float32)),
          "author_known == (author 行号 > OOV)")
    check(np.array_equal(b["video_id_known"].numpy(),
                         (b["item_idx"].numpy() != OOV).astype(np.float32)),
          "video_id_known == (item_idx != OOV)")

    # ---------- F. 覆盖率与 spec 一致 ----------
    print("\n=== F. 覆盖率与 spec 一致 ===")
    for split in SPLITS:
        dd = RankingData(proc, p, split, st)
        idx = rng.choice(len(dd), min(200000, len(dd)), replace=False)
        bb = dd.batch(idx)
        cov = spec["coverage"][split]
        devs = {f: abs(float(bb[f].numpy().mean()) - cov[f]) for f in KNOWN_FLAGS}
        devs["video_id_known"] = abs(float(bb["video_id_known"].numpy().mean())
                                     - cov["video_id_known"])
        worst = max(devs, key=devs.get)
        # 抽样 20 万行，与全量覆盖率的偏差应在抽样误差内
        check(devs[worst] < 0.005,
              f"{split}: 抽样 {len(idx):,} 行的各标志均值与 spec 全量覆盖率最大偏差 "
              f"{devs[worst]:.4f}（{worst}）")
        del dd, bb

    # ---------- G. 确定性与无副作用 ----------
    print("\n=== G. 确定性与无副作用 ===")
    b3 = data.batch(rows)
    check(all(torch.equal(b3[k], b[k]) for k in b), "同样的行号取两次，逐位相同")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "freeze_manifest.py"), "--check"],
                       capture_output=True, text=True, check=False)
    check(r.returncode == 0, "freeze_manifest --check：装载器只读，召回侧产物未变")

    # ---------- H. 捷径扫描 ----------
    print("\n=== H. 捷径扫描（实测哪些键进了模型，再比 train/valid 强度）===")
    enc_cfg = load_config("configs/single_task.yaml")
    ic = require(enc_cfg, "input")
    dl = {"max_hist": int(ic["max_hist"]), "mask_oov": bool(ic["mask_oov_in_history"])}
    dtr = RankingData(proc, p, "train", st, **dl)
    dva = RankingData(proc, p, "valid", st, **dl)
    enc = RankingEncoder(dtr, enc_cfg)
    NS = min(100000, len(dtr), len(dva))
    btr = dtr.batch(rng.choice(len(dtr), NS, replace=False))
    bva = dva.batch(rng.choice(len(dva), NS, replace=False))

    # H1. 逐个扰动，实测哪些键影响 x —— 清单自维护，不会随代码漂移
    with torch.no_grad():
        x0 = enc(btr).clone()
    in_x, prng = [], np.random.default_rng(3)
    for k, v in btr.items():
        probe = dict(btr)
        if v.dtype in (torch.int64, torch.int32):
            probe[k] = torch.from_numpy(
                prng.integers(0, max(2, int(v.max()) + 1), v.shape).astype(np.int64))
        elif v.dtype == torch.bool:
            probe[k] = ~v
        else:
            probe[k] = torch.from_numpy(prng.random(tuple(v.shape)).astype(np.float32))
        try:
            with torch.no_grad():
                if not torch.equal(enc(probe), x0):
                    in_x.append(k)
        except (ValueError, IndexError, RuntimeError):
            in_x.append(k)          # 扰动后报错也说明它被用到了
    check("item_idx_retrieval" not in in_x,
          f"召回词表行号没有进 x（进 x 的 {len(in_x)} 个键里没有它）—— "
          "它的 train/valid 强度差 0.21，进了就是那条已修的捷径")
    check(not [k for k in in_x if k.startswith("label_")], "没有任何 label_* 进 x")

    # H2. 强度扫描。二维 float 通道（user_t1 52 列 / item_t1 52 列 / user_static_num）
    #     必须逐列展开 —— 只扫标量会漏掉 104 个 T-1 列，而那正是最可能藏问题的地方。
    cols: list[tuple[str, int | None]] = []
    skipped = []
    for k in in_x:
        v = btr[k]
        if k.startswith("label_"):
            continue
        if v.ndim == 1 and v.dtype != torch.bool:
            cols.append((k, None))
        elif v.ndim == 2 and v.dtype == torch.float32:
            cols += [(k, j) for j in range(v.shape[1])]
        else:
            # 类别码（tab/hour）与索引矩阵（tags/hist）不是单调量，AUC 无意义；
            # 它们的"知不知道"由 *_known 与 mask 覆盖。
            skipped.append(k)
    gaps = []
    for k, j in cols:
        for lab in labels:
            vals = []
            for bb in (btr, bva):
                x = (bb[k] if j is None else bb[k][:, j]).numpy().astype(np.float64)
                vals.append(_fast_auc(x, bb[f"label_{lab}"].numpy().astype(np.int64)))
            name = k if j is None else f"{k}[{j}]"
            gaps.append((abs(vals[0] - 0.5) - abs(vals[1] - 0.5), name, lab, *vals))
    gaps.sort(reverse=True)
    worst = [g for g in gaps if g[0] > GAP_FAIL]
    check(not worst,
          f"进 x 的 {len(cols)} 个数值列（含二维展开）x {len(labels)} 个标签 = {len(gaps)} 对，"
          f"强度差全部 <= {GAP_FAIL}（最大 {gaps[0][0]:+.4f}，{gaps[0][1]} vs {gaps[0][2]}）")
    for g, k, lab, a_tr, a_va in worst:
        print(f"       超阈值：{k} vs {lab}  train {a_tr:.4f}  valid {a_va:.4f}  差 {g:+.4f}")
    print(f"    未扫（非单调量，由 *_known / mask 覆盖）：{sorted(set(skipped))}")
    disclose = [g for g in gaps if GAP_WARN < g[0] <= GAP_FAIL]
    print(f"    需披露（{GAP_WARN} < 差距 <= {GAP_FAIL}）共 {len(disclose)} 个，前 5：")
    for g, k, lab, a_tr, a_va in disclose[:5]:
        print(f"      {k:<20} vs {lab:<11} train {a_tr:.4f}  valid {a_va:.4f}  差 {g:+.4f}")

    # H3. 可证伪：这把尺子对着已知的坏通道必须报警，否则它只是装饰
    bad = max(
        abs(_fast_auc(btr["item_idx_retrieval"].numpy().astype(np.float64),
                      btr[f"label_{lab}"].numpy().astype(np.int64)) - 0.5)
        - abs(_fast_auc(bva["item_idx_retrieval"].numpy().astype(np.float64),
                        bva[f"label_{lab}"].numpy().astype(np.int64)) - 0.5)
        for lab in labels)
    check(bad > GAP_FAIL,
          f"可证伪：对着已知的坏通道（召回词表行号）这把尺子给出 {bad:+.4f} > {GAP_FAIL} "
          "-> 它真的能抓到那类泄漏")

    # H4. 那类泄漏的锐利特征：某一侧在 train 上正样本率恰好为 0，而 valid 上不是。
    #     按**取值**分组而不是按数组对象 —— `grp is (v == lo)` 恒为 false（每次新建
    #     ndarray），会让 lo 组永远拿 hi 组的 valid 数据比，lo 侧退化就漏掉了。
    degenerate = []
    for k, j in cols:
        vtr = (btr[k] if j is None else btr[k][:, j]).numpy()
        uniq = np.unique(vtr)
        if len(uniq) != 2:
            continue
        vva = (bva[k] if j is None else bva[k][:, j]).numpy()
        name = k if j is None else f"{k}[{j}]"
        for val in uniq:
            gtr, gva = vtr == val, vva == val
            if gtr.sum() < 100 or gva.sum() < 100:
                continue
            for lab in labels:
                if btr[f"label_{lab}"].numpy()[gtr].mean() != 0.0:
                    continue
                rate = float(bva[f"label_{lab}"].numpy()[gva].mean())
                if rate > 0.01:
                    degenerate.append((name, float(val), lab, rate))
    check(not degenerate,
          f"没有任何二值列在 train 上某一取值的正样本率恰好为 0 而 valid 上不为 0"
          f"（命中 {degenerate[:3]}）—— 这是已修 P0 的锐利特征")

    del dtr, dva, btr, bva, enc

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if failures:
        print("排序装载器验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("验算通过（行对齐 / 物品表已替换 / age 现算 / train 统计 / 掩码 / 覆盖率 / "
          "确定性 / 捷径扫描）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
