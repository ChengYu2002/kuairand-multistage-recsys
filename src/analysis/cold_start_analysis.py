"""按物品训练期曝光频次分层的召回分析（plan §3.3 / §13.4 的 Warm / Cold / Tail）。

## Protocol A 下没有 cold item —— 这是构造决定的，不是数据缺失

候选库的定义是「train 段出现过、且 `train_freq >= 5`」（`catalog.min_train_freq`），
实测候选库 `train_freq` 的最小值就是 5，66,536 条考题的答案**全部**落在候选库内
（库外 0 条）。所以「冷启动」那一列在 Protocol A 下无法填，它需要 Protocol B
（time-aware available catalog，含训练期从未出现的新 item），而
`catalog.upload_rule` 在 configs/retrieval.yaml 里标着「Protocol B 用，未实现」。

这不是小事：README §6 记录的天花板正来自这里 —— 测试期 **78.1%** 的正向事件，其目标
item 在训练段一次都没出现过，这些事件连考题都进不了。本脚本量的是**剩下那 21.9%
里部的频次梯度**，不是冷启动。两者不能混为一谈。

## 分层口径（在看到任何 Recall 数字**之前**定好）

按**考题答案**的 `train_freq` 切三段。切点是绝对值而非分位数 —— 分位数会随候选库变化
（aux 协议 `min_train_freq=1`、将来的 Protocol B），那样跨协议的数字就不可比了。
切点由答案质量分布选出，三段的考题量尽量均衡：

    tail   5 <= train_freq < 10     22,675 条（34.1%）
    mid    10 <= train_freq < 30    27,928 条（42.0%）
    warm   train_freq >= 30         15,933 条（24.0%）

## 层间的分母必须是「随机猜」，不能是热度基线

层间的 Recall **不能直接比**：warm 层的答案本来就更容易命中。需要一个与层无关的分母。

**热度基线不能当这个分母** —— 实测它在 tail 与 mid 层恒为 0，而且这是结构性的：
热度 Top-500 全部是 `train_freq >= 114` 的 item，按定义都落在 warm 层，所以它永远
打不中 tail/mid 的答案。拿 0 作分母无意义。（这一层的实测值仍然照报，它本身就是
一条信息：热度基线的全部战绩都来自 24% 的 warm 考题。）

**随机猜**才是正确的分母：每条考题只有一个正确答案，从 19.4 万候选里随机取 K 个命中
的概率是 `K / |catalog|`，与该答案属于哪一层无关。于是「倍随机猜」这一列跨层可比，
它量的正是「模型相对无信息的基准，在这一层贡献了多少信息」。

## 复用而非复制

Top-K 来自 `src.retrieval.two_tower.request_topk`（从 `evaluate_checkpoint` 里抽出的
同一份计算），指标来自 `src.evaluation.retrieval_metrics.evaluate`。这里不自己实现
任何度量 —— 否则分层表和总表可能各自正确却互相矛盾，而两张表看上去都正常。

用法：
    python -m src.analysis.cold_start_analysis
    python -m src.analysis.cold_start_analysis --strategies random exposure --force
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl
import torch

from src.evaluation.retrieval_metrics import evaluate
from src.retrieval.dataset import RetrievalData
from src.retrieval.two_tower import TwoTower, request_topk
from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)

# 切点在看到 Recall 之前定好，之后不得调整（调了就是对着结果挑分层）。
BANDS = (("tail", 5, 10), ("mid", 10, 30), ("warm", 30, None))
STRATEGY_ORDER = ("random", "inbatch", "inbatch_logq", "hybrid", "exposure")


def band_of(freq: np.ndarray) -> np.ndarray:
    out = np.full(len(freq), "", dtype=object)
    for name, lo, hi in BANDS:
        m = (freq >= lo) if hi is None else ((freq >= lo) & (freq < hi))
        out[m] = name
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    ap.add_argument("--tower", default=None, choices=("id_only", "id_side"))
    ap.add_argument("--strategies", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--results", default="results")
    ap.add_argument(
        "--force", action="store_true", help="允许局部运行覆盖范围更大的现有汇总"
    )
    a = ap.parse_args()

    cfg = load_config(a.config)
    # --tower 不只决定 checkpoint 文件名，也必须决定模型结构和按日期计算
    # item side features 的评估路径。未显式传入时必须尊重 YAML；若默认写死
    # id_only，会把 YAML 中的 id_side 静默换回另一组 checkpoint。
    tt = require(cfg, "two_tower")
    tower = a.tower or tt["item_tower_config"]
    tt["item_tower_config"] = tower
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    results, experiments = project_path(a.results), project_path("experiments")
    k_list = require(cfg, "eval", "k_list")
    catalog = pl.read_parquet(proc / f"catalog_{a.protocol}.parquet")
    cat_ids = catalog.get_column("video_id").to_numpy()
    cat_freq = catalog.get_column("train_freq").to_numpy()
    if not bool((np.diff(cat_freq) <= 0).all()):
        raise AssertionError(
            "候选库不是按 train_freq 降序存的，热度基线的 Top-K 不能取前 K 行。"
            "这会让每一层的基线全错，而表看上去正常。"
        )

    reqs = pl.read_parquet(proc / f"eval_requests_{a.protocol}.parquet")
    ans = reqs.select("video_id", "user_id").join(
        catalog.select("video_id", "train_freq"), on="video_id", how="left"
    )
    if ans["train_freq"].null_count():
        raise AssertionError(
            f"{ans['train_freq'].null_count()} 条考题的答案不在候选库内 —— "
            "Protocol A 下不该出现，分层会漏掉它们。"
        )
    freq = ans.get_column("train_freq").to_numpy()
    bands = band_of(freq)
    targets = reqs.get_column("video_id")
    users = reqs.get_column("user_id").to_numpy()
    n = len(reqs)
    counts = {b: int((bands == b).sum()) for b, _, _ in BANDS}
    if sum(counts.values()) != n:
        raise AssertionError(f"分层未覆盖全部考题：{counts} 之和 != {n}")
    log.info(
        "考题分层：%s（共 %s）",
        {b: f"{c:,} ({c / n:.1%})" for b, c in counts.items()},
        f"{n:,}",
    )

    kmax = max(k_list)
    out: dict = {
        "protocol": a.protocol,
        "item_tower_config": tower,
        "bands": {
            b: {"lo": lo, "hi": hi, "n_requests": counts[b]} for b, lo, hi in BANDS
        },
        "cold_band": None,
        "cold_band_note": (
            "Protocol A 按构造无 cold item（catalog.min_train_freq=5，"
            "实测候选库 train_freq 最小值 5，库外考题 0 条）。cold 需要 "
            "Protocol B（catalog.upload_rule，未实现）。"
        ),
        "popularity": {},
        "models": {},
    }

    # ---- 随机猜：与层无关的分母。每条考题一个答案，命中概率 = K / |catalog| ----
    n_cat = len(cat_ids)
    rand_guess = {f"recall@{k}": k / n_cat for k in k_list}
    out["random_guess"] = rand_guess
    print("\n=== 随机猜（与层无关，用作跨层可比的分母）===")
    print("".join(f"{'R@' + str(k):>14}" for k in k_list))
    print(
        "".join(f"{rand_guess[f'recall@{k}']:>14.6f}" for k in k_list)
        + f"   = K / {n_cat:,}"
    )

    # ---- 热度基线：照报，但**不作分母** —— 它在 tail/mid 层结构性为 0 ----
    pop_row = cat_ids[:kmax]
    print("\n=== 热度基线按层（不作分母：tail/mid 结构性为 0）===")
    print(
        f"{'band':<8}{'n_requests':>12}"
        + "".join(f"{'R@' + str(k):>12}" for k in k_list)
    )
    for b, _, _ in BANDS:
        m = bands == b
        t = evaluate(
            np.tile(pop_row, (int(m.sum()), 1)),
            targets.filter(pl.Series(m)),
            users[m],
            k_list,
        )
        row = {
            f"recall@{k}": float(
                t.filter(
                    (pl.col("metric") == "recall") & (pl.col("k") == k)
                ).get_column("per_request")[0]
            )
            for k in k_list
        }
        out["popularity"][b] = row
        print(
            f"{b:<8}{counts[b]:>12,}"
            + "".join(f"{row[f'recall@{k}']:>12.5f}" for k in k_list)
        )
    print(
        "  热度基线的全部战绩来自 warm 层（24% 的考题）—— 它的 Top-500 按定义全是"
        " train_freq >= 114 的 item，打不中 tail/mid。"
    )

    # ---- 各模型 ----
    strategies = a.strategies or [s for s in STRATEGY_ORDER]
    data = RetrievalData(proc, a.protocol, "test")  # 只读，各 checkpoint 共用
    for strat in strategies:
        per_seed: dict[str, dict[str, list[float]]] = {
            b: {f"recall@{k}": [] for k in k_list} for b, _, _ in BANDS
        }
        used = []
        for seed in a.seeds:
            tag = f"{strat}_{tower}_{a.protocol}_seed{seed}"
            ck = experiments / f"two_tower_{tag}.pt"
            if not ck.exists():
                log.warning("缺 %s，跳过", ck.name)
                continue
            st = torch.load(ck, weights_only=False)
            model = TwoTower(data, cfg)
            model.load_state_dict(st["state_dict"])
            model.eval()
            topk, r2 = request_topk(model, cfg, proc, a.protocol, data=data)
            if len(r2) != n:
                raise AssertionError(f"{tag} 的 request 数 {len(r2)} != {n}")
            for b, _, _ in BANDS:
                m = bands == b
                t = evaluate(topk[m], targets.filter(pl.Series(m)), users[m], k_list)
                for k in k_list:
                    per_seed[b][f"recall@{k}"].append(
                        float(
                            t.filter(
                                (pl.col("metric") == "recall") & (pl.col("k") == k)
                            ).get_column("per_request")[0]
                        )
                    )
            used.append(seed)
            del model, st, topk
        if not used:
            continue
        out["models"][strat] = {"seeds": used, "bands": {}}
        print(f"\n=== {strat}（{len(used)} seeds，mean ± std，per_request）===")
        print(
            f"{'band':<8}"
            + "".join(f"{'R@' + str(k):>22}" for k in k_list)
            + f"{'倍随机猜':>16}"
        )
        for b, _, _ in BANDS:
            row, line = {}, f"{b:<8}"
            for k in k_list:
                v = np.array(per_seed[b][f"recall@{k}"], float)
                row[f"recall@{k}"] = {
                    "mean": float(v.mean()),
                    "std": float(v.std(ddof=1)) if len(v) > 1 else None,
                    "values": v.tolist(),
                }
                line += (
                    f"{v.mean():>14.5f}±{v.std(ddof=1):.5f}"
                    if len(v) > 1
                    else f"{v.mean():>15.5f}   n/a"
                )
            ref = f"recall@{k_list[len(k_list) // 2]}"
            row["x_random_guess_at_ref"] = row[ref]["mean"] / rand_guess[ref]
            row["x_popularity_at_ref"] = (
                row[ref]["mean"] / out["popularity"][b][ref]
                if out["popularity"][b][ref]
                else None
            )
            row["ref_metric"] = ref
            out["models"][strat]["bands"][b] = row
            line += f"{row['x_random_guess_at_ref']:>15.1f}x"
            print(line)

    dst = results / f"cold_start_summary_{a.protocol}_{tower}.json"
    # 局部运行（--strategies / --seeds）不得覆盖更完整的汇总：文件名不含范围，
    # 而 README 直接引用这个文件。实测踩过一次 —— 一次 1 策略 1 seed 的调试运行
    # 把 4 策略 x 3 seeds 的结果清掉了，且没有任何报错。
    if dst.exists() and not a.force:
        try:
            old_cov = json.loads(dst.read_text("utf-8")).get("models", {})
        except (ValueError, OSError):
            old_cov = {}
        new_cov = out["models"]
        lost = {
            m: sorted(
                set(v.get("seeds", [])) - set(new_cov.get(m, {}).get("seeds", []))
            )
            for m, v in old_cov.items()
        }
        lost = {m: s for m, s in lost.items() if s or m not in new_cov}
        if lost:
            raise SystemExit(
                f"拒绝覆盖 {dst.name}：现有文件覆盖的范围更大，本次会丢掉 {lost}。"
                "局部运行请加 --force 明确表示就是要覆盖，或先把现有文件挪开。"
            )
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("汇总已写出 %s", dst.name)
    print("\n注：cold 一列在 Protocol A 下无法填 —— " + out["cold_band_note"])
    print(
        "注：层间 Recall 不可直接比（warm 层的答案本来更易命中）；跨层能比的是"
        "「倍随机猜」这一列 —— 随机猜的命中率 K/|catalog| 与答案属于哪一层无关。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
