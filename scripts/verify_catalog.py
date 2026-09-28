"""独立验算候选库与 request 集合（plan §14/§15）。

候选库的口径错误是**静默**的：门槛没生效、训练段范围取错、正向信号取错，都不会
抛异常，只会让后面所有 Recall 数字建立在一个错误的分母上，而且四种负采样策略
会一起错，对比看上去依然「正常」。

独立性说明（诚实边界）：本脚本不读 `logs_split.parquet` 的 `split` 列，而是从
`logs_main.parquet` 出发、用 data.yaml 的日期边界重算时间窗口后自行统计频次。
因此它独立校验的是**频次统计、门槛过滤、request 筛选**这套逻辑；切分边界本身的
正确性由 `leakage_check` 负责，不在这里重复。

检查分四层：
  A. 内容比对：用独立路径重算候选库与 request，与落盘文件逐行比对。
  B. 可证伪：放宽门槛 / 扩大训练段 / 收窄正向信号后，结果必须改变。
     若某项放宽后结果不变，说明该口径根本没生效，标记为「无法证伪」并判失败。
  C. 不变量：主协议 ⊆ 辅助协议、无重复、request 的 item 必在库内、
     request 的时间必落在评估窗内、meta.json 与实际文件一致。
  D. 回归锁：与 config 里冻结的实测值逐项比对。A/B/C 只能证明实现自洽 —— config
     被改动时生成与重算会**一起**改变，三层全过而口径已经不是原来那套。实测确认
     min_train_freq 与 eval_split 属于这一类漏网口径（train_splits 与 positive_signal
     分别被 B2/B3 挡住），故必须有独立的冻结值断言。

用法：
    python scripts/verify_catalog.py --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.temporal_split import end_of_day_ms
from src.utils.config import load_config, project_path, require

SPLIT_ORDER = ["warmup", "train", "valid", "test"]


def windows(cfg: dict) -> dict[str, tuple[int, int]]:
    """把 data.yaml 的日期边界还原成每个 split 的 [lo, hi) 毫秒区间。"""
    off = require(cfg, "data", "split", "timezone_offset_hours")
    ends = [require(cfg, "data", "split", f"{s}_end") for s in SPLIT_ORDER]
    cuts = [end_of_day_ms(e, off) for e in ends]
    los = [0, *cuts[:-1]]
    return {s: (lo, hi) for s, lo, hi in zip(SPLIT_ORDER, los, cuts)}


def recompute(
    logs: pl.LazyFrame,
    win: dict[str, tuple[int, int]],
    train_splits: list[str],
    eval_split: str,
    signals: list[str],
    min_freq: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """独立路径：按时间窗口（而非 split 列）重算候选库与有效 request。"""
    t = pl.col("time_ms")
    in_train = pl.any_horizontal([(t >= win[s][0]) & (t < win[s][1]) for s in train_splits])
    lo, hi = win[eval_split]
    positive = pl.any_horizontal([pl.col(c) == 1 for c in signals])

    catalog = (
        logs.filter(in_train)
        .group_by("video_id")
        .agg(pl.len().cast(pl.Int32).alias("train_freq"))
        .filter(pl.col("train_freq") >= min_freq)
        .sort("video_id")
        .collect()
    )
    reqs = (
        logs.filter((t >= lo) & (t < hi) & positive)
        .join(catalog.lazy().select("video_id"), on="video_id", how="semi")
        .select("user_id", "video_id", "time_ms", "date", "hourmin", "tab")
        .sort(["user_id", "time_ms", "video_id"])
        .collect()
    )
    return catalog, reqs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument(
        "--require-golden",
        action="store_true",
        help="缺少 catalog.expected 时判失败。流水线应当带上，手工探索可不带。",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)

    train_splits = require(cfg, "catalog", "train_splits")
    eval_split = require(cfg, "catalog", "eval_split")
    k_main = require(cfg, "catalog", "min_train_freq")
    k_aux = require(cfg, "catalog", "aux_min_train_freq")
    signals = require(cfg, "eval", "positive_signal")

    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    src = proc / "logs_main.parquet"
    if not src.is_file():
        raise FileNotFoundError(f"{src} 不存在，请先运行 src.preprocessing.preprocess")

    logs = pl.scan_parquet(src)
    win = windows(cfg)
    failures: list[str] = []
    checks = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        if ok:
            print(f"  OK   {msg}")
        else:
            failures.append(msg)
            print(f"  FAIL {msg}")

    # ---------- A. 内容比对 ----------
    print("\n=== A. 内容比对（独立路径重算 vs 落盘文件）===")
    produced: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {}
    for name, k in (("main", k_main), ("aux", k_aux)):
        cat_raw = pl.read_parquet(proc / f"catalog_{name}.parquet")
        # 落盘顺序是本模块对外的承诺之一：热度基线直接取前 K 行，不重扫日志。
        # 而 K=500 恰好切在并列组中间（第 500 名 train_freq=114，17 个并列，排名
        # 488~504），顺序一变基线数字就变。下面立刻 sort 会抹掉原始顺序，因此必须
        # 先在这里断言，否则排序被改坏验算依然全绿。
        check(
            cat_raw.equals(cat_raw.sort(["train_freq", "video_id"], descending=[True, False])),
            f"{name}: 候选库按 (train_freq desc, video_id asc) 落盘",
        )
        cat_f = cat_raw.sort("video_id")
        # request 无需额外断言：下面比对的 req_r 是排好序的，req_f 未经重排，
        # 顺序错了 equals 直接不成立。
        req_f = pl.read_parquet(proc / f"eval_requests_{name}.parquet")
        cat_r, req_r = recompute(logs, win, train_splits, eval_split, signals, k)
        produced[name] = (cat_f, req_f)
        check(cat_f.equals(cat_r), f"{name}: 候选库逐行一致（{len(cat_f):,} item）")
        check(req_f.equals(req_r), f"{name}: request 逐行一致（{len(req_f):,} 条）")

    # ---------- B. 可证伪 ----------
    print("\n=== B. 可证伪（放宽口径后结果必须改变）===")
    cat_main = produced["main"][0]
    cat_aux = produced["aux"][0]

    # B1 门槛：5 -> 1
    check(
        len(cat_aux) != len(cat_main),
        f"放宽门槛 {k_main}->{k_aux} 后候选库改变（{len(cat_main):,} -> {len(cat_aux):,}）",
    )
    # B2 训练段：train -> warmup+train
    wider = sorted(set(train_splits) | {"warmup"})
    if set(wider) == set(train_splits):
        check(False, "训练段已含 warmup，无法构造更宽的对照 -> 该口径无法证伪")
    else:
        cat_w, _ = recompute(logs, win, wider, eval_split, signals, k_main)
        check(
            len(cat_w) != len(cat_main),
            f"训练段扩为 {wider} 后候选库改变（{len(cat_main):,} -> {len(cat_w):,}）",
        )
    # B3 正向信号：收窄到第一个信号
    if len(signals) > 1:
        _, req_narrow = recompute(logs, win, train_splits, eval_split, signals[:1], k_main)
        check(
            len(req_narrow) != len(produced["main"][1]),
            f"正向信号收窄为 {signals[:1]} 后 request 改变"
            f"（{len(produced['main'][1]):,} -> {len(req_narrow):,}）",
        )
    else:
        check(False, "正向信号只有一个，无法收窄 -> 该口径无法证伪")

    # ---------- C. 不变量 ----------
    print("\n=== C. 不变量 ===")
    check(
        cat_main.join(cat_aux.select("video_id"), on="video_id", how="anti").height == 0,
        "主协议候选库 ⊆ 辅助协议候选库",
    )
    lo, hi = win[eval_split]
    for name, (cat, req) in produced.items():
        check(cat["video_id"].n_unique() == len(cat), f"{name}: 候选库 video_id 无重复")
        check(cat["train_freq"].null_count() == 0, f"{name}: 候选库无 null")
        check(
            req.join(cat.select("video_id"), on="video_id", how="anti").height == 0,
            f"{name}: 每条 request 的目标 item 都在候选库内",
        )
        check(
            bool(req["time_ms"].min() >= lo and req["time_ms"].max() < hi),
            f"{name}: request 全部落在 {eval_split} 窗口 [{lo}, {hi}) 内",
        )
        check(
            req.is_duplicated().sum() == 0,
            f"{name}: request 无完全重复行",
        )

    # meta.json 必须与实际文件对得上 —— 报告直接引用它，错了不会有人发现。
    meta = json.loads((proc / "catalog_meta.json").read_text(encoding="utf-8"))
    for name, (cat, req) in produced.items():
        m = meta["protocols"][name]
        check(m["catalog_size"] == len(cat), f"{name}: meta.catalog_size 与文件一致")
        check(m["eligible_requests"] == len(req), f"{name}: meta.eligible_requests 与文件一致")
        check(
            abs(m["request_coverage"] - len(req) / meta["eval_positive_events"]) < 1e-5,
            f"{name}: meta.request_coverage 自洽（{m['request_coverage']:.1%}）",
        )

    # ---------- D. 口径回归锁 ----------
    # A/B/C 只能证明「生成代码与独立重算一致」。但 config 本身被改动时两边会**一起**
    # 改变，三层全过而口径已经不是原来那套。实测确认过的漏网口径：min_train_freq、
    # eval_split（train_splits 与 positive_signal 分别被 B2/B3 挡住）。
    print("\n=== D. 口径回归锁（冻结实测值）===")
    expected = (cfg.get("catalog") or {}).get("expected")
    unprotected = False
    if not expected:
        if args.require_golden:
            check(False, "未配置 catalog.expected，但本次运行要求 --require-golden")
        else:
            unprotected = True
            print("  WARN 未配置 catalog.expected —— 本次运行没有口径回归保护")
    else:
        # 这两个数不在 produced 里，独立重算，不读 meta。
        t = pl.col("time_ms")
        in_train = pl.any_horizontal([(t >= win[s][0]) & (t < win[s][1]) for s in train_splits])
        lo_e, hi_e = win[eval_split]
        positive = pl.any_horizontal([pl.col(c) == 1 for c in signals])
        actual = {
            "dataset": meta.get("dataset"),
            "train_seen_items": logs.filter(in_train)
            .select(pl.col("video_id").n_unique())
            .collect()
            .item(),
            "eval_positive_events": logs.filter((t >= lo_e) & (t < hi_e) & positive)
            .select(pl.len())
            .collect()
            .item(),
            "main_catalog_size": len(produced["main"][0]),
            "main_eligible_requests": len(produced["main"][1]),
            "aux_catalog_size": len(produced["aux"][0]),
            "aux_eligible_requests": len(produced["aux"][1]),
        }
        for key, want in expected.items():
            if key not in actual:
                check(False, f"catalog.expected 含未知字段 {key!r}")
                continue
            got = actual[key]
            fmt = f"{got:,}" if isinstance(got, int) else repr(got)
            want_fmt = f"{want:,}" if isinstance(want, int) else repr(want)
            check(got == want, f"{key} = {fmt}（冻结值 {want_fmt}）")

    print(f"\n共校验 {checks} 项，失败 {len(failures)} 项。")
    if unprotected:
        print("注意：本次运行未启用口径回归锁，只证明了实现自洽，未证明口径未被改动。")
    if failures:
        print("候选库验算未通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("候选库验算通过（内容比对 / 可证伪 / 不变量 / 回归锁 四层）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
