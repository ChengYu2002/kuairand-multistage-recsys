"""候选库构建：Protocol A (warm) / Protocol B (time-aware available)（plan §14）。

候选库回答的是一个评估问题：**模型做 Top-K 检索时，从哪些 item 里挑？**
这个集合必须对所有模型（ItemCF / 四种负采样下的双塔 / 热度基线）完全一致，
否则 Recall 不可比 —— 而「四种负采样策略的公平对比」正是本项目的研究问题之一。

为什么要卡频次门槛：主建模范围内 313.7 万 item 中 64.1% 只出现过一次。
只出现一次的 item，其 ID embedding、ItemCF 共现、热度统计都不可靠；把它们放进
候选库只会让四种策略的 Recall 一起掉进噪声，差异无法归因。门槛 5 定义的是一个
「模型至少有少量历史证据可学」的 warm-item benchmark。

必须同时披露的限制（§14.4）：门槛让候选库偏向热门 item，且主协议只覆盖 12.1%
的测试正向事件。这个数字有两层成因，披露时不能混为一谈：门槛本身把覆盖率从
21.9% 压到 12.1%（少 53,879 个 request，9.8 个百分点）；而 21.9% 这个上限来自
「训练期必须见过」的前提 —— 测试期 78.1% 的正向事件其目标 item 在训练段一次都
没出现过。调门槛最多拉回 21.9%，再往上只能换协议。因此
本模块同时产出辅助协议（门槛=1），用于验证结论是否随门槛变化；真正的冷启动覆盖
要靠 Protocol B + side-feature 物品塔回答，不在本模块范围内。

产出（processed_dir 下）：
    catalog_main.parquet        video_id, train_freq   —— 主协议候选库
    catalog_aux.parquet         同上                   —— 辅助协议候选库
    eval_requests_main.parquet  一行 = 一个 retrieval request
    eval_requests_aux.parquet   同上
    catalog_meta.json           口径与规模，供报告直接引用

catalog 按 (train_freq desc, video_id asc) 排序：既是确定性顺序，也让热度基线
可以直接取前 K 行，无需重扫日志。

用法：
    python -m src.preprocessing.build_catalog --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse
import json
import time

import polars as pl

from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)


def apply_threshold(
    freq: pl.DataFrame, requests: pl.DataFrame, min_freq: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """按频次门槛切出候选库，并保留目标落在库内的 request。

    request 的过滤用 semi join 而非 is_in：后者在百万级列表上既慢又会触发
    polars 的 ambiguity 警告。
    """
    catalog = (
        freq.filter(pl.col("train_freq") >= min_freq)
        # 全序键，保证同频次时顺序仍然唯一 —— 可复现性依赖这一点（§26）。
        .sort(["train_freq", "video_id"], descending=[True, False])
    )
    eligible = (
        #这里的request就是test
        # semi join 只保留左边 requests 原有的列
        requests.join(catalog.select("video_id"), on="video_id", how="semi")
        # polars 不保证 semi join 保序；顺序必须显式固定，否则两次运行产出的文件
        # 可能逐字节不同，验收标准「同一 config 跑两次结果完全一致」就成了运气问题。
        .sort(["user_id", "time_ms", "video_id"])
    )
    return catalog, eligible


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument(
        "--data-config",
        default=None,
        help="流水线传入的数据 config；给定时校验它与 --config 里 data_config 指向同一文件。",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)

    # 本模块的数据来源由 retrieval.yaml 的 data_config 决定。若流水线的 $CFG 换成了
    # 27K 而 $RCFG 仍指向 1K，候选库会**静默地**建在错误的数据集上，且与其余步骤
    # 产出的文件互不匹配 —— 比门槛写错更难发现，因此在这里直接挡掉。

    # 防止--config：召回实验配置和--data-config：本次流水线使用的数据配置 指向不同路径数据
    ref = cfg.get("data_config")
    if args.data_config is not None:
        if ref is None:
            raise ValueError(f"{args.config} 没有 data_config 字段，无法与 --data-config 校验")
        got, want = project_path(ref).resolve(), project_path(args.data_config).resolve()
        if got != want:
            raise ValueError(
                f"数据 config 不一致：{args.config} 指向 {got}，但流水线传入的是 {want}。"
                "换数据集时 data config 与 retrieval config 必须成对更换。"
            )

    # Protocol B（time-aware available）尚未实现。若不拦截，配成 B 会得到 A 的数据
    # 却在 meta 里被标成 B，是典型的静默错标。
    protocol = require(cfg, "catalog", "protocol")
    if protocol != "A":
        raise NotImplementedError(
            f"catalog.protocol={protocol!r} 尚未实现，当前仅支持 Protocol A (warm catalog)。"
        )

    # 从配置中读取实验口径
    train_splits = require(cfg, "catalog", "train_splits")
    eval_split = require(cfg, "catalog", "eval_split")
    k_main = require(cfg, "catalog", "min_train_freq")
    k_aux = require(cfg, "catalog", "aux_min_train_freq")
    signals = require(cfg, "eval", "positive_signal")

    # 辅助协议必须真的更宽松，否则「放宽门槛后结论是否变化」这个检查无从谈起。
    if k_aux >= k_main:
        raise ValueError(f"aux_min_train_freq({k_aux}) 必须小于 min_train_freq({k_main})")
    if eval_split in train_splits:
        raise ValueError(f"eval_split({eval_split}) 不能同时出现在 train_splits({train_splits})")

    # 数据路径来自被引用的 data.yaml（load_config 会把它并入 cfg["data"]）。
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    src = proc / "logs_split.parquet"
    if not src.is_file():
        raise FileNotFoundError(f"{src} 不存在，请先运行 src.preprocessing.temporal_split")

    t0 = time.perf_counter()
    lf = pl.scan_parquet(src)
    # signal中信号为1）的 比如is_click，long_view都是postive
    positive = pl.any_horizontal([pl.col(c) == 1 for c in signals])

    # 1. 训练段频次。只数 train_splits 指定的段，这是候选库口径的全部来源。
    # 聚合统计
    freq = (
        lf.filter(pl.col("split").is_in(train_splits))
        .group_by("video_id")
        .agg(pl.len().cast(pl.Int32).alias("train_freq"))
        .collect()
    )
    log.info("训练段(%s) 出现过的 item: %s 个", "+".join(train_splits), f"{len(freq):,}")

    # 2. 评估段的全部正向事件 = 候选 request 池（尚未按候选库过滤）。
    #    带上 date / hourmin / tab：下游要靠 date join T-1 特征，靠 tab+小时做 context。
    # 取出 test 的全部正向事件，split == test  并且 is_click == 1 或 long_view == 1
    requests = (
        lf.filter((pl.col("split") == eval_split) & positive)
        .select("user_id", "video_id", "time_ms", "date", "hourmin", "tab")
        .sort(["user_id", "time_ms", "video_id"])
        .collect()
    )
    n_pos = len(requests)
    log.info("%s 段正向事件(%s): %s 个", eval_split, " or ".join(signals), f"{n_pos:,}")

    # 3. 两套门槛各切一份。
    # 只是这次实验的“说明书”，记录 这次候选库实验用了什么规则，以及数据规模是多少。
    meta: dict[str, object] = {
        "dataset": require(cfg, "data", "dataset", "name"),
        "protocol": protocol,
        "train_splits": train_splits,
        "eval_split": eval_split,
        "positive_signal": signals,
        "train_seen_items": len(freq),
        "eval_positive_events": n_pos,
        "protocols": {},
    }
    written: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {}

    print()
    print(f"{'协议':<8}{'门槛':>6}{'候选库 item':>14}{'有效 request':>15}{'request 覆盖率':>16}")
     # 根据门槛生成候选库和考题 第一次5，第二次1
    for name, k in (("main", k_main), ("aux", k_aux)):
        # 同时产生 各自的，catalog和eligible
        catalog, eligible = apply_threshold(freq, requests, k)
        if catalog.is_empty() or eligible.is_empty():
            raise AssertionError(f"协议 {name}(门槛={k}) 切出了空集合，口径一定有问题")
        # 硬不变量：候选库里不允许出现低于门槛的 item。
        if catalog["train_freq"].min() < k:
            raise AssertionError(f"协议 {name} 的候选库含 train_freq < {k} 的 item")
        # 计算覆盖率
        cov = len(eligible) / n_pos
        # 把统计数字存入 meta
        meta["protocols"][name] = {
            "min_train_freq": k,
            "catalog_size": len(catalog),
            "eligible_requests": len(eligible),
            "request_coverage": round(cov, 5),
        }
        # 保存实际 DataFrame
        written[name] = (catalog, eligible)
        print(f"{name:<8}{'>=' + str(k):>6}{len(catalog):>14,}{len(eligible):>15,}{cov:>15.1%}")
    print()

    # 门槛更松的一侧必须是超集；不成立说明过滤逻辑写反了。
    cat_main, req_main = written["main"]
    cat_aux, req_aux = written["aux"]
    if len(cat_aux) <= len(cat_main) or len(req_aux) <= len(req_main):
        raise AssertionError("放宽门槛后规模没有变大，门槛未生效")

    # 如果所有主候选视频都位于辅助库，height = 0；如果height > 0，anti join 找到异常视频，报错
    # anti join 的意思是：
    # 从左表 cat_main 中，找出无法在右表 cat_aux 匹配到的视频
    if cat_main.join(cat_aux.select("video_id"), on="video_id", how="anti").height:
        raise AssertionError("主协议候选库不是辅助协议的子集")

    # 4. 落盘。meta 单独存 json —— 报告里要引用这四个数，只打印在终端等于没存。
    for name, (catalog, eligible) in written.items():
        catalog.write_parquet(proc / f"catalog_{name}.parquet", compression="zstd")
        eligible.write_parquet(proc / f"eval_requests_{name}.parquet", compression="zstd")
    # generated_at 只作溯源用。它使 catalog_meta.json 每次运行都不同，因此
    # 「两次运行逐字节一致」这条验收只针对四个 parquet，不含本文件。
    meta["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (proc / "catalog_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log.info(
        "已写出 catalog_{main,aux}.parquet / eval_requests_{main,aux}.parquet / catalog_meta.json (%.1fs)",
        time.perf_counter() - t0,
    )
    log.info(
        "主协议：%s item / %s request / 覆盖 %.1f%% —— 报告中必须与 Recall 一并披露",
        f"{len(cat_main):,}",
        f"{len(req_main):,}",
        100 * meta["protocols"]["main"]["request_coverage"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
