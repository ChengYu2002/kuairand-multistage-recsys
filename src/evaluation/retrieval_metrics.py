"""Recall@K / HitRate@K / NDCG@K（plan §15）。

这是召回侧唯一的度量入口：ItemCF、四种负采样下的双塔、热度基线全部走同一份代码，
否则「四种策略的公平对比」无从谈起 —— 指标实现只要有一点不同，差异就无法归因。

**本模块是纯函数，不打分、不碰模型、不加载日志。** 它只接收「每个 request 的 Top-K
排好序的 item id」和「每个 request 的正确答案」。全库打分（51,746 个查询 × 194,310
个候选 = 40 GB 的分数矩阵，必须分批）属于模型侧的事，不在这里。

关于 Recall 与 HitRate：
    Recall@K   = 前 K 命中的正确答案数 / 正确答案总数
    HitRate@K  = 前 K 里只要命中任意一个就是 1
当前协议（§15：request = 每个正向事件）下每个 request 只有一个正确答案，两者**恒等**，
因此结果表只列 Recall。但本模块按「一组正确答案」实现，因为 66,536 个正向事件只落在
51,746 个不同的 (user_id, time_ms) 上（39.9% 的 request 与他人共享查询时刻）：若把
request 改判为「每个查询时刻」，一个 request 就会有 2~6 个正确答案，两个指标随即分离。
多目标情形下 NDCG 的 IDCG 必须真算（前 min(K, n_targets) 个位置的折扣和），不能省成 1。

两种平均都要报（§15）：用户活跃度差 7,900 倍，只报按 request 平均会被头部用户主导。
实测主协议覆盖 966 个用户（34 个用户的正向目标全不在候选库内，整体掉出评估），
每用户中位 50 个 request，最多 772 个，最活跃 50 人贡献 20.5% 的 request。

用法（自校准：复现热度基线）：
    python -m src.evaluation.retrieval_metrics --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import polars as pl

from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)

# 一次处理多少个 request。topk_keys 是 (block, K_max) 的 int64，block=8192、K_max=500
# 时约 33 MB —— 全量一次性展开会到数百 MB，没必要。
_BLOCK = 8192


def _flatten_targets(targets: Sequence | pl.Series, n: int) -> tuple[np.ndarray, np.ndarray]:
    """把「每个 request 的正确答案」规范成 (展开后的 item, 每个 request 的答案数)。

    接受三种写法：一维 id 数组 / List 类型的 polars Series / 列表的列表。
    """
    if isinstance(targets, pl.Series) and targets.dtype == pl.List:
        counts = targets.list.len().to_numpy().astype(np.int64)
        item = targets.explode().drop_nulls().to_numpy().astype(np.int64)
    elif isinstance(targets, pl.Series):
        item = targets.to_numpy().astype(np.int64)
        counts = np.ones(len(item), dtype=np.int64)
    else:
        arr = list(targets)
        if arr and isinstance(arr[0], (list, tuple, set, np.ndarray)):
            counts = np.array([len(x) for x in arr], dtype=np.int64)
            item = np.fromiter(
                (i for x in arr for i in x), dtype=np.int64, count=int(counts.sum())
            )
        else:
            item = np.asarray(arr, dtype=np.int64)
            counts = np.ones(len(item), dtype=np.int64)
    # counts 必须由 targets 的真实长度得出。若写成 np.ones(n)，下面这条校验就永远成立，
    # 而 numpy 会把短一截的 item 广播开 —— 模型输出与考题错位时指标照样算得出来。
    if len(counts) != n:
        raise ValueError(f"targets 覆盖 {len(counts)} 个 request，与 topk 的 {n} 行不一致")
    if len(item) != int(counts.sum()):
        raise ValueError(f"展开后的 target 数 {len(item)} 与 counts 之和 {int(counts.sum())} 不一致")
    if (counts == 0).any():
        raise ValueError("存在没有正确答案的 request —— 这类 request 应在构建考题时剔除")
    return item, counts


# item + counts
#       ↓
# 还原每个答案属于哪个 request
#       ↓
# 将 (request_id, video_id) 编码成 key
#       ↓
# 正确答案 keys 排序
#       ↓
# Top-K 也编码成相同形式 （两边大小不需要一样，因为是在做逐元素的“是否存在”查询，不是按下标一一配对。）
#       ↓
# 二分查找 Top-K key 是否存在于正确答案 keys
#       ↓
# 生成 True/False 命中矩阵
def _hit_matrix(topk: np.ndarray, item: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """(n, K_max) 的布尔矩阵：第 i 行第 j 列 = topk[i, j] 是否是第 i 个 request 的正确答案。

    做法是把 (request 下标, item id) 编码成单个 int64 键后二分查找。比逐行建 set 快，
    且天然支持一个 request 有多个正确答案。
    """
    n, k_max = topk.shape
    base = int(max(topk.max(initial=0), item.max(initial=0))) + 1
    req_idx = np.repeat(np.arange(n, dtype=np.int64), counts)
    keys = np.sort(req_idx * base + item)
    # 同一个 request 的正确答案列表里出现重复，会同时放大 Recall 的分母与 IDCG。
    if keys.size > 1 and (np.diff(keys) == 0).any():
        raise ValueError("同一个 request 的正确答案出现重复 —— 每组 target 必须互不相同")

    hits = np.empty((n, k_max), dtype=bool)
    for lo in range(0, n, _BLOCK):
        hi = min(lo + _BLOCK, n)
        # Top-K 行内必须互不相同。只靠「Recall > 1」只能抓到重复项恰好是正确答案的情形；
        # 重复的**非**正确答案不会触发它，却会把后面每个 item 的有效排名整体往后推，
        # 使 NDCG 偏小 —— 同样是静默的。整块 sort+diff 在 66,536x500 上约 0.2 秒。
        srt = np.sort(topk[lo:hi], axis=1)
        dup_rows = (np.diff(srt, axis=1) == 0).any(axis=1)
        if dup_rows.any():
            bad = lo + int(np.argmax(dup_rows))
            raise ValueError(f"第 {bad} 行 Top-K 含重复 item —— 每行必须是互不相同的 item")
        blk = np.arange(lo, hi, dtype=np.int64)[:, None] * base + topk[lo:hi].astype(np.int64)
        # 二分查找 快速查找Top-K 里的每个视频，是不是当前 request 的正确答案。
        pos = np.searchsorted(keys, blk)
        np.clip(pos, 0, len(keys) - 1, out=pos)
        hits[lo:hi] = keys[pos] == blk
    return hits


def _per_user_mean(values: np.ndarray, user_ids: np.ndarray) -> float:
    """先在用户内部取均值，再对用户取均值 —— 每个用户等权，不受活跃度影响。"""
    return (
        pl.DataFrame({"u": user_ids, "v": values})
        .group_by("u")
        .agg(pl.col("v").mean())
        .get_column("v")
        .mean()
    )


def evaluate(
    topk: np.ndarray,
    targets: Sequence | pl.Series,
    user_ids: np.ndarray,
    k_list: Sequence[int],
) -> pl.DataFrame:
    """计算召回指标。

    topk      (n_requests, K_max) 整数数组，每行按分数从高到低排好的 item id。
    targets   长度 n_requests；每项是该 request 的正确答案（单个 id 或一组 id）。
    user_ids  长度 n_requests，用于按用户平均。
    k_list    要报告的 K，全部不得超过 topk 的列数。

    返回 metric / k / per_request / per_user 四列的长表。
    """
    topk = np.asarray(topk)
    if topk.ndim != 2:
        raise ValueError(f"topk 必须是二维 (n_requests, K)，收到 {topk.shape}")
    n, k_max = topk.shape

    user_ids = np.asarray(user_ids)
    if len(user_ids) != n:
        raise ValueError(f"user_ids 长度 {len(user_ids)} 与 request 数 {n} 不一致")

    ks_raw = [int(k) for k in k_list]
    if not ks_raw:
        raise ValueError("k_list 不能为空")
    if len(set(ks_raw)) != len(ks_raw):
        raise ValueError(f"k_list 含重复值: {ks_raw}")
    if min(ks_raw) < 1:
        raise ValueError(f"k_list 必须全部 >= 1，收到 {ks_raw}")
    ks = sorted(ks_raw)
    if ks[-1] > k_max:
        raise ValueError(f"要求 K={ks[-1]} 但 topk 只有 {k_max} 列")

    # 整理正确答案：item 是所有答案，counts 是每道题的答案数
    #     targets = [
    #     [10, 20],      # request 0 有2个答案
    #     [30],          # request 1 有1个答案
    #     [40, 50, 60],  # request 2 有3个答案
    # ]
    #     item = [10, 20, 30, 40, 50, 60]
    #     counts = [2, 1, 3]

    # 它并没有丢失分组信息，因为 counts 记录了切分位置：
    # item:
    # [10, 20 | 30 | 40, 50, 60]
    #     2      1       3
    # 判断 Top-K 每个位置是否命中，输出 bool 矩阵
    item, counts = _flatten_targets(targets, n)

    hits = _hit_matrix(topk, item, counts)

    # 位置 j（0 起）的折扣是 1/log2(j+2)，即排名 r=j+1 时的 1/log2(r+1)。
    # 正确答案排第1名得1分，排得越靠后，得分越低。
    discount = (1.0 / np.log2(np.arange(2, k_max + 2))).astype(np.float64)

    # idcg_table[m] = 根据disocout，最理想情况下前 m 个位置全中时的 DCG。
    idcg_table = np.concatenate([[0.0], np.cumsum(discount)])

    # 收集不同 K 下的指标结果
    rows = []
    for k in ks:
        # 只看 Top-K 内的命中情况
        h = hits[:, :k]
        # 每个 request 命中了几个正确答案
        n_hit = h.sum(axis=1)
        # 实际排名得分：命中越靠前，DCG 越高
        dcg = h @ discount[:k]
        # 理想排名得分：正确答案全部排在最前面
        idcg = idcg_table[np.minimum(counts, k)]
        # 每个 request 的召回率
        recall = n_hit / counts
        # 兜底：重复项已在 _hit_matrix 中拦截，走到这里说明还有未预料的路径。
        if (recall > 1.0 + 1e-9).any():
            raise ValueError(
                f"K={k} 时出现 Recall > 1：Top-K 内含重复的正确答案。"
                "每行 Top-K 必须是互不相同的 item。"
            )
        # 保存每个 request 的三个指标
        per_req = {
            "recall": recall,
            "hitrate": (n_hit > 0).astype(np.float64),
            "ndcg": dcg / idcg,
        }
        # 同时计算按 request 平均和按用户平均
        for metric, v in per_req.items():
            rows.append(
                {
                    "metric": metric,
                    "k": k,
                    "per_request": float(v.mean()),
                    "per_user": float(_per_user_mean(v, user_ids)),
                }
            )
    return pl.DataFrame(rows)


def popularity_topk(catalog: pl.DataFrame, n_requests: int, k_max: int) -> np.ndarray:
    """热度基线：所有 request 推同一份榜单。

    catalog 必须已按 (train_freq desc, video_id asc) 排好 —— build_catalog 保证了这一点，
    verify_catalog 的 A 层有断言。K=500 恰好切在并列组中间（第 500 名 train_freq=114，
    17 个并列），顺序不固定这个基线就不是唯一值。
    """
    if len(catalog) < k_max:
        raise ValueError(f"候选库只有 {len(catalog)} 个 item，不足 K={k_max}")
    # .head取最大
    top = catalog.get_column("video_id").head(k_max).to_numpy()
    return np.tile(top, (n_requests, 1))


def format_table(df: pl.DataFrame, metrics: Sequence[str] = ("recall", "ndcg")) -> str:
    """默认不列 hitrate —— 单目标协议下它与 recall 恒等，两列一样的数字只会误导读者。"""
    df = df.filter(pl.col("metric").is_in(list(metrics)))
    lines = [f"{'metric':<10}{'K':>6}{'按 request 平均':>18}{'按用户平均':>16}"]
    for r in df.iter_rows(named=True):
        lines.append(
            f"{r['metric']:<10}{r['k']:>6}{r['per_request']:>18.5f}{r['per_user']:>16.5f}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    args = ap.parse_args()
    cfg = load_config(args.config)

    k_list = require(cfg, "eval", "k_list")
    # 父目录
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    catalog = pl.read_parquet(proc / f"catalog_{args.protocol}.parquet")
    reqs = pl.read_parquet(proc / f"eval_requests_{args.protocol}.parquet")

    log.info(
        "协议 %s：候选库 %s item，考题 %s 条，覆盖 %s 个用户",
        args.protocol,
        f"{len(catalog):,}",
        f"{len(reqs):,}",
        f"{reqs['user_id'].n_unique():,}",
    )

    topk = popularity_topk(catalog, len(reqs), max(k_list))
    df = evaluate(topk, reqs.get_column("video_id"), reqs.get_column("user_id").to_numpy(), k_list)

    print("\n热度基线（所有 request 推同一份榜单）")
    print(format_table(df))
    print()

    # 与其并排打印两列一样的数字让读者自己比对，不如直接把恒等关系断言出来。
    wide = df.pivot(on="metric", index="k", values="per_request")
    identical = bool(np.allclose(wide["recall"].to_numpy(), wide["hitrate"].to_numpy()))
    log.info("Recall ≡ HitRate（每个 request 只有一个正确答案）：%s —— 结果表只列前者。", identical)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
