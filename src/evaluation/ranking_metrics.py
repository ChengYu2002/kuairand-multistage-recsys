"""排序指标：AUC / GAUC / PCOC / 校准（plan §27）。

排序侧唯一的度量入口。Single-Task、MMoE、PLE、Selective Sharing 全部走这里，
否则 §27.1 的 ΔAUC 会混进指标实现的差异。

## 为什么自己实现 AUC 而不是直接调 sklearn

不是为了造轮子，是为了让验算有**两条独立路径**：这里用基于秩的实现
（Mann-Whitney U），验算脚本拿 sklearn 做交叉比对。两边对上才说明尺子可信。
并列必须用**平均秩**处理 —— 模型在初始化时或饱和时会产出大量相同的预测值，
用普通排序会让 AUC 偏离真值。

## GAUC 的两个坑

1. **标签恒定的用户必须排除** —— 他们的 AUC 无定义。稀疏任务上这个比例极高：
   实测 test 段 is_follow 只有 364/983 个用户可算（63% 被排除），
   is_comment 415/983。因此**必须与指标一并报告参与用户数**，
   否则 GAUC 会被误读成全体表现。
2. **加权方式要说清** —— 这里按用户的曝光数加权（Alibaba GAUC 的常见口径）。
   不加权的话，只有几条曝光的用户和上千条的用户等权。

## 为什么要 PCOC 和校准曲线

AUC 只看**排序**，不看**数值**。共享表示很可能让稀疏任务（is_follow 训练正样本
只有 4,922 条）失准而非失序 —— AUC 看着还行，但预测概率整体偏掉。
这正是 Week 3 负迁移分析要抓的东西，AUC 抓不到。

    PCOC = mean(预测) / mean(实际)      > 1 高估，< 1 低估，= 1 校准良好

用法：
    from src.evaluation.ranking_metrics import evaluate_ranking
    res = evaluate_ranking(pred, label, user_ids)
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """平均秩（1 起）。并列元素取它们名次的平均值。

    不能用 argsort 的名次直接当秩：预测值大量并列时（初始化、饱和、离散输出），
    普通名次会人为制造顺序，AUC 随之偏离真值。
    """
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), np.float64)
    s = x[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
            # 两个相同的分数取平均排名
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def check_label(label: np.ndarray) -> np.ndarray:
    """标签必须是一维、非空、有限、且严格取值 {0, 1}。

    不校验的话非法标签会静默算出**不可能的指标**：实测 label=2 时能得到 AUC = −1.0。
    这类错误不报错、不产生 NaN，只会让 ΔAUC 整体偏移，而四个模型共用这一个入口，
    会一起错到同一个方向。
    """
    y = np.asarray(label)
    if y.ndim != 1:
        raise ValueError(f"label 必须是一维数组，收到 shape {y.shape}")
    if y.size == 0:
        raise ValueError("label 为空")
    if not np.isfinite(y).all():
        raise ValueError(f"label 含 {int((~np.isfinite(y)).sum()):,} 个 NaN/Inf")
    bad = ~np.isin(y, (0, 1))
    if bad.any():
        vals = np.unique(y[bad])[:5]
        raise ValueError(f"label 只能是 0 或 1，发现 {int(bad.sum()):,} 个非法值，例如 {vals}")
    return y.astype(np.int8)


def _auc_unchecked(pred: np.ndarray, y: np.ndarray) -> float:
    # 标签只可能是 0 或 1，因此求和就是正样本数量。
    n_pos = int(y.sum())
    # 负样本数量 = 总样本数 - 正样本数量。
    n_neg = len(y) - n_pos
    # 缺少任意一类时没有正负样本对可供比较，AUC 无定义。
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # 按预测分数从小到大排名；相同分数使用平均秩，因此正负并列按半次胜出计算。
    r = _avg_ranks(np.asarray(pred, np.float64))

    # 取出正样本的排名，并从小到大排列。
    # 例如正样本排名是 [2, 4, 5]。
    positive_ranks = np.sort(r[y == 1])

    # 第 1/2/3 个正样本自身会分别占掉第 1/2/3 个位置。
    positive_own_positions = np.arange(1, n_pos + 1)

    # 排名减去正样本自身占据的位置，剩下的就是它前面分数更低的负样本数。
    # 示例：[2, 4, 5] - [1, 2, 3] = [1, 2, 2]。
    negative_below_each_positive = positive_ranks - positive_own_positions

    # 合计正样本正确排在负样本前面的正负样本对数：1 + 2 + 2 = 5。
    positive_wins = negative_below_each_positive.sum()

    # 全部正负样本对共有 n_pos * n_neg 个，胜出比例就是 ROC-AUC。
    total_positive_negative_pairs = n_pos * n_neg
    return float(positive_wins / total_positive_negative_pairs)


def auc(pred: np.ndarray, label: np.ndarray) -> float:
    """ROC-AUC。正负样本任一为空时返回 nan（无定义，不是 0.5）。"""
    return _auc_unchecked(pred, check_label(label))


def gauc(pred: np.ndarray, label: np.ndarray, user_ids: np.ndarray,
         weight: str = "impressions") -> tuple[float, int, int]:
    """按用户分组的 AUC。返回 (gauc, 参与用户数, 被排除用户数)。

    标签恒定的用户无法计算 AUC，直接排除 —— 稀疏任务上这个比例可以过半，
    因此参与用户数必须与指标一起报。
    """
    if weight not in ("impressions", "positives"):
        # 拼错时原本会静默落进 positives 分支 —— 你以为报的是曝光加权，实际换了口径。
        raise ValueError(f"weight 必须是 impressions 或 positives，收到 {weight!r}")
    y = check_label(label)
    df = pl.DataFrame({"u": user_ids, "p": pred, "y": y.astype(np.float64)})
    num = den = 0.0
    used = skipped = 0
    # 注意for
    for (_,), g in df.group_by("u", maintain_order=True):
        a = _auc_unchecked(g["p"].to_numpy(), g["y"].to_numpy().astype(np.int8))
        if np.isnan(a):
            skipped += 1
            continue
        # g 是当前这个用户的所有样本
        # len(g) 就是当前用户的曝光样本数量
        w = len(g) if weight == "impressions" else float(g["y"].sum())
        # 当前用户有多少条曝光，就使用多大的权重。
        # else 当前用户有多少个正样本，就使用多大的权重。

        num += a * w
        den += w
        used += 1
    return (num / den if den else float("nan")), used, skipped


def pcoc(pred: np.ndarray, label: np.ndarray) -> float:
    """预测均值 / 实际均值。要求 pred 是概率（sigmoid 之后）。"""
    y = check_label(label)
    actual = float(np.mean(y))
    return float(np.mean(pred)) / actual if actual > 0 else float("nan")


def calibration(pred: np.ndarray, label: np.ndarray, n_buckets: int = 10) -> pl.DataFrame:
    """按预测值分桶，比较每桶的预测均值与实际均值。

    用**秩**分桶而不是等宽分桶：预测值分布通常极偏（稀疏任务上大部分预测都贴近 0），
    等宽分桶会让绝大多数样本落进第一个桶。

    注意：大量并列的预测值会共享同一个平均秩，因此**实际出现的桶数可能少于 n_buckets**。
    某个桶异常大本身就是信号 —— 说明模型产出了大量相同的预测，没有在区分。
    """
    if not isinstance(n_buckets, (int, np.integer)) or n_buckets < 1:
        raise ValueError(f"n_buckets 必须是正整数，收到 {n_buckets!r}")
    y = check_label(label)
    p = np.asarray(pred, np.float64)
    r = _avg_ranks(p)
    b = np.minimum((r - 1) * n_buckets // len(p), n_buckets - 1).astype(np.int32)
    return (
        pl.DataFrame({"bucket": b, "p": p, "y": y.astype(np.float64)})
        .group_by("bucket")
        .agg(pl.len().alias("n"), pl.col("p").mean().alias("pred"),
             pl.col("y").mean().alias("actual"))
        .sort("bucket")
    )


def evaluate_ranking(pred: np.ndarray, label: np.ndarray, user_ids: np.ndarray,
                     n_buckets: int = 10, weight: str = "impressions") -> dict:
    """排序侧的统一入口。"""
    pred = np.asarray(pred, np.float64)
    label = check_label(label).astype(np.float64)
    if pred.ndim != 1:
        raise ValueError(f"pred 必须是一维数组，收到 shape {pred.shape}")
    if not (len(pred) == len(label) == len(user_ids)):
        raise ValueError(
            f"长度不一致：pred {len(pred)}, label {len(label)}, user_ids {len(user_ids)}"
        )
    if not np.isfinite(pred).all():
        raise ValueError(f"预测含 {int((~np.isfinite(pred)).sum()):,} 个 NaN/Inf")
    # 必须在归一化后 0-1的范围
    if pred.min() < 0 or pred.max() > 1:
        raise ValueError(
            f"预测必须是概率（sigmoid 之后），当前范围 [{pred.min():.4g}, {pred.max():.4g}]。"
            "PCOC 与校准曲线在 logit 上没有意义。"
        )
    g, used, skipped = gauc(pred, label, user_ids, weight)
    return {
        "auc": auc(pred, label),
        "gauc": g,
        "gauc_users": used, # 实际参与 GAUC 计算的用户数量，只有同时拥有正、负样本的用户才能计算个人 AUC
        "gauc_skipped": skipped, #计算GAUC被跳过的用户
        "pcoc": pcoc(pred, label), #预测均值与真实正样本率的比值
        "n": len(pred),
        "n_pos": int(label.sum()),
        "pos_rate": float(label.mean()),
        "calibration": calibration(pred, label, n_buckets), #校准分桶结果
    }


def format_ranking(res: dict, tag: str = "") -> str:
    c = res["calibration"]
    lines = [
        f"{tag}  n={res['n']:,}  正样本 {res['n_pos']:,}（{res['pos_rate']:.4%}）",
        f"  AUC   {res['auc']:.5f}",
        (f"  GAUC  {res['gauc']:.5f}   参与用户 {res['gauc_users']}"
         f"（排除 {res['gauc_skipped']} 个标签恒定的）"),
        f"  PCOC  {res['pcoc']:.4f}   " +
        ("高估" if res["pcoc"] > 1.05 else "低估" if res["pcoc"] < 0.95 else "校准良好"),
        "  校准分桶（预测 / 实际）:",
    ]
    for r in c.iter_rows(named=True):
        lines.append(f"    桶{r['bucket']:>2}  n={r['n']:>8,}  "
                     f"预测 {r['pred']:.5f}  实际 {r['actual']:.5f}")
    return "\n".join(lines)
