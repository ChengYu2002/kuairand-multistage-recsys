"""双塔的数据装载（阶段 0 的产物 -> 张量）。

阶段 0 已经把一切都变成了行号，所以这里没有 join、没有逐样本 Python 循环：
把几个数组读进内存，训练时按行号批量 gather 即可。

## 历史也按时刻去重

历史只取决于 (user_id, time_ms)，实测复用 5.22 倍。按样本存 padded 矩阵要 0.84 GB，
按时刻只要 0.16 GB。与特征行同一个模式：样本只记录「去第几行拿历史」。

## 掩码：padding_idx 不能替代它

index 0 是 PAD、1 是 OOV，两者都不参与 pooling。torch 的 padding_idx=0 只保证那一行
恒为零向量且不回传梯度，**不会**把它从 mean 的分母里去掉 —— 直接 mean(dim=1) 会让
历史 5 条和 50 条差 10 倍尺度。而实测训练时平均 49.67/50 条可编码、评估时只有
12.05/50，尺度会系统性错位。因此这里返回显式的 mask 与 hist_len，由用户塔自己做
masked sum / valid_len。

OOV 也被 mask 掉：评估时 82.5% 的历史条目是 OOV，若让它们参与 pooling，
结果会被同一个 OOV 向量主导，真实信号反而被淹没。代价是评估时可用历史更短，
所以 hist_len 必须作为特征交给模型 —— 它是「这个池化向量背后有多少证据」的度量。

## 内存（1K 实测）

    训练期间常驻   0.70 GB   历史矩阵 0.32 + item T-1 0.32 + 其余
    装载瞬时峰值   3.77 GB   读 4.5M 行 samples parquet 时的临时量

瞬时峰值远大于常驻，因为 samples 表里 hist_idx 是**逐样本**存的列表列。27K 上这会爆，
届时要把 hist_idx 换成 hist_row、把去重后的历史单独落一张表（与特征行同一个模式）。
1K 上 3.77 GB 不构成阻塞，故暂不改动已验过的 build_samples。

## 空历史

hist_len = 0 时历史通道输出**零向量**（不是 OOV 向量，也不除零）。模型靠 hist_len
与 user_has_history 知道这件事。实测评估时刻只有 0.37% 为空，训练时 0.005%。

用法：
    data = RetrievalData(proc, "main", "train", cfg)
    b = data.batch(np.arange(1024))
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

PAD, OOV = 0, 1


class RetrievalData:
    def __init__(self, proc: Path, protocol: str, split: str, max_hist: int = 50,
                 mask_oov: bool = True, load_item_t1: bool = False) -> None:
        self.proc, self.protocol, self.split = Path(proc), protocol, split
        self.max_hist, self.mask_oov = max_hist, mask_oov
        self.load_item_t1 = load_item_t1
        spec = json.loads((self.proc / f"encoder_spec_{protocol}.json").read_text("utf-8"))
        cols = json.loads((self.proc / f"feat_columns_{protocol}.json").read_text("utf-8"))

        # ---- T-1 特征矩阵（row 0 = 冷启动行）----
        self.user_t1 = np.load(self.proc / f"feat_user_encoded_{protocol}.npy")
        # 物品的 T-1 统计**不进物品塔**（§13.2：物品向量要离线算一次建索引）。
        # 默认不加载 —— 1K 上它是 0.32 GB 的常驻 + 每个 batch 的无用 gather，
        # 27K 上会成为大头。排序阶段（⑬）需要时显式传 load_item_t1=True。
        self.item_t1 = (
            np.load(self.proc / f"feat_item_encoded_{protocol}.npy") if load_item_t1 else None
        )
        self.user_t1_cols = cols["user"]
        self.item_t1_cols = cols["item"]

        # ---- 静态特征 ----
        us = pl.read_parquet(self.proc / f"feat_user_static_{protocol}.parquet").sort("user_id")
        num = [c for c in us.columns
               if c != "user_id" and spec["columns"].get(c, {}).get("kind") == "numeric"]
        cat = [c for c in us.columns
               if c != "user_id" and spec["columns"].get(c, {}).get("kind") == "categorical"]
        self.user_static_num = us.select(num).to_numpy().astype(np.float32)
        self.user_static_cat = us.select(cat).to_numpy().astype(np.int64)
        self.user_static_num_cols, self.user_static_cat_cols = num, cat
        self.user_static_cat_sizes = [
            spec["categorical"][c]["cardinality"] for c in cat
        ]

        it = pl.read_parquet(self.proc / f"feat_item_static_{protocol}.parquet").sort("video_index")
        n_vocab = int(it["video_index"].max()) + 1
        self.item_author = np.zeros(n_vocab, np.int64)
        self.item_author[it["video_index"].to_numpy()] = it["author_index"].to_numpy()
        self.item_duration = np.zeros(n_vocab, np.float32)
        d = it["duration_ms"].fill_null(spec["columns"]["duration_ms"]["fill"]).to_numpy()
        self.item_duration[it["video_index"].to_numpy()] = d
        self.item_has_duration = np.zeros(n_vocab, np.float32)
        self.item_has_duration[it["video_index"].to_numpy()] = (
            it["has_duration"].to_numpy().astype(np.float32)
        )
        self.item_upload = np.zeros(n_vocab, np.int64)
        self.item_upload[it["video_index"].to_numpy()] = (
            it["upload_date"].fill_null(0).to_numpy()
        )
        # tag 是变长（1~5），补成定宽矩阵，PAD=0
        self.max_tags = int(it["n_tags"].max())
        self.item_tags = np.zeros((n_vocab, self.max_tags), np.int64)
        self.item_tags[it["video_index"].to_numpy()] = _pad_ragged(
            it["tag_indices"], self.max_tags
        )

        # ---- 样本 ----
        s = pl.read_parquet(self.proc / f"samples_{split}_{protocol}.parquet")
        self.n = len(s)
        self.user_feat_row = s["user_feat_row"].to_numpy().astype(np.int64)
        self.item_feat_row = s["item_feat_row"].to_numpy().astype(np.int64)
        self.target = s["target_idx"].to_numpy().astype(np.int64)
        self.user_id = s["user_id"].to_numpy().astype(np.int64)
        self.date = s["date"].to_numpy().astype(np.int64)
        self.tab = s["tab"].to_numpy().astype(np.int64)
        self.hour = (s["hourmin"].to_numpy().astype(np.int64) // 100).clip(0, 23)
        self.labels = {c: s[c].to_numpy().astype(np.float32)
                       for c in s.columns if c.startswith("is_") or c == "long_view"}

        # ---- 历史：按 (user_id, time_ms) 去重后补成定宽矩阵 ----
        mom = (
            s.select("user_id", "time_ms", "hist_idx")
            .unique(subset=["user_id", "time_ms"], keep="first")
            .sort(["user_id", "time_ms"])
            .with_row_index("hist_row")
        )
        self.hist = _pad_ragged(mom["hist_idx"], max_hist)             # 0 = PAD
        # 必须显式保序：polars 的 left join **不保证**输出顺序与左表一致。实测当前版本
        # 碰巧保序，但一旦变了，每个样本都会拿到别人的历史 —— 而且不会报错，因为下游
        # 用的正是这份（已错位的）映射。与 build_catalog 里 semi join 那次是同一类问题。
        self.hist_row = (
            s.select("user_id", "time_ms")
            .with_row_index("_i")
            .join(mom.select("user_id", "time_ms", "hist_row"), on=["user_id", "time_ms"], how="left")
            .sort("_i")
            .get_column("hist_row").to_numpy().astype(np.int64)
        )

    def __len__(self) -> int:
        return self.n

    def item_features(self, idx: np.ndarray, date: np.ndarray | None = None) -> dict:
        """物品侧静态属性。给定词表行号（和可选的请求日）返回物品塔要的东西。"""
        out = {
            "item_idx": torch.from_numpy(idx),
            "author": torch.from_numpy(self.item_author[idx]),
            "tags": torch.from_numpy(self.item_tags[idx]),
            "duration": torch.from_numpy(self.item_duration[idx]),
            # 7.1% 的视频缺时长（已填中位数）。不给标志的话，模型无法区分
            # 「40.5 秒的视频」和「不知道多长」。
            "has_duration": torch.from_numpy(self.item_has_duration[idx]),
        }
        if date is not None:
            up = self.item_upload[idx]
            # age = 请求日 - 上传日。YYYYMMDD 不能直接相减，先转成序数天。
            age = _ymd_to_days(date) - _ymd_to_days(up)
            age = np.where(up == 0, 0.0, age).astype(np.float32)     # 无上传日 -> 0，配合 has_upload
            out["age"] = torch.from_numpy(np.log1p(np.clip(age, 0, None)))
            out["has_upload"] = torch.from_numpy((up != 0).astype(np.float32))
        return out

    def batch(self, rows: np.ndarray) -> dict:
        rows = np.asarray(rows, dtype=np.int64)
        h = self.hist[self.hist_row[rows]]
        mask = h != PAD
        if self.mask_oov:
            mask &= h != OOV
        b = {
            "user_t1": torch.from_numpy(self.user_t1[self.user_feat_row[rows]]),
            "user_static_num": torch.from_numpy(self.user_static_num[self.user_id[rows]]),
            "user_static_cat": torch.from_numpy(self.user_static_cat[self.user_id[rows]]),
            "hist": torch.from_numpy(h),
            "hist_mask": torch.from_numpy(mask),
            "hist_len": torch.from_numpy(mask.sum(1).astype(np.float32)),
            "tab": torch.from_numpy(self.tab[rows]),
            "hour": torch.from_numpy(self.hour[rows]),
            "target": torch.from_numpy(self.target[rows]),
        }
        if self.item_t1 is not None:
            b["item_t1"] = torch.from_numpy(self.item_t1[self.item_feat_row[rows]])
        b.update(self.item_features(self.target[rows], self.date[rows]))
        for k, v in self.labels.items():
            b[f"label_{k}"] = torch.from_numpy(v[rows])
        return b


def _ymd_to_days(ymd: np.ndarray) -> np.ndarray:
    """YYYYMMDD -> 自 1970-01-01 起的天数。

    YYYYMMDD 是十进制拼接，**直接相减不是天数**（20220501 − 20220430 = 71 而不是 1）。
    ymd == 0 表示缺失，返回 0，由调用方配合 has_upload 处理。
    """
    ok = ymd > 0
    y, m, d = ymd // 10000, (ymd // 100) % 100, ymd % 100
    dates = (
        (np.where(ok, y, 1970) - 1970).astype("datetime64[Y]")
        + (np.where(ok, m, 1) - 1).astype("timedelta64[M]")
        + (np.where(ok, d, 1) - 1).astype("timedelta64[D]")
    )
    return np.where(ok, dates.astype("datetime64[D]").astype(np.int64), 0)


def _pad_ragged(col: pl.Series, width: int, dtype=np.int64) -> np.ndarray:
    """polars 的变长 List 列补成定宽矩阵（PAD=0）。

    **不能用 .to_list()**：86 万行 x 50 个元素会变成 4,300 万个 Python 对象，
    实测让装载峰值多出 2.5 GB。这里全程留在 Arrow/numpy 里 —— explode 出
    (行号, 位置, 值) 三元组后一次性散射写入。
    """
    n = len(col)
    ex = (
        pl.DataFrame({"v": col})
        .with_row_index("r")
        .with_columns(pl.col("v").list.head(width))
        .with_columns(pl.int_ranges(pl.col("v").list.len()).alias("c"))
        .explode(["v", "c"], empty_as_null=False)
    )
    out = np.zeros((n, width), dtype)
    if len(ex):
        out[ex["r"].to_numpy(), ex["c"].to_numpy()] = ex["v"].to_numpy()
    return out
