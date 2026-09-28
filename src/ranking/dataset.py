"""排序侧数据装载（plan §20）。行号进、张量出，不做任何判断。

## 与召回装载器的关系

用户侧（T-1 统计、静态画像、历史池化、tab/hour）和标签，与双塔完全一样，因此这里
**组合**而不是重写 src/retrieval/dataset.py 的 RetrievalData —— 那份已经过独立验算，
复制一遍只会多一个会分叉的副本。排序只改物品侧，三处：

1. **要物品 T-1 统计**（`load_item_t1=True`）。双塔刻意不要：那 52 个数随日期变，而
   §13.2 要求物品向量能离线算一次建索引。排序没这个约束，而它是排序最有用的特征之一。
2. **物品静态属性换成排序专用表**。召回那份按 video_index 取，测试段 83.2% 的曝光是
   OOV → 取到一排零。排序这份按 video_id 直接取，作者/标签/时长/上传日期覆盖到
   64% / 97% / 93% / 100%。
3. **目标视频 ID 换成排序专用 video 词表**（P0 修复）。召回词表的成员身份部分由训练
   标签决定：train 段 OOV 样本的 is_click / long_view 正样本率**恰好为 0**，于是
   「在词表内」单独就有 train AUC 0.6981 / test 0.5190 —— 一条只在 train 成立的捷径。
   排序词表只由 train 曝光次数构建，完全不看标签。详见 rank_item_static.py。

   **历史通道仍用召回词表**，与目标不共享 embedding 表。两者在排序里没有共享的必要
   （那是双塔为了离线建索引才有的约束），而换词表救不回历史、换小词表会砍掉一半
   可编码率。历史自身不是捷径：hist_len 单特征 AUC train 0.4983 / test 0.5942。
4. **加 user x author 偏好特征**（plan §8.4）。Single-Task 在 is_follow 上的 GAUC 是
   0.49014 —— 用户内部排序≈随机，因为模型没有任何 user x author 交互特征。实测单特征
   ua_imp_7d 的 GAUC 就有 0.53784，打赢整个模型。矩阵按**样本行**对齐（pair 维度
   4,168,620 个键对 4,496,306 条样本，几乎一一对应，行号间接层省不到东西）。
5. **不抽负样本**。曝光了没点就是负样本，现成的。

因此 batch() 会把 RetrievalData 塞进来的召回侧物品静态键**显式弹掉**，再放进排序版的
同名键。弹而不是直接覆盖：万一将来 RetrievalData 改了键名，这里会抛 KeyError 而不是
静默留下一份来自错误来源的值。

## 按 video_id 直接下标，连映射表都不建

video_id 落在 0~4,371,899 且几乎连续（4,371,868 个实体，仅 31 个空洞），所以排序静态表
可以摊成「下标 = video_id」的数组，取特征就是一次数组索引。训练循环里没有 join ——
449 万行 x 5 轮 x 5 个任务，每批 join 一次的话光备料就比训练慢。

空洞位置初始化成 OOV 而不是 PAD：实测没有任何样本落在空洞上（验算会断言），
但万一落上了，语义应该是「不认识这个作者」，而不是「这一格是填充位」。

## 一份静态表供三个 split 共用

RankItemStatic 与 split 无关（约 160 MB）。三个 split 各建一份就是白占 480 MB，
所以它单独构造、当参数传进来。

物品 T-1 矩阵（345 MB）目前仍是每个 split 各读一份 —— 1K 上三份约 1 GB，32 GB 机器
不阻塞，故不动已验过的 RetrievalData。27K 上必须改成共享或 mmap。

## 时长的标准化用的是 train 统计量

fill / mean / std 全部来自 rank_item_spec 里由 **train 段样本**估计的值，在 __init__
里对全表算一次，不在每个 batch 里重复算。

## video_age 必须现算

age = 请求日 − 上传日，同一个视频在不同请求日的 age 不同，不能预先算成固定值。
表里只存原始 upload_date，age 在 batch() 里按请求日算。

用法：
    st = RankItemStatic(proc, "main")
    tr = RankingData(proc, "main", "train", st)
    b = tr.batch(np.arange(2048))
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.retrieval.dataset import OOV, PAD, RetrievalData, _pad_ragged, _ymd_to_days

# RetrievalData.item_features() 塞进来的召回侧物品静态键。排序一律换成排序表的版本。
_RETRIEVAL_ITEM_KEYS = ("author", "tags", "duration", "has_duration", "age", "has_upload")

# 顺序固定：进 x 的顺序由 features.FLAG_KEYS 沿用，改顺序 = 改输入指纹
KNOWN_FLAGS = ("video_id_known", "author_known", "tag_known", "duration_known",
               "upload_date_known")


class RankItemStatic:
    """排序专用物品元数据，摊成按 video_id 下标的数组。与 split 无关，多个 split 共用。"""

    def __init__(self, proc: Path | str, protocol: str = "main") -> None:
        proc = Path(proc)
        self.protocol = protocol
        self.spec = json.loads(
            (proc / f"rank_item_spec_{protocol}.json").read_text("utf-8"))
        t = pl.read_parquet(proc / f"item_static_rank_{protocol}.parquet")
        vid = t.get_column("video_id").to_numpy()
        n = int(vid.max()) + 1
        self.n_rows = n
        self.max_tags = int(self.spec["max_tags"])
        self.n_authors = int(self.spec["vocabs"]["author"]["embedding_rows"])
        self.n_tags_vocab = int(self.spec["vocabs"]["tag"]["embedding_rows"])
        self.n_videos_vocab = int(self.spec["vocabs"]["video"]["embedding_rows"])
        if self.spec["vocabs"]["video"].get("built_from") != "train_exposures_only_no_labels":
            raise ValueError(
                "rank_item_spec 里的 video 词表来源不是 train_exposures_only_no_labels。"
                "召回词表的成员身份由训练标签决定（train 段 OOV 的 click 正样本率恰好为 0），"
                "拿它当目标 ID 通道会引入 train AUC 0.6981 / test 0.5190 的捷径。"
            )

        # 空洞初始化成 OOV（作者）/ PAD（标签）：语义是「不认识」而不是「填充位」
        self.author = np.full(n, OOV, np.int32)
        self.author[vid] = t.get_column("author_rank_idx").to_numpy()
        self.video = np.full(n, OOV, np.int32)
        self.video[vid] = t.get_column("video_rank_idx").to_numpy()
        self.tags = np.full((n, self.max_tags), PAD, np.int32)
        self.tags[vid] = _pad_ragged(t.get_column("tag_rank_idx"), self.max_tags,
                                     dtype=np.int32)

        # 时长：按 spec（train 段估计）填充 + log1p 标准化，全表算一次
        d = self.spec["columns"]["duration_ms"]
        raw = t.get_column("duration_ms").fill_null(d["fill"]).to_numpy().astype(np.float64)
        z = ((np.log1p(raw) - d["mean"]) / d["std"]).astype(np.float32)
        self.duration = np.zeros(n, np.float32)
        self.duration[vid] = z

        self.upload = np.zeros(n, np.int32)      # YYYYMMDD，0 表示缺失
        self.upload[vid] = t.get_column("upload_date").to_numpy()

        self.flags = {}
        for f in KNOWN_FLAGS:
            arr = np.zeros(n, np.float32)
            arr[vid] = t.get_column(f).to_numpy().astype(np.float32)
            self.flags[f] = arr

    def features(self, video_id: np.ndarray, date: np.ndarray) -> dict:
        """给定原始 video_id 与请求日（YYYYMMDD），返回排序侧的物品静态通道。"""
        tags = self.tags[video_id]
        up = self.upload[video_id]
        # age = 请求日 − 上传日。YYYYMMDD 不能直接相减，先转序数天。缺上传日记 0，
        # 由 upload_date_known 告诉模型这 0 是「不知道」而不是「今天刚发」。
        age = _ymd_to_days(date) - _ymd_to_days(up)
        age = np.where(up == 0, 0.0, np.clip(age, 0, None)).astype(np.float32)
        out = {
            # 目标视频的 ID：**排序词表**行号（只由 train 曝光构建，不看标签）
            "item_idx": torch.from_numpy(self.video[video_id].astype(np.int64)),
            "author": torch.from_numpy(self.author[video_id].astype(np.int64)),
            "tags": torch.from_numpy(tags.astype(np.int64)),
            "tag_mask": torch.from_numpy(tags != PAD),
            "tag_len": torch.from_numpy((tags != PAD).sum(1).astype(np.float32)),
            "duration": torch.from_numpy(self.duration[video_id]),
            "age": torch.from_numpy(np.log1p(age)),
        }
        for f in KNOWN_FLAGS:
            out[f] = torch.from_numpy(self.flags[f][video_id])
        return out


class RankingData:
    def __init__(self, proc: Path | str, protocol: str, split: str,
                 item_static: RankItemStatic, max_hist: int = 50,
                 mask_oov: bool = True, pair_features: bool = True) -> None:
        proc = Path(proc)
        self.proc, self.protocol, self.split = proc, protocol, split
        self.item_static = item_static
        # 用户侧 / 历史 / 标签全部沿用已验过的召回装载器；item_t1 是排序才要的
        self.inner = RetrievalData(proc, protocol, split, max_hist=max_hist,
                                   mask_oov=mask_oov, load_item_t1=True)
        # 进输入指纹：这些口径一变，x 的含义就变了，四个模型必须一致
        self.params = {"max_hist": int(max_hist), "mask_oov_in_history": bool(mask_oov),
                       "pair_features": bool(pair_features)}
        # ---- user x author 偏好（§8.4）。按样本行对齐，不走行号间接 ----
        self.pair = None
        self.pair_cols: list[str] = []
        if pair_features:
            spec = json.loads((proc / f"pair_spec_{protocol}.json").read_text("utf-8"))
            self.pair_cols = list(spec["columns"])
            self.pair = np.load(proc / f"feat_pair_{split}_{protocol}.npy")
            if self.pair.shape[1] != len(self.pair_cols):
                raise AssertionError(
                    f"pair 矩阵列数 {self.pair.shape[1]} 与 spec 的 {len(self.pair_cols)} 不符")
        s = pl.read_parquet(proc / f"samples_{split}_{protocol}.parquet",
                            columns=["video_id"])
        self.video_id = s.get_column("video_id").to_numpy().astype(np.int64)
        if self.pair is not None and len(self.pair) != len(self.inner):
            raise AssertionError(
                f"pair 矩阵行数 {len(self.pair)} != 样本行数 {len(self.inner)} —— "
                "两者不是同一次预处理的产物，按行号取会整体错位且不会报错")
        if len(self.video_id) != len(self.inner):
            raise AssertionError(
                f"samples 行数不一致：video_id {len(self.video_id)} vs "
                f"装载器 {len(self.inner)}")
        hi = self.video_id.max()
        if hi >= item_static.n_rows:
            raise AssertionError(
                f"{split} 段有 video_id {hi} 超出排序静态表范围 {item_static.n_rows}，"
                "静态表与 samples 不是同一次预处理的产物")
        # 底层装载器用 user_id 直接索引静态画像数组（user_static_num[user_id]），
        # 这假设 user_id 是从 0 起的连续编号。1K 上成立（993 个用户，0..999），
        # 27K 抽 2-3k 用户后**未必**成立 —— 那时会静默取到别人的画像，而不是报错。
        n_users = self.inner.user_static_num.shape[0]
        u_hi = int(self.inner.user_id.max())
        if u_hi >= n_users:
            raise AssertionError(
                f"{split} 段最大 user_id {u_hi} >= 静态画像行数 {n_users}。"
                "底层装载器按 user_id 直接索引，必须先把 user_id 重编号成 0..n-1，"
                "否则每个用户都会拿到别人的画像且不会报错。")

    def __len__(self) -> int:
        return len(self.inner)

    @property
    def user_id(self) -> np.ndarray:
        return self.inner.user_id

    @property
    def labels(self) -> dict[str, np.ndarray]:
        return self.inner.labels

    def batch(self, rows: np.ndarray) -> dict:
        rows = np.asarray(rows, dtype=np.int64)
        b = self.inner.batch(rows)
        # 弹掉召回侧的物品静态键（按 video_index 取，测试段 83.2% 是一排零）。
        # 弹而不是覆盖：键名若变动要立刻 KeyError，不能静默留下错来源的值。
        for k in _RETRIEVAL_ITEM_KEYS:
            del b[k]
        # 召回词表行号只留作诊断，**绝不进编码器**：它的成员身份由训练标签决定。
        # 改名而不是留在 item_idx 上，是为了任何误用都会在指纹与验算里现形。
        b["item_idx_retrieval"] = b.pop("item_idx")
        del b["target"]
        vid = self.video_id[rows]
        # item_idx（排序词表行号）与 video_id_known 都由排序静态表给出
        b.update(self.item_static.features(vid, self.inner.date[rows]))
        if self.pair is not None:
            b["pair"] = torch.from_numpy(self.pair[rows])
        return b
