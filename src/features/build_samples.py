"""训练样本表：特征 join（plan 阶段 0 的 ⑥）。

## 不物化宽表，存索引

样本只碰得到 item 特征表的 **7.1%**（166 万 / 2,353 万行），而同一个 (video, date) 的
52 个特征平均被 4.2 个样本共用。物化宽表要存 705 万 x 137 列 float32 ≈ 3.9 GB，
其中绝大部分是重复。改为：

    特征矩阵   预先编码好的 float32 数组，常驻内存（item 0.32 GB / user 6 MB）
    样本表     每个样本只存两个 int32 行号，8 字节 —— 比物化省 68 倍
    取数       feat[idx]，纯数组索引，训练循环里没有 join

## 第 0 行是冷启动行

每个特征矩阵的 **row 0 专门存冷启动填充值**，join 不到的样本 `idx = 0`。好处是
冷启动填充**只发生一次、只填一行**，不会出现「训练时填了评估时忘了」或者「两处填法不同」
这类 bug —— 而 56.3% 的样本走的正是这条路径。那一行由 apply_spec 作用在一整行 null 上
得到，与普通行**完全同一个代码路径**，不存在第二套填充逻辑。

## 已知限制：27K 会顶不住

裁剪后的 item 矩阵在 1K 上是 0.32 GB。27K 的日志约 35 倍，同比例估算约 11 GB，
放不进内存。届时要把 .npy 换成 np.memmap 按行读，**样本表与查表接口不变**，
只有加载方式要改。别到那时当成新问题重新设计。

## 裁剪依据不构成泄漏

裁剪用的是「哪些 (video, date) 出现在样本里」，不含任何标签信息；而推理时本来就知道
自己要给哪些 item 打分。特征值本身是 T-1 聚合，归一化统计量只用 train 段（见 ⑤）。

用法：
    python -m src.features.build_samples --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from src.features.build_vocab import OOV
from src.features.feature_encoder import apply_spec, classify
from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)

COLD_ROW = 0


def encode_matrix(
    df: pl.DataFrame, feat_cols: list[str], spec: dict
) -> np.ndarray:
    """套用编码规格并转成 (n+1, k) 的 float32 矩阵，row 0 为冷启动行。

    冷启动行由 apply_spec 作用在一整行 null 上得到 —— 与普通行同一条代码路径，
    因此不可能与普通行的填充逻辑不一致。
    """
    cold = pl.DataFrame(
        {c: [None] for c in feat_cols}, schema={c: pl.Float64 for c in feat_cols}
    )
    cold_vals = apply_spec(cold, spec).select(feat_cols).to_numpy().astype(np.float32)

    body = (
        apply_spec(df.select(feat_cols).cast(pl.Float64), spec)
        .select(feat_cols)
        .to_numpy()
        .astype(np.float32)
    )
    out = np.vstack([cold_vals, body])
    if not np.isfinite(out).all():
        bad = [feat_cols[j] for j in np.unique(np.where(~np.isfinite(out))[1])]
        raise AssertionError(f"编码后出现 NaN/Inf，列：{bad[:5]}")
    return out


def map_history(samples: pl.DataFrame, vocab: dict[int, int]) -> pl.Series:
    """把历史里的原始 video_id 换成词表行号；不在词表的落 OOV。

    必须保序（hist 是「新的在前」）。explode 会丢掉组内顺序信息，因此先记下位置，
    聚合时再按位置排回来。
    """
    vm = pl.DataFrame(
        {"vid": list(vocab.keys()), "vidx": list(vocab.values())},
        schema={"vid": pl.Int32, "vidx": pl.Int32},
    )
    return (
        samples.select("_rid", "hist")
        .with_columns(pl.int_ranges(pl.col("hist").list.len()).alias("pos"))
        # empty_as_null=False：空历史必须**不产生任何行**。默认的 True 会为空列表产出一个
        # null 行，随后被 fill_null(OOV) 填成 [OOV] —— 于是「没有历史」变成了「有一个未知
        # 视频的历史」，长度语义就错了（实测 train 212 / valid 35 / test 5 条）。
        .explode(["hist", "pos"], empty_as_null=False)
        .join(vm, left_on="hist", right_on="vid", how="left")
        .with_columns(pl.col("vidx").fill_null(OOV))
        .group_by("_rid")
        .agg(pl.col("vidx").sort_by("pos").alias("hist_idx"))
        .join(samples.select("_rid"), on="_rid", how="right")
        .sort("_rid")
        .get_column("hist_idx")
        .fill_null([])
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--protocol", default="main", choices=("main", "aux"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = project_path(require(cfg, "data", "dataset", "processed_dir"))
    t0 = time.perf_counter()

    spec = json.loads((proc / f"encoder_spec_{args.protocol}.json").read_text(encoding="utf-8"))
    labels = require(cfg, "data", "labels", "tasks")
    splits = ["train", "valid", "test"]

    logs = pl.scan_parquet(proc / "logs_split.parquet").filter(pl.col("split").is_in(splits))
    need_item = logs.select("video_id", "date").unique().collect()
    need_user = logs.select("user_id", "date").unique().collect()

    # ---- 特征矩阵（裁剪 -> 编码 -> 加冷启动行）----
    mats, lookups, feat_cols_all = {}, {}, {}
    for side, fname, key, prefix in (
        ("user", "feat_user_daily", "user_id", "user"),
        ("item", "feat_item_daily", "video_id", "item"),
    ):
        need = need_user if side == "user" else need_item
        tbl = (
            pl.scan_parquet(proc / f"{fname}.parquet")
            .join(need.lazy(), on=[key, "date"], how="semi")
            .sort([key, "date"])          # 全序键：保证行号可复现
            .collect()
        )
        feat_cols = [c for c in tbl.columns if classify(c, prefix)]
        feat_cols_all[side] = feat_cols
        mats[side] = encode_matrix(tbl, feat_cols, spec)
        # 行号从 1 开始，0 留给冷启动行
        lookups[side] = tbl.select(
            key, "date", (pl.int_range(pl.len(), dtype=pl.Int32) + 1).alias("row")
        )
        log.info("%s 特征: %s 行 x %d 列 -> 矩阵 %s (%.2f GB)",
                 side, f"{len(tbl):,}", len(feat_cols), mats[side].shape,
                 mats[side].nbytes / 1024**3)

    for side in ("user", "item"):
        np.save(proc / f"feat_{side}_encoded_{args.protocol}.npy", mats[side])
        lookups[side].write_parquet(proc / f"feat_row_{side}_{args.protocol}.parquet", compression="zstd")
    # 接口约定写进文件，避免「规格写了但没人知道怎么取」
    (proc / f"feat_columns_{args.protocol}.json").write_text(
        json.dumps({
            "cold_row": COLD_ROW,
            "protocol": args.protocol,
            **{k: v for k, v in feat_cols_all.items()},
            "interface": {
                "user_t1": "feat_user_encoded_{p}.npy[samples.user_feat_row]",
                "item_t1": "feat_item_encoded_{p}.npy[samples.item_feat_row]",
                "user_static": "feat_user_static_{p}.parquet 按 user_id 取",
                "item_static": "feat_item_static_{p}.parquet 按 video_index 取"
                               "（含 author_index / tag_indices / upload_date / 编码后的 duration_ms）",
                "history": "samples.hist_idx，已是词表行号；0=PAD 1=OOV，pooling 时两者都要 mask",
                "target": "samples.target_idx，1 表示目标不在词表",
                "age": "src.features.video_age.age_days(samples.date, item_static.upload_date) 现算",
            },
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 静态特征：套用同一份规格后落盘 ----
    # 规格里还有 33 列不属于 T-1（用户画像 31 / duration / 三个缺失标记）。它们是**按实体**
    # 而不是按 (实体, 日期) 的，塞进上面的矩阵会重复几百万遍。但「规格写了、模型拿不到」
    # 同样不行 —— 因此在这里就把规格套好落盘，塔只管按 id 取，不需要自己再懂编码规则。
    #   feat_user_static_*.parquet   1,000 行，按 user_id 取
    #   feat_item_static_*.parquet   897,503 行，按 video_index 取
    uf = apply_spec(pl.read_parquet(proc / "user_features.parquet"), spec).drop(
        [c for c in spec["dropped"]]
    )
    uf.write_parquet(proc / f"feat_user_static_{args.protocol}.parquet", compression="zstd")
    it_static = apply_spec(
        pl.read_parquet(proc / f"item_static_{args.protocol}.parquet"), spec
    )
    it_static.write_parquet(proc / f"feat_item_static_{args.protocol}.parquet", compression="zstd")
    log.info("静态特征: user %s 行 x %d 列 / item %s 行 x %d 列（已套用同一份编码规格）",
             f"{len(uf):,}", len(uf.columns), f"{len(it_static):,}", len(it_static.columns))

    # ---- 样本表 ----
    vocab = pl.read_parquet(proc / f"vocab_video_{args.protocol}.parquet")
    vmap = dict(zip(vocab["id"].to_list(), vocab["index"].to_list()))
    # 必须按连接键去重：原始日志里存在 5 组重复曝光（用户 413 在同一毫秒
    # 1650424936111 下有 5 个视频各记录了两次，共 10 行，全在 train 段）。两边都重复时
    # left join 会扇出成 2x2，凭空多出 10 行。hist 只依赖 (user_id, time_ms)，
    # 重复行的 hist 完全相同，去重是无损的。
    hist = (
        pl.scan_parquet(proc / "user_history.parquet")
        .select("user_id", "video_id", "time_ms", "hist")
        .unique(subset=["user_id", "video_id", "time_ms"])
    )

    print()
    print(f"{'split':<8}{'样本':>12}{'user 冷启动':>14}{'item 冷启动':>14}{'目标在词表':>12}")
    for sp in splits:
        s = (
            logs.filter(pl.col("split") == sp)
            .select("user_id", "video_id", "time_ms", "date", "tab", "hourmin", *labels)
            .join(hist, on=["user_id", "video_id", "time_ms"], how="left")
            .join(lookups["user"].lazy().rename({"row": "user_feat_row"}),
                  on=["user_id", "date"], how="left")
            .join(lookups["item"].lazy().rename({"row": "item_feat_row"}),
                  on=["video_id", "date"], how="left")
            .with_columns(
                pl.col("user_feat_row").fill_null(COLD_ROW).cast(pl.Int32),
                pl.col("item_feat_row").fill_null(COLD_ROW).cast(pl.Int32),
                pl.col("hist").fill_null([]),
            )
            .sort(["user_id", "time_ms", "video_id"])   # 全序键，保证可复现
            .collect()
            .with_row_index("_rid")
        )
        # 硬断言：一连串 left join 之后行数必须与源日志完全相同。任何一边在连接键上
        # 不唯一都会静默扇出 —— 上面那 5 组重复曝光就是这么被发现的。
        n_src = int(
            logs.filter(pl.col("split") == sp).select(pl.len()).collect().item()
        )
        if len(s) != n_src:
            raise AssertionError(
                f"{sp}: join 后 {len(s):,} 行 != 源日志 {n_src:,} 行，某个连接键不唯一"
            )
        s = s.with_columns(
            map_history(s, vmap).alias("hist_idx"),
            pl.col("video_id").replace_strict(vmap, default=OOV, return_dtype=pl.Int32)
            .alias("target_idx"),
        ).drop("_rid", "hist")

        s.write_parquet(proc / f"samples_{sp}_{args.protocol}.parquet", compression="zstd")
        n = len(s)
        print(f"{sp:<8}{n:>12,}"
              f"{(s['user_feat_row'] == COLD_ROW).mean():>13.1%}"
              f"{(s['item_feat_row'] == COLD_ROW).mean():>14.1%}"
              f"{(s['target_idx'] != OOV).mean():>12.1%}")
    print()
    log.info("完成 (%.1fs)。矩阵 row 0 = 冷启动；样本只存行号，训练时 feat[idx] 取数。",
             time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
