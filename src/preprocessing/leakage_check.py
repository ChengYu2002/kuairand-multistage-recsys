"""泄漏检查（plan §7）—— P0。

时间切分与特征快照的错误都是**静默**的：不会报错，只会让指标虚高。
因此每次重建数据后都应跑一遍本检查，任何一条不通过即退出非零。

当前覆盖：
  1. 曝光后字段（§7.1）未出现在建模表中
  2. 官方整月统计文件（§7.2）未被物化进 processed/
  3. date 列与 time_ms 构造上一致（原始 date 把 23 点记为第二天）
  4. 时间切分严格有序（§6.2）

特征管线完成后，这里还应加入「Day T 样本只使用 <= T-1 的聚合」的校验。

用法：
    python -m src.preprocessing.leakage_check --config configs/data.yaml
"""

from __future__ import annotations

import argparse

import polars as pl

from src.utils.config import load_config, project_path, require
from src.utils.logger import get_logger

log = get_logger(__name__)

ORDER = ["warmup", "train", "valid", "test"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = project_path(require(cfg, "dataset", "processed_dir"))
    failures: list[str] = []

    # 1. 曝光后字段
    forbidden = set(cfg.get("leakage", {}).get("forbidden_fields", []))
    for name in ("logs_main.parquet", "logs_holdout.parquet", "logs_split.parquet", "logs_random.parquet"):
        f = proc / name
        if not f.is_file():
            continue
        cols = set(pl.scan_parquet(f).collect_schema().keys())
        bad = forbidden & cols
        if bad:
            failures.append(f"{name} 含曝光后字段: {sorted(bad)}")
        else:
            log.info("OK  %-22s 无曝光后字段", name)

    # 2. 被禁用的官方统计文件
    for pat in cfg.get("leakage", {}).get("excluded_files", []):
        hits = list(proc.glob(f"*{pat}*"))
        if hits:
            failures.append(f"被禁用的统计文件已进入 processed/: {[h.name for h in hits]}")
    log.info("OK  官方整月统计文件未物化")

    # 3. date 与 time_ms 一致性 —— T-1 特征的「Day T 只用 <= T-1」依赖这一点
    tz = cfg.get("split", {}).get("timezone_offset_hours")
    if tz is not None:
        for name in ("logs_main.parquet", "logs_split.parquet", "logs_random.parquet"):
            f = proc / name
            if not f.is_file():
                continue
            n_bad = (
                pl.scan_parquet(f)
                .filter(
                    pl.col("date")
                    != (pl.from_epoch("time_ms", time_unit="ms") + pl.duration(hours=tz))
                    .dt.strftime("%Y%m%d")
                    .cast(pl.Int32)
                )
                .select(pl.len())
                .collect()
                .item()
            )
            if n_bad:
                failures.append(f"{name} 有 {n_bad:,} 行的 date 与 time_ms 不符")
            else:
                log.info("OK  %-22s date 与 time_ms 一致", name)

    # 4. 时间切分
    split_file = proc / "logs_split.parquet"
    if not split_file.is_file():
        failures.append("logs_split.parquet 不存在，请先运行 temporal_split")
    else:
        s = (
            pl.scan_parquet(split_file)
            .group_by("split")
            .agg(pl.col("time_ms").min().alias("lo"), pl.col("time_ms").max().alias("hi"))
            .collect()
        )
        by = {r["split"]: r for r in s.iter_rows(named=True)}
        present = [n for n in ORDER if n in by]
        for a, b in zip(present, present[1:]):
            if by[a]["hi"] >= by[b]["lo"]:
                failures.append(f"时间泄漏: max({a}) >= min({b})")
            else:
                log.info("OK  max(%-6s) < min(%s)", a, b)

    if failures:
        log.error("泄漏检查未通过：")
        for f in failures:
            log.error("  - %s", f)
        return 1
    log.info("全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
