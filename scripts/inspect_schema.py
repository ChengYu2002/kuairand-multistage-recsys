"""KuaiRand Schema Inspection —— Week 1 的第一步。

回答四个在 plan 里悬而未决、且阻塞后续所有决策的问题：

  1. §14.1  upload_dt 是什么粒度？ -> 候选库该用 `<` 还是 `<=`
  2. §7.1   哪些字段是曝光后产物？ -> 泄漏字段清单定稿
  3. §10.1  不同 tab 下 is_click 语义一致吗？ -> 统一建模 or 只用部分场景
  4. §6.1   每日曝光量如何分布？ -> train/val/test 边界怎么切

用法：
    python scripts/inspect_schema.py
    python scripts/inspect_schema.py --data-dir data/raw/KuaiRand-27K/data

结果同时打印到终端并写入 results/schema_report.md。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent

# 曝光后才产生，只能当 label / 分析对象（plan §7.1）
POST_INTERACTION = ["play_time_ms", "profile_stay_time", "comment_stay_time", "is_profile_enter"]
# 候选标签（plan §10）
LABELS = ["is_click", "long_view", "is_like", "is_follow", "is_comment", "is_forward", "is_hate"]

_report: list[str] = []


def say(line: str = "") -> None:
    print(line)
    _report.append(line)


def header(title: str) -> None:
    say()
    say(f"## {title}")
    say()


def find_files(data_dir: Path) -> dict[str, list[Path]]:
    """按前缀归类数据文件；27K 的日志是分片的，所以每类返回列表。"""
    groups: dict[str, list[Path]] = {
        "log_standard": [],
        "log_random": [],
        "user_features": [],
        "video_features_basic": [],
        "video_features_statistic": [],
    }
    for path in sorted(data_dir.glob("*.csv")):
        for key in groups:
            if path.name.startswith(key):
                groups[key].append(path)
                break
    return groups


def scan(paths: list[Path]) -> pl.LazyFrame:
    """始终用 lazy scan —— 27K 的日志有 3 亿行，eager read 会直接 OOM。"""
    return pl.concat([pl.scan_csv(p) for p in paths], how="vertical_relaxed")


def section_files(groups: dict[str, list[Path]], data_dir: Path) -> None:
    header("1. 文件清单")
    say("| 文件 | 大小 |")
    say("|---|---:|")
    for paths in groups.values():
        for p in paths:
            say(f"| `{p.name}` | {p.stat().st_size / 1024**2:,.0f} MB |")


def section_log_schema(lf: pl.LazyFrame) -> list[str]:
    header("2. 标准日志 Schema")
    schema = lf.collect_schema()
    n_rows = lf.select(pl.len()).collect().item()
    say(f"行数: **{n_rows:,}**，列数: **{len(schema)}**")
    say()
    say("| 字段 | 类型 | 用途 |")
    say("|---|---|---|")
    for name, dtype in schema.items():
        if name in POST_INTERACTION:
            role = "**曝光后产物 —— 禁止作为特征**"
        elif name in LABELS:
            role = "label 候选"
        else:
            role = "特征 / 上下文"
        say(f"| `{name}` | {dtype} | {role} |")
    return list(schema.keys())


def section_nulls(lf: pl.LazyFrame, cols: list[str]) -> None:
    header("3. 缺失率")
    n_rows = lf.select(pl.len()).collect().item()
    nulls = lf.null_count().collect().row(0)
    rows = [(c, n) for c, n in zip(cols, nulls) if n > 0]
    if not rows:
        say("标准日志无缺失值。")
        return
    say("| 字段 | 缺失数 | 缺失率 |")
    say("|---|---:|---:|")
    for col, n in sorted(rows, key=lambda x: -x[1]):
        say(f"| `{col}` | {n:,} | {n / n_rows:.2%} |")


def section_time(lf: pl.LazyFrame) -> None:
    header("4. 时间覆盖（决定 split 边界 —— plan §6.1）")
    daily = (
        lf.group_by("date")
        .agg(
            pl.len().alias("impressions"),
            pl.col("user_id").n_unique().alias("users"),
            pl.col("video_id").n_unique().alias("items"),
            pl.col("is_click").mean().alias("click_rate"),
        )
        .sort("date")
        .collect()
    )
    say(f"日期范围: **{daily['date'].min()} ~ {daily['date'].max()}**，共 **{len(daily)}** 天")
    say()
    say("| date | 曝光数 | 活跃用户 | distinct item | click 率 |")
    say("|---|---:|---:|---:|---:|")
    for r in daily.iter_rows(named=True):
        say(
            f"| {r['date']} | {r['impressions']:,} | {r['users']:,} "
            f"| {r['items']:,} | {r['click_rate']:.4f} |"
        )
    say()
    say("> 注意：T-1 特征有 7 天窗口（§8.2），开头约 7 天只能作为特征预热期，不可进训练集。")


def section_tab(lf: pl.LazyFrame) -> None:
    header("5. Tab / Scene 语义检查（plan §10.1 —— P0）")
    say(
        "官方文档说明 `is_click` 在**双列 UI** 下是点击，在**单列 UI** 下等价于 valid_play。"
        "若某个 tab 的 click 率极高且 `P(long_view | click)` 接近 1，该 tab 多半是单列 UI。"
    )
    say()
    tab = (
        lf.group_by("tab")
        .agg(
            pl.len().alias("n"),
            pl.col("is_click").mean().alias("click_rate"),
            pl.col("long_view").mean().alias("long_view_rate"),
            (pl.col("long_view").filter(pl.col("is_click") == 1).mean()).alias("lv_given_click"),
        )
        .sort("n", descending=True)
        .collect()
    )
    total = tab["n"].sum()
    say("| tab | 曝光数 | 占比 | click 率 | long_view 率 | P(long_view\\|click) |")
    say("|---:|---:|---:|---:|---:|---:|")
    for r in tab.iter_rows(named=True):
        lv_c = r["lv_given_click"]
        say(
            f"| {r['tab']} | {r['n']:,} | {r['n'] / total:.2%} | {r['click_rate']:.4f} "
            f"| {r['long_view_rate']:.4f} | {lv_c:.4f} |"
            if lv_c is not None
            else f"| {r['tab']} | {r['n']:,} | {r['n'] / total:.2%} | {r['click_rate']:.4f} "
            f"| {r['long_view_rate']:.4f} | - |"
        )


def section_labels(lf: pl.LazyFrame, cols: list[str]) -> None:
    header("6. 标签正样本率（plan §10.2）")
    present = [c for c in LABELS if c in cols]
    stats = lf.select([pl.col(c).mean().alias(c) for c in present]).collect().row(0)
    say("| Task | Positive Rate | 可用性 |")
    say("|---|---:|---|")
    for col, rate in zip(present, stats):
        if rate is None:
            note = "全空"
        elif rate < 0.001:
            note = "极稀疏，考虑排除"
        elif rate < 0.01:
            note = "稀疏"
        else:
            note = "可用"
        say(f"| `{col}` | {rate:.5f} | {note} |" if rate is not None else f"| `{col}` | - | {note} |")
    say()
    say("> §10 要求最终保留 4–5 个任务，据此表决定砍哪个。")


def section_video(paths: list[Path]) -> None:
    header("7. 视频特征（upload_dt 粒度 / tag 格式 —— plan §14.1）")
    if not paths:
        say("未找到 video_features_basic 文件。")
        return
    lf = scan(paths)
    schema = lf.collect_schema()
    say(f"列: {', '.join(schema.keys())}")
    say()
    sample = lf.select(["video_id", "author_id", "upload_dt", "tag", "video_duration"]).head(5).collect()
    say("样例：")
    say("```")
    say(str(sample))
    say("```")
    say()
    up = lf.select(pl.col("upload_dt")).head(1000).collect()["upload_dt"]
    has_time = any(":" in str(v) for v in up if v is not None)
    say(f"- `upload_dt` 含时分秒: **{has_time}**")
    if not has_time:
        say("  -> 只有日期粒度，候选库必须用 `upload_date < request_date`（严格小于）。")
    tags = lf.select(pl.col("tag")).head(1000).collect()["tag"]
    multi = sum(1 for v in tags if v is not None and "," in str(v))
    say(f"- `tag` 为逗号分隔多值：1000 条样本中 **{multi}** 条含多个 tag，需要拆分后做 multi-hot。")


def section_statistic(paths: list[Path]) -> None:
    header("8. 官方统计文件（plan §7.2 —— 确认泄漏，不作为特征）")
    if not paths:
        say("未找到 video_features_statistic 文件。")
        return
    size = sum(p.stat().st_size for p in paths) / 1024**3
    schema = pl.scan_csv(paths[0]).collect_schema()
    say(f"体积 **{size:.1f} GB**，列数 {len(schema)}（仅读取 schema，不扫描内容）。")
    say()
    say(
        "> 官方文档明确：这些是**整月每日平均**统计量。对任一天的样本而言都含未来信息，"
        "属于严重泄漏，因此本项目不使用该文件，所有物品统计特征统一从 past-only 日志自建（§7.2/§8.3）。"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/KuaiRand-1K/data")
    ap.add_argument("--out", default="results/schema_report.md")
    args = ap.parse_args()

    data_dir = (ROOT / args.data_dir).resolve()
    if not data_dir.is_dir():
        print(f"数据目录不存在: {data_dir}", file=sys.stderr)
        print("请先下载并解压 KuaiRand，见 README Quick Start。", file=sys.stderr)
        return 1

    groups = find_files(data_dir)
    if not groups["log_standard"]:
        print(f"{data_dir} 下没找到 log_standard_*.csv", file=sys.stderr)
        return 1

    say(f"# KuaiRand Schema Report")
    say()
    say(f"数据目录: `{args.data_dir}`")

    section_files(groups, data_dir)
    lf = scan(groups["log_standard"])
    cols = section_log_schema(lf)
    section_nulls(lf, cols)
    section_time(lf)
    section_tab(lf)
    section_labels(lf, cols)
    section_video(groups["video_features_basic"])
    section_statistic(groups["video_features_statistic"])

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(_report) + "\n", encoding="utf-8")
    print(f"\n报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
