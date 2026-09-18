"""T-1 日级用户聚合特征（plan §8.2）。

产出 (user_id, date) -> 过去 1/3/7 天的曝光、各标签计数、比率（朴素 + 平滑）、
活跃天数、消费广度。其中 date 的含义是「这组特征适用于哪一天的样本」，
数值来自该日期**之前**的日志。

用法：
    python -m src.features.daily_user_features --config configs/data.yaml
"""

from __future__ import annotations

from src.features._driver import build

if __name__ == "__main__":
    raise SystemExit(build("user_id", "user", "feat_user_daily.parquet", __name__))
