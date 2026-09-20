"""T-1 日级物品聚合特征（plan §8.3）。

结构与用户维度相同，另含热度趋势 `item_trend`（近 1 天曝光 / 7 天日均）与
冷启动标记 `item_has_history` —— 本数据集 65% 的视频仅出现过一次，缺历史是常态，
必须让模型能区分「统计值为 0」和「没有统计值」。

用法：
    python -m src.features.daily_item_features --config configs/data.yaml
"""

from __future__ import annotations

from src.features._driver import build

if __name__ == "__main__":
    raise SystemExit(build("video_id", "item", "feat_item_daily.parquet", __name__))
