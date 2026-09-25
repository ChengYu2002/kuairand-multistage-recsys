#!/usr/bin/env bash
# Week 1: sample users -> preprocess -> temporal split -> history -> catalog -> labels
set -euo pipefail
CFG=${1:-configs/data.yaml}
# 候选库的口径（门槛 / 训练段范围 / 正向信号）属于评估协议，存在 retrieval.yaml。
RCFG=${2:-configs/retrieval.yaml}
python -m src.preprocessing.sample_users    --config "$CFG"
python -m src.preprocessing.preprocess      --config "$CFG"
python -m src.preprocessing.temporal_split  --config "$CFG"
python -m src.preprocessing.build_history   --config "$CFG"
python -m src.preprocessing.build_catalog   --config "$RCFG" --data-config "$CFG"
python -m src.preprocessing.build_labels    --config "$CFG"
python -m src.preprocessing.leakage_check   --config "$CFG"

# T-1 日级特征（plan §8）。验算必须跟在生成之后：口径错误不会抛异常。
python -m src.features.daily_user_features  --config "$CFG"
python -m src.features.daily_item_features  --config "$CFG"
python scripts/verify_t1_features.py        --config "$CFG"
python scripts/verify_history.py            --config "$CFG"
python scripts/verify_catalog.py            --config "$RCFG" --require-golden
