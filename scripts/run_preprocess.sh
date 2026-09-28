#!/usr/bin/env bash
# Week 1-2: 数据管线。顺序即依赖顺序，不能随意调换 —— 下游模块会静默读到上一次的旧文件。
set -euo pipefail
CFG=${1:-configs/data.yaml}
# 候选库 / 词表 / 编码的口径属于评估协议，存在 retrieval.yaml。
RCFG=${2:-configs/retrieval.yaml}

# ---- 原始日志 -> 切分 -> 历史 ----
python -m src.preprocessing.sample_users    --config "$CFG"
python -m src.preprocessing.preprocess      --config "$CFG"
python -m src.preprocessing.temporal_split  --config "$CFG"
python -m src.preprocessing.build_history   --config "$CFG"
python -m src.preprocessing.build_labels    --config "$CFG"
python -m src.preprocessing.leakage_check   --config "$CFG"

# ---- T-1 日级特征（plan §8）。验算紧跟生成：口径错误不会抛异常 ----
python -m src.features.daily_user_features  --config "$CFG"
python -m src.features.daily_item_features  --config "$CFG"
python scripts/verify_t1_features.py        --config "$CFG"
python scripts/verify_history.py            --config "$CFG"

# ---- 候选库与词表 ----
python -m src.preprocessing.build_catalog   --config "$RCFG" --data-config "$CFG"
python scripts/verify_catalog.py            --config "$RCFG" --require-golden
# 词表依赖候选库，且必须落盘 —— 重算时顺序一变，checkpoint 就静默失效。
python -m src.features.build_vocab          --config "$RCFG"
python scripts/verify_vocab.py              --config "$RCFG" --require-golden

# ---- 物品静态属性（② video_age / ③ tag multi-hot）。依赖词表 ----
python -m src.features.video_age            --config "$RCFG"
python scripts/verify_item_static.py        --config "$RCFG" --require-golden

# ---- 编码规格（⑤）。**依赖上面的 T-1 特征文件**，顺序不能提前 ----
python -m src.features.feature_encoder      --config "$RCFG"
python scripts/verify_encoder.py            --config "$RCFG"

# ---- 特征 join（⑥）。依赖编码规格 + 词表 + T-1 特征 ----
python -m src.features.build_samples        --config "$RCFG"
python scripts/verify_samples.py            --config "$RCFG"
