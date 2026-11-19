#!/usr/bin/env bash
# DIN-softmax target-aware history pooling 的预注册实验（已完成）。
#
# 只在代码 review 通过后运行。正式范围已经在 results/din_prediction.json 固定为
# 五任务 x seeds 42/43/44；开发阶段只评 valid，不传 --eval-test。
set -euo pipefail

python scripts/verify_din.py
python scripts/verify_ranking_data.py \
  --split valid \
  --encoder-config configs/din.yaml

for SEED in 42 43 44; do
  python -m src.ranking.din --config configs/din.yaml --seed "$SEED"
done

python -m src.evaluation.din_analysis --protocol main --split valid
