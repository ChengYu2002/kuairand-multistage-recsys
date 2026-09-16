#!/usr/bin/env bash
# Week 2-3: 四种负采样策略在完全相同的设置下对比
set -euo pipefail
CFG=${1:-configs/retrieval.yaml}
python -m src.retrieval.itemcf --config "$CFG"
for NEG in random inbatch exposure hybrid; do
  python -m src.retrieval.two_tower --config "$CFG" --negative "$NEG"
done
python -m src.evaluation.negative_sampling_analysis --config "$CFG"
