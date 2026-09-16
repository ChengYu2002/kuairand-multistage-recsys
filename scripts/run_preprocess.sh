#!/usr/bin/env bash
# Week 1: sample users -> preprocess -> temporal split -> history -> catalog -> labels
set -euo pipefail
CFG=${1:-configs/data.yaml}
python -m src.preprocessing.sample_users    --config "$CFG"
python -m src.preprocessing.preprocess      --config "$CFG"
python -m src.preprocessing.temporal_split  --config "$CFG"
python -m src.preprocessing.build_history   --config "$CFG"
python -m src.preprocessing.build_catalog   --config "$CFG"
python -m src.preprocessing.build_labels    --config "$CFG"
python -m src.preprocessing.leakage_check   --config "$CFG"
