#!/usr/bin/env bash
# Week 3-4: Single-Task / MMoE / PLE / Selective Sharing，3 seeds
set -euo pipefail
for SEED in 42 43 44; do
  python -m src.ranking.single_task       --config configs/single_task.yaml --seed "$SEED"
  python -m src.ranking.mmoe              --config configs/mmoe.yaml       --seed "$SEED"
  python -m src.ranking.ple               --config configs/ple.yaml        --seed "$SEED"
  python -m src.ranking.selective_sharing --config configs/selective.yaml  --seed "$SEED"
done
python -m src.evaluation.multitask_analysis
python -m src.analysis.multitask_transfer
