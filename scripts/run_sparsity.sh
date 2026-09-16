#!/usr/bin/env bash
# Week 4: Controlled Label Sparsity，固定架构/特征/切分，只改正样本信号量
set -euo pipefail
python -m src.analysis.controlled_sparsity --config configs/sparsity.yaml
