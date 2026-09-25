#!/usr/bin/env bash
# Week 2-3: 四种负采样策略在完全相同的设置下对比
set -euo pipefail
CFG=${1:-configs/retrieval.yaml}

# 尺子必须先于任何模型验过：retrieval_metrics 是 ItemCF / 双塔 / 四种负采样共用的
# 唯一度量入口，它算错的话所有模型会一起错到同一个方向，对比表看上去依然正常。
python scripts/verify_retrieval_metrics.py --config "$CFG"

# ItemCF 的实现验算必须先于结果：它在本数据上低于热度基线，而「实现错了」和
# 「共现太稀疏」在指标上长得一样，只有验算能把两者切开。
python scripts/verify_itemcf.py --config "$CFG"

# 全库检索：双塔评估要用它把向量变成 Top-K。与模型无关，先验过再谈模型。
# （exposure 负样本直接取自「曝光但未正向」的日志，不经过打分器，别混为一谈。）
python scripts/verify_ann_index.py --full-scale

python -m src.retrieval.itemcf --config "$CFG"                                  # ItemCF-50   主基线
python -m src.retrieval.itemcf --config "$CFG" --iuf true                       # ItemCF-50-IUF 活跃度惩罚消融
python -m src.retrieval.itemcf --config "$CFG" --history all_before --score-block 200  # ItemCF-All 历史长度消融
for NEG in random inbatch exposure hybrid; do
  python -m src.retrieval.two_tower --config "$CFG" --negative "$NEG"
done
python -m src.evaluation.negative_sampling_analysis --config "$CFG"
