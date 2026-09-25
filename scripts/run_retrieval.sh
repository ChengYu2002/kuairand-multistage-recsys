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

# 两个塔：掩码不变性是重点 —— padding_idx 不能替代 mask，写错会让训练/评估尺度差约 4 倍。
python scripts/verify_towers.py --config "$CFG"

# 损失与负采样：accidental hit 屏蔽、采样器接口契约、端到端确定性。
# 契约不统一的话，Week 3 插 exposure / hybrid 就得改别处代码，验收标准第 3 条当场失效。
python scripts/verify_two_tower.py --config "$CFG"

python -m src.retrieval.itemcf --config "$CFG"                                  # ItemCF-50   主基线
python -m src.retrieval.itemcf --config "$CFG" --iuf true                       # ItemCF-50-IUF 活跃度惩罚消融
python -m src.retrieval.itemcf --config "$CFG" --history all_before --score-block 200  # ItemCF-All 历史长度消融
# 只跑已实现的策略。exposure / hybrid 是 Week 3 的内容，现在列进来会让
# set -e 在第三轮直接终止整个流水线。加上时把它们补进这个列表即可。
for NEG in random inbatch; do
  python -m src.retrieval.two_tower --config "$CFG" --negative "$NEG"
done
python -m src.evaluation.negative_sampling_analysis --config "$CFG"
