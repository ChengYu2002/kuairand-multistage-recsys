#!/usr/bin/env bash
# Week 2-3: 五个负采样实验臂在完全相同的设置下对比（§16.6）
#
# 这个脚本就是 §7.1 / §7.2 那几张表的复现路径，所以它跑的范围必须与表一致：
# 五个臂 x 3 个 seed。单臂单 seed 跑不出 mean ± std，也跑不出配对差值。
# 代价：双塔部分约 5 x 3 x 40 分钟 ≈ 10 小时（CPU，单机）。要缩范围就改下面两个
# 变量，但那时 negative_sampling_analysis 会按护栏拒绝覆盖完整的汇总 —— 那是故意的。
set -euo pipefail
CFG=${1:-configs/retrieval.yaml}
ARMS=${ARMS:-"random inbatch inbatch_logq exposure hybrid"}
SEEDS=${SEEDS:-"42 43 44"}

# 尺子必须先于任何模型验过：retrieval_metrics 是 ItemCF / 双塔 / 各负采样臂共用的
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
python scripts/verify_two_tower.py --config "$CFG"

# 曝光池（CSR 结构、空池 fallback、浮点边界）与 Hybrid 的切分布局。
python scripts/verify_exposure_negative.py --config "$CFG"

# logQ：Q 表的口径（全局训练正样本频率，非 batch 经验频率）与「正负样本都要校正」。
python scripts/verify_logq.py --config "$CFG"

python -m src.retrieval.itemcf --config "$CFG"                                  # ItemCF-50   主基线
python -m src.retrieval.itemcf --config "$CFG" --iuf true                       # ItemCF-50-IUF 活跃度惩罚消融
python -m src.retrieval.itemcf --config "$CFG" --history all_before --score-block 200  # ItemCF-All 历史长度消融

for NEG in $ARMS; do
  for SEED in $SEEDS; do
    python -m src.retrieval.two_tower --config "$CFG" --negative "$NEG" --seed "$SEED"
  done
done

# 三张表，口径各不相同，缺一张就读不出机制：
#   negative_sampling_analysis  聚合 Recall + 配对差值（谁高谁低）
#   cold_start_analysis         频次分层（好在哪一层、代价付在哪一层）
#   global_component_rho        ū·v_i 与 train_freq 的 ρ（学到的热度先验朝哪边）
python -m src.evaluation.negative_sampling_analysis --config "$CFG"
python -m src.analysis.cold_start_analysis --config "$CFG"
python scripts/global_component_rho.py --config "$CFG" --check
