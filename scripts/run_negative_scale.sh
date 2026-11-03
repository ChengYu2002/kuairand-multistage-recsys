#!/usr/bin/env bash
# 负样本规模实验：固定「最佳配置」，只改负样本数量（§16.5 之外的独立一项）
#
# 这不是 run_retrieval.sh 的一部分，也不能并进它。理由：
#   run_retrieval.sh 回答的是「负样本从哪儿来」—— 五个臂，n_neg 一律 10（§16.6
#   把 negatives_per_positive 列为受控变量）。本脚本回答的是「负样本有多少个」，
#   它**故意**动了那个受控变量，所以必须是独立的入口、独立的汇总文件。
#   negative_sampling_analysis.py 里有一处显式 `continue` 把 inbatch_full_logq
#   排除在五臂表之外，那是同一个决定的另一半，别去掉。
#
# 受控对比（除负样本数量外一切相同）：
#   基线  inbatch_logq       n_neg=10        已跑完，本脚本不重跑
#   实验  inbatch_full_logq  B-1 = 1023      本脚本跑
# 两者共用 id_side 物品塔、同一 lr / batch_size / temperature / max_steps / seed。
#
# 代价：实测 139 ms/step x 20000 步 ≈ 46 分钟/seed，3 个 seed ≈ 2.3 小时（CPU，单机）。
# 比 n_neg=50 更便宜（外推 223 ms/step ≈ 74 分钟/seed）却多拿 20 倍负样本 ——
# 因为每步有 79% 的固定开销（Adam 更新 1.73 亿参数），与 n_neg 无关。
set -euo pipefail
CFG=${1:-configs/retrieval.yaml}
TOWER=${TOWER:-id_side}
SEEDS=${SEEDS:-"42 43 44"}
PY=${PY:-.venv/bin/python}

# 先确认基线在：配对差值需要同 seed 的两份结果，缺了的话跑完 2.3 小时才发现。
for SEED in $SEEDS; do
  BASE="experiments/two_tower_inbatch_logq_${TOWER}_main_seed${SEED}.pt"
  if [ ! -f "$BASE" ]; then
    echo "缺基线 $BASE —— 配对对比需要同 seed 的 n_neg=10 结果。" >&2
    echo "先跑：$PY -m src.retrieval.two_tower --config $CFG \\" >&2
    echo "        --negative inbatch_logq --item-tower-config $TOWER --seed $SEED" >&2
    exit 1
  fi
done

# 尺子先于结果。四把尺子各自挡住一类「不会靠 loss 曲线暴露」的错：
$PY scripts/verify_retrieval_metrics.py --config "$CFG"   # 共用度量入口，错了所有臂一起错
$PY scripts/verify_towers.py --config "$CFG"              # 掩码不变性；id_side 依赖物品侧特征
$PY scripts/verify_two_tower.py --config "$CFG"           # 端到端确定性 —— 旧 sampled 路径未被动过
$PY scripts/verify_logq.py --config "$CFG"                # Q 的口径：全局训练正样本频率
$PY scripts/verify_full_inbatch.py --config "$CFG"        # 本臂专属：日期分组、重复屏蔽、logQ 的轴

for SEED in $SEEDS; do
  $PY -m src.retrieval.two_tower --config "$CFG" \
    --negative inbatch_full_logq --item-tower-config "$TOWER" --seed "$SEED"
done

# 两张表，缺一张读不出机制：
#   negative_scale_analysis   整体 Recall + 配对差值（规模到底换来了什么）
#   cold_start_analysis       频次分层（换来的东西落在 tail 还是 warm）
# 没有 global_component_rho：它对 id_side 直接拒绝运行（item 向量随评估日变化，
# ū·v_i 不是单一向量）。要这个数得先定义「按哪一天」，那是另一个决定。
$PY scripts/negative_scale_analysis.py --config "$CFG" --tower "$TOWER" --seeds $SEEDS
$PY -m src.analysis.cold_start_analysis --config "$CFG" --tower "$TOWER" \
  --strategies inbatch_logq inbatch_full_logq --seeds $SEEDS
