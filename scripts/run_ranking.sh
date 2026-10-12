#!/usr/bin/env bash
# 排序侧（Track B）。顺序即依赖顺序 —— 下游会静默读到上一次的旧文件。
#
# 当前跑完整的 Single-Task / MMoE / CGC / PLE / Selective Sharing 主线。
# 每个模型先验算再训练；set -e 保证任何结构检查失败都会中止，不会带病出表。
set -euo pipefail
CFG=${CFG:-configs/data.yaml}
SEEDS=${SEEDS:-42}          # 开发期单 seed；正式实验 SEEDS="42 43 44"（plan §26）

# ---- 排序侧专用物品元数据（词表只用 train 建，且不看标签；召回侧文件一个不动）----
# 验算紧跟生成：这张表错了不报错，只会让四个模型一起错到同一个方向。
python -m src.features.rank_item_static  --config "$CFG"
python scripts/verify_rank_item_static.py --config "$CFG"

# ---- User x Author 偏好特征（plan §8.4）。补它是因为 is_follow 的 GAUC 只有 0.49014，
# 用户内部排序≈随机 —— 模型没有任何 user x author 交互特征。
python -m src.features.pair_preference    --config "$CFG"
python scripts/verify_pair_preference.py  --config "$CFG"
# 装载器只读，但搬错了不报错：行对齐与捷径扫描必须每次验
python scripts/verify_ranking_data.py     --config "$CFG"

# ---- 模型与输入层的一次性验算（不依赖具体 seed）----
# 指标实现是四个模型共用的唯一入口：它错了 ΔAUC 会整体偏移，所以也要进常设回归。
python scripts/verify_ranking_metrics.py
python scripts/verify_single_task.py

# ---- Single-Task（§27.1 里 ΔAUC 的分母）----
# 默认不评 test：开发期反复看 test 会把它变成第二个 validation，而 test 只能用一次。
# 架构与超参全部冻结后，再统一加 --eval-test 跑一次。
for SEED in $SEEDS; do
  python -m src.ranking.single_task --config configs/single_task.yaml --seed "$SEED"
done

# ---- MMoE（plan §23）。输入指纹与 Single-Task 逐字相同，ΔAUC 才可归因 ----
python scripts/verify_mmoe.py
for SEED in $SEEDS; do
  python -m src.ranking.mmoe --config configs/mmoe.yaml --seed "$SEED"
done

# ---- CGC / PLE（plan §24）。一个文件两个模型：--levels 1 = CGC，2 = PLE ----
# 专家宽度已压到 [128,96] 使 dense 参数与 MMoE 可比（§25.1）：CGC 0.81x / PLE 1.17x
python scripts/verify_ple.py
for SEED in $SEEDS; do
  python -m src.ranking.ple --config configs/ple.yaml --seed "$SEED" --levels 1
  python -m src.ranking.ple --config configs/ple.yaml --seed "$SEED"
done

# ---- Selective Sharing（plan §25）：共享表示逐维门控 + 每任务私有表示 ----
python scripts/verify_selective_sharing.py
for SEED in $SEEDS; do
  python -m src.ranking.selective_sharing --config configs/selective.yaml --seed "$SEED"
done

# ---- 汇总：逐任务 mean ± std 与配对 ΔAUC / ΔGAUC ----
python -m src.evaluation.multitask_analysis

# ---- 以下未实现（Week 4-5）----
# python -m src.analysis.multitask_transfer
