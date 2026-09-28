#!/usr/bin/env bash
# 排序侧（Track B）。顺序即依赖顺序 —— 下游会静默读到上一次的旧文件。
#
# 当前只跑 Single-Task。MMoE / PLE / Selective Sharing 与后续分析还是 stub，
# 已改成显式抛错（退出码 1），所以配合 set -e 不会出现「脚本全绿但什么都没跑」。
# Week 3 实现后把下面 MTL 段的注释解开。
set -euo pipefail
CFG=${CFG:-configs/data.yaml}
SEEDS=${SEEDS:-42}          # 开发期单 seed；正式实验 SEEDS="42 43 44"（plan §26）

# ---- 排序侧专用物品元数据（词表只用 train 建，且不看标签；召回侧文件一个不动）----
# 验算紧跟生成：这张表错了不报错，只会让四个模型一起错到同一个方向。
python -m src.features.rank_item_static  --config "$CFG"
python scripts/verify_rank_item_static.py --config "$CFG"
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

# ---- 以下均未实现（Week 3-5）。解开注释前它们会以退出码 1 中止本脚本 ----
# for SEED in $SEEDS; do
#   python -m src.ranking.mmoe              --config configs/mmoe.yaml       --seed "$SEED"
#   python -m src.ranking.ple               --config configs/ple.yaml        --seed "$SEED"
#   python -m src.ranking.selective_sharing --config configs/selective.yaml  --seed "$SEED"
# done
# python -m src.evaluation.multitask_analysis
# python -m src.analysis.multitask_transfer
