#!/usr/bin/env bash
# DIN-raw（不做 softmax）的预注册实验。
#
# 只在代码 review 通过后运行。正式范围已在 results/din_raw_prediction.json 固定为
# 五任务 x seeds 42/43/44；开发阶段只评 valid，不传 --eval-test。
set -euo pipefail

# 已完成的 DIN-softmax checkpoint 显示 attention 模块塌缩。DIN-raw 正式训练必须先完成
# 仪表化短跑并人工解释，随后显式落盘 go/no-go；仅有代码验算通过还不够。
if [[ ! -f results/din_raw_go_no_go.json ]]; then
  echo "拒绝启动：缺少 results/din_raw_go_no_go.json。" >&2
  echo "先 review 并运行 scripts/diagnose_din_training_dynamics.py，再根据轨迹决定修法。" >&2
  exit 1
fi
python - <<'PY'
import json
from pathlib import Path

decision = json.loads(Path("results/din_raw_go_no_go.json").read_text("utf-8"))
if decision.get("decision") != "run":
    raise SystemExit("拒绝启动：din_raw_go_no_go.json 的 decision 不是 'run'")
PY

python scripts/verify_din.py
python scripts/verify_ranking_data.py \
  --split valid \
  --encoder-config configs/din_raw.yaml

for SEED in 42 43 44; do
  python -m src.ranking.din_raw --config configs/din_raw.yaml --seed "$SEED"
done

python scripts/diagnose_din_attention.py --protocol main
python -m src.evaluation.din_raw_analysis --protocol main --split valid
