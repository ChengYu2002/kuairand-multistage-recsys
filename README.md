# KuaiRand Multi-Stage Recommender

> Industrial-style multi-objective short-video recommendation system built on KuaiRand:
> global temporal splitting, T-1 feature snapshots, two-tower retrieval, and multi-task ranking.

**状态：🚧 骨架搭建中（Week 1）。** 完整设计见 [plan_architecture.md](plan_architecture.md)。

---

## 1. 两个核心研究问题

1. **Negative Sampling Study** — 不同负样本构造策略（random / in-batch / exposure / hybrid）
   如何影响双塔召回在全库检索中的效果？
2. **Multi-Task Learning + Selective Sharing** — 在不同任务稀疏度与可比参数预算下，
   不同共享机制（Single-Task / MMoE / PLE / Selective Sharing）如何影响正迁移与负迁移？

## 2. System Architecture

```text
Raw KuaiRand Logs
        ↓
Feature / Data Pipeline        (user sampling → temporal split → T-1 snapshots)
        ↓
Multi-channel Retrieval        (Two-Tower + ItemCF → candidate union)
        ↓
[Optional] Coarse Ranking      (LightGBM, P1)
        ↓
Multi-task Fine Ranking        (Single-Task / MMoE / PLE / Selective Sharing)
        ↓
Score Fusion → Re-ranking
        ↓
Offline Evaluation
```

架构是完整链路，但离线实验**不要求所有阶段 end-to-end 串联训练**，分为
Track A (Retrieval) 与 Track B (Ranking & Multi-task) 两条轨道。

## 3. Dataset & Split

| | |
|---|---|
| 开发数据集 | KuaiRand-1K |
| 主实验数据集 | KuaiRand-27K，按用户抽样 2k–3k users |
| 抽样单位 | **user**（保留完整历史，禁止按行随机抽样） |
| 切分方式 | Global Temporal Split，`max(train) < min(val) < min(test)` |

具体日期在 EDA 之后确定，不提前写死。

## 4. Leakage Control

严格区分 **prediction-time feature** 与 **post-interaction outcome**。
播放时长、主页/评论区停留时长、是否进入主页等曝光后行为**只能作为 label 或分析对象**。
官方每日统计文件不直接作为特征，统一从 past-only logs 自建。

> Note: snapshot-style user features (follow / fan counts 等) may contain limited future
> information relative to earlier interactions.

## 5. T-1 Feature Pipeline

Day T 的样本只允许使用 **Day T-1 及更早**日志生成的聚合特征，按
`user+date` / `item+date` / `author+date` / `tag+date` 预聚合后按日期 join 回样本。

## 6. Candidate Catalog & Request Unit

- **Protocol A — Warm Catalog**：只含 train/val 阶段出现过的 item。
- **Protocol B — Time-aware Available Catalog**：允许 `upload_date < request_date` 的未见新 item。
- **Request Unit**：测试窗口内**每一个正向点击 / 有效播放事件** = 一个 retrieval request。

> The sampled catalog is an offline approximation and inherits selection bias from the
> sampled users and logged exposures.

## 7. Results

### 7.1 Negative Sampling (Two-Tower)

| Strategy | Recall@50 | Recall@100 | Recall@500 | NDCG@100 |
|---|---:|---:|---:|---:|
| Random | | | | |
| In-batch | | | | |
| Exposure | | | | |
| Hybrid | | | | |

### 7.2 Item Tower / Cold Start

| Item Tower | Overall R@100 | Warm R@100 | Cold R@100 | Tail R@100 |
|---|---:|---:|---:|---:|
| ID-only | | | | |
| ID + Side Features | | | | |

### 7.3 Multi-task Ranking (mean ± std over 3 seeds)

| Model | Click AUC | Long-view AUC | Like AUC | Follow AUC | Comment AUC |
|---|---:|---:|---:|---:|---:|
| Single-Task | | | | | |
| MMoE | | | | | |
| PLE | | | | | |
| Selective Sharing | | | | | |

### 7.4 Controlled Label Sparsity

正样本整行下采样至 100% / 10% / 1% / 0.1%，固定架构、特征、用户集、时间切分与种子，
只改变可用正向训练信号，观察各共享机制的 ΔAUC 与负迁移敏感性。

## 8. Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 把 KuaiRand 原始文件放到 data/raw/ 后：
bash scripts/run_preprocess.sh configs/data.yaml
bash scripts/run_retrieval.sh  configs/retrieval.yaml
bash scripts/run_ranking.sh
bash scripts/run_sparsity.sh
```

## 9. Repository Structure

```text
├── configs/        实验配置（data / retrieval / mmoe / ple / selective / sparsity ...）
├── data/           raw / processed / cache（不入库）
├── src/
│   ├── preprocessing/  抽样、清洗、时间切分、历史序列、候选库、标签、泄漏检查
│   ├── features/       T-1 日级 user / item / author / tag 特征、video age、编码
│   ├── retrieval/      ItemCF、Two-Tower、四种负采样、LogQ、ANN
│   ├── ranking/        Single-Task、MMoE、PLE、Selective Sharing、DIN(P2)
│   ├── prerank/        LightGBM 粗排 (P1)
│   ├── analysis/       受控稀疏、负迁移、冷启动、参数预算
│   ├── reranking/      分数融合、类目/作者多样性
│   ├── evaluation/     召回与排序指标、结果汇总
│   └── utils/          config / logger / seed
├── scripts/        端到端运行脚本
├── notebooks/      EDA 与图表
├── experiments/    运行产物（不入库）
├── results/        最终表格与图（不入库）
└── tests/
```

## 10. Scope Guardrail

Retrieval P0 只含 ItemCF + Two-Tower + 四种负采样 + Recall/NDCG 评估 + 简单 candidate union。
在多任务主线（MMoE / PLE / Selective Sharing / Controlled Sparsity / 3-seed）全部完成前，
**不新增** popularity / author / tag / freshness 等启发式召回源。时间预算约为
data 25% / retrieval 25% / MTL 35% / analysis 15%。

## 11. License

MIT（见 [LICENSE](LICENSE)）。KuaiRand 数据集版权归原作者所有，本仓库不包含任何原始数据。
