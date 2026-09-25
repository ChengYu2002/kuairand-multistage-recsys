# KuaiRand Multi-Stage Recommender

> Industrial-style multi-objective short-video recommendation system built on KuaiRand:
> global temporal splitting, T-1 feature snapshots, two-tower retrieval, and multi-task ranking.

**状态：🚧 Week 2 进行中 —— 召回主线已跑通，排序尚未开始。**
完整设计见本地个人笔记 `note/plan_architecture.md`（不入库）。

已完成：数据管线、候选库与词表、召回指标、ItemCF、全库检索、双塔（random / in-batch）。
未完成：Single-Task 排序基线、exposure / hybrid 负采样、MMoE / PLE / Selective Sharing。

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
Multi-channel Retrieval        (Two-Tower / ItemCF —— 见下方说明)
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

> **关于 candidate union**：上图是工业链路的完整形态，真实系统会把多路召回合并成
> 一个候选集再交给排序。但本项目**不用 union 产出任何上报的召回指标**，每一路召回
> 都在同一个候选库上独立评估。原因是研究问题一要比较四种负采样策略对双塔的影响 ——
> 一旦把 ItemCF 的结果并进候选集，双塔的 Recall 就不再可归因，策略间的差异会被另一路
> 召回的贡献掩盖。ItemCF 在本项目中的定位是**经典非神经基线**，不是召回源。
> Track B（排序与多任务）在打过标的曝光日志上训练与评估，也不消费 union，因此
> union 目前没有实现，且不是 P0 交付物。

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

- **Protocol A — Warm Catalog**：只含 **train 段**出现过、且曝光次数达到频次门槛的 item。
  不含 warmup 与 valid：valid 保持干净调参集的身份。口径与实测规模（KuaiRand-1K）：

  | 协议 | 门槛 | 候选库 item | 有效 request | request 覆盖率 |
  |---|---|---:|---:|---:|
  | 主协议 | `train_freq >= 5` | 194,310 | 66,536 | 12.1% |
  | 辅助协议 | `train_freq >= 1` | 1,708,902 | 120,415 | 21.9% |

- **Protocol B — Time-aware Available Catalog**：允许 `upload_date < request_date` 的未见新
  item（`upload_dt` 仅日期粒度，故为严格小于）。**尚未实现**。
- **Request Unit**：测试窗口内**每一个正向点击 / 有效播放事件**（`is_click or long_view`）
  = 一个 retrieval request。测试窗共 550,316 个正向事件，其中 66,536 个的目标 item 落在
  主协议候选库内。注意 66,536 个事件只对应 51,746 个不同的 `(user, time_ms)` 查询时刻 ——
  一次请求返回一批视频，用户可能点击其中多个，这些事件共享同一份查询上下文。

> 必须与 Recall 一并披露的限制：频次门槛使候选库偏向较热门 item，主协议仅覆盖 12.1%
> 的测试正向事件。这个低覆盖率有**两层成因，不能只归给门槛，也不能说与门槛无关**：
>
> 1. 门槛本身让覆盖率从 21.9% 降到 12.1%，即少掉 53,879 个 request（9.8 个百分点）；
> 2. 21.9% 这个**上限**则来自 Protocol A「训练期必须见过」的前提 —— 测试期 78.1% 的
>    正向事件，其目标 item 在训练段一次都没出现过（本数据集 72.6% 的视频是 31 天窗口
>    期内上传的，内容换代极快）。
>
> 也就是说：调门槛最多把覆盖率拉回 21.9%，再往上只能靠 Protocol B + 带 side feature
> 的物品塔。
>
> The sampled catalog is an offline approximation and inherits selection bias from the
> sampled users and logged exposures.

## 7. Results

### 7.1 Negative Sampling (Two-Tower)

**非神经基线（双塔必须跨过的地板）** —— Protocol A，66,536 条考题，按 request 平均：

| Baseline | Recall@50 | Recall@100 | Recall@500 | NDCG@100 |
|---|---:|---:|---:|---:|
| Random guess | — | 0.000515 | — | — |
| **Popularity** | **0.00222** | **0.00490** | **0.02301** | **0.00101** |
| ItemCF-50（主基线） | 0.00095 | 0.00150 | 0.00942 | 0.00036 |
| ItemCF-50-IUF（消融） | 0.00092 | 0.00167 | 0.00899 | 0.00038 |
| ItemCF-All（消融） | 0.00005 | 0.00017 | 0.00532 | 0.00003 |

观察记录（非实现缺陷，已由 19 项独立验算排除实现错误）：

- **标准余弦 ItemCF 在本数据上低于热度基线。** 机制是极稀疏共现下的冷门偏置：每个
  item 平均只被 6.2 个用户看过，绝大多数共现次数为 1，`|Ui∩Uj| / √(|Ui||Uj|)` 的分子
  近似常数，排序几乎完全由分母决定，于是余弦退化成「谁更冷门谁排前面」。实测
  ItemCF-50 的 Top-100 中位 `train_freq` = 7，而考题答案中位 = 14、候选库全体中位 = 8
  —— 推荐分布与目标分布方向相反，因此可以低于随机猜测。
- **历史越长反而越差**（ItemCF-All < ItemCF-50）：历史从 50 条放大到中位 4,271 条后，
  被翻出来的超冷门 item 更多，冷门偏置被进一步放大。
- **IUF 影响很小且方向不一致**（@100 略升、@500 略降），不进主线。
- **UserCF 一次性诊断**：R@100 ≈ 0.0087（静态用户画像口径，未做正式实现与验算），
  用于确认协同信号确实存在、排除「CF 在本数据上全盘失效」。因「最相似用户」这一量
  在 1,000 用户抽样下被抽样本身扭曲（而 item-item 共现是物品目录的真实属性，样本量
  增大会收敛），UserCF 不进主线，仅作记录。

必须与上表一并披露的口径：

- **补位**：ItemCF-50 有 **13.28%** 的查询时刻非零候选不足 500，尾部由 `pad_order`
  （默认 `catalog_asc`，热门优先）填充，这部分 Top-K 不由 ItemCF 决定。改用
  `catalog_desc` 时 Recall@500 约差 5%（0.00942 vs 0.00894）。
- **不排除已看视频**：为使各召回方法的后处理完全一致，一律不排除用户历史视频。
  代价是热度基线 Top-50 中有 26.3% 是该用户训练期已曝光的项。
- **查询单位**：66,536 条考题只对应 51,746 个不同的 `(user_id, time_ms)`；同一时刻的
  考题共享同一份 Top-K。

**四种负采样策略对比** —— 同一候选库、同一考题集、除负采样外所有变量固定（§16.6）。
20,000 步，embedding_dim 64，temperature 0.05，10 负样本/正样本，item tower = ID-only：

| Strategy | Recall@50 | Recall@100 | Recall@500 | NDCG@100 | 末 100 步 loss |
|---|---:|---:|---:|---:|---:|
| **Random** | **0.00513** | **0.00888** | **0.03523** | **0.00196** | 0.2476 |
| In-batch | 0.00277 | 0.00531 | 0.02579 | 0.00112 | 0.2977 |
| Exposure | *未实现* | | | | |
| Hybrid | *未实现* | | | | |

Random 达到热度基线的 **1.81 倍**，In-batch 仅 1.08 倍。

> ⚠️ **当前结果只有 seed 42 一个种子。** Random 与 In-batch 的差距有多少来自种子噪声
> 尚未验证；§26 要求的 3 seeds + mean ± std 对负采样对比同样适用，报告前必须补齐。

训练与评估口径：正样本取 train 段正向事件且目标落在候选库内（121 万条，占全部正样本
的 58%）。**正负样本必须同空间** —— 若正样本可在词表任意位置而负样本只来自候选库，
「不在候选库」就完美预测「是正样本」（占 42%），而成员身份恰好由 index 区间编码。

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
│   ├── retrieval/      ItemCF、Two-Tower、负采样、全库检索（暴力分块，非 ANN）、数据装载
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

Retrieval P0 只含 ItemCF + Two-Tower + 四种负采样 + Recall/NDCG 评估。**不含 candidate
union** —— 每一路召回独立评估，理由见 §2。全库检索用**精确的分块暴力**而非 ANN：
19.4 万候选暴力只要 13.3 秒，而 ANN 的近似误差会混进四种策略的对比里（§16.6 要求
除采样外无任何差异）。在多任务主线（MMoE / PLE / Selective Sharing /
Controlled Sparsity / 3-seed）全部完成前，**不新增** popularity / author / tag / freshness
等启发式召回源，也不新增第二个协同过滤基线（UserCF 已作一次性诊断，结论见 §7.1）。时间预算约为
data 25% / retrieval 25% / MTL 35% / analysis 15%。

## 11. License

MIT（见 [LICENSE](LICENSE)）。KuaiRand 数据集版权归原作者所有，本仓库不包含任何原始数据。
