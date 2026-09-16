# Industrial-style 短视频多目标推荐系统：最终执行版

## 1. 项目定位

本项目定位为：

> **Industrial-style Multi-Objective Short-Video Recommendation System**

目标不是“在公开数据集上跑几个模型”，而是构建一个结构完整、数据切分严谨、实验问题清晰、可用于推荐算法求职与多任务研究的短视频推荐项目。

系统主链路：

```text
Raw KuaiRand Logs
        ↓
Feature / Data Pipeline
        ↓
Multi-channel Retrieval
        ↓
[Optional] Coarse Ranking
        ↓
Multi-task Fine Ranking
        ↓
Score Fusion
        ↓
Re-ranking
        ↓
Offline Evaluation
```

需要明确：

> **系统架构是完整链路，但离线实验不要求所有阶段严格 end-to-end 串联训练。**

实际分成两条主要实验轨道：

```text
Track A — Retrieval
Two-Tower
+ ItemCF
+ Negative Sampling
+ Warm / Cold Retrieval Analysis

Track B — Ranking & Multi-task
Single-Task
+ MMoE
+ PLE
+ Selective Sharing
+ Controlled Sparsity
```

---

# 2. 项目只保留两个核心亮点

为了控制范围，项目不再包装成“四个亮点”。

## 2.1 核心亮点一：Two-Tower Negative Sampling Study

研究问题：

> 不同负样本构造策略如何影响双塔召回在全库检索中的效果？

比较：

```text
Random Negative
vs
In-batch Negative
vs
Exposure Negative
vs
Hybrid Negative
```

P1：

```text
In-batch + LogQ Correction
```

重点指标：

```text
Recall@K
HitRate@K
NDCG@K
```

核心原则：

> 不预设 Exposure Negative 一定更优。

曝光未点击样本更接近排序边界，但它本身已经经过上游推荐系统筛选；而 retrieval 需要从更大的候选空间中排除大量明显无关物品。

因此：

```text
Hybrid Negative
```

是 P0 核心实验。

---

## 2.2 核心亮点二：Multi-Task Learning + Selective Sharing

研究问题：

> 在不同任务稀疏度与可比参数预算下，不同共享机制如何影响多任务推荐中的正迁移与负迁移？

比较：

```text
Single-Task
vs
MMoE
vs
PLE
vs
Selective Sharing
```

重点分析：

```text
Per-task AUC
ΔAUC vs Single-Task
Negative Transfer
Controlled Label Sparsity
Parameter Count
Training Cost
```

Selective Sharing 是 P0。

---

# 3. 其他部分的定位

以下内容保留，但不作为“主卖点”。

## 3.1 Dynamic User Tower

这是 KuaiRand 用户数相对有限、但单用户序列很长时的合理架构选择。

## 3.2 Feature-aware Item Tower

用于利用 KuaiRand 的 author / tag / duration / upload time 等 side information，并辅助分析冷启动和长尾。

## 3.3 Warm / Cold / Tail Recall

这是 retrieval 的辅助分析，不是独立研究主线。

## 3.4 LightGBM

属于 P1 的系统完整性模块，不是主线核心。

---

# 4. Dataset

## 4.1 主数据集

正式实验优先使用：

```text
KuaiRand-27K
```

选择理由：

- 提供真实全局时间戳
- 用户与物品特征丰富
- 标准日志是推荐曝光后的交互日志
- 未点击曝光可以作为真实曝光负样本来源
- 提供多种行为标签
- 另有随机曝光日志
- 适合时间感知特征工程
- 适合 MMoE / PLE / Selective Sharing
- item 空间大，可做 warm / cold retrieval 分析

---

## 4.2 开发与兜底数据集

开发使用：

```text
KuaiRand-1K
```

用途：

- preprocessing debug
- schema validation
- 时间切分
- retrieval / ranking pipeline 开发
- MMoE / PLE / Selective Sharing 调试

---

# 5. 数据规模策略

KuaiRand 单用户交互非常密集，因此正式规模不能按 Tenrec 的用户数思路设计。

## 5.1 Development

```text
KuaiRand-1K
```

目标：

> 先在 1K 上把 P0 主流程跑通。

---

## 5.2 Main Experiment

从：

```text
KuaiRand-27K
```

中按用户抽取：

```text
2k–3k users
```

作为主实验起点。

资源允许时再扩到：

```text
3k–5k users
```

如果过重：

```text
只保留后三周日志
```

原则：

```text
完整实验
>
更大用户数
```

---

## 5.3 用户抽样原则

必须：

```text
按用户抽样
```

不能：

```text
按行随机抽样
```

流程：

```text
Sample Users
    ↓
Keep Full Selected-user Histories
    ↓
Global Temporal Split
    ↓
Build Retrieval / Ranking Samples
```

---

## 5.4 Week-1 兜底规则

如果 Week 1 结束时 27K 仍无法稳定完成：

```text
sample
join
split
EDA
cache
```

则：

```text
最终实验直接使用 KuaiRand-1K
```

P0 砍项顺序：

```text
1. 27K 主实验退回 1K
2. 再砍一种负采样策略
3. 最后才考虑动 Selective Sharing 或 3 Seeds
```

---

# 6. Global Temporal Split

KuaiRand 提供真实时间戳，因此采用：

```text
Global Temporal Split
```

原则：

```text
Train < Validation < Test
```

严格按全局时间分离。

---

## 6.1 日期如何确定

具体日期不提前写死。

先做 EDA：

```text
daily impressions
daily users
daily items
daily new items
daily positive rates
daily tab distribution
```

然后选择连续时间窗。

---

## 6.2 切分要求

必须：

```text
max(train_time)
<
min(validation_time)
<
min(test_time)
```

---

# 7. Leakage Control

这是 P0。

必须区分：

```text
Prediction-time Feature
vs
Post-interaction Outcome
```

---

## 7.1 禁止作为输入的字段

例如：

```text
播放时长
主页停留时长
评论区停留时长
曝光后观看结果
是否进入主页
曝光后行为
未来统计量
```

这些只能作为：

```text
label
analysis target
post-hoc analysis
```

---

## 7.2 官方统计文件

默认不直接使用官方每日视频统计作为模型特征。

统一从：

```text
Past-only Logs
```

自己构造。

---

## 7.3 Snapshot User Features

例如：

```text
关注数
粉丝数
social statistics
```

可能是采集时的快照。

可以保留，但 README 中明确说明：

> snapshot-style user features may contain limited future information relative to earlier interactions.

---

# 8. Feature Engineering：统一采用 T-1 日级快照

不要对几千万行做“每条曝光逐行滚动窗口”。

统一使用：

> **T-1 Daily Snapshot**

即：

```text
Day T 的样本
只能使用 Day T-1 及更早日志生成的聚合特征
```

这样：

```text
无 future leakage
+
计算成本可控
+
更接近工业离线特征管线
```

---

## 8.1 计算方式

推荐使用：

```text
Polars / PyArrow / DuckDB
```

先按：

```text
user + date
item + date
author + date
tag + date
```

聚合，再按日期 join 回样本。

---

## 8.2 User Daily Features

例如：

```text
过去 1 天点击率
过去 3 天点击率
过去 7 天点击率

过去 1 天互动次数
过去 3 天互动次数
过去 7 天互动次数

过去 7 天活跃天数
过去 7 天 distinct item 数
```

所有 Day T 特征最多使用：

```text
<= T-1
```

---

## 8.3 Item Daily Features

例如：

```text
过去 1 / 3 / 7 天曝光量
过去 1 / 3 / 7 天点击量
过去 1 / 3 / 7 天 CTR
近期热度
热度变化率
```

---

## 8.4 User-Author Preference

例如：

```text
过去 7 天对作者曝光次数
过去 7 天对作者点击次数
过去 7 天对作者互动率
```

---

## 8.5 User-Tag Preference

例如：

```text
过去 7 天对 tag/category 的曝光数
过去 7 天点击数
过去 7 天互动率
```

---

## 8.6 Video Age

构造：

```text
video_age =
request_date - upload_date
```

如果 upload time 只有日期粒度，则按日期计算，不假设日内先后顺序。

---

## 8.7 Exposure Context

例如：

```text
tab
scene
context fields
```

---

# 9. 用户历史序列

用户塔需要最近行为历史。

不使用逐行 Python loop。

推荐：

```text
按 user 排序
↓
向量化构造历史
↓
取当前曝光之前最近 N 个正向点击/有效观看
```

例如：

```text
recent_click_items_N
```

P0：

```text
Embedding Average / Mean Pooling
```

P2：

```text
DIN-style Attention
```

---

# 10. Labels

候选：

```text
click / valid play
long view
like
follow
comment
```

最终选择：

```text
4–5 tasks
```

---

## 10.1 Scene / Tab Semantics Check

P0。

不同 tab / UI 场景下：

```text
click
```

可能语义不同。

因此必须：

```text
按 tab / scene 做 EDA
```

检查：

```text
label definition
positive rate
behavior distribution
```

再决定：

```text
统一建模
or
仅使用语义一致场景
```

---

## 10.2 Positive Rate Table

| Task | Positive Rate |
|---|---:|
| Click / Valid Play | |
| Long View | |
| Like | |
| Follow | |
| Comment | |

---

# 11. Controlled Label Sparsity

用于研究：

> 稀疏训练信号本身如何影响 Single-Task / MMoE / PLE / Selective Sharing。

---

## 11.1 Task Selection

从最密集且语义稳定的任务中选一个：

```text
Click
or
Long View
```

---

## 11.2 构造

通过：

```text
正样本整行下采样
```

构造：

```text
Original
10%
1%
0.1%
```

不是把正样本改成负样本。

---

## 11.3 固定变量

固定：

```text
architecture
feature set
user set
time split
evaluation set
random seeds
```

只改变：

```text
available positive training signal
```

---

# 12. Overall Architecture

```text
                         KuaiRand
                            │
                            ▼
                    Data / Feature Pipeline
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
        Two-Tower                      ItemCF
              │                           │
              └────── Candidate Union ────┘
                            │
                            ▼
                 [Optional] LightGBM
                      Pre-ranking
                            │
                            ▼
                  Logged Ranking Samples
                            │
      ┌─────────────────────┼─────────────────────┐
      │                     │                     │
      ▼                     ▼                     ▼
 Single-Task              MMoE                   PLE
      │                     │                     │
      └──────────────┬──────┴──────────────┬──────┘
                     │                     │
                     ▼                     ▼
            Selective Sharing       Controlled
               Variant              Sparsity Study
                     │                     │
                     └──────────┬──────────┘
                                ▼
                     Multi-task Probabilities
                                │
                                ▼
                          Score Fusion
                                │
                                ▼
                           Re-ranking
```

---

# 13. Track A — Two-Tower Retrieval

## 13.1 User Tower

用户塔不依赖：

```text
user_id-only embedding
```

而使用：

```text
Recent Positive History
+
User Features
+
T-1 Daily User Features
+
Context
       ↓
History Aggregation
       ↓
MLP
       ↓
Dynamic User State
```

这属于合理架构选择，不作为独立研究亮点。

---

## 13.2 Item Tower

### Config A — ID-only

```text
item_id
↓
embedding
```

### Config B — ID + Side Features

```text
item_id
author_id
tags
duration
upload age
music/type
metadata
↓
feature fusion
↓
item vector
```

P1：

```text
ID Dropout / Rare-ID Masking
```

---

## 13.3 Item Tower 对比定位

只作为 retrieval 的辅助分析：

```text
ID-only
vs
ID + Side Features
```

重点报告：

```text
Overall Recall
Warm Recall
Cold Recall
Tail Recall
```

不单独包装成第三个主研究问题。

---

# 14. Candidate Catalog

这是必须明确的评估口径。

---

## 14.1 上传时间粒度

优先检查 KuaiRand 实际字段。

如果 upload time 只有日期粒度，则规定：

```text
upload_date < request_date
```

而不是：

```text
upload_date <= request_date
```

避免同一天无法判断先后顺序的问题。

---

## 14.2 Protocol A — Warm Catalog

只包含：

```text
train / validation 阶段已见 item
```

用于评价：

```text
classic warm-item retrieval
```

---

## 14.3 Protocol B — Time-aware Available Catalog

允许：

```text
在 request date 之前已上传
```

但训练日志里没出现过的新 item。

用于评价：

```text
new / cold-start retrieval
```

---

## 14.4 抽样数据的候选库偏差

正式实验只抽：

```text
2k–3k users
```

因此候选 item 主要来自：

```text
这些用户对应的日志 + 可关联到的 item metadata
```

这并不等价于完整线上全站 catalog。

README 必须报告：

```text
candidate catalog size
number of warm items
number of cold items
number of tail items
catalog construction rule
```

并说明：

> the sampled catalog is an offline approximation and inherits selection bias from the sampled users and logged exposures.

---

# 15. Retrieval Request Unit

统一定义：

> **测试窗口内每一个正向点击 / 有效播放事件，作为一个 retrieval request。**

对于 request at time t：

```text
输入：
t 之前的用户历史与 T-1 特征

目标：
当前正向 item

候选库：
满足 catalog protocol 的可用 item
```

这样 Recall@K 的单位清晰。

不使用：

```text
每用户每天一个 request
```

作为默认口径。

---

# 16. Negative Sampling Study

P0。

---

## 16.1 Random Negative

从：

```text
available catalog
```

中采样随机非正样本。

---

## 16.2 In-batch Negative

同 batch 其他正样本作为 negative。

---

## 16.3 Exposure Negative

来自：

```text
标准日志中未点击曝光
```

---

## 16.4 Hybrid Negative

P0。

例如：

```text
Random + Exposure
```

或：

```text
In-batch + Exposure
```

---

## 16.5 In-batch + LogQ

P1。

---

## 16.6 公平比较原则

固定：

```text
same architecture
same embedding dimension
same optimizer
same learning rate
same batch size
same negatives per positive
same loss form
same training steps
same user subset
same candidate catalog
same seeds
```

---

# 17. Retrieval Evaluation

主表：

| Strategy | Recall@50 | Recall@100 | Recall@500 | NDCG@100 |
|---|---:|---:|---:|---:|
| Random | | | | |
| In-batch | | | | |
| Exposure | | | | |
| Hybrid | | | | |

辅助表：

| Item Tower | Overall R@100 | Warm R@100 | Cold R@100 | Tail R@100 |
|---|---:|---:|---:|---:|
| ID-only | | | | |
| ID + Side Features | | | | |

---

# 18. ItemCF

传统召回源。

```text
User History
     ↓
Item Similarity
     ↓
Candidate Aggregation
     ↓
Top-K
```

可使用：

```text
Co-occurrence
Cosine Similarity
Normalized Co-visitation
```

---

# 19. ANN / FAISS

P1。

```text
Available Items
     ↓
Item Tower
     ↓
Offline Embeddings
     ↓
FAISS Index
```

---

# 20. Track B — Ranking / Multi-task

重要：

> Multi-task supervision 来自 logged exposure samples。

因此：

```text
Single-Task
MMoE
PLE
Selective Sharing
```

都基于曝光日志训练与评估。

不把无标签 retrieval candidate 强行当 supervised negative。

---

# 21. LightGBM

P1。

如果实现：

```text
Logged Exposure Samples
        ↓
Feature Table
        ↓
LightGBM
```

特征：

```text
T-1 user statistics
T-1 item statistics
user-author preference
user-tag preference
video age
tab/context
retrieval scores
```

---

# 22. Single-Task

P0。

分别训练：

```text
DNN_click
DNN_long_view
DNN_like
DNN_follow
DNN_comment
```

实际保留 4–5 个任务。

---

# 23. MMoE

P0。

```text
Input
  ↓
Shared Experts
  ↓
Task-specific Gates
  ↓
Task Towers
```

核心：

```math
h_t =
\sum_{k=1}^{K}
g_{tk}(x)E_k(x)
```

---

# 24. PLE

P0。

核心：

```text
Shared Experts
+
Task-Specific Experts
```

研究：

> 更明确的 shared / task-specific capacity 是否减少 Negative Transfer？

---

# 25. Selective Sharing

P0。

代码开发时间提前到 Week 3。

比较：

```text
Full Shared
vs
Selective Task-Specific Representation
vs
PLE-style Task-specific Expert Capacity
```

---

## 25.1 Parameter Budget

记录：

```text
Embedding Parameters
Expert Parameters
Tower Parameters
Total Parameters
```

尽可能保持可比。

---

# 26. Multi-seed Requirement

最终核心实验：

```text
Single-Task
MMoE
PLE
Selective Sharing
Controlled Sparsity
```

必须：

```text
3 Seeds
```

报告：

```text
mean ± std
```

开发阶段可单 seed。

---

# 27. Core Multi-task Evaluation

| Model | Click AUC | Long-view AUC | Like AUC | Follow AUC | Comment AUC |
|---|---:|---:|---:|---:|---:|
| Single-Task | | | | | |
| MMoE | | | | | |
| PLE | | | | | |
| Selective Sharing | | | | | |

---

## 27.1 ΔAUC

```math
\Delta AUC_t =
AUC_{MTL,t}
-
AUC_{ST,t}
```

---

# 28. Negative Transfer Analysis

分析：

```text
哪些 task 受损？
损失是否超过 seed variance？
MMoE 与 PLE 是否不同？
Selective Sharing 是否缓解？
稀疏度是否提高敏感性？
```

---

# 29. Random Exposure Robustness

P1。

使用 KuaiRand 随机曝光日志进行：

```text
less policy-biased robustness check
```

比较：

```text
Standard Logged Test
vs
Random Exposure Test
```

观察：

```text
AUC ordering
ΔAUC direction
Negative Transfer pattern
```

不宣称完全无偏。

---

# 30. Score Fusion

输出：

```text
p_click
p_long_view
p_like
p_follow
p_comment
```

融合：

```math
Score = \sum_t w_t p_t
```

---

# 31. Re-ranking

简单规则：

```text
Repeated-item suppression
Category diversity
Author diversity
Consumed-item suppression
```

不做复杂 RL。

---

# 32. Repository Structure

```text
kuairand-recsys/
│
├── README.md
├── plan_architecture.md
├── requirements.txt
│
├── configs/
│   ├── data.yaml
│   ├── retrieval.yaml
│   ├── single_task.yaml
│   ├── mmoe.yaml
│   ├── ple.yaml
│   ├── selective.yaml
│   ├── lightgbm.yaml
│   └── din.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
│
├── src/
│   ├── preprocessing/
│   │   ├── sample_users.py
│   │   ├── preprocess.py
│   │   ├── temporal_split.py
│   │   ├── build_history.py
│   │   ├── build_catalog.py
│   │   ├── build_labels.py
│   │   └── leakage_check.py
│   │
│   ├── features/
│   │   ├── daily_user_features.py
│   │   ├── daily_item_features.py
│   │   ├── author_preference.py
│   │   ├── tag_preference.py
│   │   ├── video_age.py
│   │   └── feature_encoder.py
│   │
│   ├── retrieval/
│   │   ├── itemcf.py
│   │   ├── two_tower.py
│   │   ├── user_tower.py
│   │   ├── item_tower.py
│   │   ├── random_negative.py
│   │   ├── exposure_negative.py
│   │   ├── inbatch_negative.py
│   │   ├── hybrid_negative.py
│   │   ├── logq_correction.py
│   │   └── ann_index.py
│   │
│   ├── ranking/
│   │   ├── single_task.py
│   │   ├── mmoe.py
│   │   ├── ple.py
│   │   ├── selective_sharing.py
│   │   └── din.py
│   │
│   ├── prerank/
│   │   └── lightgbm.py
│   │
│   ├── analysis/
│   │   ├── controlled_sparsity.py
│   │   ├── multitask_transfer.py
│   │   ├── cold_start_analysis.py
│   │   └── parameter_budget.py
│   │
│   ├── reranking/
│   │   ├── score_fusion.py
│   │   ├── category_diversity.py
│   │   └── author_diversity.py
│   │
│   ├── evaluation/
│   │   ├── retrieval_metrics.py
│   │   ├── ranking_metrics.py
│   │   ├── negative_sampling_analysis.py
│   │   ├── multitask_analysis.py
│   │   ├── random_exposure_eval.py
│   │   └── efficiency.py
│   │
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── seed.py
│
├── scripts/
├── experiments/
├── notebooks/
├── tests/
└── results/
```

---

# 33. 五周执行计划

## Week 1 — 只做数据地基

必须完成：

```text
KuaiRand-1K 下载 / 读取
Schema Inspection
Global Temporal Split
Scene / Tab Label Check
基础 EDA
最小 T-1 Daily Features
最近 N 个点击历史
数据缓存
```

同时：

```text
27K 下载后台进行
```

Week 1 不要求：

```text
27K 全部处理完
复杂特征工程
冷启动完整分析
```

Week 1 结束只回答：

> 1K pipeline 是否稳定？27K 是否值得继续接入？

---

## Week 2 — Retrieval + Single-Task

完成：

```text
Two-Tower baseline
ID-only Item Tower
ID + Side Features
Random Negative
In-batch Negative
ItemCF
Single-Task DNN
```

先在 1K 跑通。

---

## Week 3 — MTL + Selective Sharing 提前开发

完成：

```text
Exposure Negative
Hybrid Negative

MMoE
PLE
Selective Sharing
```

目标：

> Week 3 结束时，四个多任务比较模型都能跑。

P1：

```text
FAISS
LogQ
ID Dropout
```

---

## Week 4 — Final Protocol + 3 Seeds

正式规模：

```text
27K sample 2k–3k users
```

若不稳定：

```text
1K fallback
```

完成：

```text
Negative Sampling Final Runs
Single-Task
MMoE
PLE
Selective Sharing
Controlled Sparsity
3 Seeds
```

---

## Week 5 — 分析与收尾

重点：

```text
mean ± std
ΔAUC
Negative Transfer
Warm / Cold / Tail Recall
Controlled Sparsity Figure
Parameter Budget
README
Architecture Diagram
Resume Bullets
Interview Notes
```

P1 有余力再补：

```text
LightGBM
Random Exposure Robustness
FAISS latency
Score Fusion
Re-ranking
```

DIN：

```text
默认不做
```

---

# 34. P0 / P1 / P2

## P0

```text
KuaiRand-1K stable pipeline
Global temporal split
Leakage control
Scene / label semantics check
T-1 daily features
Recent-N history

Two-Tower
ItemCF
ID-only Item Tower
ID + Side-feature Item Tower

Random Negative
In-batch Negative
Exposure Negative
Hybrid Negative

Single-Task
MMoE
PLE
Selective Sharing

Controlled Label Sparsity
3 Seeds
mean ± std
ΔAUC
Negative Transfer

Candidate Catalog Definition
Request Unit Definition
Warm / Cold Recall
README
Final Figures
```

---

## P1

```text
27K 2k–3k user main run
ID Dropout / Rare-ID Masking
FAISS
LogQ
LightGBM
Random Exposure Robustness
Parameter Count
Training Time
Score Fusion
Category / Author Re-ranking
```

---

## P2

```text
DIN
DIN + MMoE / PLE
Hard Negative Mining
ANN Index Tuning
Inference Latency
3k–5k User Scale Check
```

---

# 35. 最低成功版本

```text
KuaiRand-1K
   ↓
Global Temporal Split
   ↓
T-1 Daily Features
   ↓
Two-Tower + ItemCF
   ↓
Random / In-batch / Exposure / Hybrid
   ↓
Single-Task
   ↓
MMoE
   ↓
PLE
   ↓
Selective Sharing
   ↓
Controlled Sparsity
   ↓
3-seed ΔAUC
   ↓
Negative Transfer Analysis
```

这个版本已经足够成为第一份推荐算法实习的核心项目。

---

# 36. README 第一屏

建议展示：

1. 项目一句话定位
2. 两个核心研究问题
3. System Architecture
4. Dataset & Global Temporal Split
5. Leakage Control
6. T-1 Feature Pipeline
7. Candidate Catalog / Request Definition
8. Negative Sampling Comparison
9. Multi-task Result Table
10. Controlled Sparsity
11. Selective Sharing
12. Warm / Cold Retrieval
13. Engineering Cost
14. Quick Start

---

# 37. 最终简历表达目标

> Built an industrial-style short-video recommender on KuaiRand with global temporal splitting, T-1 feature snapshots, Two-Tower retrieval, and multi-task ranking. Compared random, in-batch, exposure, and hybrid negatives under a controlled retrieval protocol, and evaluated MMoE, PLE, and selective sharing with 3-seed task-level transfer analysis.

如果 cold-start 辅助实验明显：

> Also compared ID-only and feature-aware item towers and separately reported warm-item and cold-item Recall@K.

---

# 38. 项目成功标准

项目成功不定义为：

```text
模型越多越好
```

而定义为：

```text
两个核心问题讲清楚
+
Global Temporal Split 正确
+
无 Leakage
+
T-1 特征工程可复现
+
Negative Sampling Study 公平
+
Single-Task / MMoE / PLE / Selective Sharing 完整
+
3 Seeds
+
Controlled Sparsity
+
Negative Transfer Analysis
+
Warm / Cold Retrieval 口径清楚
+
面试可深入解释
```

最终希望体现：

> 我不仅会实现推荐模型，还能正确处理时间切分、特征快照、候选库定义、请求单位、负样本偏差、多任务负迁移和任务专属容量，并通过受控实验验证设计选择。

# 39. Retrieval Scope Guardrail

为了避免项目重新陷入类似 OTTO 的：

```text
candidate source
+ heuristic
+ weight tuning
+ rule stacking
+ endless candidate engineering
```

本项目对 Retrieval 设置明确的范围限制。

---

## 39.1 Retrieval P0 只允许以下内容

```text
ItemCF
+
Two-Tower
+
Random / In-batch / Exposure / Hybrid Negative Sampling
+
Recall@K / NDCG@K Evaluation
+
Simple Candidate Union
```

到这里即视为 Retrieval 主线完成。

---

## 39.2 默认不新增额外召回源

除非所有以下模块已经完成：

```text
Single-Task
MMoE
PLE
Selective Sharing
Controlled Sparsity
3-seed Final Runs
Negative Transfer Analysis
```

否则不新增：

```text
Popularity Recall
Author Recall
Tag Recall
Freshness Recall
Multi-hop Co-visitation
Rule-based Recall
Manual Boosting
Complex Source Weighting
```

---

## 39.3 Retrieval Freeze Condition

满足以下条件后冻结 Retrieval：

```text
1. ItemCF baseline 可复现
2. Two-Tower 可稳定训练
3. 四种 Negative Sampling 对比完成
4. Recall@100 / Recall@500 稳定
5. Candidate Union 可运行
6. Warm / Cold Recall 口径明确
```

冻结后：

```text
不再为了提高少量 Recall
继续增加 heuristic candidate source
```

后续时间优先投入：

```text
MMoE
PLE
Selective Sharing
Controlled Sparsity
Multi-seed Analysis
```

---

## 39.4 时间预算

建议总体时间占比：

```text
Data / Feature Pipeline        ~25%
Retrieval                      ~25%
MTL / Selective Sharing       ~35%
Analysis / README             ~15%
```

原则：

> Retrieval 是必要基础设施，但不是整个项目的终点。

本项目真正的差异化来自：

```text
Negative Sampling Study
+
Multi-task Transfer / Selective Sharing
```

而不是不断堆叠 candidate heuristics。

