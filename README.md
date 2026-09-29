# KuaiRand Multi-Stage Recommender

> Industrial-style multi-objective short-video recommendation system built on KuaiRand:
> global temporal splitting, T-1 feature snapshots, two-tower retrieval, and multi-task ranking.

**状态：🚧 Week 3 进行中 —— 召回主线、Single-Task 基线与 MMoE 均已跑通。**
完整设计见本地个人笔记 `note/plan_architecture.md`（不入库）。

已完成：数据管线、候选库与词表、召回指标、ItemCF、全库检索、双塔（random / in-batch）、
排序侧物品元数据、User x Author 偏好特征（§8.4）、排序指标（AUC / GAUC / PCOC / 校准）、
Single-Task 基线、MMoE（各 3 seeds，只评 valid）。
未完成：PLE、Selective Sharing、User x Tag 偏好（§8.5）、exposure / hybrid 负采样、
test 段评估。

> ⚠️ **稀疏任务上的结论受样本量限制，不是 seed 数不够。** is_follow 的 ΔAUC 合并 std
> 是 0.0169，比文献效应量（+0.0016~+0.0045）大一个量级 —— valid 仅 1,457 个正样本、
> 395 个可算 GAUC 的用户。再加 seed 只会把 std 估得更准，要在 follow / comment 的 GAUC
> 上下结论只能上 27K。

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

### 7.3 Multi-task Ranking

**当前有 Single-Task（§27.1 里 ΔAUC 的分母）与 MMoE，各 3 seeds（42/43/44），只在 valid 上
评估。** PLE / Selective Sharing 未实现。test 只在架构与超参全部冻结后用一次 ——
开发期反复看 test 会把它变成第二个 validation。

**AUC（mean ± std over 3 seeds）**

| Model | Click | Long-view | Like | Follow | Comment |
|---|---:|---:|---:|---:|---:|
| Single-Task | 0.73642±0.00047 | 0.74196±0.00060 | 0.92135±0.00093 | 0.81613±0.00901 | 0.88227±0.00053 |
| MMoE | 0.73786±0.00072 | 0.74151±0.00122 | 0.91735±0.00152 | 0.80433±0.01434 | 0.87511±0.00080 |
| PLE | | | | | |
| Selective Sharing | | | | | |

**ΔAUC 相对 Single-Task（§27.1）** —— **配对差值**：四个模型用同一组 seed、同一份数据、
同一个行序，seed 是受控的配对因子，所以先算每个 seed 的 `MMoE_s − ST_s`，再求 mean ± std。

| Model | Click | Long-view | Like | Follow | Comment |
|---|---:|---:|---:|---:|---:|
| MMoE | +0.00144±0.00119 | −0.00045±0.00167 | **−0.00400±0.00060** | **−0.01180±0.00543** | **−0.00716±0.00110** |
| 三个 seed 同号？ | 同号(+) | 异号 | **同号(−)** | **同号(−)** | **同号(−)** |

粗体 = 三个 seed 方向一致且 \|mean\|/std > 2。**不做显著性判定** —— 3 个 seed 估出的 std
本身极不可靠，这里只提供方向一致性与量级供判断。

**GAUC（mean ± std）与 ΔGAUC —— 五个任务全部落在噪声内**

| Model | Click | Long-view | Like | Follow | Comment |
|---|---:|---:|---:|---:|---:|
| Single-Task | 0.58962±0.00091 | 0.61925±0.00078 | 0.59584±0.00751 | 0.51024±0.00226 | 0.52202±0.00968 |
| MMoE | 0.58905±0.00229 | 0.61819±0.00278 | 0.59475±0.00766 | 0.50582±0.00682 | 0.52706±0.01214 |
| ΔGAUC（配对） | −0.00056±0.00211 | −0.00106±0.00238 | −0.00110±0.00200 | −0.00442±0.00862 | +0.00505±0.00876 |
| 三个 seed 同号？ | 异号 | 异号 | 异号 | 异号 | 异号 |

**ΔGAUC 五个任务全部异号**，即差值的符号在 seed 之间就会翻转。注意这只能说「**测不出**
MMoE 对用户内部排序有稳定影响」，**不能说「保住了用户内部排序」**。

#### 限制：1K 上 follow / comment 的 GAUC 结论做不出来

这不是「效应不存在」，是**测量能力不足**，而且两个稀疏任务与两个稠密任务的原因不同：

**稀疏任务：GAUC 已经贴在随机这条地板上，能损失的上限 ≈ 噪声。**

| task | ST GAUC | 离 0.5 的余量 | 配对 ΔGAUC 的 std | 可算用户 | 正样本/用户 |
|---|---:|---:|---:|---:|---:|
| is_follow | 0.51024 | **0.01024** | 0.00862 | 395 | **3.7** |
| is_comment | 0.52202 | **0.02202** | 0.00876 | 452 | 10.8 |

is_follow 的 GAUC 只比随机高 0.010，而噪声是 0.0086 —— **能丢的总量和测不准的量几乎
相等**，任何结论都无从谈起。根源在每个可算用户平均只有 **3.7 个正样本**（曝光 1,610 次）：
用 3~4 个正样本估一个用户的 AUC 本身就极噪，395 个这样的估计平均起来也压不下去。

**稠密任务：余量充足，但效应本身小于噪声。**

| task | ST GAUC | 余量 | ΔGAUC | 配对 std | 正样本/用户 |
|---|---:|---:|---:|---:|---:|
| is_click | 0.58962 | 0.0896 | −0.00056 | 0.00211 | 608.5 |
| long_view | 0.61925 | 0.1193 | −0.00106 | 0.00238 | 437.9 |

**为什么 AUC 测得出而 GAUC 测不出**：同一个模型差异，is_comment 在 AUC 上是
−0.00716±0.00110（6.5 倍 std），在 GAUC 上是 +0.00505±0.00876（0.6 倍）。差别在有效
样本量 —— AUC 把 132 万行放进一个排序算一个数，GAUC 要算 452 个用户各自的 AUC 再加权。
实测 is_comment 的 GAUC seed std 是 AUC 的 **18.1 倍**，is_like 是 **8.1 倍**。

**可以说的**：MMoE 在 AUC 上的损失（3/3 seed 同号）主要发生在**跨用户**成分，因为 AUC
含跨用户与用户内两部分而 GAUC 只有用户内。
**不能说的**：「用户内排序没受影响」—— 那需要测量能力支撑，而这里没有。

**解决路径只有增加有效样本量**，不是加 seed：27K 抽 2~3k 用户可把 is_follow 的正样本从
1,457 提到约 4,000~6,000、每用户约 10~15 个，那时 GAUC 才具备测量能力。在 1K 上应停止
尝试从 follow / comment 的 GAUC 得出结论。

**PCOC（mean over 3 seeds；1.0 为校准良好）**

| Model | Click | Long-view | Like | Follow | Comment |
|---|---:|---:|---:|---:|---:|
| Single-Task | 1.1078 | 1.1952 | 1.3866 | 1.9356 | 1.9317 |
| MMoE | **1.0390** | **1.1168** | **1.3798** | 2.7168 | 2.1529 |

（PCOC 粗体 = 更接近 1）

#### 聚合指标（记录用，不作为判据）

五个任务聚合成一个数的三种口径。**报告正文不使用它们**，理由见下方。

```text
简单平均        = Σ(任务指标) / 任务数
按用户数加权     = Σ(任务指标 × 该任务 GAUC 参与用户数) / Σ(参与用户数)
按正样本数加权   = Σ(任务指标 × 该任务 valid 正样本数) / Σ(正样本数)
```

（下表基于 3-seed 均值）

| 口径 | | Single-Task | MMoE | Δ |
|---|---|---:|---:|---:|
| **AUC** | 简单平均 | 0.81963 | 0.81523 | −0.00440 |
| | 按 GAUC 用户数加权 | 0.80517 | 0.80238 | −0.00280 |
| | 按正样本数加权 | 0.74442 | 0.74490 | **+0.00047** |
| **GAUC** | 简单平均 | 0.56739 | 0.56697 | −0.00042 |
| | 按 GAUC 用户数加权 | 0.58184 | 0.58130 | −0.00053 |
| | 按正样本数加权 | 0.60134 | 0.60058 | −0.00076 |

**三种加权给出相反的结论**，而且没有客观标准去选哪一个 —— 这正是不把聚合指标当判据的
原因。按正样本数加权时权重极度倾斜：

| task | valid 正样本 | 权重 | ΔAUC | 对加权结果的贡献 |
|---|---:|---:|---:|---:|
| is_click | 599,421 | 56.31% | +0.00144 | +0.000811 |
| long_view | 429,997 | 40.40% | −0.00045 | −0.000182 |
| is_like | 28,736 | 2.70% | −0.00400 | −0.000108 |
| is_comment | 4,866 | 0.46% | −0.00716 | −0.000033 |
| is_follow | 1,457 | **0.14%** | −0.01180 | **−0.000016** |

点击与长播占 96.7% 的权重，is_follow 只占 0.14%：MMoE 在 follow 上 −0.0118 的损失被稀释
700 倍后只剩 −0.000016，等于消失。而关注 / 评论**恰恰因为稀少才珍贵**，按频次加权等于把
权重和业务价值搞反。工业上的做法相反 —— PLE 线上的排序分是
`score = p_VTR^w_VTR × p_VCR^w_VCR × ... × p_CMR^w_CMR × f(video_len)`，那些 `w` 是按线上
实验调出来的**业务权重**，不是频次。离线拿不到业务权重，所以**逐任务报是唯一诚实的做法**。

MMoE / PLE / STEM 三篇论文也都逐任务报，没有一篇报聚合指标。

#### MMoE 的形态：只有最稠密的任务受益，中等稀疏的任务稳定受损

3 seeds 的配对差值（逐 seed 的 MMoE−ST，再求 mean ± std）：

```text
is_like     (2.17%)  [-0.00394,-0.00343,-0.00463]  -0.00400±0.00060   3/3 负，|mean|/std 6.6
is_comment  (0.37%)  [-0.00589,-0.00787,-0.00772]  -0.00716±0.00110   3/3 负，|mean|/std 6.5
is_follow   (0.11%)  [-0.00563,-0.01391,-0.01586]  -0.01180±0.00543   3/3 负，|mean|/std 2.2
is_click   (45.28%)  [+0.00269,+0.00032,+0.00132]  +0.00144±0.00119   3/3 正，|mean|/std 1.2
long_view  (32.48%)  [+0.00141,-0.00095,-0.00182]  -0.00045±0.00167   符号翻转
```

**三个稀疏任务在三个 seed 上全部下降**，这是本项目对负迁移最一致的证据。
is_click 的 +0.00144 量级与 PLE Table 1 中 MMoE 的 +0.0016 接近，但它只有自身 std 的
1.2 倍，不足以称为「稳定收益」。

> ⚠️ **配对与否会改变结论。** 早先版本用「两模型 std 的合并」作尺度（非配对），
> 得到 is_follow 的尺度 0.0169 > 效应 0.0118，于是判成「读不出、且注定读不出」；
> 改成配对后 std 是 0.0054，三个 seed 全负。反过来 is_click 的比值从 1.6 降到 1.2。
> seed 是受控的配对因子，**必须配对计算**。

**⚠️ 单 seed 时的读法是错的，这里更正**：seed 42 单独看时五个任务的符号"完全按稀疏度
分开"（两个稠密任务为正、三个稀疏任务为负），看起来与等权 loss 的机制完美吻合。
3 seeds 之后 long_view 变为 −0.00045（负号且不显著），说明"稠密任务受益"这一半**只有
is_click 成立**，seed 42 上 long_view 的 +0.00141 是运气。**在符号结构上做解释之前必须
先补 seed** —— 单次运行的符号排列极易自圆其说。

等权 loss 的机制本身仍然成立且值得记录：五个任务的轮末 BCE 分别是 0.57976 / 0.53113 /
0.05594 / 0.01762 / 0.00663 —— click 是 follow 的 **87 倍**，取均值时共享底座几乎由稠密
任务塑造。等权是**有意的选择**：任务加权是另一个研究问题，混进来会让「共享机制的影响」
不可归因；三个 MTL 变体必须用同一套。

这不是实现缺陷，而是文献中有名字的现象：PLE 称之为 **seesaw phenomenon**，其 Table 1 中
MMoE 相对 Single-Task 是 VTR +0.0016 / VCR −0.0001（一正一负）；MMoE 原论文 Table 1/2 中
MMoE 赢主任务、输辅助任务；STEM (AAAI-24) Figure 1 显示在两任务反馈量相当的样本子集上
MMoE 与 PLE 均**劣于** Single-Task。

#### seed 方差随稀疏度暴涨，决定了哪些任务能下结论

Single-Task 的 3-seed std（本项目第一次真实测量）：

| | is_click | long_view | is_like | is_comment | is_follow |
|---|---:|---:|---:|---:|---:|
| AUC std | 0.00047 | 0.00060 | 0.00093 | 0.00053 | **0.00901** |
| GAUC std | 0.00091 | 0.00078 | **0.00751** | **0.00968** | 0.00226 |

对照文献的效应量（+0.0016~+0.0045）：**稠密任务的噪声远小于效应，最稀疏任务的噪声远大于
效应**。is_follow 的 ΔAUC −0.0118 看着最大，但合并 std 是 0.0169 —— 比效应还大。
**这不是 seed 不够，是样本量不够**（valid 仅 1,457 个正样本、395 个可算 GAUC 的用户）；
再加 seed 只会把 std 估得更准，不会让它变小。要在 follow 上下结论只能上 27K。

**GAUC 的 seed 方差比 AUC 大一个数量级**（is_comment 0.0097 vs 0.0005），导致五个任务的
ΔGAUC 全部读不出。所以尽管 GAUC 在**度量意义**上更贴近排序模型的实际工作（用户内排序），
在 1K 规模上能支撑结论的反而是 AUC。这一点必须与结果一并披露。

> 方法论备注：诊断阶段曾用「单次训练内部、各评估点之间的波动」当作噪声尺度的代用品，
> 实测它系统性**高估 4~8 倍**（is_click AUC：代用品 0.0036 vs 真实 seed std 0.00047）。
> 它量的是优化过程的抖动，收敛后会被平均掉，与 seed 方差不是一回事。显著性判定只能用
> seed std。

#### PCOC：共享让最稀疏的任务明显失准

```text
is_click    1.1078 -> 1.0390   改善
long_view   1.1952 -> 1.1168   改善
is_like     1.3866 -> 1.3798   基本不变
is_comment  1.9317 -> 2.1529   恶化
is_follow   1.9356 -> 2.7168   明显恶化（高估从 94% 升到 172%）
```

AUC 只看排序不看数值，抓不到这件事。共享表示让稀疏任务**失准而非失序**正是 §28 负迁移
分析的抓手 —— 注意 is_follow 的 ΔAUC/ΔGAUC 都读不出，但 PCOC 的恶化幅度远超其他任务。

#### 一个尚未解耦的混杂因素

本项目的 Single-Task 是 **5 张独立 embedding 表**，MMoE 是 **1 张共享表**。因此
ΔAUC(MMoE − ST) 同时混了两个变量：(a) 专家 / 塔是否共享；(b) **embedding 是否共享**。
STEM (AAAI-24) 的核心论点正是 (b) 才是负迁移主因 —— 现有 MMoE / PLE 都属于
shared-embedding paradigm，该文为此专门构造 ME-MMoE / ME-PLE 来分离这两个变量。
§25 的 Selective Sharing 若在 **embedding 层**也设一档（如稀疏任务给专属 embedding），
即可同时覆盖这个前沿方向。

- **稀疏任务的高 AUC 主要来自用户之间的差异，不是「这个视频值得关注」。**
  is_follow 的 AUC 是 0.826 而 GAUC 只有 0.512 —— 绝大部分判别力来自模型学会了
  「哪些用户爱关注」，而不是「什么视频值得关注」。若只看 AUC 会得出「follow 比 click
  做得好得多（0.83 vs 0.74）」这个完全错误的结论。因此研究问题二真正要看的是
  **GAUC**，而不是 ΔAUC 的绝对值。
- **PCOC 随稀疏度单调恶化**：1.12 → 1.22 → 1.44 → 1.91 → 1.96。最稀疏的两个任务把概率
  高估约 95%。AUC 只看排序不看数值，抓不到这件事 —— 而共享表示最可能让稀疏任务
  **失准而非失序**，这是 §28 负迁移分析的抓手。
- **GAUC 参与用户数必须与指标一起报**：985 → 982 → 780 → 452 → 395。标签恒定的用户
  AUC 无定义，一律排除；follow 上只有 40% 的用户可算。
- **只有 1 个 seed。** is_follow 的轮内曲线在 0.823~0.842 之间抖动（valid 仅 1,457 个正
  样本），单 seed 的 0.01 差异读不出任何东西。§26 要求的 3 seeds + mean ± std 必须补齐
  后才能谈 follow / comment 上的迁移。

#### User x Author 偏好特征把 follow 拉离了「随机」这条地板

最初的基线里 is_follow 的 GAUC 是 **0.49014**、is_comment **0.50537** —— 用户内部排序与
随机无异。原因是模型完全没有 user x author 交互特征：用户侧只有历史视频的**池化均值**，
物品侧只有作者 embedding，靠 MLP 反推「是否同一作者」极难。实测一个计数特征
（该用户过去 7 天看该作者几次）单独的 GAUC 就有 0.53784 / 0.53764 —— **打赢整个 3,750 万
参数的模型**，证明信号存在、只是没进输入。

补上 §8.4 之后（15 列，窗口 3/7 天，T-1 口径）：

| task | ΔAUC | **ΔGAUC** |
|---|---:|---:|
| is_click | +0.00134 | +0.00455 |
| long_view | +0.00389 | +0.00410 |
| is_like | +0.00210 | +0.00516 |
| is_comment | +0.00029 | +0.00810 |
| **is_follow** | **−0.00056** | **+0.02211** |

**is_follow 的 AUC 降了、GAUC 升了 0.022** —— 改善完全发生在用户内部。若特征只是让模型
更会判断「哪些用户爱关注」，AUC 会涨而 GAUC 不动；实际方向相反。ΔGAUC 的梯度也符合
预期：稠密任务 +0.004~0.005，follow +0.022（五倍）。

仍要披露的两点：

- **follow 远没解决。** 0.51225 只是勉强高于随机，而且 `ua_imp_7d` 单特征的 0.53784
  **仍然高于加了该特征的整个模型**。3,750 万参数没把这 15 列用好（74% 的行没有近期
  历史、取先验值；embedding 可能在别处过拟合把信号淹了）。这个缺口恰好是研究问题二
  的一个具体假设：共享表示若真能帮稀疏任务，应表现为它把这些列用得比 Single-Task 更好。
- **PCOC 在两个稀疏任务上恶化**（1.827→1.962、1.823→1.912）。排序变好、数值更偏。

口径偏离（都由实测驱动，写在 configs/data.yaml 的 `pair_preference` 注释里）：窗口不取
1 天（(user,author,date) 日表 8,371,862 行而原始日志 9,015,279 行，平均 1.08 次/三元组，
1 天窗口对约 96% 的样本恒为 0）；只给 is_click / long_view 算平滑比率（pair 维度曝光量
0~4，给 0.11% 的事件算比率是纯噪声）。缺历史时计数填 0、平滑比率填**全局先验 g** ——
平滑公式在 num=imp=0 处恰好等于 g，填充与公式天然一致。

#### 物品侧在测试期几乎是空的

排序训练在曝光日志上，而本数据集内容换代极快（72.6% 的视频在 31 天窗口内上传）。
实测测试段的物品侧覆盖率：

| 标签 | 时长 | 上传日期（→ video age） | 作者 | **目标视频 ID** |
|---:|---:|---:|---:|---:|
| 96.6% | 93.3% | ~100% | 64.1% | **10.8%** |

视频 ID 有 89.2% 对不上，这**修不了也不该修**：昨天刚上传的视频不可能有学过的专属向量。
模型从「不知道是什么」变成「不知道是哪一个，但知道谁拍的、什么类、多长、多新」——
这正是真实系统处理新视频的方式。为此排序侧单独建了一张物品元数据表（只用合法的 basic
元数据，官方统计表全表禁用），三个词表全部只由 train 段构建，召回侧文件一个字节未改。

#### 一条已修的捷径，一条残留的漂移

**已修**：目标视频 ID 最初沿用召回词表，而召回词表 =「候选库 ∪ train 段正向视频」——
成员身份**部分由训练标签决定**。实测 train 段 962,054 条 OOV 样本里 is_click 与 long_view
的正样本率**恰好为 0**（任何正样本都会把该视频送进词表），于是「在词表内」这一个比特
单独就有 train AUC 0.6981 / test 0.5190。is_like 不受影响（0.5043），因为召回词表的
positive_signal 只含 is_click 与 long_view —— 机制完全对得上。目标通道已换成只由 train
曝光次数构建的词表（194,310 个，与候选库逐个相同）；历史通道仍用召回词表，两者各一张
embedding 表。

**残留（披露而非掩盖）**：新的 `video_id_known` 仍带轻微热度信号，train 0.5509 /
valid 0.5248（强度差 0.026）。它是合法的 prediction-time 特征（train 期曝光相对 test
是过去数据），但差距不为零。对比修复前的 0.179，性质不同。验算里有一层常设的**捷径
扫描**：先实测哪些通道真的进了模型，再比 train/valid 单特征 AUC 强度，超 0.15 报错、
0.05~0.15 逐条打印披露。

#### 过拟合位置与训练预算

原设置（5 轮、无正则）下 valid AUC 从第 1 轮起单调下滑，is_follow 更是从峰值掉 0.0915。
逐个消融后定位到**历史 embedding**（897,505 行 × 32 = 2,870 万参数，占 77%）：

| 手段 | 打在哪 | is_click 1 轮末 AUC | 有效 |
|---|---|---:|---|
| 基准 | — | 0.72780 | — |
| dropout=0.2 | 塔（12 万参数） | 0.72734 | ✗ |
| id_dropout=0.3 | 目标 ID（620 万） | 0.72864 | ✗ |
| 去掉整条目标 ID 通道 | — | 0.72817 | ✗ |
| lr 1e-3 → 3e-4 | — | 0.72503 | ✗ 更差 |
| **weight_decay=1e-5** | **全部 embedding** | **0.73372** | ✓ |

预算（epochs=1 / weight_decay=1e-5）一次性在 **valid** 上选定，四个模型共用同一套值，
完整依据写在 `configs/single_task.yaml` 的注释里。取 1e-5 而非 1e-4 是因为后者在 is_follow
上掉 0.024 —— 只看 is_click 会选错。顺带结论：目标 ID 通道对 valid AUC 的贡献≈0（拿掉
只动 0.0004），与它在 test 段 89.2% 是 OOV 一致；保留它是为了与 MTL 对比时结构一致。

#### 一个必须披露的不对称

Single-Task 是 5 个**完全独立**的模型（不共享 embedding），MMoE / PLE / Selective 是 1 个
模型带 5 个头。因此总计算量与参数量都差约 5 倍 —— 这是「共享 vs 不共享」这个比较的固有
形态，不是算力对齐实验。§25.1 里 Single-Task 是**参照点**，参数预算可比性只在三个 MTL
变体之间要求。

**checkpoint 选择必须对称**：ST 可以为 5 个任务各挑一次最佳轮，MTL 只能挑一次；若允许
ST 挑 5 次，ΔAUC 会系统性偏向 ST、凭空造出负迁移。因此四个模型一律**固定轮数、报最后
一步**，每 500 步记曲线但不据此选择。

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
