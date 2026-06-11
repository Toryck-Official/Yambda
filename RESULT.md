# 0408Yambda: Yambda-HSRL Baseline 结果与汇报说明

## 一句话定位

这个项目目前完成的是 **HSRL/SID 思路在 Yambda 上的端到端 baseline 迁移**：

1. 先把 Yambda 的 item embedding 映射成离散语义 ID, 即 SID。
2. 再用用户历史序列监督训练 HPN/SASRec actor, 让 actor 能预测下一个 item 的 SID。
3. 然后训练一个 UserResponse 模型作为离线 RL 的用户反馈模拟器。
4. 最后用 DDPG 风格的 actor-critic 在模拟器里微调，并用离线候选集排序做 sanity check。

当前结论要谨慎表述：

- **可确认**：RQVAE SID + HPN warmstart 在 Yambda 上是可用的，100 个候选的随机负采样评估里明显强于随机。
- **不可确认**：DDPG/RL 微调带来了稳定增益。当前训练曲线没有显示出明确提升，最终排序结果很可能主要来自 HPN warmstart。
- **需要补充**：RQVAE 码本利用率、SID 碰撞率、相似 item 的 SID 局部性、HPN-only 与 RL actor 的直接 ablation、hard negative/full-catalog 评估。

---

## 流水线总览

| Stage | 脚本 | 做什么 | 当前状态 |
|---|---|---|---|
| 01 | `01_build_codebook.py` | 训练 RQKMeans/RQ codebook 备选产物 | 跑通，但最终 SID 不是用这个 npz 编码的 |
| 02 | `02_build_item_sid.py` | 把全量 item 映射为 dense id + 4 层 SID | 最终使用 RQVAE checkpoint 编码 772 万 item |
| 03 | `03_split_data.py` | 把 `multi_event.parquet` 切成 train/val/test 样本 | 跑通，但当前 split 统计口径有异常，需要复核 |
| 04 | `04_train_hpn_warmstart.py` | 监督训练 HPN/SASRec actor 预测 target SID | 有效，是当前最可信的模型能力来源 |
| 05 | `05_train_user_response.py` | 训练离线用户反馈模拟器 | 可用但有过拟合和校准不足 |
| 06 | `06_train_yambda_sid.py` | 在模拟器里用 DDPG 风格 actor-critic 微调 | 未看到稳定增益，且实现细节需要复查 |
| 07 | `07_eval_candidate_ranking.py` | 1 正 + 随机负样本候选排序评估 | sanity check 很强，但不是完整推荐评估 |

一个关键口径：

**最终这版 SID 使用的是 RQVAE checkpoint, 不是 Stage 01 训练出的 `yambda_rq_codebook.npz`。**

Stage 01 的 codebook 可以理解成原论文 RQKMeans 路线的备选/对照产物；Stage 02 里实际传入了 `RQVAE_CKPT`, 因此最终 item SID 来自 RQVAE encoder + RQVAE codebooks。

---

## Stage 01: Codebook 训练

### 脚本作用

`01_build_codebook.py` 的目标是复现原 HSRL 里“先把 item embedding 离散化”的前置步骤。它对 item embedding 做残差量化，每一层产生一个 token，最后每个 item 有一条类似 `[z1, z2, z3, z4]` 的 SID。

当前配置：

| 参数 | 值 |
|---|---:|
| embedding column | `normalized_embed` |
| item 总数 | 7,721,749 |
| 采样训练向量 | 200,000 |
| embedding dim | 128 |
| SID 层数 | 4 |
| 每层 codebook size | 256 |
| max_iter | 30 |

### 为什么后来用 RQVAE 而不是 RQKMeans

原论文的 RQKMeans/RQ codebook 在这个数据上效果不好，所以后续改成 RQVAE。这个选择可以这样解释：

- RQKMeans 是直接在 embedding 空间里逐层聚类，比较依赖原始 embedding 空间的几何形状。
- RQVAE 是用神经网络 encoder + residual quantization 联合学习，可以在离散化前先学习一个更适合量化的表示。
- 对大规模 item catalog, 只靠 KMeans 容易出现码字利用不均、碰撞、语义局部性差等问题；RQVAE 通常更容易控制重构、熵和码字利用率。

### 快速 SID 质量诊断

需要先说明口径：本地目前缺少最终 RQVAE 产出的全量 `dense_item2sid.npy` 和 RQVAE checkpoint, 所以还不能对 **最终 RQVAE SID** 复算全量指标。下面这组数是用 Stage 01 的 `yambda_rq_codebook.npz` 对 `embeddings.parquet` 前 100,000 个 item 做的 RQKMeans 备选码本抽样诊断，结果保存在 `artifacts/codebook/rqkmeans_sid_quality_sample.json`。它可以作为“为什么要继续检查 SID 质量”的辅助证据，但不能等同于最终 RQVAE SID 的质量。

| 指标 | RQKMeans 备选码本抽样结果 | 怎么解读 |
|---|---:|---|
| 每层码字利用率 | L1-L4 都是 256/256, 即 100% | 每层 token 都被用到了，单看利用率没有塌缩 |
| 每层 token 分布熵 | L1 7.986 / L2 7.977 / L3 7.975 / L4 7.976 bits | 最大熵是 8 bits, 熵比例都约 99.7%-99.8%, 分布接近均匀 |
| full SID 碰撞率 | 100,000 样本中 unique path 95,710, excess collision rate 4.29% | 有一定碰撞；pipeline 注释里 RQVAE `best_entropy_e20` 碰撞率约 3.88%, 需要用最终映射复算确认 |
| embedding 近邻的 SID 距离 | 1,000 个 anchor 的最近邻平均 Hamming distance 2.875/4, same L1 65.6%, same L1-L2 20.9%, same full SID 1.2% | 粗粒度第一层有一定局部性，但完整 SID 对 embedding 近邻并不稳定 |

这组数的重点不是证明 RQKMeans 好，而是说明要同时看三类问题：

- **利用率/熵**：token 有没有被充分使用。
- **碰撞率**：不同 item 是否被压成同一个完整 SID。
- **局部性**：embedding 空间里相似的 item, SID 是否也相近。

最终汇报时可以说：目前对备选 RQKMeans 码本做了快速抽样检查，码字利用率和熵都不错，但 full-SID 碰撞和近邻局部性仍然需要关注；最终 RQVAE SID 的同类指标还需要恢复全量 `dense_item2sid.npy` 后复算。

### 为什么“相似 item 的 SID 相距太远”是坏事

HSRL/SID 的核心假设不是只要 ID 唯一就行，而是希望 SID 有层级语义：

- 第 1 层最好表示粗粒度语义。
- 后几层再逐步区分细粒度 item。
- 相似 item 应该共享部分前缀或至少 token 距离较近。

如果两个 embedding 非常相似的 item 被分到完全不相关的 SID, actor 在预测一个 item 时学到的概率质量就不能自然泛化到另一个相似 item。结果是：

- 模型需要更多样本去记住每个 item。
- 候选打分时相似替代品拿不到概率提升。
- 语义检索会变得不连续。

所以低碰撞率和高码字利用率只是必要条件，不是充分条件；还需要检查 SID 是否保留了 embedding 空间里的相似性。

---

## Stage 02: 全量 Item SID 构建

### 脚本作用

`02_build_item_sid.py` 做两件事：

1. 把 Yambda 稀疏的原始 `item_id` 映射为连续 dense id。
2. 用 RQVAE encoder 给每个 dense item 生成 4 层 SID。

当前结果：

| 指标 | 值 |
|---|---:|
| 编码模式 | RQVAE |
| 原始 item_id 范围 | 2 到 9,390,623 |
| dense item 数量 | 7,721,749 |
| SID 层数 | 4 |
| 每层 vocab size | 256 |
| embedding dim | 128 |

### 该阶段怎么汇报

可以这样说：

> 我们把 Yambda 的 772 万 item 全量重映射成 dense id, 并通过 RQVAE 编码成 4 层语义 ID。这样后面的 HSRL actor 不直接在 772 万 item 上分类，而是预测 4 个 0-255 的 token, 把超大 action space 分解成层级语义动作空间。

### 当前不足

Stage 02 目前只能说明“映射构建成功”，不能说明“SID 质量足够好”。尤其缺少：

- 每层 token 直方图。
- full path 碰撞数量。
- 近邻 item 的 SID 前缀共享率。
- 同一 artist/genre/embedding-neighbor 是否有语义聚集。

另外，当前 git 里没有拉下完整的 `dense_item2sid.npy` / `orig2dense.npy` 大文件，只有 meta 文件。因此本地这份报告不能现场复算码本利用率和碰撞率。

---

## Stage 03: 数据切分

### 脚本作用

`03_split_data.py` 把 `multi_event.parquet` 里的用户多事件序列转换成训练样本。

Yambda 的特点是同一个用户对同一个 item 可能有多个时间戳上的事件，例如 listen、like、dislike、unlike。因此脚本没有把每个事件都单独当成一个 target，而是按用户时间线做了 episode 聚合：

- 对每个用户，先把所有事件按时间排序。
- 对同一个 item, 如果相邻事件间隔不超过 `close_gap_seconds=3600`, 就合成同一个 user-item episode。
- 一个 episode 里聚合出一个 target item 和一个 reward。
- reward 当前为 `max_play_ratio + like - dislike`。
- 每个样本的历史是 episode 开始前的最近 50 个事件。

### 有没有破坏用户历史序列

严格说，没有打乱全局时间顺序；脚本会按时间排序，并且 history 取的是当前 episode 开始前的事件。

但它确实做了一个简化：

- episode 内部的多步交互被压缩成一个 target + 一个 reward。
- 例如用户先 listen 后 like, 这些内部动作不会作为多个连续 step 进入 HPN warmstart。
- 这更像是“把一次 user-item 完整交互压成一个推荐反馈样本”，而不是完整复原用户行为轨迹。

这个简化对 warmstart 是可以接受的，因为 HPN warmstart 只是为了学一个初始 actor, 不是最终还原真实环境。但如果后面要严格做 sequential RL, 需要更细的状态转移建模。

### 当前数据规模

报告中记录：

| 指标 | 值 |
|---|---:|
| 活跃用户 | 10,000 |
| raw events | 23,898,321 |
| kept events | 23,100,324 |
| missing mapping | 1,570,553 |
| train rows | 41,513,562 |
| val rows | 10,000 |
| test rows | 10,000 |

这里有一个必须标注的风险：

**train rows 大于 kept events, 按当前 episode 逻辑通常不应发生。**

因为每个 episode 至多生成一条样本，而 episode 数不应超过事件数。这个统计可能来自 resume/追加写入、meta 口径不一致或 TSV 重复累积。汇报时不要把这个数字当成已经完全确认的数据质量证明，建议说：

> 当前 split 已经跑通，但 split count 的统计口径还要复核，尤其是 train rows 与 kept events 的关系。

---

## Stage 04: HPN Warmstart 监督训练

### 脚本作用

`04_train_hpn_warmstart.py` 训练的是 HSRL 里的 actor 初始策略，也就是 HPN/SASRec policy。

输入：

- 用户历史 item embedding 序列，长度 50。

输出：

- target item 的 4 层 SID token logits。

训练目标：

- 用真实 target item 的 SID 当 label。
- 对每一层 token 做 cross entropy。
- 这是监督学习/behavior cloning 风格的 warmstart。

它不是严格意义上的 autoregressive teacher forcing, 因为当前 actor 不是“输入上一层真 token 再预测下一层”，而是用同一个历史状态输出 4 个 token head。但它的思想确实是用离线行为数据把 actor 先预训练到一个可用状态。

### 为什么要训练 HPN warmstart

如果直接用 RL 从随机 actor 开始，在 772 万 item 的 action space 上探索几乎不可行。HPN warmstart 的目的就是：

1. 先学会“给定用户历史，下一个可能 item 的语义 ID 大概是什么”。
2. 给 RL 一个不是随机的初始策略。
3. 让后续 actor-critic 只需要微调，而不是从零学习推荐行为。

### 当前结果

| Epoch | train_loss | val_loss | val l1 acc | val l2 acc | val l3 acc | val l4 acc | val full path acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.7423 | 4.8472 | 16.01% | 4.42% | 2.93% | 2.27% | 0.020% |
| 2 | 4.5094 | 4.7773 | 18.93% | 5.24% | 3.41% | 2.00% | 0.010% |
| 3 | 4.4514 | 4.7418 | 19.40% | 5.55% | 3.66% | 2.20% | 0.020% |

随机猜单层 token 的准确率约为 `1/256 = 0.39%`。因此 l1 到 l4 都明显高于随机，说明 actor 确实学到了用户历史和 SID 之间的关系。

full path acc 很低不意外，因为完整预测 4 个 token 等价于精确命中一个 item 的 SID。推荐系统里只看 full path acc 会过于苛刻，因为模型可以把真实 item 排到前面，即使 argmax 的完整 SID 没完全相同。

### 参数不够完善的地方

- 只训练了 100,000 条样本，而 split 里有千万级样本，训练量明显偏小。
- `train_positive_only=false`, `min_train_reward=-1e9`, 会把负反馈 item 也作为 target 学习；如果目标是推荐正反馈 item, 后续应尝试只用正反馈或按 reward 加权。
- 只训 3 epoch, 没有学习率调度、early stopping、不同模型规模对比。
- 缺少 HPN-only candidate ranking 与 RL actor 的直接对比，所以目前不能量化 RL 增益。

---

## Stage 05: UserResponse 用户反馈模拟器

### 脚本作用

`05_train_user_response.py` 训练的是离线环境里的用户反馈模型。

在 RL 推荐系统里，agent 的动作是“推荐哪个 item”；用户模拟器不一定要输出一个新动作，而是要对 agent 推荐的 item 返回反馈。当前模型做的是：

输入：

- 用户历史 item embedding 序列。
- agent 推荐的候选 item embedding。

输出：

- 一个连续 reward 预测值，也就是这个用户对这个 item 的偏好分数。

在环境里：

- `direct_score` 模式直接把预测值当 immediate reward。
- 同时用 `pred > 0` 构造一个二值 response。

所以它不是在模拟“用户下一步做什么动作”，而是在模拟“用户对推荐 item 的反馈/收益是多少”。

### 当前结果

| Epoch | train_loss | val_mse |
|---:|---:|---:|
| 1 | 0.2239 | 0.2217 |
| 2 | 0.1972 | 0.2227 |
| 3 | 0.1943 | 0.2292 |
| 4 | 0.1924 | 0.2296 |
| 5 | 0.1908 | 0.2303 |

best val MSE 在第 1 epoch, 之后训练集继续下降但验证集变差，说明有过拟合。当前 checkpoint 采用 epoch 1 是合理的。

### 不足

- 没有 mean predictor / popularity predictor baseline, 所以 0.2217 MSE 的绝对质量不好判断。
- reward 是 `max_play_ratio + like - dislike`, 还没有做分布校准。
- 没有分开建模 listen / like / dislike / skip 等多种反馈动作。
- UserResponse 的质量会直接限制 RL 训练；如果模拟器偏差大，DDPG 学到的是模拟器偏好，不一定是真实用户偏好。

---

## Stage 06: DDPG/RL 微调

### 脚本作用

`06_train_yambda_sid.py` 把前面组件串起来做 RL 微调：

1. 用 Stage 04 的 HPN warmstart 初始化 actor。
2. actor 根据用户历史输出 4 层 SID token 分布。
3. `SIDFacade_credit` 在候选 item 池里，根据候选 item 的 SID 给每个 item 打分并选出推荐 item。
4. `YambdaEnvironment_GPU_HAC` 用 UserResponse 给这个推荐 item 返回 reward。
5. transition 被写入 replay buffer。
6. critic 学 Q 值，actor 根据 token-level policy gradient 风格的 loss 更新。

当前配置：

| 参数 | 值 |
|---|---:|
| n_iter | 10,000 |
| episode_batch_size | 32 |
| batch_size | 128 |
| candidate_items | 50,000 |
| buffer_size | 100,000 |
| actor_lr | 1e-4 |
| critic_lr | 1e-3 |
| gamma | 0.9 |
| entropy_coef | 0.01 |
| bc_coef | 0.1 |
| initial_greedy_epsilon | 0 |
| final_greedy_epsilon | 0 |

### 当前训练现象

从 `yambda_sid.report` 解析：

| 指标 | 前 10 个 report 平均 | 后 10 个 report 平均 | 说明 |
|---|---:|---:|---|
| average_total_reward | 2.68 | 2.63 | 没有稳定提升 |
| critic_loss | 0.077 | 0.076 | critic loss 稳定 |
| entropy | 4.33 | 4.78 | 策略熵上升 |
| bc_loss | 0.0 | 0.0 | BC 实际没有起作用 |

best reward 出现在 step 5900, 约 3.06；最终 step 9900 是 2.92。整体更像震荡，而不是稳定收敛提升。

### 需要谨慎的实现问题

当前不能把 Stage 06 解释成“DDPG 显著提升了 HSRL”。主要原因：

- 没有 HPN-only actor 与 RL actor 的同口径排序对比。
- `bc_loss` 全程为 0, 说明 `bc_coef=0.1` 这个约束没有真正发挥作用。
- 当前 DDPG 代码里的 `done_mask` bootstrap 逻辑需要复查；如果 `done=true` 时还 bootstrap, TD target 会有偏差。
- candidate pool 是 50,000 个 item, 不是全 772 万 item。
- exploration epsilon 为 0, 行为主要依赖当前策略和 entropy 项，探索不足。
- reward 来自 UserResponse 模拟器，不是真实在线反馈。

汇报建议：

> Stage 06 目前证明了 actor-critic 流程可以跑通，但还没有证明 RL 微调带来稳定收益。当前最终排序指标主要应归因于 HPN warmstart。

---

## Stage 07: 候选集排序评估

### 脚本作用

`07_eval_candidate_ranking.py` 做一个离线 sanity check：

1. 对 test.tsv 中每条样本，取真实 target item 作为正样本。
2. 随机采样 99 个负样本。
3. 共 100 个候选 item。
4. actor 输出 4 层 SID logits。
5. 对每个候选 item, 查它的 4 层 SID, 把各层 log probability 相加作为候选分数。
6. 看真实 target item 排第几。

这个评估回答的是：

> 给定 1 个真实 target + 99 个随机负样本，actor 能不能把真实 target 排到前面？

它不等价于：

- 论文里的 Total Reward / Depth。
- 全量召回。
- 真实线上推荐效果。
- hard negative 排序。

### 当前结果与修正后的随机 baseline

| 指标 | 随机 baseline | 当前模型 |
|---|---:|---:|
| Mean Rank | 50.5 | 5.28 |
| MRR | 0.0519 | 0.647 |
| HR@1 | 1.0% | 52.0% |
| HR@5 | 5.0% | 80.5% |
| HR@10 | 10.0% | 88.3% |
| HR@20 | 20.0% | 93.9% |
| NDCG@1 | 1.0% | 52.0% |
| NDCG@5 | 2.95% | 67.5% |
| NDCG@10 | 4.54% | 70.0% |
| NDCG@20 | 7.04% | 71.4% |
| token_acc_l1 | 0.39% | 25.7% |
| token_acc_l2 | 0.39% | 8.4% |
| token_acc_l3 | 0.39% | 5.1% |
| token_acc_l4 | 0.39% | 2.9% |
| full_path_acc | 约 2.33e-8% | 0.25% |

原报告里有两处需要修正：

- MRR 的随机 baseline 不是 0.50, 而是 `H_100 / 100 = 0.0519`。
- full_path_acc 不是 1/100, 因为它不是候选排序随机命中率，而是 4 层 SID token 全部 argmax 命中的概率；均匀随机约为 `1 / 256^4`。

### 怎么解释这个结果

可以说：

> 在随机负采样的 100 候选评估下，actor 明显能把 held-out target 排到前面，说明 SID actor 学到了用户历史和目标 item 之间的关系。

但不要说：

> 这个指标证明 RL 微调有效。

因为 Stage 07 当前只评估了最终 actor, 没有同时评估 HPN warmstart actor。

---

## 当前最重要的问题清单

### 1. RQVAE SID 质量还没完整证明

需要补：

- 每层码字利用率。
- 每层 token entropy。
- full SID 碰撞率。
- embedding 近邻的 SID 距离。
- 相似 item 的前缀共享率。

### 2. split 统计需要复核

`train rows > kept events` 按当前逻辑不合理。需要检查：

- TSV 是否重复追加。
- resume 统计是否混入旧 run。
- `sequence_id` 是否唯一。
- episode 数是否真的超过事件数。

### 3. HPN warmstart 训练还偏弱

需要补：

- 更多训练样本。
- positive-only 或 reward-weighted training。
- HPN-only candidate ranking。
- 不同 SID 编码方式的对比。

### 4. UserResponse 只是初版模拟器

需要补：

- baseline MSE。
- reward 分布和校准。
- 多反馈类型建模。
- 验证它对 unseen item/user 的泛化。

### 5. DDPG 微调需要修正和 ablation

需要补：

- 修正/确认 done mask。
- 修正 BC loss 不生效的问题。
- 加 HPN-only vs RL actor 对比。
- 加 random-init RL 对比。
- 加不同 `candidate_items`, `entropy_coef`, `bc_coef`, exploration epsilon 的 ablation。

### 6. Stage 07 还只是 easy negative 评估

需要补：

- popular negatives。
- same-user/history negatives。
- in-batch negatives。
- 50k candidate ranking。
- full-catalog retrieval approximation。

---

## 汇报时推荐说法

可以按这段讲：

> 这几天主要做的是把 HSRL 的 SID actor-critic 框架迁移到 Yambda。第一步是解决 item action space 过大的问题，我们把 772 万 item 通过 RQVAE 编码成 4 层语义 ID, 每层 256 个 token。这样 actor 不需要直接预测 772 万分类，而是预测 4 个离散语义 token。

> 第二步是处理 Yambda 的 multi-event 数据。因为同一个用户对同一个 item 会有 listen、like、dislike 等多次事件，我们把同一用户同一 item 在一小时窗口内的完整交互聚合成一个 episode, 用 episode 前的历史作为状态，用聚合 reward 作为反馈。这个简化保留了用户时间顺序，但压缩了 episode 内部动作。

> 第三步是 HPN warmstart。我们用监督学习训练 SASRec actor, 输入用户历史 embedding 序列，输出 target item 的 4 层 SID token。这个阶段结果最稳定，token accuracy 显著高于随机，说明 SID 和用户历史之间有可学习关系。

> 第四步是训练 UserResponse 作为离线 RL 环境。它不是输出用户动作，而是对 agent 推荐的 item 返回一个连续 reward 和二值 response。这个模拟器目前能用，但有过拟合，后续要做校准。

> 第五步是用 DDPG 风格的 actor-critic 微调 SID actor。当前流程已经跑通，但 reward 曲线没有稳定提升，因此不能声称 RL 带来了显著收益。最终 100 候选随机负采样排序表现很好，更多应归因于 HPN warmstart。

> 所以当前 baseline 的结论是：Yambda 上的 SID 表示和 HPN warmstart 是有效的；RL 微调和 simulator 还需要进一步修正、消融和更强评估。

---

## 可以预期会被问到的问题

### Q1: 为什么不用原论文 RQKMeans？

因为在当前 Yambda embedding 上，RQKMeans 的离散化效果不理想。我们改用 RQVAE, 让 encoder 和 residual codebooks 联合学习，更容易获得可用的离散语义 ID。但这还需要补充 RQKMeans vs RQVAE 的定量对比。

### Q2: RQVAE 的 SID 一定好吗？

不一定。现在只能说明下游 HPN 能学到 SID, 还不能说明 SID 本身最优。还要检查码字利用率、碰撞率、相似 item 的 SID 距离和前缀共享率。

### Q3: 用户反馈模拟器为什么不输出“动作”？

在推荐 RL 里，agent 的动作是推荐 item；用户模拟器的角色是对推荐 item 给出反馈，例如 click/reward/like。当前 UserResponse 输出的是连续 reward, 环境再用 `pred > 0` 得到二值 response。

### Q4: 最终 HR@10=88.3% 是否说明推荐效果很好？

只能说明在 99 个随机负样本的候选排序里很好。随机负样本相对容易，不代表 full-catalog 或 hard-negative 场景。它是 sanity check, 不是最终推荐指标。

### Q5: DDPG 是否有效？

目前不能这么说。当前 DDPG reward 没有稳定上升，BC loss 没起作用，且缺少 HPN-only 对照。更稳妥的结论是：actor-critic 训练流程跑通，但 RL 增益待验证。
