# 03 语义 ID 阶段性总结

本文记录当前 `0626 Predictor` 工作区中语义 ID、候选桶、soft next state、数据切分和 Bellman-style value 的阶段性结论。本文只总结当前实现和已验证现象，不替代完整方法设计。

## 1. 当前语义 ID 实现

当前正式数据使用：

```text
mapping_root:
  /root/autodl-tmp/0626/0626 Predictor/01_data/processed/raw_rqkmeans

dense_item2sid.npy:
  shape = [7721750, 4]

dense_item_features.npy:
  shape = [7721750, 128]
```

含义：

```text
7721749 个有效 item
4 层语义 ID
每层 256 个 code
每个 item 对应一个 SID path，例如 [z1, z2, z3, z4]
```

当前码本来源是 RQKMeans，即 residual quantization KMeans。每一层是一个聚类中心集合，也可以叫一层 codebook。第 1 层对原始 embedding 聚类，第 2 层以后对上一层没有解释掉的残差继续聚类。

当前码本训练完成后是冻结的。后续 HPN、predictor、value/rerank 都只读取 `dense_item2sid.npy` 和 item embedding，不再更新码本。

## 2. 和 HSRL 原始设定的差距

当前实现参考了 HSRL 风格的 residual balanced KMeans，但仍需要谨慎表述。

可以说：

```text
we use a fixed residual KMeans semantic codebook
```

不建议过度说：

```text
we exactly reproduce the original HSRL semantic ID construction
```

原因：

```text
1. 当前项目输入是 Yambda 的 item embedding parquet。
2. 当前训练脚本使用 reservoir sampling 训练码本，而不是每次全量 item 直接聚类。
3. 当前 0626 Predictor 的 HPN 是自包含实现，用来服务 future-aware rerank；它不是 0408 Regret 主线 HSAC 的完整替代。
```

这不一定是漏洞，但论文和实验记录里要把“固定 RQKMeans 码本”说清楚。

## 3. 已完成的语义 ID 验证

诊断脚本：

```text
/root/autodl-tmp/0626/0626 Predictor/04_eval/eval_sid_codebook.py
```

结果文件：

```text
/root/autodl-tmp/0626/0626 Predictor/artifacts/sid_diagnostics/sid_codebook_diagnostics.json
/root/autodl-tmp/0626/0626 Predictor/artifacts/sid_diagnostics/sid_diag_v2.log
```

当前验证结论：

```text
有效 item 数: 7,721,749
ID roundtrip 正确率: 100%
train 前 100,000 条 target/history SID 覆盖率: 100%
```

每层 code 使用情况：

```text
4 层都使用了全部 256 个 code
没有 dead code
每层 entropy ratio 约 0.991 - 0.994
单个 code 最大占比约 0.8% - 1.6%
```

解释：

```text
这是 balanced KMeans 的预期结果，说明没有某几个 code 吞掉大部分 item。
这是好事，因为它避免码本坍塌，也让 HPN 每层分类更平衡。
但它不等于热门歌曲会集中到少数 token；热门程度不是当前聚类目标。
```

当前聚类目标是 item embedding 的几何相似性，不是曝光量、播放量或流行度。很多热门歌可能风格不同，因此不会落到同一类；很多冷门歌如果 embedding 相似，也可能共享相同前缀。

## 4. 碰撞和候选桶问题

虽然理论组合空间是：

```text
256^4 = 4,294,967,296
```

但实际只用到了其中很小一部分路径。

全量统计：

```text
unique full SID paths: 4,784,986
collision item share: 50.8074%
unique item share: 49.1926%
最大 full SID bucket: 3,402 items
```

val/test/replay_val 目标 item 的桶大小：

```text
val:
  target collision share: 58.71%
  p50 bucket size: 2
  p90 bucket size: 18
  p99 bucket size: 347
  max bucket size: 1904
  bucket <= 32 share: 93.78%

test:
  target collision share: 59.37%
  p50 bucket size: 2
  p90 bucket size: 18
  p99 bucket size: 347
  max bucket size: 1282
  bucket <= 32 share: 93.82%

replay_val:
  target collision share: 58.63%
  p50 bucket size: 2
  p90 bucket size: 19
  p99 bucket size: 347
  max bucket size: 3230
  bucket <= 32 share: 93.80%
```

结论：

```text
如果系统动作被定义成唯一 item，那么当前 full SID 可寻址性不足。
如果系统动作被定义成 SID 候选桶，再用 rerank 选具体 item，则当前结果可以继续尝试。
```

不建议把碰撞桶内 item 随机选为最终推荐。原因是 val/test 目标中约 59% 都在碰撞桶里，随机选择会让策略动作变得不稳定。

## 5. 为什么提高每层 code 数不能完全解决共编码

提高每层 code 数，例如从 256 提到 512 或 1024，通常会降低碰撞，但不能保证彻底解决。

原因：

```text
1. 理论组合空间大，不等于实际路径都会被使用。
2. RQKMeans 是逐层最近中心分配，很多组合路径永远不会出现。
3. 当前 balanced KMeans 只平衡每一层的边际 token 使用，不平衡完整 path 使用。
4. 如果 embedding 本身把多个 item 放得很近，码本仍可能把它们编码到相同或相近路径。
```

提高 code 数的代价：

```text
1. HPN 每层分类空间变大，训练更难。
2. beam decode 和候选索引变大，推理成本上升。
3. 粗语义可能变碎，前几层 token 的语义解释性可能下降。
```

阶段判断：

```text
可以做小规模对照实验验证 512 vocab 是否降低碰撞。
但这不是优先级最高的问题。
当前更稳妥的路线是把 full SID 视为候选桶，并补强候选桶 rerank。
```

## 6. 语义相似性验证

embedding 最近邻和 SID 前缀关系：

```text
embedding 最近邻平均 cosine: 0.90096
最近邻 same level-1 token: 70.75%
随机 pair same level-1 token: 0.5%
最近邻 same level-1/2 prefix: 29.1%
最近邻 same level-1/2/3 prefix: 7.7%
最近邻 full SID 相同: 2.5%
```

解释：

```text
第 1 层具有明显粗语义聚类能力。
第 2 层仍有一定相似性。
第 3/4 层更偏细粒度残差编码，不应期待相似物品都共享完整路径。
```

所以当前码本不是完全失败；问题主要在 full SID 作为唯一 item action 的可寻址性不足。

## 7. soft next state 是什么

soft next state 是一个临时预测状态，不是真实用户历史。

它接收：

```text
当前状态 s_t
候选动作 a
predictor 预测出的 response_probs / played_ratio / reward / regret_probs
```

输出：

```text
soft_next_state(s_t, a)
```

作用：

```text
在不真实执行推荐、不写入用户历史的情况下，估计“如果推荐这个候选 item，下一状态大概会变成什么样”。
```

例子：

```text
当前用户喜欢 A、B，最近对 C 播放很低。
候选 item D 的 predictor 预测结果是高播放、低 dislike 风险。
soft next state 就把 D 和预测反馈压成一个临时下一状态表示。
value head 再估计这个临时下一状态的长期价值。
```

边界：

```text
soft next state 只能用于候选打分和 Bellman-style value target。
不能永久写入用户真实 history。
真实 history 只能由真实日志或真实交互更新。
```

## 8. val/test/replay_val 是什么

当前 0626 Predictor 直接读取 0408 Regret transitions。

数据位置：

```text
/root/autodl-tmp/0626/0626 Predictor/01_data/processed/regret_current_data
```

其中：

```text
train:
  训练集，用户有较长连续 step，可用于 predictor/value 训练。

val:
  验证集，当前每个用户只有 1 条，适合单步响应预测，不适合验证多步未来价值。

test:
  测试集，当前每个用户也只有 1 条，适合单步最终评估，不适合验证多步 future_return。

replay_val:
  经验回放式验证集，用户平均约 49 条连续 step，更适合检查 future_return、soft next state、value head 的多步意义。
```

这里的 replay 不是强化学习在线 replay buffer，而是从历史序列中保留下来的连续轨迹验证材料。它用于评估“多步未来预测”是否有意义。

当前发现：

```text
val/test 中 future_return 基本等于当前 reward。
train/replay_val 中 future_return 才真正包含后续 step。
```

因此后续不能只看 val/test 的 value 结果。

## 9. Bellman 最优方程当前实现和差距

当前实现是 Bellman-style 近似，不是完整贝叶斯最优价值函数。

当前代码目标近似：

```text
V(s_t)
≈ max_{a in C_t} [
    predicted_reward(s_t, a)
    + gamma * V(soft_next_state(s_t, a))
  ]
```

如果 `sample_m > 1`，则近似：

```text
average_m max_{a in C_t} [...]
```

差距：

```text
1. sigma 只是单个 predictor 的输出分布，不是真正后验分布。
2. 默认 sample_m=1，很多运行实际上没有多场景采样。
3. reward/play 没有完整随机采样，目前主要是 response/regret 采样。
4. 没有稳定的 target value network。
5. predictor 自身还没有证明在候选反事实 item 上可靠。
6. HPN 候选召回很低时，value/rerank 无法发挥作用。
```

需要区分两种优先级：

```text
方法实现优先级:
  P0: 先确认语义 ID 和候选桶机制可用。
  P1: 修正 predictor 专用 train/val/test 切分，保证验证集和测试集有连续 step。
  P2: 完整验证 predictor 的每类反馈预测能力，不能只看总 acc。
  P3: 用连续验证集检查 future_return、soft next state 和 value head。
  P4: 引入 target value network，稳定 Bellman-style target。
  P5: 再考虑 sample_m > 1、多模型 ensemble 或 dropout 多次前向，增强不确定性估计。

端到端推荐优先级:
  HPN 候选召回仍然是最终 rerank 能否生效的瓶颈。
  但在 predictor/value 还没有验证完整之前，不应先把主要精力放在 HPN 调参。
```

## 10. 当前阶段建议

当前不要急着重建码本，也不要直接跑大规模 predictor 训练。

建议顺序：

```text
1. 接受当前 RQKMeans 语义 ID 作为候选桶索引，而不是唯一 item 地址。
2. 先为 predictor 建立合理的连续 train/val/test 验证口径。
3. 单独验证 predictor 的 response/play/reward/regret 预测质量。
4. 再验证 soft next state 和 Bellman-style value。
5. 最后回到 HPN 候选召回和端到端 rerank。
6. 如果候选召回仍受 SID 碰撞或桶长尾影响，再做 512 codebook 对照实验。
```



## 11. 2026-06-29 predictor 口径修正

本轮决定把 predictor 工作区的播放反馈口径改为：

```text
只要存在 listen，就把播放项视为非负正反馈。
低播放不再单独作为后悔类别。
显式 dislike / unlike 仍然保留为负反馈和后悔风险。
```

新的 predictor-side reward 口径：

```text
play_reward = clip(max_play_ratio, 0, 1)    如果存在 listen
play_reward = 0                             如果不存在 listen

reward =
  play_reward
+ 0.8 * effective_like
- 1.2 * effective_dislike
- 0.6 * effective_unlike
+ 0.2 * effective_undislike
```

注意：如果一个 step 同时有低播放和显式 dislike，最终 reward 仍可能为负。这表示负值来自显式负反馈，不是低播放本身。

本轮新增：

```text
01_data/build_predictor_seq_splits.py
```

作用：

```text
读取已经构造好的 Regret step-level transitions。
不重新切 session。
不重新聚合同 item step。
只按每个用户的时间顺序重新切 predictor 专用 train/val/test。
默认比例为 80% / 10% / 10%。
```

本轮还扩展了 `FutureIterableDataset`：

```text
默认 reward_mode = positive_play
默认 regret_mode = explicit_negative
输出 next_history_* 字段，用于 soft next state 监督。
```

value 训练现在包含：

```text
target_value_head:
  用慢更新目标网络稳定 Bellman-style target。

soft_state_loss:
  用真实 next_history 的编码监督 soft_next_state(s, a)，避免 soft state 只是随机变换。
```

验证指标也从单一 response accuracy 扩展为：

```text
response_by_class:
  listen / like / dislike / unlike / undislike 的 precision / recall / F1

regret_by_class:
  none / low_play / dislike / unlike 的分类召回
```

阶段判断：

```text
下一步应该先生成 predictor_seq_data 并做 predictor 专用训练验证。
不要先调 HPN。
HPN 是端到端 rerank 的后续瓶颈，但 predictor/value 口径必须先独立站稳。
```
