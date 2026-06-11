# 2026-05-15 Regret Debug Todo

本文件是给 assistant 自己看的排查控制板。除非用户明确要求，后续不再把聊天过程持续追加到这里；真正的工作重点转到代码和实验诊断。

## 工作边界

- 允许直接做：读代码、读日志、读 meta、写文档、做只读统计、跑不改变产物的诊断脚本。
- 需要先向用户确认：修改 reward 公式、修改失败记忆池写入规则、修改 RAPI 介入公式、重切数据、重训 simulator、重训 policy、重训 semantic ID。
- 不使用软链接作为新的实验入口。后续命令都使用真实路径。

## 术语说明

- `simulator`：用户反馈模拟器。给定当前用户历史和推荐 item，预测用户会不会听、播放比例、like、dislike、unlike、undislike 等反馈。这里它相当于离线训练 policy 时的环境。
- `policy`：推荐策略网络。输入用户历史，输出下一次推荐的 semantic ID，然后映射成 item。
- `actor`：policy 里的动作生成部分。它负责生成推荐 item。
- `critic`：policy 里的价值评估部分。它估计当前状态或动作的长期收益，用来稳定 actor 更新。
- `rollout`：把 policy 放进 simulator 里连续交互若干步。例如推荐 20 次，每次 simulator 返回反馈和 reward。
- `reward`：奖励值。当前 paper 公式里大致是 `play_ratio + like - dislike`，如果启用 RRCA effective reward，还会加 revision correction。
- `base_reward`：不含 RRCA 修正的基础奖励。
- `effective_reward`：训练 actor/critic 时实际使用的奖励，可以包含 RRCA 修正。
- `RRCA`：Retrospective Revision Credit Assignment，回顾式修正奖励。意思是用户后面出现 unlike 或 undislike 时，回头修正相关决策的奖励。
- `RAPI`：Revision-Aware Policy Intervention，生成 item 时用失败记忆池干预 policy logits，避免推荐和失败路径相似的 semantic ID。
- `B_rev`：论文里的用户失败记忆池。这里也叫 regret memory pool。
- `memory snapshot`：离线预计算的每个用户在当前 step 前的失败记忆池状态。它用于解决 dataloader shuffle 后无法按用户在线维护历史的问题。
- `semantic ID`：语义 ID。一个 item 被表示成多层 token，例如 `[12, 33, 7, 201]`，从粗到细描述 item。
- `SID token`：semantic ID 的某一层编号。例如 `[12, 33, 7, 201]` 里的 `12` 是第 0 层 token。
- `eta`：RAPI 介入强度。eta 越大，失败路径对 logits 的惩罚越强。
- `phi`：失败记忆池里某条失败路径的强度。phi 越大，说明这条失败路径越需要被避开。
- `gamma`：衰减因子。时间越久的失败反馈，影响越小，通常用 `gamma^Delta`。
- `low_play`：播放比例很低的隐式失败信号。
- `explicit negative`：显式负反馈，目前主要指 `dislike` 和有效 `unlike`。
- `invalid revision`：无效撤销信号。例如没有历史 dislike，却出现 undislike；这种不应该给正向修正。

## 当前数据流例子

假设用户 `U1` 的真实事件流是：

```text
时间 100: item A, listen, play_ratio=0.90, like=1
时间 130: item B, listen, play_ratio=0.20
时间 160: item B, dislike=1
时间 220: item C, listen, play_ratio=0.70
```

切分阶段会把连续同 item 的事件聚成 RL step：

```text
step 1: item A, positive
step 2: item B, low_play + dislike
step 3: item C, positive
```

如果失败记忆池使用 `explicit_negative`，那么 step 3 前的 memory snapshot 应该只记：

```text
item B 的 semantic ID，例如 [10, 8, 99, 7]
regret_type_id=2，表示 dislike
phi=dislike_strength * gamma^Delta
```

它不应该因为 item B 播放低而额外写入 low_play。当前排查的核心之一就是确认 split 端是否真的做到这一点。

policy 训练时的数据流是：

```text
transition batch
-> env.reset_from_batch 得到当前用户历史
-> actor 根据历史生成 semantic ID
-> decoder 把 semantic ID 映射成 item
-> simulator 返回 listen/play/like/dislike/unlike/undislike
-> 根据反馈组成 base_reward 和 effective_reward
-> critic 估值，actor 根据 advantage 更新
```

评估时的数据流是：

```text
同一个 actor
-> base 分支：不使用 RAPI memory
-> RAPI 分支：使用失败记忆池 soft mask
-> 两边分别进入 simulator rollout
-> 比较 avg reward、step、negative rate、failure rate
```

当前发现的问题是：训练分支可以加载 precomputed memory snapshot，但评估分支目前主要从当前 50 条 history 初始化 memory。两者不完全一致。

## 2026-05-19 当前状态

- 已按用户确认继续推进，不再停留在参数 sweep。
- 已修正 simulator 反馈采样和显式负反馈失败口径后，reward 量级回到约 8-9；这主要来自 simulator 分布修正，不应归功于 RAPI。
- 已定位旧 RAPI 的关键问题：token 级 soft mask 不一定改变 decoder 最终 item。
- 已新增 candidate-level RAPI rerank：对合法 item 候选按失败路径相似度扣分。
- 已把新 RAPI 接入训练、评估、诊断三条链路。
- 已完成小样本 smoke eval：16 episodes，`delta_reward=+0.0688`，只作为跑通证据。
- 当前要跑正式评估，判断新 RAPI 是否有稳定收益。

## 今日逐层排查清单

### L0. 证据冻结和文档化

- [x] 写入今日待办文档。
- [x] 写入北京时间阶段日志。
- [x] 汇总当前关键 artifact 的真实路径，不用 `artifacts/current` 软链接。
- [x] 汇总当前最新训练和评估的 meta 指标。

判断标准：

- 文档中明确哪些是事实，哪些是推断，哪些需要确认后再改。
- 后续任何长任务都有 screen 名称、log 路径、meta 路径。

### L1. 当前实验结果体检

- [x] 读取 `policy_hsac_main_timehour_r1_expneg_failure.meta.json`。
- [x] 读取 `main_v1_timehour_r1_expneg_failure_eta0p02_test_1000.meta.json`。
- [x] 对比 100 episode eta sweep 和 1000 episode 正式评估。
- [x] 确认当前 actor、simulator、transition root 是否都是预期真实路径。

重点指标：

```text
base_reward
rapi_reward
delta_reward
base_neg / rapi_neg
failure_low_play_rate
failure_dislike_rate
invalid_revision_mass_per_step
raw_negative_rate
```

目前已知判断：

- `eta=0.02` 在 1000 episodes 上没有稳定正提升。
- 当前不是 RAPI 符号明显反了。
- 当前 reward 量级低，主要来自 simulator 给出的 play_ratio 偏低、dislike 偏高。

### L2. RAPI memory snapshot 管线

目标：确认失败记忆池是否真正是用户级、显式负反馈、训练评估一致。

- [x] 检查 split meta 中 `regret_memory_scope`。
- [x] 统计 snapshot 里 low_play / dislike / unlike 的实际占比。
- [x] 统计过滤成 explicit_negative 后，每个样本还剩多少 memory entry。
- [x] 对比 precomputed snapshot 和 current history 初始化出来的 memory 覆盖率。
- [x] 检查 train 是否加载 snapshot。
- [x] 检查 eval 是否加载 snapshot。

需要确认后才允许做的逻辑修改：

- [ ] 修改 `02_split_transitions.py`，让 `REGRET_MEMORY_SCOPE=explicit_negative` 在 split 端真正排除 low_play。
- [ ] 重新生成一个新 transition root，例如 `artifacts/transitions/main_v1_data_timehour_r1_explicitmem`。
- [ ] 修改 `09_eval_simulator_rollout.py`，让 eval 支持和 train 一样加载 precomputed memory snapshot。

不确认就不做：

- 不覆盖 `artifacts/transitions/main_v1_data_timehour_r1`。
- 不直接改已有正式 meta。

### L3. RAPI 介入公式和第 0 层问题

目标：判断 RAPI 是太弱、太强、还是太粗。

- [x] 统计 RAPI memory 存在时 action 真实改变比例。
- [x] 按 eta 统计 action changed rate。
- [x] 按 semantic ID 层级统计 token changed rate。
- [ ] 统计第 0 层 token 分布是否过于集中。
- [ ] 统计失败 item 和正反馈 item 在第 0 层 prefix 上是否混杂严重。

当前代码事实：

- 第 0 层没有 prefix，所以相似度被设成 1。
- 这不是说所有推荐 token 都一定在失败池里。
- 它的含义是：memory 中出现过的第 0 层 token 会被无条件惩罚。
- 如果第 0 层 token 很粗，一个失败 item 可能误伤同粗类下大量好 item。

需要确认后才允许做的逻辑修改：

- [ ] 禁用第 0 层 RAPI intervention。
- [ ] 给第 0 层设置更小权重。
- [ ] 改成只在更深层 prefix 命中后才惩罚。
- [ ] 改 eta 默认值或 layer weights 默认值。
- [x] 新增 candidate-level rerank，让 RAPI 直接影响最终合法 item 排序。

当前新增证据：

```text
256 episode 静态诊断:
active_memory_rows = 21/256 = 8.2%
旧 token-RAPI item changed = 0/256
新 candidate-RAPI item changed = 1/256
active memory 内 changed = 1/21 = 4.76%

案例:
user_id = 19400
base_item = 19898
old_token_rapi_item = 19898
new_candidate_rapi_item = 6355763
base_sid = [186, 208, 0, 243]
new_sid = [186, 208, 0, 218]
```

解释：

- 旧 RAPI 不是一定没有改变 logits，而是改变后仍可能被 decoder 映射回同一个最终 item。
- 新 RAPI 已能改变最终 item，但改变率仍低。
- 短期先用正式 eval 验证收益；如果仍不稳，再查 memory 稀疏和 candidate overlap。

### L4. Simulator 分布校准

目标：解释为什么当前 reward 从历史 7 左右降到 4-5。

- [ ] 统计真实 test transition 的 play_ratio、like、dislike、unlike、undislike 分布。
- [ ] 统计 simulator rollout 的同类分布。
- [ ] 比较真实数据和 simulator 的 play_ratio 均值、dislike rate、listen rate。
- [ ] 单独统计 policy 推荐 item 是否偏离训练数据分布。
- [ ] 检查 simulator 是否把 organic 行为信息作为输入或目标的一部分。

当前已知公式拆解：

```text
当前 eval:
mean_play_ratio≈0.391
like_rate≈0.020
dislike_rate≈0.163
reward_per_step≈0.391 + 0.020 - 0.163 = 0.248
avg_step≈19.17
total_reward≈4.75
```

所以 reward 低不是显示错误，而是 simulator 当前反馈分布偏严格。

需要确认后才允许做的逻辑修改：

- [ ] 改 simulator 训练目标。
- [ ] 改 simulator loss 权重。
- [ ] 改 simulator 采样策略。
- [ ] 加入 organic 特征。
- [ ] 重训 simulator。

### L5. Semantic ID 健康度检查

目标：判断是不是语义 ID 本身导致 RAPI 失败。

- [ ] 统计各层 SID token 的频率分布。
- [ ] 统计 item 到第 0 层 token 的覆盖是否极端不均。
- [ ] 统计正反馈和显式负反馈在 SID prefix 上的可分性。
- [ ] 抽样查看同 prefix 下 item embedding 是否相似。
- [ ] 检查 decoder top-k 是否经常回退到 fallback item。

当前判断：

- 暂时不能因为第 0 层相似度全 1 就断定 semantic ID 坏了。
- 这更可能先是 RAPI 第 0 层介入太粗。
- 重训 semantic ID 是最大动作，只能在诊断证明 SID 结构明显有问题后再讨论。

需要确认后才允许做的逻辑修改：

- [ ] 重训 SID。
- [ ] 改 RQKMeans/RQVAE 生成方式。
- [ ] 改 semantic ID 层数或 vocab size。

### L6. Policy 重新训练与正式评估

目标：验证新的 memory 设计是否真的提升主线。

前置条件：

- [ ] L2 memory snapshot 管线已修正并确认。
- [ ] eval 和 train memory 初始化一致。
- [ ] RAPI 诊断指标已经写入 meta。
- [ ] simulator 是否重训已经单独确认。

正式训练前需要确认：

- [ ] 新 transition root 名称。
- [ ] 新 simulator checkpoint 路径。
- [ ] 新 actor save prefix。
- [ ] eta 和 layer weights。
- [ ] 是否禁用第 0 层 intervention。

## 必须停下来问用户的确认点

1. 是否重切数据或重新生成 memory snapshot。
2. 是否把失败池范围固定为 `explicit_negative`。
3. 是否让 eval 加载 precomputed memory snapshot。
4. 是否修改第 0 层 RAPI 介入。
5. 是否重训 simulator。
6. 是否把 organic 加入 simulator 输入或训练目标。
7. 是否重训 policy。
8. 是否回头检查或重训 semantic ID。

## 下一步建议

先跑新 candidate-RAPI 的正式评估，不再只看 16 episode smoke test。若正式评估仍无稳定收益，再继续查两个方向：

- memory 稀疏：显式负反馈太少，很多用户没有可用 `B_rev`。
- candidate overlap：decoder top-k 合法候选和失败路径相似度重合太少，RAPI 即使生效也很少改变 item。

## L1/L2 只读诊断结果

本节结果来自只读读取 meta 和 parquet，不涉及逻辑修改。

### 当前真实 artifact 路径

```text
transition root:
/root/autodl-tmp/0408Yambda/Regret/artifacts/transitions/main_v1_data_timehour_r1

simulator:
/root/autodl-tmp/0408Yambda/Regret/artifacts/user_response/main_v1_timehour_r1_simulator/regret_user_response.pt

actor:
/root/autodl-tmp/0408Yambda/Regret/artifacts/models/policy_hsac_main_timehour_r1_expneg_failure_actor

critic:
/root/autodl-tmp/0408Yambda/Regret/artifacts/models/policy_hsac_main_timehour_r1_expneg_failure_critic

train meta:
/root/autodl-tmp/0408Yambda/Regret/artifacts/models/policy_hsac_main_timehour_r1_expneg_failure.meta.json

1000 episode eval meta:
/root/autodl-tmp/0408Yambda/Regret/artifacts/evals/main_v1_timehour_r1_expneg_failure_eta0p02_test_1000.meta.json
```

### L1 训练和评估现状

训练 meta：

```text
loss=0.6735
base_reward_per_step=0.3610
effective_reward_per_step=0.3609
negative_rate=0.1018
failure_low_play_rate=0.0
failure_dislike_rate=0.1017
failure_unlike_rate=0.0001
memory_low_play_per_step=0.0
memory_dislike_per_step=0.1017
invalid_revision_mass_per_step=0.4980
raw_negative_rate=0.4794
callback_per_step=0.000117
```

解释：

- 训练期 low_play 没有进入 failure/memory，符合当前显式负反馈设定。
- `effective_reward` 和 `base_reward` 很接近，说明 RRCA 没有再把 reward 虚假抬爆。
- `invalid_revision_mass` 和 `raw_negative_rate` 仍然很高，说明 simulator 的原始反馈分布仍然偏脏。

1000 episode eval：

```text
base_reward=4.7550
rapi_reward=4.7363
delta_reward=-0.0187
base_reward_per_step=0.2481
rapi_reward_per_step=0.2467
base_neg=0.1630
rapi_neg=0.1639
base_fail=0.1630
rapi_fail=0.1639
base_invalid_revision_mass_per_step=0.4077
rapi_invalid_revision_mass_per_step=0.4070
base_raw_negative_rate=0.5113
rapi_raw_negative_rate=0.5117
```

解释：

- RAPI 没有稳定提升，且 negative/failure 还略高。
- 这不是训练崩溃，而是 RAPI 介入目前没有有效改善 simulator rollout。

### L2 memory snapshot 现状

split meta：

```text
regret_memory_scope=all_failed
total memory insertions=13,498,113
low_play=13,183,196
dislike=102,115
unlike=212,802
```

test split 的 snapshot 统计：

```text
rows=10,000
snapshot all entries=59,134
snapshot low_play=57,299
snapshot dislike=598
snapshot unlike=1,237
snapshot mean entries per row=5.91
snapshot explicit mean entries per row=0.18
snapshot explicit p50=0
snapshot explicit p90=0
snapshot any explicit rate=6.17%
snapshot any low_play but no explicit rate=64.19%
snapshot full20 rate=13.97%
snapshot full20 but no explicit rate=12.61%

history explicit mean entries per row=1.15
history explicit p50=0
history explicit p90=2
history any explicit rate=15.10%
```

train part-00000 的 snapshot 统计：

```text
rows=50,000
snapshot all entries=301,301
snapshot low_play=296,165
snapshot dislike=1,664
snapshot unlike=3,472
snapshot mean entries per row=6.03
snapshot explicit mean entries per row=0.10
snapshot explicit p50=0
snapshot explicit p90=0
snapshot any explicit rate=4.75%
snapshot any low_play but no explicit rate=68.71%
snapshot full20 rate=12.57%
snapshot full20 but no explicit rate=11.87%

history explicit mean entries per row=0.18
history explicit p50=0
history explicit p90=0
history any explicit rate=5.93%
```

解释：

- 当前 snapshot 明确是 `all_failed`，不是纯 `explicit_negative`。
- snapshot 的 entry 绝大多数是 low_play。
- test 里有 64.19% 的样本存在 low_play memory 但没有显式负反馈 memory。
- test 里 full20 且没有 explicit 的比例是 12.61%，这就是实际的 memory slot 挤占。
- 如果训练时再过滤 explicit negative，会导致很多样本 memory 直接变空。
- eval 当前从 history 初始化，test history explicit 覆盖率 15.10%，反而高于 snapshot explicit 覆盖率 6.17%。这证明 train/eval memory 来源不一致会影响判断。

### L2 结论

RAPI 当前结果不稳，优先不是调 eta，而是修正 memory 管线设计后再讨论 eta：

1. split 端需要真正生成 explicit-negative snapshot，而不是先 all_failed 后过滤。
2. eval 需要和 train 一样支持 precomputed snapshot，或者 train/eval 都统一成 history-only。按论文 `B_rev` 设定，更合理的是统一成 precomputed snapshot。
3. 这两项都属于逻辑修改，需要用户确认后才能做。
