# Regret Daily Log Beijing Time

这份是给用户看的简版阶段记录，只保留核心脉络。除非用户明确要求，后续不再持续追加。

## 核心阶段

### UserResponse 重构

- 问题：早期 simulator 直接预测 scalar reward，容易塌缩到均值，无法判断错误来自 listen、play、like 还是 dislike。
- 处理：改成 feedback-first simulator，分别预测 listen、play_ratio、like、dislike、unlike、undislike，再按 reward 公式组合。
- 意义：simulator 更像真实用户反馈环境，policy 训练时能拿到更可解释的反馈。

### time_hour 主线

- 问题：论文里的 `Delta_tr` 是时间距离，旧实现更接近 step index 距离。
- 处理：主线 split 使用 `time_hour`，原始 timestamp 单位按 5 秒换算成小时。
- 主线数据路径：`/root/autodl-tmp/0408Yambda/Regret/artifacts/transitions/main_v1_data_timehour_r1`。

### RRCA revision gate

- 问题：simulator 曾大量预测无历史依据的 `undislike`，导致 `eff_r` 被虚假抬高。
- 处理：`unlike` 必须有同 item 历史 like，`undislike` 必须有同 item 历史 dislike，才允许进入 RRCA 修正。
- 结果：虚假 reward 抬升被压住，但也暴露出有效 revision 信号很稀疏。

### low_play 与失败记忆池

- 用户判断：音乐推荐里短播放仍然是一次交互，不应直接当成 RAPI 失败路径。
- 当前共识：RAPI 失败记忆池只记显式负反馈，即 dislike 和有效 unlike；low_play 不写入失败路径池。
- 训练侧现状：`expneg_failure` policy 训练里 `failure_low_play_rate=0.0`，`memory_low_play_per_step=0.0`，说明训练时显式负反馈开关生效。

### 当前 RAPI 问题

- 1000 episode 评估：`base_reward=4.7550`，`rapi_reward=4.7363`，RAPI 没有稳定正提升。
- 发现：旧 split 的 precomputed memory snapshot 仍是 `all_failed`，low_play 数量远大于 dislike/unlike，会挤占 memory slot。
- 发现：训练可以加载 precomputed snapshot，评估之前主要从当前 history 初始化 memory，二者不一致。
- 当前修正方向：让 split 真正生成 `explicit_negative` snapshot，并让 eval 和 train 使用一致的 precomputed memory 初始化。

## 当前不要轻易动的大项

- 不直接重训 semantic ID。先诊断 SID token 分布和 RAPI 第 0 层介入是否过粗。
- 不直接大修 simulator。先比较真实数据和 simulator rollout 的 play_ratio、like、dislike 分布。
- 不覆盖旧 transition root。新实验使用真实新路径，不用软链接。

## 2026-05-19 BJT 简要更新

### reward 量级问题

- 复查后确认：reward 从 4-5 回升到 8-9，主要不是 RAPI 带来的，而是 simulator 反馈采样和失败信号口径修正后，`dislike` 不再被过度放大。
- 当前主线仍使用论文固定权重：`reward = play_ratio + like - dislike`，`omega/lambda` 默认都是 1，不是学习出来的。
- 当前 simulator 仍是核心环境：policy 训练和最终评估都会用它产生用户反馈。

### RAPI 失效诊断与修改

- 问题：旧 RAPI 只改每层 SID token logits，但 decoder 最后会在合法 SID 组合里选 item，导致 token 级变化经常没有改变最终 item。
- 修改：新增 candidate-level rerank。先生成合法 item 候选，再根据候选 SID 和用户失败记忆池 `B_rev` 的相似度扣分。
- 公式口径：`adjusted_score = decoder_score - eta * failure_path_penalty`。
- 训练和评估脚本都已接入：默认 `RAPI_CANDIDATE_RERANK=1`，`RAPI_CANDIDATE_ETA=1.0`。

### 修改前后案例

- 诊断样本：256 条 test episode。
- 有非空 memory 的行：`21/256 = 8.2%`，说明 memory 本身仍偏稀疏。
- 旧 token-RAPI：最终 item 改变率为 0。
- 新 candidate-RAPI：改变 `1/256` 条，active memory 行里改变 `1/21 = 4.76%`。
- 具体例子：用户 `19400`，base 和旧 RAPI 都推荐 item `19898`，新 RAPI 改为 item `6355763`。
- 小样本 smoke eval：16 episodes 上 `base_reward=8.5187`，`rapi_reward=8.5875`，`delta=+0.0688`；样本太小，只能说明新逻辑能跑通并且不再完全不动。

### 当前判断

- RAPI 已从“几乎无效”变成“能改变最终 item”，但还没证明能稳定提升。
- 当前更可能的瓶颈是：显式负反馈 memory 太稀疏，以及合法候选 item 与失败路径重合太少。
- 下一步先跑正式评估，不急着重训 simulator、policy 或 semantic ID。
