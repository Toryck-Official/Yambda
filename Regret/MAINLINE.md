# main_v1 正式主线

这个文件只记录当前正式主线。旧实验文件和长名字先保留，但后续汇报、复现、继续开发都优先看这里。

## 一句话流程

```text
原始事件流 -> 按 1 小时切 session -> 连续同 item 聚合成 RL step -> 训练 simulator -> 训练 HSAC 策略 -> 用 simulator rollout 评估 reward / depth / negative rate
```

## 以后只用这 5 个入口

```bash
./scripts/01_split_data.sh
./scripts/02_train_simulator.sh
./scripts/03_train_policy.sh
./scripts/04_eval_policy.sh
./scripts/05_sweep_eta.sh
```

长任务用 screen：

```bash
screen -dmS main_policy bash -lc 'cd /root/autodl-tmp/0408Yambda/Regret && ./scripts/03_train_policy.sh'
screen -r main_policy
```

主线配置：

```text
configs/main.env
```

## 短路径

当前已验证产物通过短路径访问：

```text
artifacts/current/data
artifacts/current/simulator
artifacts/current/policy_actor
artifacts/current/policy_critic
artifacts/current/policy_meta.json
artifacts/current/eval_eta010_test1000.json
artifacts/current/eta_sweep_test500.tsv
```

这些短路径是软链接，指向历史长名字产物。这样不破坏旧日志，同时日常不用再记长名字。

如果重新跑实验，默认输出到：

```text
artifacts/transitions/main_v1_data
artifacts/user_response/main_v1_simulator
artifacts/models/policy_hsac_main_actor
artifacts/models/policy_hsac_main_critic
artifacts/models/policy_hsac_main.meta.json
```

确认新结果更好后，再手动更新 `artifacts/current/*` 软链接。不要直接覆盖当前已验证结果。

## 数据定义

原始 timestamp 单位是 5 秒。session 切分规则：

```text
相邻事件时间差 > 3600 秒，则断开成新 session
单个 session 最大跨度 21600 秒
连续同 item 最多聚合 100 个原始事件
```

RL step 定义：

```text
同一个 session 内，连续出现的同一个 item 聚合成一个 step
例如 AAABBBCDE -> A, B, C, D, E 共 5 个 step
每个 step 的时间戳取该连续 run 的最后一个原始事件时间
```

这样做的目的：让“系统推荐一个 item”和“用户对这个 item 的一组连续反馈”一一对应。

公式 8 的 `Delta_tr`：

```text
B_rev 是按用户维护的跨 session 失败记忆池。
主线预计算记忆池使用 REGRET_MEMORY_DELTA_MODE=time_hour：
Delta_tr = (当前 step_time - 失败 step_time) * 5秒 / 3600秒
Phi = gamma^Delta_tr * psi
```

说明：

```text
step_time 仍取连续同 item run 的最后一个原始事件 timestamp。
如果改成 REGRET_MEMORY_DELTA_MODE=step，则退回旧实现：Delta_tr = user_step_idx 差。
如果改成 REGRET_MEMORY_DELTA_MODE=time_tick，则直接用原始 timestamp tick 差，通常衰减过快，只建议做消融。
```

## RRCA 和 RAPI

RRCA 是训练时的回顾式 reward 修正：

```text
base reward = omega_listen * played_ratio + omega_like * like - omega_dislike * dislike
psi = - lambda_unlike * unlike + lambda_undislike * undislike
effective reward = base reward + gamma^Delta * psi
```

当前主线是 `rrca_apply_to=effective_reward`，也就是把后悔/修正信号写进有效 reward，再影响优势函数。固定主线取 `omega/lambda=1`；可学习权重只作为消融，通过 `LEARN_REWARD_WEIGHTS=1` 打开。

RRCA 的回溯位置：

```text
主线：RRCA_CALLBACK_TARGET=current
含义：发现 unlike / undislike 后，只修正当前五元组的 effective reward。

消融：RRCA_CALLBACK_TARGET=latest_same_item
含义：在当前 rollout 记录里找最近一次相同 item，把 gamma^Delta * psi 回写到那个历史 step 的 effective reward。
```

当前论文版本更接近 `current`；`latest_same_item` 是“真正回调历史 reward”的对照实验，不默认启用。

Revision 有效性门控：

```text
GATE_REVISION_BY_HISTORY=1
unlike 只有在当前历史里同 item 曾经出现 like 时，才作为有效撤销信号。
undislike 只有在当前历史里同 item 曾经出现 dislike 时，才作为有效正向撤销信号。
```

原因：当前 simulator 容易过度预测 `undislike`。如果不做门控，会把不存在前置 dislike 的无效撤销也加成 `+psi`，导致 `effective_reward` 被虚假抬高，训练 loss 和 advantage 失真。

RAPI 是生成时的失败记忆池介入：

```text
失败记忆池记录用户历史里失败的语义路径
当前主线把 low_play / dislike / unlike 都放进失败记忆池
生成下一个语义 token 时，对和失败路径重叠的 token 加 soft mask
```

RAPI 消融分两层：

```text
推理期消融：04_eval_policy.sh 已经自动比较 base vs RAPI。
训练期消融：设置 TRAIN_USE_RAPI=0，单独训练一个 no-RAPI actor。
```

两者回答的问题不同：

```text
base vs RAPI：同一个 actor，推理时加不加失败记忆池 soft mask。
TRAIN_USE_RAPI=0：训练 rollout 过程中完全不用失败记忆池，看学出来的 actor 本身是否变差。
```

旧文件名里的 `brev` 就是论文里的 `B_rev`，以后中文统一叫“失败记忆池”。旧文件名里的 `af` 是 `all_failed`，表示 low_play / dislike / unlike 都进入失败记忆池。

## 当前固定参数

```text
history_len=50
session_gap_seconds=3600
max_session_span_seconds=21600
max_run_events=100
reward_version=v2

regret_memory_size=20
regret_memory_gamma=0.9
regret_memory_scope=all_failed
regret_memory_delta_mode=time_hour

simulator=train_rows=1m, epochs=3, decoupled_reward_model

hsac_episodes=100000
hsac_epochs=1
hsac_max_steps=10
hsac_decode_top_k=4
hsac_gamma=0.9
train_use_rapi=1
rrca_apply_to=effective_reward
rrca_signal_scope=revision_only
rrca_callback_target=current
gate_revision_by_history=1
base_reward_mode=paper
omega/lambda fixed at 1.0 by default

eval_eta=0.10
eval_episodes=1000
eval_max_steps=20
eval_decode_top_k=8
```

## 当前主结果

注意：下面结果来自 `REGRET_MEMORY_DELTA_MODE=time_hour` 引入前的已验证产物；它仍可作为历史 benchmark，但如果要严格对齐公式 8 的跨 session 时间差，需要先重新切分数据再训练/评估。

1000 episodes simulator rollout，RAPI eta=0.10：

```text
base_reward=7.2940
rapi_reward=7.3314
delta_reward=+0.0374

base_step=19.737
rapi_step=19.785
delta_step=+0.048

base_neg=0.3004
rapi_neg=0.3006
delta_neg=+0.00018
```

结论：

```text
RAPI 对累计 reward 和交互 depth 有小幅正收益。
negative rate 没有明显下降，说明失败记忆池目前主要在改善整体收益，不是稳定抑制负反馈。
```

## 和论文仍未完全一致的地方

```text
1. simulator 目前一次推荐主要给一个聚合反馈，不是真实用户可能连续多动作的完整过程。
2. lambda / omega 可学习权重已经有消融开关，但还没有跑正式消融结果。
3. RAPI 的失败记忆池已包含 low_play / dislike / unlike，但介入强度 eta 仍需要 sweep。
4. 当前评估依赖 simulator，不能等同于真实在线 A/B。
5. 旧 offline SID/replay 训练仍保留做消融，不是当前主线。
```

## 脚本说明

详细脚本用途看：

```text
scripts/README.md
```

## 旧文件处理规则

```text
run_*.sh：旧包装入口已删除；当前主线只用 01-05 短脚本
scripts/User-response/：已合并进 08_train_yambda_simulator.py 并删除
06_train_sid_offline.py 和 07_eval_regret_pool_replay.py：旧 offline/replay 消融，不是主线
```

后续不要再新增 `run_*` 包装脚本；需要新入口时沿用 `NN_中文可读动作.sh` 的命名方式。
