# 04 主实验计划

本文记录 `/root/autodl-tmp/0626/0626 Predictor` 独立化之后的 base 主实验计划。

## 1. 当前目录分工

```text
00_preprocess/
  RQKMeans 码本训练和 item -> semantic ID 映射。

01_data/
  数据准备。默认使用本项目内的 01_data/processed。

02_model/
  HPN、future predictor、soft state、value model。

03_train/
  predictor、HPN、value 的离线训练。

04_eval/
  SID、predictor、value、rerank 的离线诊断。

06_rl/
  正式 RL 主实验：simulator、HSAC/HPN 策略训练、RRCA、RAPI、rollout 评估。
```

## 2. 当前已经具备的能力

```text
1. 数据和 RQKMeans 映射可以在本项目内读取。
2. predictor 可以单独训练和评估。
3. 0408 Regret 的 simulator / HSAC / rollout 代码已经复制到 06_rl。
4. 06_rl 默认使用本项目内的数据路径，不再默认依赖 0408。
5. baseline no-predictor 主实验现在可以从 06_rl 启动。
```

当前还没完成：

```text
predictor 尚未接入 06_rl 的 HSAC rollout 动作选择。
因此 with-predictor 的正式 RL 消融还不能直接跑。
```

## 3. 数据准备

如果 `01_data/processed/raw_rqkmeans` 和 `01_data/processed/regret_current_data` 已存在，可以跳过。

```bash
screen -dmS data_formal bash -lc 'cd "/root/autodl-tmp/0626/0626 Predictor" && COPY_TRANSITIONS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./01_data/prepare_formal_data.sh > 01_data/logs/prepare_formal_data.log 2>&1'
```

监控：

```bash
screen -r data_formal
tail -f "/root/autodl-tmp/0626/0626 Predictor/01_data/logs/prepare_formal_data.log"
```

## 4. SID / RQKMeans 验证

```bash
screen -dmS sid_diag bash -lc 'cd "/root/autodl-tmp/0626/0626 Predictor" && python3 04_eval/eval_sid_codebook.py --mapping_root 01_data/processed/raw_rqkmeans --transition_root 01_data/processed/regret_current_data --transition_split train --transition_max_rows 100000 --target_bucket_max_rows 0 --out_dir artifacts/sid_diagnostics > artifacts/logs/sid_diag_full.log 2>&1'
```

输出：

```text
artifacts/sid_diagnostics/sid_codebook_diagnostics.json
artifacts/logs/sid_diag_full.log
```

## 5. Predictor 训练

先跑 1M 行版本，作为 base predictor。

```bash
screen -dmS predictor_base bash -lc 'cd "/root/autodl-tmp/0626/0626 Predictor" && OUT_DIR=artifacts/predictor_base_1m MAX_TRAIN_ROWS=1000000 MAX_VAL_ROWS=50000 EPOCHS=1 ./run_predictor_only.sh > artifacts/logs/predictor_base_1m.log 2>&1'
```

监控：

```bash
screen -r predictor_base
tail -f "/root/autodl-tmp/0626/0626 Predictor/artifacts/logs/predictor_base_1m.log"
```

## 6. Predictor 离线验证

```bash
screen -dmS predictor_eval bash -lc 'cd "/root/autodl-tmp/0626/0626 Predictor" && CKPT=artifacts/predictor_base_1m/future_predictor.pt MAX_ROWS=50000 ./eval_predictor_only.sh > artifacts/logs/predictor_base_1m_eval.log 2>&1'
```

重点看：

```text
response_by_class
reward_mae
future_return_mae
future_regret_f1
oracle candidate contrast
predictor value ablation
```

## 7. Simulator 训练

```bash
screen -dmS sim_base bash -lc 'cd "/root/autodl-tmp/0626/0626 Predictor/06_rl" && MAIN_ID=base_v1 SIM_TRAIN_OUT_DIR=artifacts/user_response/base_v1_sim ./scripts/02_train_simulator.sh'
```

监控：

```bash
screen -r sim_base
tail -f "/root/autodl-tmp/0626/0626 Predictor/06_rl/artifacts/user_response/base_v1_sim/train.log"
```

输出：

```text
06_rl/artifacts/user_response/base_v1_sim/regret_user_response.pt
```

## 8. Baseline 策略训练

不接 predictor，只跑 HPN/HSAC + simulator + RRCA + RAPI。

```bash
screen -dmS policy_base bash -lc 'cd "/root/autodl-tmp/0626/0626 Predictor/06_rl" && MAIN_ID=base_no_predictor SIMULATOR_CHECKPOINT=artifacts/user_response/base_v1_sim/regret_user_response.pt POLICY_SAVE_PREFIX=artifacts/models/base_no_predictor ./scripts/03_train_policy.sh'
```

监控：

```bash
screen -r policy_base
tail -f "/root/autodl-tmp/0626/0626 Predictor/06_rl/artifacts/logs/base_no_predictor_policy_train.log"
```

输出：

```text
06_rl/artifacts/models/base_no_predictor_actor
06_rl/artifacts/models/base_no_predictor_critic
06_rl/artifacts/models/base_no_predictor.meta.json
```

## 9. Baseline Rollout 评估

```bash
screen -dmS eval_base bash -lc 'cd "/root/autodl-tmp/0626/0626 Predictor/06_rl" && MAIN_ID=base_no_predictor SIMULATOR_CHECKPOINT=artifacts/user_response/base_v1_sim/regret_user_response.pt ACTOR_CHECKPOINT=artifacts/models/base_no_predictor_actor ROLLOUT_EPISODES=1000 ./scripts/04_eval_policy.sh'
```

监控：

```bash
screen -r eval_base
tail -f "/root/autodl-tmp/0626/0626 Predictor/06_rl/artifacts/logs/base_no_predictor_eta010_test_1000.log"
```

主要指标：

```text
base avg cumulative reward
RAPI avg cumulative reward
interaction step depth
negative rate
delta_rapi_minus_base
```

## 10. Predictor 接入后消融

这一部分需要先改 `06_rl/scripts/10_train_hsac_simulator_rollout.py` 和
`06_rl/scripts/09_eval_simulator_rollout.py`：

```text
HPN actor 生成 top-k SID/item 候选
-> predictor 对每个候选 item 预测 reward / future_return / future_regret
-> final_score = policy_score + beta * predictor_score
-> 选择 final_score 最高或按 final_score 采样
-> simulator 给反馈
-> HSAC 更新
```

建议先实现最小可控版本：

```text
predictor_score = predicted_future_return - eta * future_regret_prob
beta in {0.1, 0.3, 1.0}
```

实现后计划命令：

```bash
screen -dmS policy_pred_b01 bash -lc 'cd "/root/autodl-tmp/0626/0626 Predictor/06_rl" && MAIN_ID=with_predictor_b01 USE_FUTURE_PREDICTOR=1 PREDICTOR_CHECKPOINT="../artifacts/predictor_base_1m/future_predictor.pt" PREDICTOR_BETA=0.1 SIMULATOR_CHECKPOINT=artifacts/user_response/base_v1_sim/regret_user_response.pt POLICY_SAVE_PREFIX=artifacts/models/with_predictor_b01 ./scripts/03_train_policy.sh'
```

评估：

```bash
screen -dmS eval_pred_b01 bash -lc 'cd "/root/autodl-tmp/0626/0626 Predictor/06_rl" && MAIN_ID=with_predictor_b01 USE_FUTURE_PREDICTOR=1 PREDICTOR_CHECKPOINT="../artifacts/predictor_base_1m/future_predictor.pt" PREDICTOR_BETA=0.1 SIMULATOR_CHECKPOINT=artifacts/user_response/base_v1_sim/regret_user_response.pt ACTOR_CHECKPOINT=artifacts/models/with_predictor_b01_actor ROLLOUT_EPISODES=1000 ./scripts/04_eval_policy.sh'
```

## 11. 最小主实验表

至少需要以下实验：

```text
Exp A: baseline no predictor
Exp B: with predictor beta=0.1
Exp C: with predictor beta=0.3
Exp D: with predictor beta=1.0
```

可选但建议保留：

```text
Exp E: no RAPI
Exp F: RAPI eta sweep
```

最终汇总字段：

```text
avg cumulative reward
avg interaction step
negative rate
explicit regret rate
RAPI delta reward
RAPI delta step
simulator metrics
predictor metrics
```
