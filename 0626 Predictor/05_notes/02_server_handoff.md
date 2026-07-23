# 02 服务器交接说明

本文档给接手 `/root/autodl-tmp/0626/0626 Predictor` 的同事使用。当前项目 focus 已经从“先把 HPN 跑通”调整为“先把预测器完整实现并验证清楚”。本文只记录当前 focus、关键修改、已跑结果、下一步建议。

## 0. 远端历史交接补充

远端 `main` 在本次同步前已有一版交接说明，核心提醒是：离线训练可以使用日志里的后续真实行为作为监督信号；在线推荐或离线模拟在线评估时，不能读取真实未来行为，只能使用 predictor 基于当前历史和候选 item 预测出的未来信息。远端记录的早期实现提交是 `27bf4e7 Add future-aware predictor workspace`，本次同步前远端最新交接提交是 `4a6da24 Add server handoff notes`。当前服务器实际工作路径以 `/root/autodl-tmp/0626/0626 Predictor` 为准。

## 1. 当前 focus

当前主线是 action-conditioned future predictor，也就是：

```text
输入：当前用户状态 s_t + 一个候选动作 a_t
输出：这个候选 item 可能带来的用户反馈、播放比例、即时 reward、未来累计 reward、未来后悔风险
```

这里的动作不是 `like/dislike/unlike`，动作是“推荐哪个 item”。在 HPN 体系里，这个 item 可以由 SID path 表示；但预测器本身不应该依赖 HPN 才能训练。

当前已经明确：

```text
1. predictor 和 simulator 都应该从真实日志转出来的 transition 数据训练。
2. predictor/simulator 的训练不需要 HPN。
3. HPN 只是后续策略网络或推荐模型的一种实现，可替换成 baseline。
4. 最终论文重点应看 RL recommendation 的 rollout reward、交互深度、负反馈率、后悔信号，而不是 exact next-item recall。
5. exact item recall 可以作为诊断指标，但不能作为当前方法成败的唯一核心指标。
```

## 2. 预测器的输入输出

预测器读取：

```text
01_data/processed/predictor_seq_data
```

一个样本大致是：

```text
history_item_ids / history_feedbacks / history_event_type_ids
  -> 当前用户历史，也就是 s_t

action_features / target_dense_item_id / target_sid
  -> 当前要评估的候选 item，也就是 a_t

response_targets
  -> 多标签反馈：listen, like, dislike, unlike, undislike

play_ratio / reward / regret_type_id
  -> 当前 step 的播放比例、公式 reward、当前后悔类型

future_return / future_regret_any
  -> 从当前 step 往后看的未来累计 reward 和未来是否有显式后悔
```

预测器输出：

```text
response_probs: 五类反馈概率，sigmoid 多标签，不是单选 softmax
predicted_play_ratio: 播放比例预测
predicted_reward: 即时 reward 预测
regret_probs: 当前后悔类型概率
predicted_future_return: 未来累计 reward 预测
future_regret_prob: 未来显式后悔概率
candidate_logit: 当前候选 item 是否像真实下一步 item 的候选区分分数
```

注意：`listen` 极多，所以单看 `response_acc` 会虚高。必须看分类型 precision/recall/F1，尤其是 like、dislike、unlike、undislike。

## 3. 当前已经确认的问题

已经连续验证了两件事：

```text
1. 预测器能学到“当前状态整体会发生什么”：
   response_micro_f1 可以到 0.97 左右
   reward_mae 可以到 0.28 左右
   future_return_mae 可以到 0.76 左右
   future_regret_f1 可以到 0.41-0.48 左右

2. 预测器仍没有学到“同一个状态下哪个 candidate item 更值得推荐”：
   candidate_k=8 时随机 top1=0.125
   candidate_logit / predicted_reward / predicted_future_return 的 top1 仍接近或低于随机
   pairwise_true_gt_negative 仍在 0.49 左右，没有明显高于 0.5
```

这说明当前瓶颈不是普通反馈预测，而是 action-conditioned 区分能力。换句话说，模型知道“这个用户大概率会听歌”，但还不能可靠判断“给这个用户推荐 A 比推荐 B 更好”。

## 4. 已做的关键修改

修改文件：

```text
02_model/predictor.py
03_train/train_predictor.py
04_eval/eval_predictor_oracle_contrast.py
run_predictor_only.sh
eval_predictor_only.sh
```

备份目录：

```text
Temporary/predictor_hardneg_backup_20260703_1116
Temporary/predictor_actionfix_backup_20260703_1505
```

新增候选负样本模式：

```text
inbatch
  普通 batch 内其他 item 作为负样本。

history
  从同一个用户历史里取最近交互过、但不是当前 target 的 item 作为负样本。

history_inbatch
  奇数负样本槽位用用户历史 item，偶数槽位回退到 batch 内 item。

semantic_inbatch
  从 batch 内找 SID 前缀相同的 item 作为语义相近负样本。

history_semantic_inbatch
  用户历史负样本、语义相近负样本、batch 回退负样本混合。
```

当前 wrapper 默认：

```text
CANDIDATE_NEGATIVE_MODE=history_semantic_inbatch
SEMANTIC_PREFIX_LEVEL=1
CANDIDATE_CE_WEIGHT=1.0
CANDIDATE_CE_K=8
```

训练和评估会输出负样本来源占比：

```text
candidate_history_negative_share
candidate_semantic_negative_share
candidate_fallback_negative_share
```

`02_model/predictor.py` 里还加了 action-fusion 修复：对 state 和 action 分别归一化，并额外加入 action residual 分支，避免 action 信息在 joint MLP 里被冲淡。这个修复确实让模型输出对不同候选 item 有更明显数值变化，但没有解决候选排序指标。

## 5. 已完成验证和结果

静态检查已通过：

```bash
python3 -m py_compile 03_train/train_predictor.py 04_eval/eval_predictor_oracle_contrast.py
bash -n run_predictor_only.sh eval_predictor_only.sh
```

### 5.1 hard-negative 1M

训练产物：

```text
artifacts/predictor_hardneg_1m/future_predictor.pt
artifacts/predictor_hardneg_1m/metrics.json
artifacts/logs/predictor_hardneg_1m.log
```

50k validation 主要结果：

```text
response_micro_f1 = 0.9733
reward_mae = 0.2881
future_return_mae = 0.7706
future_regret_f1 = 0.4816

candidate_logit.top1 = 0.1318
candidate_logit.pairwise_true_gt_negative = 0.4971
predicted_reward.top1 = 0.1303
predicted_future_return.top1 = 0.1324
```

解释：点预测头能用，但候选排序基本随机。`candidate_logit.top1` 略高于 0.125 不能算有效，因为 pairwise 仍低于 0.5。

### 5.2 action-fusion 1M

训练命令：

```bash
cd "/root/autodl-tmp/0626/0626 Predictor" && screen -dmS pred_actionfix_1m bash -lc 'mkdir -p artifacts/logs && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OUT_DIR=artifacts/predictor_actionfix_1m MAX_TRAIN_ROWS=1000000 MAX_VAL_ROWS=50000 EPOCHS=1 BATCH_SIZE=256 DEVICE=cuda ./run_predictor_only.sh > artifacts/logs/predictor_actionfix_1m.log 2>&1'
```

训练产物：

```text
artifacts/predictor_actionfix_1m/future_predictor.pt
artifacts/predictor_actionfix_1m/future_predictor_epoch1.pt
artifacts/predictor_actionfix_1m/metrics.json
artifacts/logs/predictor_actionfix_1m.log
```

训练内 validation：

```text
response_micro_f1 = 0.9731
reward_loss = 0.1307
future_return_mae = 0.7606
future_regret_f1 = 0.4128
candidate_ce_top1 = 0.1178
candidate_ce_mrr = 0.3330
candidate_history_negative_share = 0.4286
candidate_semantic_negative_share = 0.2635
candidate_fallback_negative_share = 0.3080
```

独立 eval：

```text
response_micro_f1 = 0.9731
reward_mae = 0.2816
future_return_mae = 0.7606
future_regret_f1 = 0.4128

candidate_logit.top1 = 0.1180
candidate_logit.pairwise_true_gt_negative = 0.4917
predicted_reward.top1 = 0.1102
predicted_future_return.top1 = 0.1122
expected_formula_reward.top1 = 0.1110
```

解释：action-fusion 修复让模型对 action 更敏感，但没有带来候选排序收益。候选 top1 低于随机基线 0.125，说明继续单纯调这个 candidate CE 目标意义不大。


### 5.3 action-value label 诊断和 predictor 消融

新增脚本：

```text
04_eval/diagnose_action_value_labels.py
04_eval/eval_predictor_value_ablation.py
```

5 万行 action-value label 诊断产物：

```text
artifacts/logs/action_value_label_diag_50k.json
artifacts/logs/action_value_label_diag_50k.log
```

关键结论：

```text
candidate_negative_mode = history_semantic_inbatch
candidate_k = 8

按即时 reward 做 oracle 排序：
  top1 = 0.1242
  随机基线 = 0.125
  target_tie_rate = 0.6643

按 future_return 做 oracle 排序：
  top1 = 0.1999
  随机基线 = 0.125
  target_tie_rate = 0.2250
```

解释：即时 reward 大量并列，几乎不能支撑 candidate CE 这种“真实 next item 一定排第一”的训练目标；future_return 里有更明显的排序信号，所以主线消融应该优先围绕 future_return / Bellman value，而不是 candidate_logit 或即时 reward。

5 万行 predictor value ablation 产物：

```text
artifacts/logs/predictor_value_ablation_50k.json
artifacts/logs/predictor_value_ablation_50k.log
```

候选设置：

```text
candidate_negative_mode = semantic_inbatch
semantic_prefix_level = 1
candidate_k = 8
source_share: semantic 0.8072, fallback 0.0678, target 0.1250
```

以 `future_return` 作为离线 label 时：

```text
random selected_label_mean = 2.630543
logged_target selected_label_mean = 2.628715
predicted_reward selected_label_mean = 2.646959  (+0.016416 vs random, +0.018244 vs target)
predicted_future_return selected_label_mean = 2.646865  (+0.016322 vs random, +0.018150 vs target)
expected_formula_reward selected_label_mean = 2.648374  (+0.017831 vs random, +0.019659 vs target)
mean_history_cosine selected_label_mean = 2.657735  (+0.027192 vs random, +0.029020 vs target)
```

以即时 `reward` 作为离线 label 时：

```text
random selected_label_mean = 0.649212
logged_target selected_label_mean = 0.644677
predicted_reward selected_label_mean = 0.660054  (+0.010842 vs random, +0.015377 vs target)
expected_formula_reward selected_label_mean = 0.656473  (+0.007261 vs random, +0.011796 vs target)
mean_history_cosine selected_label_mean = 0.661657  (+0.012445 vs random, +0.016980 vs target)
```

解释：预测器不是完全没信号。它在这个离线候选消融里能带来小幅正提升，尤其是 `predicted_reward`、`predicted_future_return` 和 `expected_formula_reward` 对 future_return 都高于 random 和 logged target。但简单的 `mean_history_cosine` 更强，所以最终主线应该把 predictor 作为 future-aware rerank 的一项增益特征，而不是单独替代策略分数。

工程结论：

```text
1. 全链路技术上能跑通：数据 -> predictor -> value/soft state -> rerank/eval 都已有脚本。
2. predictor 有弱正向消融证据，可以进入主线实验。
3. 不能宣称 predictor 单独已经是强 reranker；它应该和策略分数、相似度或 HPN 分数组合。
4. 下一步主实验应做 base scorer vs base + predictor_future_score 的 rollout 或 rerank 消融。
```

## 6. 下一步建议

不要继续盲跑 HPN，也不要继续只调 `candidate_ce_weight`。当前应该把问题拆成两层：

```text
第一层：预测器作为 outcome model
  判断给定 s_t 和已发生的 a_t 时，能不能预测反馈、reward、future_return、future_regret。
  这一层目前基本可用，但显式小类还需要继续优化。

第二层：预测器作为 action scorer
  判断同一个 s_t 下，候选 a_t 哪个更好。
  这一层目前没有成立，不能直接用 candidate_logit / predicted_reward 去 rerank。
```

下一步优先级：

```text
1. 暂停 candidate_logit 作为核心目标，不要再用 exact next-item 对比当作唯一训练方向。
2. 保留 predictor 的 outcome heads：response、play、reward、future_return、future_regret。
3. 重新设计 action-value 训练信号：
   - 如果要做候选排序，需要真实曝光负样本、同状态候选集、或从同用户相近时间窗口构造更可信的偏好对。
   - 仅用“日志下一首歌是正样本，其他 item 是负样本”在音乐场景里噪声很大。
4. 接入策略网络时，先把 predictor 当作 simulator/transition prediction 的一部分，而不是直接当 reranker。
5. 如果仍想做 rerank，需要先定义清楚候选 item 的反事实标签来源，否则 top1 会继续随机。
```

推荐下一次小改动不是重训，而是先做数据诊断脚本，回答：

```text
同一个 user 的历史 item 和真实 next item，在 reward / future_return / regret 标签上是否真的有可学习差异？
同一 SID 前缀下的候选 item 是否存在稳定偏好差异？
是否能从数据里构造“同用户同状态附近 A 明显好于 B”的 pairwise 标签？
```

## 7. 当前要避免的误区

不要把 HPN exact recall 低直接等同于 predictor 失败。当前阶段 predictor 是独立模块，先验证它是否能在同一个用户状态下区分真实 action 和 hard negative action。

不要只看 `response_acc`。listen 太多，acc 高可能只是学会“多数情况下会 listen”。

不要把 `candidate_logit` 直接称为最终 future score。它只是“候选 item 像不像真实下一步 item”的监督信号。最终用于策略网络时，应该结合：

```text
predicted_reward
predicted_future_return
future_regret_prob
candidate_logit
policy / HPN 原始分数
```

但在当前评估结果下，`candidate_logit` 还不能直接用于 rerank。

## 8. 关于文件改动范围

本轮实际操作和修改集中在：

```text
/root/autodl-tmp/0626/0626 Predictor
```

我没有主动修改 `/root/autodl-tmp/0408Yambda`、`/root/autodl-tmp/0330Yambda` 等其他项目目录。

需要注意：当前 git status 里显示 `../01_build_codebook.py` 和 `../paper/` 也有状态变化，但它们在 `0626 Predictor` 的上级 `/root/autodl-tmp/0626` 下，不是我这轮 predictor 修改的目标文件。接手时请不要默认这些变化属于本轮 predictor 修改。

当前工作区本来就有大量未提交改动和未跟踪文件，接手前建议先用：

```bash
cd "/root/autodl-tmp/0626/0626 Predictor" && git status --short
```

确认哪些是历史遗留、哪些是当前 predictor 修改。

## 9. 最短接手路线

```text
1. 先读本文档。
2. 看 02_model/predictor.py 的 _predict_flat，确认 state-action fusion。
3. 看 03_train/train_predictor.py 的 make_action_conditioned_candidates。
4. 看 04_eval/eval_predictor_oracle_contrast.py 的同名候选构造。
5. 看 artifacts/predictor_hardneg_1m 和 artifacts/predictor_actionfix_1m 的结果。
6. 不要继续直接放大 candidate CE 训练；先诊断反事实 action 标签是否可靠。
7. predictor 可以先作为 outcome model 保留，后续接策略网络时用它预测 reward / future_return / future_regret。
8. 只有 action scorer 的标签定义清楚后，再考虑 HPN rerank 或策略候选重排。
```

## 10. 2026-07-03 独立项目化更新

当前已经把 0408 Regret 中正式主实验需要的最小 RL 代码闭包复制到本项目：

```text
06_rl/
  configs/main.env
  regret_core/
  scripts/01_split_data.sh
  scripts/02_train_simulator.sh
  scripts/03_train_policy.sh
  scripts/04_eval_policy.sh
  scripts/05_sweep_eta.sh
  scripts/02_split_transitions.py
  scripts/08_train_yambda_simulator.py
  scripts/09_eval_simulator_rollout.py
  scripts/10_train_hsac_simulator_rollout.py
```

`06_rl` 的默认数据入口已经改成本项目内部路径：

```text
01_data/processed/raw_rqkmeans
01_data/processed/regret_current_data
```

RL 训练产物默认写入：

```text
06_rl/artifacts/
```

这意味着后续主实验可以只在 `/root/autodl-tmp/0626/0626 Predictor` 内执行，不需要再跳回 `/root/autodl-tmp/0408Yambda/Regret` 找训练脚本。0408 只作为历史来源和对照实现。

本次没有把 0408 的诊断脚本、事件分布分析脚本、旧 SID 离线训练脚本全部复制过来。原因是这些不是 base 主实验闭环必要部分，继续混进主线目录会重新制造混乱。需要时再按具体问题补到 `04_eval/` 或 `06_rl/scripts/`。
