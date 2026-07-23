# 02 服务器交接说明

本文档给服务器端继续接手 `0626 Predictor` 的同事使用。它不重复展开完整方法论，重复内容直接看：

```text
/Users/Toryck/Coding/Yambda/Yambda/0626 Predictor/6.26 大纲.md
```

本文只保留接手时最需要知道的：当前实现状态、文件位置、运行顺序、检查指标、已知风险。

> 2026-07-17 更新：predictor 数据入口已经改为 `session_run` 聚合时间步；共享状态编码器已经加入当前会话短期状态和历史会话长期状态；主要目标已经改为五维非互斥反馈概率。旧段落若仍写“原始事件级 builder”，均已失效。

## 1. 当前目标

本工作区要在原 HSRL / HPN 推荐框架前后加入一个 `future-aware predictor`，让推荐不只看当前候选 item 的原始打分，还额外考虑：

```text
当前真实历史 s_t
+ 候选动作 a，也就是候选 item / SID path
-> 预测该 item 可能带来的未来响应、播放比例、即时奖励、后悔风险、临时未来状态
-> 再用 value head 估计长期价值
-> 对 HPN 候选集重排
```

完整概念边界、为什么不是简单模拟器、为什么不是直接使用真实未来行为，请看 `6.26 大纲.md`。

## 2. 最重要的边界

必须保持以下边界，否则方法会变成未来信息泄漏：

```text
离线训练：
  可以用日志里的后续真实行为作为监督信号。

在线推荐 / 离线评估模拟在线：
  不能使用真实未来行为。
  只能使用 predictor 根据当前历史和候选 item 预测出的 future representation。
```

动作定义：

```text
动作不是 like / dislike / unlike。
动作是推荐哪个 item。
在 HPN 中动作进一步表示为 SID path。
```

Yambda 原始响应只有：

```text
listen
like
dislike
unlike
undislike
```

`recommend` 不是 Yambda 原始字段，不能作为新 predictor 的真实标签来源。

`listen` 是否发生是一个二值响应标签；发生播放后，`played_ratio_pct` 另外表示连续播放完成度。超过 100% 通常表示回拉或重复听，因此不能在状态数据中直接丢弃；reward 中是否裁剪是另一个问题。

## 3. 当前实现状态

核心工作区已经推送到：

```text
git@github.com:Toryck-Official/Yambda.git
```

核心实现提交：

```text
27bf4e7 Add future-aware predictor workspace
```

如果本文档后续也已提交，请以仓库最新 `main` 为准。

当前已实现：

```text
1. future data 构造
2. v2 reward 口径
3. embedding store 构造
4. future predictor
5. soft state update
6. HPN 自包含实现
7. HPN SID 候选召回
8. value head
9. Bellman-style 多采样训练目标
10. HPN 候选 rerank 评估脚本
11. smoke 脚本和 server pipeline 脚本
```

当前没有完成或不能过度宣称：

```text
1. 没有在服务器上跑过全量训练。
2. 还没有验证全量指标是否提升。
3. 当前 future data 已复用 `session_run` 的同会话连续同物品聚合，但仍需在全量数据上审计窗口长度和显式反馈稀疏度。
4. 当前 value 训练是 Bellman-Jensen-inspired，不应直接宣称已经完整复现 Bellman-Jensen。
5. HPN 候选生成有 prefix fallback / root fallback，是工程近似。
```

这些风险的详细解释也见 `6.26 大纲.md`。

## 4. 目录结构

主要文件：

```text
0626 Predictor/
  00_preprocess/
    run_preprocess.sh
    README.md

  01_data/
    reward.py
    build_future_data.py
    build_embed_store.py
    future_dataset.py

  02_model/
    hpn.py
    hpn_candidates.py
    predictor.py
    soft_state.py
    value.py

  03_train/
    train_hpn.py
    train_predictor.py
    train_value.py

  04_eval/
    eval_predictor.py
    eval_rerank.py
    eval_hpn_future_rerank.py

  05_notes/
    01_status.md
    02_server_handoff.md

  configs/
    default.json

  run_smoke.sh
  run_server_pipeline.sh
  6.26 大纲.md
```

建议先读顺序：

```text
1. 05_notes/02_server_handoff.md
2. 05_notes/01_status.md
3. 6.26 大纲.md
4. run_server_pipeline.sh
5. 01_data/build_future_data.py
6. 02_model/predictor.py
7. 03_train/train_value.py
8. 04_eval/eval_hpn_future_rerank.py
```

## 5. 服务器运行顺序

先进入工作区：

```bash
cd "/Users/Toryck/Coding/Yambda/Yambda/0626 Predictor"
```

准备数据：

```bash
bash 00_preprocess/run_preprocess.sh
```

跑完整服务器链路：

```bash
bash run_server_pipeline.sh
```

这条链路的意图是：

```text
1. 训练 HPN
2. 训练 predictor
3. 用 HPN 候选训练 value head
4. 用 HPN top-k 候选做 future-aware rerank 评估
```

如果服务器路径不同，需要先改脚本里的数据路径和输出路径。

## 6. 重点看哪些结果

predictor 先看：

```text
reward_mae
play_ratio_mae
response_exact
response_micro_f1
regret_acc
```

value 训练先看：

```text
loss
empty_candidate_rows
target_value_mean
pred_value_mean
```

HPN future rerank 评估先看：

```text
hpn_recall
future_rerank_recall
empty_candidate_rows
```

最关键判断：

```text
future_rerank_recall 是否高于 hpn_recall。
empty_candidate_rows 是否接近 0。
predictor 的 reward/play/response 是否不是随机水平。
```

如果 `future_rerank_recall` 没提升，不要先改大结构，先检查 predictor 是否学到了有用信号，以及 SID 候选召回是否足够覆盖真实目标。

## 7. 本地已经验证过什么

本地 Mac 只做过 smoke 级别验证：

```text
1. Python 语法编译通过。
2. fake SID mapping 下 future data 能构造。
3. HPN 能跑 1 个小 epoch。
4. predictor 能跑小样本训练。
5. value 能使用 HPN 候选和多采样目标训练。
6. HPN future rerank 脚本能跑通，empty_candidate_rows=0。
```

本地 smoke 的指标没有性能意义，因为使用的是小样本和 fake SID，只能说明链路没有明显运行错误。

## 8. 需要特别小心的地方

第一，`response_probs` 使用 sigmoid 多标签头，因为一个物品响应窗口可以同时出现 `listen` 和 `like`。主要损失权重为 1.0，播放、奖励、后悔三个辅助目标默认各为 0.1。

第二，`soft_next_state` 是候选打分和 value target 里的临时预测状态，不是线上真实状态。线上真实状态只能由用户实际反馈更新。

第三，当前 `CandidateScorer` 里的总分包含：

```text
base_logit
predicted_reward
gamma * V(soft_next_state)
regret penalty
uncertainty penalty
```

这些项的量纲是否平衡，需要在服务器训练后通过验证集调参，不应直接假定默认权重最优。

第四，当前实现可以称为：

```text
model-based offline RL inspired reranking
```

更保守中文表述是：

```text
基于预测模型的离线强化学习式候选重排。
```

不要直接写成“完整贝叶斯最优价值函数已实现”。目前是受 Bellman-Jensen 启发的工程版本。

## 9. 建议接手者下一步

建议按这个顺序做：

```text
1. 在服务器上跑通 run_server_pipeline.sh。
2. 记录 predictor / value / rerank 的完整日志。
3. 如果内存或速度出问题，先降低 max_events、batch_size、top_k、sample_m。
4. 检查 HPN 候选召回。如果召回很低，rerank 不可能救回来。
5. 检查 predictor 指标。如果预测接近随机，value head 也没有可靠输入。
6. 审计不同 `history_len` 下，先前会话长期状态的有效覆盖比例。
7. 最后再调 alpha、gamma、regret_weight、uncertainty_weight 等 rerank 权重。
```

更细的方法设计和争议点，请不要在本文档里找，直接看：

```text
/Users/Toryck/Coding/Yambda/Yambda/0626 Predictor/6.26 大纲.md
```
