# SASRec / HAC / HSRL 接入方案

## 当前判断

SASRec、HAC、HSRL 这几类 baseline 不应该和当前 predictor/RAPI 主线混在一起。

原因：

- predictor 主线关注“未来反馈预测器如何辅助 HPN/critic”。
- RAPI 主线关注“负反馈记忆如何干预策略”。
- SASRec、HAC、HSRL 更适合作为“离线序列推荐/策略 baseline”。

因此第一阶段建议：

- 不接 simulator；
- 不接 predictor；
- 不接 RAPI；
- 使用同一份 transition split；
- 使用同一套 SID/codebook；
- 用离线 candidate ranking 指标比较。

## 从 HSRL 代码看到的可复用点

路径：`/root/autodl-tmp/0626/HSRL`

关键脚本：

| 文件 | 作用 |
|---|---|
| `04_train_hpn_warmstart.py` | 训练 HPN/SID warm-start |
| `06_train_yambda_sid.py` | HSRL/SID 主训练入口 |
| `07_eval_candidate_ranking.py` | 离线候选排序评估 |
| `adapter/model/facade/SIDFacade_credit.py` | 把 SID token 分布映射到候选 item |
| `adapter/model/agents/DDPG.py` | HSRL 的 actor-critic 更新 |

`06_train_yambda_sid.py` 有两种模式：

| 模式 | 是否依赖 UserResponse/simulator | 适合当前主实验吗 |
|---|---|---|
| `online_env` | 依赖 | 暂时不适合 |
| `offline_transition` | 不依赖 | 适合 |

所以当前要接的是 `offline_transition` 路线。

## 三个 baseline 怎么定义

### 1. SASRec baseline

目标：只验证序列编码能力。

输入：

- history item embedding
- history feedback/event 可先关闭，做纯序列 baseline

输出：

- target item 的 SID token logits

训练：

- next-item SID cross entropy
- 可加 candidate CE

评估：

- HR@K
- NDCG@K
- MRR
- target rank

是否需要 simulator/predictor/RAPI：

- 不需要。

实现方式：

- 可以基于当前 `02_model/hpn.py` 拆一个 `SASRecSIDPolicy`。
- 和 HPN 的区别是：SASRec 只用最后 state 一次性预测各层 SID；不做 HPN 的逐层 residual credit。

### 2. HAC baseline

这里要区分两个含义。

如果按 HSRL 原始在线 HAC 路线：

- 需要 environment；
- 需要 UserResponse；
- 本质还是 simulator 训练。

这条线会和当前 predictor/simulator 主线混在一起，不适合作为第一阶段 baseline。

如果按当前可控实验路线：

- 用 transition parquet 做 offline HAC-style actor-critic；
- reward 直接来自 logged transition 的 `paper_base_reward` 或 `paper_effective_reward`；
- 不使用 UserResponse。

这条适合作为 baseline。

是否需要 simulator/predictor/RAPI：

- 在线 HAC 需要 simulator；
- 离线 HAC 不需要。

当前建议：只做离线 HAC。

### 3. HSRL baseline

目标：复现 HSRL 的 SID actor + token critic 结构。

输入：

- transition parquet
- dense item features
- dense item -> SID

输出：

- HSRL actor checkpoint
- HSRL critic checkpoint

训练：

- 使用 `06_train_yambda_sid.py --train_mode offline_transition`
- 正 reward transition 做 imitation
- 负 reward transition 做 avoid
- critic 拟合 logged reward

是否需要 simulator/predictor/RAPI：

- 不需要。

## 当前接入坑

### HPN checkpoint 兼容性

当前 Predictor 的 HPN checkpoint：

`artifacts/hpn/hpn.pt`

格式是：

```python
{
  "model_state": ...,
  "config": ...,
  "item_dim": ...
}
```

但 HSRL 的 `load_hpn_checkpoint` 只认：

```python
ckpt.get("model_state_dict", ckpt)
```

所以如果直接把 Predictor 的 `hpn.pt` 传给 HSRL，可能不会真正加载权重。

需要修复为：

```python
state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
```

### 架构命名不完全一致

Predictor 的 `FutureHPNPolicy` 和 HSRL 的 `SIDPolicy_credit` 结构相似，但字段名不完全一致：

- Predictor: `encoder`, `input_norm`, `event_emb`
- HSRL: `transformer`, `emb_norm`

因此 checkpoint 不能假设完全互通。

更稳妥的做法：

- SASRec/HPN baseline 用 Predictor 自己的 `FutureHPNPolicy`；
- HSRL baseline 用 HSRL 自己的 `SIDPolicy_credit`；
- 二者共享数据和评估，不强行共享 checkpoint。

## 推荐实验组织

统一数据：

- data: `01_data/processed/predictor_seq_data`
- transition: `01_data/processed/regret_current_data`
- item features: `01_data/processed/raw_rqkmeans/dense_item_features.npy`
- SID: `01_data/processed/raw_rqkmeans/dense_item2sid.npy`

统一评估：

- candidate ranking
- HR@1/5/10/20
- NDCG@K
- MRR
- mean rank

不要把这些 baseline 放到 simulator rollout reward 表里。

## 第一阶段实验表

| 方法 | 数据 | simulator | predictor | RAPI | 评估 |
|---|---|---|---|---|---|
| SASRec-SID | predictor_seq_data | 否 | 否 | 否 | candidate ranking |
| HPN-SID | predictor_seq_data | 否 | 否 | 否 | candidate ranking |
| HSRL offline | regret_current_data | 否 | 否 | 否 | candidate ranking + reward-gated offline metrics |
| HAC online | 暂缓 | 是 | 否 | 否 | 暂不进入主表 |

## 下一步代码工作

1. 新增或复用一个统一 baseline 训练入口：
   - `SASRec-SID`
   - `HPN-SID`
   - `HSRL-offline`

2. 修复 HSRL checkpoint 加载兼容：
   - 支持 `model_state`

3. 统一 candidate ranking eval：
   - 输入 actor checkpoint；
   - 输入模型类型；
   - 输出同一套 HR/NDCG/MRR。

4. 暂时不跑 online HAC：
   - 避免重新引入 simulator 问题；
   - 避免和 predictor/RAPI 主线混淆。
