# 01 当前状态

## 机器评估

当前机器：

```text
MacBook Air
Apple M5
10 核 CPU: 4 性能核 + 6 能效核
10 核 GPU
16 GB 内存
约 693 GiB 可用磁盘
```

Python 环境：

```text
Python 3.12.10
PyTorch 2.11.0
MPS built=True
MPS available=False
```

结论：

```text
可以跑完整代码链路。
不建议直接在本机跑全量多步大训练。
默认使用 smoke / small 参数，确认逻辑正确后再放大。
```

## 当前实现范围

已经实现：

```text
00_preprocess/run_preprocess.sh
  服务器/本机预处理入口
  顺序执行 codebook -> item SID -> future data -> embedding store

01_data/reward.py
  v2 reward
  五类原始响应
  played_ratio 连续值
  regret 派生标签

01_data/build_future_data.py
  从 multi_event.parquet 构造 future predictor 样本
  支持 future_horizon
  支持 user split 和 smoke 用 row split

01_data/build_embed_store.py
  从 embeddings.parquet 构造 numpy embedding store
  支持只抽取 needed item

01_data/future_dataset.py
  流式读取 parquet
  embedding 查找
  batch collate

02_model/predictor.py
  Transformer history encoder
  action-conditioned future predictor
  response / play / reward / regret / soft token heads

02_model/hpn.py
  自包含 HPN
  输入历史状态，输出 SID token 分布
  支持 beam decode SID path

02_model/hpn_candidates.py
  SID path 到候选 item 的索引
  支持完整路径匹配和前缀回退

02_model/soft_state.py
  soft_next_state 构造

02_model/value.py
  value head
  candidate scorer
  Bellman-style value loss

03_train/train_predictor.py
  predictor 训练入口

03_train/train_hpn.py
  HPN 训练入口
  使用 future_data 中的 target_sid

03_train/train_value.py
  soft_next_state + value head 训练入口

04_eval/eval_predictor.py
  predictor heads 评估
  response 使用多标签指标

04_eval/eval_rerank.py
  future-aware rerank 评估

04_eval/eval_hpn_future_rerank.py
  HPN top-k SID path
  SID path 映射候选 item
  predictor + soft_next_state + value head 重排

run_server_pipeline.sh
  服务器训练与评估流水线
```

## Smoke 结果

样本构造：

```text
users_seen: 1
rows_written: 240
train: 192
val: 24
test: 24
needed_items: 173
reward_version: v2
```

embedding store：

```text
needed_items: 173
matched_embeddings: 166
missing_embeddings: 7
embedding_dim: 128
```

predictor smoke：

```text
epochs: 1
train_rows: 160
val_rows: 24
device: cpu
val_response_acc: 0.7917
val_regret_acc: 0.9167
val_reward_mae: 0.3212
val_play_mae: 0.2498
```

value / rerank smoke：

```text
candidate_k: 4
logged_action_top1_rate: 0.25
logged_action_mean_rank: 2.5
```

这些数字只证明链路可运行，不代表模型效果。

## 2026-06-26 复查修正

发现并修正的问题：

```text
1. value 训练原先默认用批内候选构造 Bellman target。
   已支持 --candidate_source hpn，服务器流水线已切到 HPN 候选。

2. Bellman-style target 原先只有 M=1 确定性版本。
   已加入 --sample_m，多场景时执行 average_m max_a。

3. response_probs 原先是 softmax 单分类。
   已改为 sigmoid 多标签概率，支持 listen 与 like 等响应同时出现的建模口径。

4. SID path 精确匹配过严，未训练 HPN 容易生成空候选。
   已加入前缀回退，完整路径不存在时回退到更短 SID 前缀。

5. 有 SID 映射时，如果样本缺失 target_sid，原先可能污染 HPN 训练。
   已改成默认丢弃缺失 SID 的样本，除非显式传 --keep_missing_sid。

6. eval_predictor 原先只给 response top-1 acc。
   已补充多标签 exact match / micro precision / recall / F1。
```

新增 smoke：

```text
HPN smoke:
  train_hpn.py 跑通

HPN candidate value smoke:
  train_value.py --candidate_source hpn --sample_m 2 跑通
  empty_candidate_rows = 0

HPN future rerank smoke:
  eval_hpn_future_rerank.py 跑通
  empty_candidate_rows = 0
```

## GitHub 目标

完成到服务器可复现状态后，推送到：

```text
git@github.com:Toryck-Official/Yambda.git
```

本地确认：

```text
/Users/Toryck/Coding/Yambda/Yambda 是 git 仓库
origin 已经指向 git@github.com:Toryck-Official/Yambda.git
```

当前不立即 push。原因：

```text
0626 Predictor 还在构建中
HPN top-k 和预处理闭环尚未完成
artifacts 已经被 .gitignore 排除
```

## 已知问题

1. 当前 Python 环境不能使用 MPS，训练按 CPU 估算。
2. subset embedding store 需要扫描完整 embeddings.parquet，速度慢。
3. 小样本里有少数 item 找不到 embedding，目前置零。
4. rerank smoke 使用批内候选，不是 HPN top-k 候选。
5. HPN top-k rerank 代码已经实现并通过 smoke，但还没有在完整 SID 映射和 HPN checkpoint 上跑服务器级验证。
6. 预处理脚本复用外层现有码本/SID 脚本，输出落在 `0626 Predictor/artifacts/preprocess`。
7. current response target 仍来自单个原始事件；多事件聚合成一次 item-level response episode 还没做成默认数据构造。
