# Factorized SNMPP Failure Diagnosis Report

## 1. Hypothesis

检验旧 Tiny Overfit 失败是否主要来自 group-time 与 feedback mark 共用四个 coupled intensity 的参数化冲突。唯一修正是共享原 SNMPP temporal context 后，以独立轻量 head 输出 Lambda 与 q，并定义 lambda_k=Lambda*q_k。

## 2. Frozen Contract

- 完全复用 6,413 target groups / 38,363 target events、seed=2026、D_SID、grouped shared pre-history、冻结码本、event representation、psi/phi/delay、raw event-level full-history sum、Q=64、hour 与 696-hour horizon。
- 未改 class weight、采样、optimizer、loss weight、history、burst、SID 或 sequence encoder。保留旧 gradient clip=10；它不是本轮修复。
- 新输出 heads 共 25 参数，替换旧 4 个 baseline logits 后净增 21 参数。

## 3. Old Gradient Conflict

- 初始化 shared gradient cosine = -0.0869；5 个参数组中 4 个为负。
- psi cosine = -0.0869；phi cosine = -0.9790。
- 失败 checkpoint 的 shared cosine 转为正值，但这是全正 influence 与多数类退化后的方向一致，不能反证初始化冲突。

## 4. Shared-group Feedback Oracle

- 全部组 oracle floor = 0.245290 CE/event。
- singleton = 0.000000；multi-event = 0.267476。
- Tiny 类别比例：like 29.33%，dislike 16.66%，unlike 41.05%，undislike 12.96%。
- 旧模型 99.92% 预测为 unlike；这就是 majority-class collapse。

## 5. Unit Tests

7 个测试函数覆盖 10 项 contract，全部通过：Lambda 正且 finite、q 非负且和为 1、sum lambda=Lambda、singleton identity、permutation invariance、shared pre-history、delayed update、Q64、四 mark gradients、冻结 codebook。

## 6. Old vs New

| 指标 | OLD coupled final (lr=1e-5) | NEW factorized initial | NEW factorized final (lr=1e-05) |
|---|---:|---:|---:|
| total loss | 7.505963 | 6.128556 | 6.167570 |
| time loss/group | 6.129873 | 4.742262 | 4.762732 |
| feedback loss/event | 1.376090 | 1.386294 | 1.404838 |
| feedback excess | 1.130800 | 1.141004 | 1.159548 |
| Macro-F1 | 0.145945 | 0.113401 | 0.145516 |
| Lambda P99 | 16.846274 | 0.021490 | 3.057884 |
| Lambda max | 147.261642 | 0.021490 | 26.568062 |

- NEW recall: like=0.0000, dislike=0.0000, unlike=1.0000, undislike=0.0000。
- NEW argmax distribution: [0.0, 0.0, 1.0, 0.0]。
- 完全匹配旧正式配置 lr=1e-5 时，factorized Lambda max=26.568，而 OLD 为 147.262；数值敏感性 lr=1e-3 时仍达 191.229。
- 训练中最佳 transient time loss=4.610499、最佳 feedback loss=1.365539，说明解耦有部分数值收益；但最终均反弹且类别始终 collapse。
- NEW signed influence 仍为全正：negative mass max=0，未恢复 excitation/inhibition 两侧。

## 7. Burst Diagnostic

- previous-group >500（17 组）：Lambda mean=1.205，time loss mean=7.388，absolute influence mean=49.767，gradient norm median=122.377。
- <=500 reference（17 组）：对应为 0.034、3.978、5.137、13.535。
- 全部 finite，但 raw full-history burst amplification 仍显著存在。

## 8. Final Gate Answers

1. 旧模型存在明显初始化梯度冲突：是。
2. Tiny oracle floor：0.245290 CE/event。
3. unlike collapse 是多数类 collapse：是。
4. factorized Lambda/q 数学测试：全部通过。
5. 阻止 Lambda 爆炸：在完全匹配的 1e-5 对照中做到，但 1e-3 敏感性仍爆到百级，说明 factorization 改善稳定性但并非无条件稳定。
6. time loss 真正下降：前 3 轮短暂下降，最终反弹并高于初始化，因此未稳定下降。
7. feedback excess 明显下降：否，反而略升。
8. Macro-F1 / 四类 recall 摆脱 collapse：否。
9. psi 恢复正负 influence：否，仍全正。
10. >500 burst amplification：仍存在。
11. Minimal SNMPP Stage B：不批准。

## 9. Status

factorized_snmpp_tiny_passed = false
time_mark_coupling_failure_supported = false
history_aggregation_gate_needed = true
minimal_snmpp_stageB_approved = false

解释：检测到 coupling 对数值不稳定有贡献，但证据不支持它是 Tiny 失败的主要或唯一原因。

本轮到此 STOP；未启动 Stage B、Hierarchical SID、HPN/BOLA 或 full-scale training。
