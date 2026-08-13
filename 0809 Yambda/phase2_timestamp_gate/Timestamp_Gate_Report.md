# SNMPP Phase 2：Gate 2 与 Timestamp Likelihood Gate 报告

状态：本轮已完成并 STOP；没有训练 Minimal SNMPP，没有加入 Hierarchical SID head。

## 1. Hypothesis

验证正式 SID 能否无损物化到 D_SID，以及方案 B 能否在不虚构同刻组内顺序的前提下，正确实现 group-arrival time likelihood 与给定观测 cardinality 的 feedback multiset 条件伪似然。

## 2. Data Contract

- 数据：exact-deduplicated D_all 的 real-audio explicit-feedback subset，记为 D_SID；
- 事件：like / dislike / unlike / undislike，不使用 listen；
- split：沿用 global cutoff `22,172,245 / 24,154,245`；
- 时间协议：同一 `uid+timestamp` 构成 group，组内共享严格前序历史，整体评分后再进入历史；
- Timestamp Gate：train-only 固定小样本，765 个目标组；无训练、无调参。

## 3. Data Counts

| 项目 | 数量 |
|---|---:|
| events | 121,819,651 |
| users | 854,649 |
| items | 2,367,341 |
| timestamp groups | 77,565,283 |
| like | 85,325,332 |
| dislike | 11,061,046 |
| unlike | 23,667,107 |
| undislike | 1,766,166 |

## 4. Gate 2 Results

正式冻结 full-fit audio-only RQKMeans：4 层、每层 256 code、normalized 128-d audio embedding、seed 2026。Item catalog 保留 embedding；event table 只保存 item_id 与四层 SID，不复制 embedding。

12 项守恒检查全部通过：catalog 行数、SID 一对一映射和范围、event/feedback/user/item 数量、cutoff 与 split、timestamp group membership/size/count、无组内顺序、codebook freeze、全量 item→SID hash 均正确。正式 manifest：`phase2_gate2_sid/gate2_manifest.json`。

## 5. Train-only Timestamp Scale

| 项目 | 结果 |
|---|---:|
| train timestamp groups | 61,535,193 |
| positive adjacent group gaps | 60,759,338 |
| group-level zero gaps | 0 |
| min / mode / GCD | 5 / 5 / 5 秒 |
| P50 | 2,845 秒 |
| P90 | 389,800 秒 |
| P95 | 751,985 秒 |
| P99 | 2,494,400 秒 |
| P99.9 | 8,536,335 秒 |

冻结决定：raw unit 为秒、5 秒量化网格；model unit 为小时；prediction horizon 为 train P99 向上取整到整天，即 2,505,600 秒（29 天），覆盖 99.0103% train positive group gaps。没有查看 validation/test 来决定尺度或 horizon。

## 6. Timestamp Likelihood Gate

硬性测试全部通过：

| 测试 | 结果 |
|---|---|
| permutation invariance | PASS，最大差 0 |
| shared pre-history | PASS |
| delayed group update | PASS |
| integral uniqueness | PASS |
| no fake zero-delta transition | PASS |
| singleton equivalence | PASS，最大误差 0 |
| numerical stability | PASS |
| four-feedback gradient visibility | PASS |
| controlled target-cardinality stability | PASS |

控制实验固定相同 history、time 和 mark composition，只复制 target multiset：time loss 与 intensity 完全不变；理论 mark-sum 随观测 M 线性相加，这是条件伪似然定义，不是时间项重复计数；mark loss/event 保持不变。

## 7. Historical Burst Stress

这里使用 raw SNMPP sum，没有 mean/sqrt(M) normalization，也没有 clipping。

| 上一历史组大小 | abs influence 中位数 | intensity 中位数 | gradient norm 中位数 |
|---|---:|---:|---:|
| 1 | 0.0969 | 2.2457 | 1.0375 |
| 2–5 | 0.1794 | 2.2363 | 2.7076 |
| 6–20 | 0.5935 | 2.2408 | 6.2349 |
| 21–100 | 2.3177 | 2.6588 | 20.8680 |
| 101–500 | 11.2350 | 5.6234 | 149.6412 |
| >500 | 55.0979 | 35.9590 | 656.7604 |

`>500` 档无 NaN/Inf，但相对 `101–500` 档，三个中位数分别放大约 4.90、6.39、4.39 倍；最大 gradient norm 为 17,872.17。结论是“数值仍有限，但 raw sum 对 extreme burst 有明显尺度敏感性”，不能解释为毫无风险。根据协议，本轮只登记为 sensitivity/exclusion candidate，不删除、不归一化。

## 8. Evaluation Contract

- singleton group：后续可报告 feedback Accuracy/Macro-F1、SID TokenAcc、PrefixAcc、JointExact；
- multi-event group：不得随机制造唯一 next event；必须使用 Group Recall@K、Group Precision@K、feedback composition NLL、semantic event coverage@K、SID prefix coverage@K；
- 当前 B 没有建模 `p(M|H)`，不能声称可生成完整 group cardinality。

## 9. Conclusion

```text
timestamp_B_approved_for_minimal_snmpp = true
```

批准理由是 SID 守恒、历史/积分记账、排列不变性、singleton 等价性和有限梯度全部通过。该结论只批准方案 B 的实现契约，不代表 SNMPP 已有效，也不消除 extreme-burst 风险。本轮在训练前 STOP。
