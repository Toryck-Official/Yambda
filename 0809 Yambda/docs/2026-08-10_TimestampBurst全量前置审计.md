# Timestamp Burst 全量前置审计

日期：2026-08-10

状态：全量只读审计完成；15 项守恒检查全部通过；未删除事件、未生成 batch_event_flag。

## 1. Hypothesis

验证极端 timestamp group 在完整 explicit-feedback 日志中是否只影响极少数据，并量化不同分布阈值对事件、用户、四类反馈和严格 revision pair 的影响。

## 2. Data Contract

- 数据：完整 5B sequential 日志筛出的四类 explicit feedback；
- 事件：like、dislike、unlike、undislike；
- group key：`(uid,timestamp)`；
- 同 group 不推断内部顺序；
- 当前统计针对 exact duplicate 去重前的原始 explicit rows；
- threshold 语义：仅排除 `group_size > T` 的完整 groups；
- 本审计不执行过滤，也不在假想过滤后重新配对 strict revision。

## 3. Data Counts

| 统计 | 数量 |
|---|---:|
| users | 857,499 |
| events | 136,292,476 |
| timestamp groups | 80,259,999 |
| like | 89,334,605 |
| dislike | 11,579,143 |
| unlike | 32,944,520 |
| undislike | 2,434,208 |
| strict like→unlike | 6,364,678 |
| strict dislike→undislike | 308,204 |
| same-item same-time ordering ambiguity groups | 3,612,417 |

以上全部与 Phase 0 冻结统计一致。

## 4. Model Input

不适用。本阶段禁止模型训练。

## 5. Model Output

不适用。本阶段输出 group-size histogram、分位点和候选阈值影响。

## 6. Training Objective

不适用。

## 7. Metrics

- timestamp group size 的 P50/P75/P90/P95/P99/P99.9/P99.99/max；
- threshold 排除的 groups/events/users；
- 排除事件的 feedback composition；
- 任一端点落入被排除 group 的现有 strict revision pairs。

## 8. Results

### 8.1 Group-size distribution

| 分位点 | group size |
|---|---:|
| P50 | 1 |
| P75 | 1 |
| P90 | 3 |
| P95 | 5 |
| P99 | 11 |
| P99.9 | 42 |
| P99.99 | 197 |
| max | 10,000 |

分布高度长尾：绝大多数 group 很小，但少量 group 包含大量事件。

### 8.2 Distribution-derived threshold sensitivity

下表中的规则均为“保留 group size 小于等于阈值，排除大于阈值的完整 group”。

| 候选 | 阈值 T | 排除 groups | 排除 events | 事件占比 | 影响 users | 用户占比 |
|---|---:|---:|---:|---:|---:|---:|
| P99 | 11 | 694,447 | 20,407,012 | 14.9730% | 192,789 | 22.4827% |
| P99.9 | 42 | 77,377 | 9,088,974 | 6.6687% | 40,392 | 4.7104% |
| P99.99 | 197 | 8,000 | 3,879,401 | 2.8464% | 5,937 | 0.6924% |

### 8.3 Feedback composition affected

| 候选 | like | dislike | unlike | undislike |
|---|---:|---:|---:|---:|
| P99, T=11 | 8,984,436 | 1,848,123 | 8,725,730 | 848,723 |
| P99.9, T=42 | 4,232,910 | 694,904 | 3,693,224 | 467,936 |
| P99.99, T=197 | 2,198,397 | 102,368 | 1,434,160 | 144,476 |

P99.99 尾部被排除事件中 unlike 与 undislike 的占比高于全量分布，因此 burst removal 不是反馈类型中性的清洗。

### 8.4 Strict revision pairs touched

“Touched”表示既有 strict pair 的 origin 或 revision 任一端点属于被排除 group；没有在过滤后重新建立新的配对。

| 候选 | like→unlike | 占全部 like→unlike | dislike→undislike | 占全部 dislike→undislike |
|---|---:|---:|---:|---:|
| P99, T=11 | 1,530,370 | 24.0447% | 145,737 | 47.2859% |
| P99.9, T=42 | 723,323 | 11.3646% | 104,147 | 33.7916% |
| P99.99, T=197 | 285,872 | 4.4915% | 27,375 | 8.8821% |

## 9. Failure / Anomaly

1. 最大 group size 为 10,000，确有明显超出普通人工逐项操作速度的 burst。
2. 但极端尾部承载的事件和 revision 信号并不少。仅按分位点删除会对本文重点研究的 preference revision 产生选择偏差。
3. 当前统计在 deterministic exact dedup 前完成。重复副本可能放大部分 group size；实际 materialize `batch_event_flag` 前必须在去重后重算。
4. 本审计只能识别统计 burst，不能凭 group size 单独证明某用户是机器人或数据错误。

## 10. Conclusion

- `D_all` 应保留，不能被 burst-cleaned 数据静默替代。
- P99 和 P99.9 都过于激进，不适合作为默认 human-like threshold。
- P99.99 是三个分布候选中最保守的，但仍删除 2.85% 事件，并触及 4.49%/8.88% 的两类 strict revision pairs，不能直接确定采用。
- `D_human_like` 只能作为明确标注的 sensitivity dataset；阈值需在 exact dedup 后复算并由用户确认。

## 11. 是否足以进入下一阶段

足以制定 Phase 1 的确定性去重与 group/flag 实现，但不足以自动确定 `D_human_like` 阈值。

## 12. 下一阶段建议

1. 先只执行 exact duplicate dedup，并重新计算去重后的 group-size histogram；
2. `D_all` 作为完整观测版本；
3. 若需要 human-like sensitivity，优先复核 `T=197` 附近而不是 T=11/42，但不得在用户确认前物化为主版本；
4. threshold 确认后重新构造 strict revision subset，不能直接用本审计的 touched 数量当作过滤后 pair 数量。

