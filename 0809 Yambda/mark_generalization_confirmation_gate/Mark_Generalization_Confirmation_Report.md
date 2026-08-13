# Mark Generalization Confirmation Gate

## 1. 结论

本轮 Gate 通过。

- `mark_long_history_generalization_confirmed = true`
- `group_snmpp_design_approved = true`
- test 未读取；Time 未重复；Group-SNMPP 未设计或训练。

这里的“通过”只表示：在本轮固定的 2,000 用户、三个随机种子和预先规定的 early-stopping 协议下，Full-history GRU 的 validation CE/event 均低于 previous-group composition 与 last-group-only MLP。

## 2. 数据与选择协议

- 用户：2,000
- train target groups：101,241；target events：180,400
- validation target groups：44,365；target events：70,890
- 用户资格只使用 train 期信息：至少 6 个 train timestamp groups。
- 用户按 seed=2026 的稳定哈希选择，不使用 validation label 或模型表现。
- 每用户 train target 不超过 64 个；超出时按时间位置等距抽样，不读取 feedback label。
- validation 保留所选用户的全部 validation-period targets，并使用严格 rolling history。
- 组内使用 permutation-invariant representation，不制造同 timestamp 内顺序。
- 不删除 >500 burst，不使用 listen，不看 test。

范围限制：当前可复用物化池只有 7,343 用户，且其中一部分源自早期 Tiny 的 train-only 分层选择。因此本轮证明的是“固定物化池内可重复”，还不能直接外推到全部 854,649 名 D_SID 用户。

## 3. 模型选择协议

- 模型配置沿用上一轮：hidden size 64、Adam、learning rate 1e-3。
- batch size 128；最多 12 epochs。
- early-stopping metric：validation CE/event。
- patience=3，min_delta=1e-4。
- checkpoint 只按 validation CE 选择；test 完全禁用。
- seeds：2026、2027、2028。

## 4. Validation 主要结果

| 方法 | CE/event | Macro-F1 | 说明 |
|---|---:|---:|---|
| Full-train global feedback prior | 1.158664 | 0.174688 | 完整 D_SID train 类别频率 |
| Previous-group composition | 0.728521 | 0.589132 | 最近一组反馈构成，加 full-train Dirichlet strength 1 |
| Last-group-only MLP | 0.692678 ± 0.001835 | 0.557634 ± 0.001667 | 仅上一组特征 |
| Last-5 GRU | 0.595891 ± 0.000505 | 0.588572 ± 0.001678 | 最近 5 组，轻量诊断 |
| Full-history GRU | **0.557663 ± 0.000523** | **0.589810 ± 0.003863** | 完整严格前序 group history |

Full-history CE 相对改善：

- 相对 Previous-group composition：23.45%
- 相对 Last-group-only MLP：19.49%
- 相对 Last-5 GRU：6.42%

Macro-F1 需要单独谨慎解释：Full-history 的 0.589810 仅比 previous-group composition 的 0.589132 高约 0.00068，不能声称 Macro-F1 有实质提升。本轮稳定、明显的增益证据来自 CE，也就是对真实 group feedback composition 的概率分配更好。

Train/validation CE：

| 方法 | Train CE | Validation CE | Validation - Train |
|---|---:|---:|---:|
| Last-group-only MLP | 0.760001 | 0.692678 | -0.067323 |
| Last-5 GRU | 0.644689 | 0.595891 | -0.048799 |
| Full-history GRU | 0.618673 | 0.557663 | -0.061010 |

validation CE 低于 train CE 不是“泛化优于训练”的结论：train targets 是每用户按时间位置抽取的最多 64 条，而 validation 保留全部目标；两者真实反馈组成也不同（train 为 59.62%/15.44%/22.54%/2.40%，validation 为 53.70%/22.72%/22.03%/1.55%）。因此该 gap 同时包含目标采样与时间分布变化。

## 5. 三随机种子一致性

| Seed | Last-group CE | Last-5 CE | Full-history CE | Full best epoch |
|---:|---:|---:|---:|---:|
| 2026 | 0.695265 | 0.596602 | **0.556996** | 7 |
| 2027 | 0.691550 | 0.595482 | **0.557721** | 6 |
| 2028 | 0.691219 | 0.595589 | **0.558273** | 5 |

三颗种子方向完全一致，Full-history 同时胜过 previous composition、last-group MLP 和额外的 last-5 诊断模型。没有单类 collapse 或非有限数值。

## 6. 类别诊断

Full-history 三种子 validation 平均 Recall：

| Feedback | Recall |
|---|---:|
| like | 0.8956 |
| dislike | 0.8443 |
| unlike | 0.5476 |
| undislike | **0.0277** |

Full-history 的 group argmax 预测分布约为：like 67.90%、dislike 22.78%、unlike 9.23%、undislike 0.09%。

因此不能把本轮解释成“四类反馈都已解决”。模型没有塌成单一 like，但仍明显偏向 like，且 undislike 几乎不可预测。CE 的稳定优势主要证明历史对整体反馈组成有信息，并没有消除极少数类问题。

## 7. History-length 诊断

| Validation history length | Targets | Full CE | Last-5 CE | Full 相对优势（绝对 CE） |
|---|---:|---:|---:|---:|
| 6–20 | 606 | 0.895823 | 0.906052 | 0.010229 |
| 21–50 | 1,949 | 0.608279 | 0.621681 | 0.013402 |
| 51–100 | 3,718 | 0.612657 | 0.649631 | 0.036974 |
| 101–500 | 17,724 | 0.600199 | 0.641309 | 0.041110 |
| >500 | 20,368 | 0.494752 | 0.533997 | 0.039245 |

Full-history 在所有非空 history-length 切片都优于 last-5；优势从历史超过约 50 组后明显增大。因此性能不只是来自最近 1–2 个 groups，长历史确实提供了额外泛化信息。

注意：由于用户资格要求至少 6 个 train groups，本轮 validation 没有 history length 1 或 2–5 的样本。

## 8. Gate 回答

- A. Full-history 是否稳定优于 previous-group composition：是，3/3 seeds。
- B. 是否稳定优于 last-group-only MLP：是，3/3 seeds。
- C. 三个 seed 是否一致：是，CE 标准差约 0.00052。
- D. 是否仍严重偏向 like / undislike 几乎不可预测：是；undislike Recall 只有约 2.77%。
- E. 长历史是否提供额外信息：是；Full-history 在所有可用 history-length 切片均优于 last-5，且长序列优势更明显。

本轮到此 STOP，不继续实现 Group-SNMPP。
