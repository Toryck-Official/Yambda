# Minimal SNMPP Pilot Report

## 1. Hypothesis

在固定 Phase 2 v1.1 协议下，先验证全历史 signed temporal SNMPP 能否在 Tiny 同集上学习非平凡的时间与四类反馈信号；只有通过后才允许 Pilot 泛化、基线比较与扩展性分析。

## 2. Data Contract

- D_SID；仅 like/dislike/unlike/undislike；不使用 listen 或 missing-audio item。
- 同 timestamp 组共享严格前序历史；整组评分后再更新历史。
- 4x256 frozen audio-only RQKMeans codebook；SID 只进入历史表示，无 SID 输出头。
- hour 单位、696-hour horizon、Q=64；train stratified-random，evaluation midpoint。

## 3. Data Counts

- D_SID: 121,819,651 events / 854,649 users / 2,367,341 items / 77,565,283 groups。
- Tiny: 6,413 target groups，38,363 target events。
- Pilot manifest: 100,225 train targets / 12,633 validation targets；test 未使用。

## 4. Model Input / Output

输入为完整前序事件的 feedback embedding、冻结 SID codeword 向量和连续时间；输出仅为下一 timestamp-group 时间分布和组内 feedback 分布。

## 5. Training Objective

normalized optimization objective = mean_group(time pseudo-likelihood loss) + mean_event(feedback cross-entropy)。它不是原始严格 NLL。

## 6. Tiny Overfit Results

- PASS: False
- Loss: 6.130785 -> 7.505963 (relative drop -22.43%)
- Feedback Accuracy / Macro-F1: 0.410578 / 0.145945
- Per-class recall: like=0.0008, dislike=0.0000, unlike=0.9996, undislike=0.0000
- Q64 stochastic time-loss CV: 4.992e-07

### 数值与退化诊断

- Feedback loss/event: 1.386262 -> 1.376090（略降，但主要来自多数类 unlike）。
- Time loss/group: 4.744523 -> 6.129873（恶化）。
- total intensity 最大值: 0.021489 -> 147.261642 / hour。
- 四类目标的 baseline、feedback embedding 与 delay 均有非零有限梯度；失败不是断图。
- psi、phi、delay 均偏离初始化，但 signed mass 退化成全正累积，并未形成可信的 excitation/inhibition 结构。
- 已保留 lr=1e-3、1e-4 的摆动日志；最终 lr=1e-5 仍失败。

### Previous-group >500 诊断

17 个 burst 目标的 time loss/group 均值 22.646、total intensity 均值 6.192、gradient norm 中位数 2294.6；对照分别为 3.885、0.042、101.6。均保持 finite，但 burst 明显放大 raw sum；样本仅 17 组，不能把全局失败只归因于 burst。

## 7. Pilot / Baseline / Scalability

未执行：Tiny Overfit Gate 未通过时必须 STOP。

## 8. Failure / Anomaly

Tiny Overfit Gate failed; Stage B and scalability were not authorized.

## 9. Conclusion

minimal_snmpp_pilot_passed = false
hierarchical_sid_head_approved = false
full_scale_snmpp_approved = false

## 10. Gate Questions

1. Tiny overfit：失败。
2. 非平凡 signal：没有形成；参数虽更新，但输出退化。
3. Pilot validation vs feedback baseline：未执行，Stage A 失败后无授权。
4. Time NLL/MAE vs baseline：未执行泛化比较；Tiny time loss 已恶化。
5. Collapse：发生 unlike collapse，不是 like collapse，但同样属于单类退化。
6. psi/phi/delay：均有更新，但方向退化，不能视为有效学习。
7. >500 burst：保持 finite，但强度、损失与梯度显著放大。
8. Q64 noise：可接受，CV 约 5e-7，不是本次失败原因。
9. Full-history 扩展性：未进入正式 profiling；当前 raw sum 已暴露数值风险，full scale 不批准。
10. Hierarchical SID：不批准。

## 11. Reproducibility

- code revision: efd4a2e0abc2a403d86c59fa8b754141d28f54e5
- subset SHA256: 6417bd993dbf3c4831e5921a906d79a2c7b928c3af2b1321eaa3f38432f74c6f
- 已保存 config、checkpoint、曲线、所有被拒绝数值尝试日志。
