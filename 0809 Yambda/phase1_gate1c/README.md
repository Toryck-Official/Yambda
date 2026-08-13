# Gate 1C：Explicit Collaborative → Audio Semantic Space

Gate 1A/1B 已封存。本目录研究仅使用 `global train period` 的四类显式反馈图，
能否为缺失 audio embedding 的物品恢复 item-specific 音乐语义。

当前阶段只做 cutoff-independent 覆盖审计：正式 global chronological split
比例尚未确认，因此 60%/70%/80%/90% 只作为候选覆盖曲线，不是最终协议。

硬约束：

- 只使用 `like/dislike/unlike/undislike`，不使用 listen；
- 协同图只允许包含正式 train cutoff 以前的边；
- 同一 timestamp 不跨 split；
- audio embedding 只作为有真实 embedding 物品的监督标签；
- 评估 item 的 audio embedding 不得进入映射器输入；
- 交互频率与用户集合来自 train period，validation/test 事件禁止使用；
- 未确认 cutoff 前不训练 collaborative embedding 或映射器；
- 不进入 Gate 2，不分配最终 SID。

后续若获准训练，优先级为：

1. 先构造保留四种反馈关系的简单 collaborative baseline；
2. 再学习 `collaborative embedding → frozen audio embedding`；
3. 以 Prefix@1/2 和 NN@10/50 为主要语义恢复证据；
4. Prefix@3/4 与 ExactSemanticSID 继续报告，但不作为唯一标准。

