# Phase 1 Canonical Explicit-Feedback Dataset

本目录物化经批准的 `D_all`：

- 事件仅含 `like/dislike/unlike/undislike`；
- 仅删除完全相同的 `(user,item,timestamp,feedback)` 多余副本；
- 不删除状态异常、同刻多反馈或 timestamp burst；
- global chronological split 为 80/10/10 event-count cutoff；
- cutoff timestamp 的完整事件组进入同一 split；
- 四类事件共享去重数组，各 split 通过每用户绝对边界引用数组，不复制事件。

`is_organic` 不属于当前 canonical 事件契约，也不参与去重后的模型输入。

