# 0809 Yambda 主线工作区

本目录是 2026-08-09 起确认的 Yambda 显式反馈主线唯一写入位置。

当前边界：

- 主事件仅包含 `like / dislike / unlike / undislike`，不包含 `listen`。
- 当前停留在 Phase 0 完成后的人工确认点。
- 尚未执行清洗、数据切分、RQKMeans 重训、SNMPP 或 HPN 训练。
- `/root/autodl-tmp/0804 Yambda` 仅作为历史实现和元数据补全方法的只读参考。

目录：

- `docs/`：固定研究协议、阶段说明与人工决策记录。
- `phase0_5b_audit/`：完整 5B 显式反馈审计脚本、来源分片、中间数据和结果。
- `dataprocess/`：后续经确认后实施的清洗、特征补全与 RQKMeans 工作；当前尚未创建实现。

