# Phase 1 前置审计

本目录只保存 Phase 1 开始物化数据前的只读审计和可复现脚本。这里不训练 SNMPP，不训练 RQKMeans，也不生成最终 train/validation/test。

## Scripts

- `scripts/audit_timestamp_burst_sensitivity.py`：流式扫描完整 explicit event staging，统计 timestamp group 分布及候选阈值对事件、用户、反馈构成和 strict revision pair 的影响。
- `scripts/validate_timestamp_burst_audit.py`：将 burst 审计结果与 Phase 0 冻结统计逐项守恒核对。
- `scripts/audit_existing_sid_contract.py`：核验旧 SID 层数、码本 shape、消歧层、唯一恢复、旧 HPN 使用方式和数据契约。

## Official artifacts

- `artifacts/timestamp_burst_sensitivity.json`：全量 burst 详细审计，含直方图；后台任务完成后生成。
- `artifacts/timestamp_burst_validation.json`：burst 守恒校验和精简阈值影响；校验通过后生成。
- `artifacts/existing_sid_contract.json`：旧 SID/RQKMeans/HPN 契约审计。

## Runtime files

- `artifacts/timestamp_burst_progress.json`：screen 任务进度，不是实验结果。
- `logs/`：运行日志。

带 `smoke1000` 的文件仅用于脚本小样本自检，不属于正式结果；全量守恒校验通过后删除。

## Decision documents

- `../docs/2026-08-10_Phase1前置审计与SID重建决策清单.md`
- `../docs/2026-08-10_Phase1清洗与分组实施契约.md`
- `../docs/2026-08-10_SID重建与Proxy验证实施方案.md`

当前禁止擅自决定的事项：

1. `D_human_like` 的 burst threshold；
2. global chronological split 比例；
3. SID identity/disambiguation 方案。

