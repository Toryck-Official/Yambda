# Gate 1B：SID 可恢复性补充验证

本目录只验证三件事：RQKMeans 对 embedding 扰动的稳定性、学习式
metadata/context 补全能否超过质心、直接预测 SID 是否进一步改善。

边界：

- 使用 Gate 1A 的 real-only candidate codebook，保持冻结；
- 不训练或物化 Gate 2 最终码本；
- 不给 missing item 分配最终 SID；
- 不训练 SNMPP 或 HPN；
- `item_tokens.json` 是历史 SID，不作为 metadata 输入；
- 当前本地官方静态字段只有 artist-item 与 album-item 关系。

主要输出：

- `artifacts/rq_perturbation_stability.json`
- `artifacts/metadata_information_audit.json`
- `artifacts/learned_embedding_metrics.json`
- `artifacts/direct_sid_metrics.json`
- `artifacts/gate1b_metrics.json`
- `reports/Gate1B_SID可恢复性报告.md`

