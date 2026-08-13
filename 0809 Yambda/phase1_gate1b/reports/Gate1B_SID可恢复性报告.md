# Gate 1B：SID 可恢复性补充验证

结论分类：**C（在当前 artist/album-only 输入下，学习模型仍不足以可靠恢复多层 SID）**。这不是“任何 metadata 都无法恢复 SID”的普遍结论。Gate 2 未启动。

## 固定协议

- 100,000 个真实 embedding 用于 RQ 扰动校准。
- 学习任务使用 300,000 / 50,000 / 50,000 个严格 LOO train/validation/test item。
- 三个 seed；同一 real-only 冻结 candidate RQKMeans；测试集沿用 Gate 1A 固定 50k。
- 模型输入只有 artist/album LOO centroid、可用性和上下文规模；item ID 与显式交互频率不进入模型。
- Direct SID 测试严格自回归，API 不接受真实 prefix。

## A. RQKMeans 稳定性

| 实际 cosine | Prefix@1 | Prefix@2 | Prefix@3 | Prefix@4 |
|---:|---:|---:|---:|---:|
| 0.99 | 96.541% | 85.743% | 69.117% | 52.045% |
| 0.95 | 92.264% | 67.755% | 39.410% | 19.102% |
| 0.90 | 88.898% | 52.508% | 22.055% | 7.096% |
| 0.85 | 85.607% | 39.732% | 12.333% | 2.843% |
| 0.80 | 82.490% | 28.209% | 6.629% | 1.048% |
| 0.70 | 75.900% | 12.585% | 1.706% | 0.159% |

cosine≈0.85 时 Prefix@4 只剩 2.843%，说明残差量化本身高度敏感；但 Prefix@1 仍有 85.607%。Gate 1A 在相近 cosine 下 Prefix@1 只有 52.694%，所以 centroid 还存在方向结构偏差，不能把全部损失归因于 RQ 边界。低 margin 样本的 token flip 显著更多。

## B. Metadata 信息能力

本地固定版本真实存在的 item 静态关系只有 artist 与 album；没有 track title、genre、release 字段。2,315,335 / 2,367,341 个 real item 可构造严格 LOO context。

507,730 个 missing item 只有 159,627 种 artist/album context；390,718 个 item 的可用 context 与其他 item 完全相同，最大桶 30,893。因此仅靠这些输入无法逐首恢复完整音乐语义。

## C/D. 固定测试集结果

| 方法 | Cosine | MSE | Prefix@1 | Prefix@2 | Prefix@3 | Prefix@4 | NN@10 | NN@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Conditional centroid | 0.834337 | 0.002588 | 52.694% | 17.036% | 5.510% | 2.008% | 24.756% | 37.900% |
| Learned embedding MLP（3-seed mean） | 0.837161 | 0.002544 | 53.117% | 16.697% | 4.890% | 1.423% | 24.300% | 38.020% |
| Direct SID（3-seed mean） | — | — | 52.774% | 16.957% | 4.934% | 1.428% | — | — |

MLP 的 cosine 与 Prefix@1 有很小但统计可检出的提升；任务优先的 Prefix@2/3/4 反而显著下降。Direct SID 没有进一步改善完整层级路径。后层独立 TokenAcc 较高不等于路径可用：只要前缀错了，后层单 token 猜对也不能恢复真实 SID。

未继续上 DeepSets：当前 Gate 只要求先验证简单 learned predictor；更重要的是，同一 artist/album context 的 item 会收到相同的允许输入，DeepSets 不能凭空补回缺失的 track-specific 信息。是否引入新的 item-specific metadata/embedding 来源属于后续新方案，需要另行确认。

## 最终三个回答

1. **RQKMeans 对小扰动高度敏感：是。** 特别是残差后层和低 codeword margin 样本。
2. **Learned embedding 明显优于 centroid：否。** 只有 cosine/第一层小幅改善，核心多层 SID 与 NN 指标没有整体改善。
3. **Direct SID 进一步优于 embedding→RQKMeans：否。** Prefix@3/4 无显著提升，并且仍明显低于 centroid。

**STOP：Gate 2 未启动；没有给 missing item 分配最终 SID；没有训练 SNMPP/HPN。**
