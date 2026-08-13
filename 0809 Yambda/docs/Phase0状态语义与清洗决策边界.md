# Phase 0 状态语义与清洗决策边界

日期：2026-08-09

状态：仅解释与提出审计需求，尚未执行任何清洗。

## 1. 官方能够确认的语义

- `like`：用户点赞物品。
- `dislike`：用户点踩物品。
- `unlike`：用户撤销对物品的点赞。
- `undislike`：用户撤销对物品的点踩。

来源：Yandex 官方 Yambda 数据卡的 File Descriptions：
`https://huggingface.co/datasets/yandex/yambda`。

官方材料没有进一步承诺：

- `like` 与 `dislike` 在平台状态中是否始终严格互斥；
- 重复 `like` 或重复 `dislike` 是否一定是重复日志；
- 所有撤销事件的原始操作是否都位于当前观测窗口内。

因此不能仅凭常识补全业务状态机。

## 2. 当前“严格 revision 配对”的精确定义

对每一个 `(user_id, item_id)` 分别维护：

- 活跃 like 状态及其建立时间；
- 活跃 dislike 状态及其建立时间。

`like -> unlike` 只有在同一用户、同一物品已经于严格更早的时间戳建立活跃 like 时才记为一对。`dislike -> undislike` 同理。要求严格更早意味着延迟必须大于 0；同时间戳事件共享前序历史，不人为排序。同一 `(user,item,timestamp)` 出现多个反馈类型时，整组只记为歧义组，不参与状态更新和严格配对。

这是一种保守的可审计配对口径，不等同于平台完整真实状态。

## 3. Phase 0 中各异常指标的定义

### 完全重复的多余事件

同一 `(user_id, item_id, timestamp, feedback_type)` 出现多行时，第一行保留为原事件，其余行计入“多余事件”。例如三条完全相同的 like 记录计 2 条多余事件。

### 同 user-item-timestamp 出现不同反馈的组

同一用户、同一物品、同一时间戳下出现至少两种反馈类型。例如同刻出现 `like+unlike`、`like+dislike` 或 `dislike+undislike`。这类组没有可观察的组内顺序，不能解释为先 like 再 unlike。

### 涉及冲突的事件

上述歧义组内所有原始事件行的总数。一个歧义组可以包含 2 条或更多事件，因此该数大于歧义组数。这里“冲突”是审计技术用语，表示无法确定组内状态顺序，不等于已经证明业务数据错误。

### 重复 like / dislike

在同一 `(user,item)` 的观察窗口内，已有尚未被撤销的活跃 like/dislike，后面严格更晚时间又出现同类操作。它可能是日志重复、跨设备同步或平台状态重置，也可能是当前状态机假设不完整，不能在 Phase 0 直接删除。

### 无活跃前序状态的 unlike / undislike

在当前观察窗口内找不到同一 `(user,item)` 的未撤销严格前序 like/dislike。可能原因包括观察窗口左截断、同刻歧义、日志缺失或异常操作。该标签不等于“确定脏数据”。

### like 与 dislike 的跨状态现象

- `dislike_while_liked_groups`：359,921 组。
- `like_while_disliked_groups`：88,906 组。

它们说明在当前独立双状态审计器下确实能观察到交叉状态，但在没有官方互斥规则前，不能自动清洗。

## 3.1 全量补充审计结果

补充审计仍使用相同的 136,292,476 条显式事件，没有改变数据协议。按同一
`(user,item,timestamp)` 分组后：

| 精确反馈组合 | 组数 | 组内事件行 |
|---|---:|---:|
| like + dislike | 48,050 | 98,815 |
| like + unlike | 2,078,702 | 4,575,232 |
| dislike + undislike | 239,934 | 522,820 |
| like + undislike | 218,158 | 436,652 |
| dislike + unlike | 789,468 | 1,579,932 |
| unlike + undislike | 143,939 | 288,895 |
| like + dislike + unlike | 10,585 | 37,554 |
| like + dislike + undislike | 19,598 | 61,253 |
| like + unlike + undislike | 52,145 | 157,014 |
| dislike + unlike + undislike | 10,089 | 30,462 |
| 四类同刻 | 1,749 | 8,982 |

所有多反馈组合合计 3,612,417 组、7,797,611 行，与原歧义统计精确一致。

从可观察前序状态拆分后：

- 活跃 like 尚未撤销时再次 like：1,291,984 组；
- 活跃 dislike 尚未撤销时再次 dislike：79,806 组；
- 活跃 like 时出现 dislike：359,921 组；
- 活跃 dislike 时出现 like：88,906 组；
- 只有 active like、没有 active dislike 时出现 undislike：28,669 组；
- 只有 active dislike、没有 active like 时出现 unlike：9,161 组；
- 无窗口内 active like 的 unlike：22,984,129 组；
- 无窗口内 active dislike 的 undislike：1,394,859 组。

最后两项包含左截断，不能解释为同等数量的日志错误。补充字段已合并进唯一正式
`phase0_5b_audit/artifacts/data_audit.json`。

## 4. 当前建议的清洗层级

### 可直接提出删除候选，但仍需用户确认

1. 完全重复的多余行：删除副本、保留一条，不改变可观察事件语义。
2. 明显极端的单用户同时间戳批量操作：先定义阈值并做用户级、事件类型级敏感性统计，再决定删除整组还是整用户。

### 不应直接删除

1. `unlike` 前无窗口内 active like。
2. `undislike` 前无窗口内 active dislike。
3. 重复 like / dislike。
4. like 与 dislike 交叉活跃。
5. 同刻不同反馈类型。

这些事件应先保留在 canonical 原始层，同时增加状态质量标签。对需要严格 revision 链的专项分析，可以只使用 `strict_valid` 子集；这不等于从主事件表物理删除其他事件。

## 5. SID 与缺失 embedding 的后续边界

完整 5B 显式物品集合为 2,875,071 个，旧 SID 仅覆盖 244,553 个，因此旧 SID 不能直接作为当前主码本。

旧 0804 方法可作为批判性参考：

1. 对有真实音频 embedding 的显式物品使用源向量；
2. 对缺失物品，利用官方 artist/album 映射，在已有源向量的同作者或同专辑物品上计算质心代理；
3. 只用训练期显式交互构造协同兜底，避免验证/测试交互泄漏；
4. 用有真实向量的物品做留一遮蔽，评估代理向量 MSE、余弦相似度和语义前缀恢复；
5. 没有任何依据的物品保持 cold item，不伪造向量；
6. 保留 `feature_source`、`imputation_confidence` 和代理重复标记；
7. 在确定显式物品特征库后重新训练 RQKMeans，并额外审计重构误差、四层碰撞、完整 SID 唯一性和邻近保持。

旧实验只覆盖约 24.5 万物品，其权重和覆盖结论不能直接外推到当前 287.5 万显式物品。

本轮只读元数据覆盖审计已经得到：

- 缺少源音频 embedding：507,730 个显式物品、13,433,721 条事件；
- 存在同作者或同专辑的源向量依据：450,594 个物品、12,450,538 条事件；
- 没有这种元数据依据：57,136 个物品、983,183 条事件；
- artist 映射覆盖全部缺失物品，但只有 448,289 个物品所属作者组含源向量；
- album 映射覆盖 163,768 个缺失物品，其中 68,757 个所属专辑组含源向量。

这些只是“有构造质心的依据”，不代表代理向量准确。正式补全前仍须做留一遮蔽重建，再决定是否启用训练期显式协同兜底和如何重训码本。完整结果在 `phase0_5b_audit/artifacts/missing_embedding_metadata_audit.json`。

## 6. 尚待补充的必要统计

Phase 0 已经覆盖多数状态异常，但为了决定清洗规则，还需按同一 `(user,item)` 严格时间组补充：

- 完整状态转换矩阵及各路径计数；
- 同一 `(user,item,timestamp)` 的反馈组合计数，而不只是用户时间戳级组合；
- `like+dislike`、`like+unlike`、`dislike+undislike` 等歧义组的明细与规模分布；
- 重复 like/dislike 的间隔分布及是否集中于异常用户；
- 无前序 revision 在用户观察窗早期与后期的占比，用于判断左截断解释力；
- 缺失 embedding 物品的事件量、用户量、artist/album 映射覆盖率和长尾分布。

补充审计仍然只统计，不执行清洗。
