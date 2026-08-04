# HPN 历史消融：listen 与 is_organic（2026-08-04）

## 协议

- 三份数据集行集完全一致：train 564,319 / validation 100,210 / test 111,512；
- 历史窗口 200 事件；资格判定只看显式历史（listen 不参与资格）；
- 训练 200,000 行 × 6 轮，种子 2026，前缀自回归语义 HPN（4 层 × 256）；
- 测试 50,000 行 × 100 固定负例，排除历史物品；
- 三个变体只差历史输入：
  1. `control_h200`：只有显式反馈（like/dislike/unlike/undislike）；
  2. `listen_h200`：显式反馈 + listen（第 5 类事件，历史中 listen 占约 72%）；
  3. `listen_source_h200`：同 listen，并给每个历史事件加 is_organic 源标记
     （1=推荐驱动，2=organic）。

## 结果

| 方法 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| control_h200 | 0.4735 | 0.3691 | 0.6206 | 0.4172 | 0.7699 | 0.4577 | 0.3890 |
| listen_h200 | 0.5368 | 0.4235 | 0.6836 | 0.4716 | 0.8203 | 0.5087 | 0.4373 |
| listen_source_h200 | 0.5393 | 0.4241 | 0.6861 | 0.4721 | 0.8202 | 0.5086 | 0.4369 |
| 同预算流行度 | 0.3780 | 0.2925 | 0.4777 | 0.3255 | 0.5630 | 0.3490 | 0.3074 |

## 分层

训练见过物品 Recall@10 / NDCG@10 / MRR：

| 方法 | Recall@10 | NDCG@10 | MRR |
|---|---:|---:|---:|
| control_h200 | 0.7105 | 0.5003 | 0.4677 |
| listen_h200 | 0.7599 | 0.5504 | 0.5144 |
| listen_source_h200 | 0.7612 | 0.5506 | 0.5144 |

训练未见物品 Recall@10 / NDCG@10 / MRR：

| 方法 | Recall@10 | NDCG@10 | MRR |
|---|---:|---:|---:|
| control_h200 | 0.4334 | 0.2320 | 0.1996 |
| listen_h200 | 0.5202 | 0.2904 | 0.2465 |
| listen_source_h200 | 0.5249 | 0.2913 | 0.2453 |

## 结论

1. **listen 进入 HPN 历史显著提升下一显式交互物品检索**：
   Recall@10 +0.063、NDCG@10 +0.054、MRR +0.048；
   已见物品和未见物品都受益，未见物品提升更大（Recall@10 +0.087）。
   在该监督检索任务上，"listen 进来就没用"的说法不成立。
2. **is_organic 源标记的增量可忽略**：listen_source 相对 listen 仅
   Recall@10 +0.0025、NDCG@10 +0.0005，且 listen 提升远大于源标记。
   说明区分"推荐驱动 vs organic 事件"这个二值标签本身对历史编码贡献有限；
   曝光代理（同物品推荐驱动 listen 的滞后）仍是另一个待验证的候选级特征。
3. 边界：这是有界规模（20 万行 × 6 轮）的监督检索对照，不是全量正式实验，
   也不构成 RL/曝光可控结论；listen 作为 SNMPP 的事件标记与作为 HPN 历史
   是两回事，本结果只回答后者。

## 产物

- 数据：`artifacts/global_time_explicit_hpn_h200/`、
  `artifacts/global_time_explicit_hpn_listen_h200/`（行集一致，manifest 指纹绑定）；
- 训练：`artifacts/global_time_explicit_hpn_{h200,listen_h200,listen_source_h200}_pilot_seed2026/`；
- 评估：三个 run 目录下的 `evaluation_test_sampled_100_negatives.json`。
