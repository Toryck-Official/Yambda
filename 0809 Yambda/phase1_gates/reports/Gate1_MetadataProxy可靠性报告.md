# Gate 1：Metadata Proxy 可靠性报告

日期：2026-08-10

结论：**C。当前 metadata proxy 不足以作为四层 semantic SID 的可靠来源。Gate 2 因缺少获准 proxy rule 暂停。**

## 1. Hypothesis

Artist/album metadata centroid 能否在严格 leave-one-out 条件下，把缺失 audio embedding 的 item 放回其真实 embedding 所属的四层 coarse-to-fine semantic region。

## 2. Data Contract

- 目标只从原本具有真实 normalized audio embedding 的 item 中抽取；
- 样本 100,000 items，validation/test 各 50,000；
- 目标自身 embedding 从每个 artist/album centroid 中严格排除；
- alpha 只在 validation 的 both-available 子集选择；
- test 只在 alpha 固定后评一次；
- 不使用 collaborative information；
- 不修改 explicit events；
- 不训练 SNMPP/HPN。

## 3. Data Counts

| 可重构类型 | 全部 real items | Gate 1 sample |
|---|---:|---:|
| artist-only | 598,684 | 32,293 |
| album-only | 14,427 | 10,128 |
| artist + album | 1,702,224 | 57,579 |
| 合计 | 2,315,335 | 100,000 |

样本还按下列维度分层：

- artist/album group：small=1–2、medium=3–9、large≥10 个其他 real items；
- explicit frequency：tail≤P50=2，mid=3–37，head>37 events；
- validation/test 在联合 strata 内 seeded 50/50 切分。

## 4. Model Input

真实 normalized 128-d audio embedding 只作 evaluation truth；proxy 输入只来自其他歌曲的 official artist/album group embedding。

## 5. Model Output

- Artist centroid；
- Album centroid；
- Conditional rule：both available 时使用 validation 选择的 artist/album 权重；只有一方时回退到该单一 proxy。

## 6. Training Objective

Proxy 本身无监督训练。Validation 在 both-available 子集上按以下字典序选择 album weight：PrefixAcc@4、@3、@2、@1、cosine、负 MSE。

选中 `alpha=1.0`，即 both available 时 album centroid 优于任何真正的 artist/album 混合。

Prefix 量化使用与 proxy targets 不重叠的 100,000 个 real items 拟合的 candidate RQKMeans：4 levels、256 codes/level、128-d、25 iterations、seed 2026。四层均使用 256 codes，最终 reconstruction MSE 为 0.00118577。

## 7. Metrics

任务优先级固定为：

1. SID PrefixAcc；
2. exact nearest-neighbor consistency；
3. cosine similarity；
4. normalized-vector MSE。

NN 使用 50,000 个与 targets 不重叠的 real embedding reference items，每种方法最多 10,000 个 test query，FAISS IndexFlatIP exact search。

## 8. Results

### 8.1 Test overall

| Method | Test rows | Cosine mean | MSE | Prefix@1 | Prefix@2 | Prefix@3 | Prefix@4 | NN@10 | NN@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Artist centroid | 44,935 | 0.8277 | 0.002693 | 50.72% | 14.75% | 4.31% | 1.49% | 21.67% | 35.07% |
| Album centroid | 33,854 | 0.8466 | 0.002397 | 54.75% | 18.49% | 6.18% | 2.24% | 26.40% | 39.89% |
| Conditional rule | 50,000 | 0.8343 | 0.002588 | 52.69% | 17.04% | 5.51% | 2.01% | 24.83% | 37.87% |

Cosine 看似较高，但随着 SID 层数增加，一致率迅速从约 53% 降到 2%。因此 proxy 可以捕捉部分粗粒度方向，却不能可靠恢复完整四层 semantic path。

### 8.2 Conditional rule by availability

| Stratum | Rows | Cosine | MSE | Prefix@1 | Prefix@2 | Prefix@3 | Prefix@4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| artist-only | 16,146 | 0.8087 | 0.002990 | 48.38% | 13.99% | 4.11% | 1.53% |
| album-only | 5,065 | 0.7572 | 0.003794 | 39.05% | 9.67% | 2.27% | 0.65% |
| both | 28,789 | 0.8623 | 0.002151 | 57.51% | 20.04% | 6.86% | 2.51% |

Both-available 相对最好，但 Prefix@4 仍只有 2.51%。Album-only 明显最弱，不能形成可信 conditional acceptance tier。

### 8.3 Group-size and frequency stability

Conditional Prefix@4：

- artist group：small 3.61%、medium 1.68%、large 2.14%；
- album group：small 2.83%、medium 1.97%、large 2.26%；
- frequency：tail 1.90%、mid 1.97%、head 2.85%。

没有任何 group-size/frequency stratum 显示出稳定的四层恢复能力；large group 并未单调改善结果。

### 8.4 Candidate-codebook sensitivity

使用旧的、独立拟合于 210,985 个 real embedding items 的 historical codebook 复算：

| Codebook | Prefix@1 | Prefix@2 | Prefix@3 | Prefix@4 |
|---|---:|---:|---:|---:|
| New Gate-1 candidate | 52.69% | 17.04% | 5.51% | 2.01% |
| Historical real-only candidate | 53.74% | 18.89% | 6.84% | 2.54% |

两套 real-only codebook 给出相同结论，排除了单次 RQ 初始化造成结论的主要可能性。

### 8.5 Potential missing-item coverage

| Rule | Missing items | Missing events |
|---|---:|---:|
| Artist support | 448,289 | 12,234,686 |
| Album support | 68,757 | 3,719,259 |
| Either/conditional | 450,594 | 12,450,538 |
| No supported metadata | 57,136 | 983,183 |

Conditional proxy 理论上覆盖 88.75% missing items、92.68% missing events；但覆盖率不能抵消语义质量失败。

## 9. Failure / Anomaly

1. Prefix@1 尚有粗粒度信号，但 Prefix@2–4 快速坍缩；
2. NN@10 仅约 25%，proxy 与 truth 的局部音乐邻域差异明显；
3. Validation 最优 alpha 为 1.0，说明 artist+album blend 没有超过 album 本身；
4. Album-only 稀有层最弱，不能通过简单 conditional threshold 获得可靠子集；
5. 200k Gate-1 candidate fit 在 2-GiB 环境超出内存，最终使用 100k candidate；该资源调整不改变 100k proxy validation sample，也由第二套 codebook sensitivity 验证了结论。

## 10. Conclusion

技术判断为 **C**：不建议把 metadata proxy 当作当前四层 semantic SID 来源。

它最多能作为 coarse auxiliary metadata feature，不能在未修改协议的情况下替代真实 audio embedding。不能仅因 cosine≈0.83 或潜在 coverage≈88.75% 就宣布 Gate 通过。

## 11. 是否足以进入下一阶段

**否。** Gate 2 要求先得到 Gate-1-confirmed proxy rule，再给 450,594 metadata-available items 分配 semantic SID。当前没有获得该规则。

## 12. 下一阶段建议

需要用户在以下方向中明确选择一个，不能由实现自行替换：

1. 接受 Gate 1 失败，将全部 507,730 missing-embedding items 作为 unknown semantic representation；
2. 改变 representation 协议，只让 proxy 提供 coarse/side feature，而不是四层 SID；
3. 明确授权即使 Gate 1 失败仍强制使用 conditional proxy，作为带质量风险的 sensitivity branch；
4. 为 missing items 寻找新的真实音频 embedding 来源。

在确认前，不拟合 Gate 2 最终码本、不生成 full SID、不训练 SNMPP/HPN。

