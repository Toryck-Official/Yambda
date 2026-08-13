# SNMPP Phase 2 数学协议与实现设计

状态：**Gate 2、Timestamp Likelihood Gate 与 Numerical Integration Accuracy Audit 均已完成；Minimal SNMPP Pilot 获准，但本轮 STOP，尚未训练**  
版本：v1.1，2026-08-11  
范围：Gate 2 物化协议、Timestamp Likelihood Gate、最小 SNMPP、层级语义 ID 预测及第一轮评估。  
明确排除：missing-audio 补全、UNK SID、消歧 suffix、exact-item resolver、HPN、BOLA、强化学习、用户模拟器。

---

## 0. 本协议解决什么问题

目标不是立刻跑一个“大而全”的模型，而是先固定两个最容易出错的契约：

1. **D_SID 到底包含什么，语义 ID 如何永久冻结；**
2. **timestamp quantization 与大量 tied timestamps 如何进入时间似然而不虚构组内顺序。**

只有 Timestamp Likelihood Gate 通过后，才允许实现最小 SNMPP；只有最小 SNMPP 的时间和反馈建模通过后，才允许增加层级 SID 输出头。

---

## 1. 已冻结的研究边界

### 1.1 事件与数据范围

显式反馈类型固定为：

```text
0 = like
1 = dislike
2 = unlike
3 = undislike
```

不使用 listen，不使用 is_organic，不引入用户模拟器或强化学习。

原始去重主表记为 `D_all`。Phase 2 只处理其中具有真实音频 embedding 的事件：

$$
D_{SID}=\{e\in D_{all}: item(e)\in I_{real\_audio}\}.
$$

这里的 `D_SID` 是明确的 **real-audio explicit-feedback subset**，不是完整 D_all，也不是完整音乐库。

现有审计给出的物化前守恒基准是：

| 项目 | D_SID 物化前基准 |
|---|---:|
| events | 121,819,651 |
| users | 854,649 |
| items | 2,367,341 |
| like | 85,325,332 |
| dislike | 11,061,046 |
| unlike | 23,667,107 |
| undislike | 1,766,166 |
| timestamp groups | 77,565,283 |

必须保留此前偏差结论：D_SID 保留约 96% 的 like/dislike，但只保留约 73% 的 unlike/undislike。因此 Phase 2 的结论只适用于 D_SID；不得把它描述成 D_all 上无偏的完整显式反馈建模。

### 1.2 全局时间切分不可改变

沿用 D_all 已冻结的 global chronological split：

```text
train cutoff inclusive      = 22,172,245
validation cutoff inclusive = 24,154,245
```

过滤到 D_SID 后仍按相同 cutoff 判断 split。不能为了接近 80/10/10 而重新找 cutoff。

同一个 `uid + timestamp` group 不得跨 split。

### 1.3 Timestamp Order Recoverability Audit 的冻结结论

最新全量审计已经固定以下事实：

- parquet/list row order 工程上可重复，但不存在可辩护的业务时序依据；
- 不允许用 parquet 行序、反序或随机 tie-break 构造 chronology；
- tied events 中，属于“整个时间组可唯一排序”的事件仅约 1.96%；
- 不采用 mixed protocol；
- 主协议正式选择 `grouped shared pre-group history`；
- 组内全部事件共享严格前序历史 $\mathcal H_{t^-}$；
- 整个 group 评分完成后才整体进入历史；
- 不创建任何组内 $\Delta t=0$ transition；
- `group_size > 500` 只登记为 sensitivity/exclusion candidate，不从主数据删除。

官方字段证据和 train-only Timestamp Scale Audit 已共同确认：raw timestamp 以秒记录并量化到 5 秒网格；`model_time_unit=hour`；prediction horizon 固定为 696 小时（29 天）。该配置不得使用 validation/test 重新估计。

### 1.4 语义 SID 的含义

正式语义 SID 使用上一阶段 full-fit audio-only RQKMeans：

```text
4 semantic levels
256 codes per level
128-dimensional normalized audio input
25 KMeans iterations per residual level
seed = 2026
fit items = all 2,367,341 real-audio explicit items
```

语义 SID 为：

$$
SID(i)=(s_{i1},s_{i2},s_{i3},s_{i4}),\qquad s_{i\ell}\in\{0,\ldots,255\}.
$$

四层 SID **不是 exact item identity**。当前 full-fit B0 的 collision item ratio 为 43.1204%。Phase 2 第一阶段只预测 semantic item，不宣称能唯一恢复 item_id。

---

## 2. Step 1：Gate 2 正式 SID materialization 协议

本节既是执行契约，也是已完成 Gate 2 的守恒基准。正式产物见 `phase2_gate2_sid/gate2_manifest.json`；12 项检查全部通过。

### 2.1 正式冻结输入

冻结上一阶段 B0 产物：

```text
sid_feasibility_supplement/artifacts/B0_audio_only/codebooks.npy
sid_feasibility_supplement/artifacts/B0_audio_only/codes.uint8.npy
```

Gate 2 执行后复制到不可变正式目录，并生成：

```text
codebook_sha256
codes_sha256
source_embedding_sha256
item_universe_sha256
protocol_version
fit configuration
```

正式报告中不再称其为 candidate。

### 2.2 物化文件分层

不把 128 维 audio embedding 重复写入 1.218 亿条事件。采用两个表：

**Item catalog**

```text
item_id
sid_1
sid_2
sid_3
sid_4
audio_embedding[128]
```

恰好 2,367,341 行，`item_id` 唯一。

**Event table**

```text
uid
timestamp
group_id
feedback_type
item_id
sid_1
sid_2
sid_3
sid_4
split
```

事件表只冗余四个 uint8 token，不冗余 128 维向量。

### 2.3 Gate 2 必须通过的守恒检查

1. `item catalog rows = 2,367,341`；
2. 所有 D_SID item 均且仅均能 join 一个 SID；
3. `sid_1...sid_4` 全部位于 `[0,255]`；
4. join 前后 event count 都是 121,819,651；
5. 四类 feedback count 分别守恒；
6. user count、item count守恒；
7. train/validation/test cutoff 数值不变；
8. 每个事件的 split assignment 不变；
9. `uid + timestamp` group 数、group size、group membership 不变；
10. 排序只允许稳定按 `(uid,timestamp)`，不得为组内事件创造先后次序；
11. 码本加载后 `requires_grad=False`，下游 checkpoint 不得保存可训练副本；
12. 随机抽样及全量哈希检查 `item_id -> SID` 与冻结 codes 完全一致。

任一守恒检查失败，Gate 2 停止，不进入 Timestamp Gate。

---

## 3. Step 2：Timestamp Likelihood Gate

### 3.1 同刻事件的形式化定义

对用户 $u$，将相同 timestamp 的显式事件构成一个组：

$$
G_{u,j}=\{(f_{u,j,m},i_{u,j,m},SID(i_{u,j,m}))\}_{m=1}^{M_{u,j}},
$$

组时间为 $\tau_{u,j}$，且：

$$
\tau_{u,1}<\tau_{u,2}<\cdots.
$$

严格前序历史定义为：

$$
\mathcal H^-_{u,j}=\bigcup_{r<j}G_{u,r}.
$$

组内所有事件共享 $\mathcal H^-_{u,j}$。整个 $G_{u,j}$ 评分结束后，才整体进入后续历史。

观测窗口左侧可能存在未记录历史，因此每个用户的第一个可用 group 默认只作为 warm-start history，不计算“从未知观察起点到第一组”的时间损失。从第二个 group 开始形成 next-group target。

global split 采用滚动历史评估：validation target 可以使用其之前的 train groups 和更早 validation groups；test target 可以使用其之前的 train/validation groups 和更早 test groups。所有这些都只是目标时间以前已经观察到的真实历史，不允许使用当前或未来 group。

### 3.2 方案 A：完整 group-level point process

方案 A 将一个 group 视为一次 compound occurrence：

$$
\log p(\Gamma_u)=
\sum_j\left[
\log \Lambda_G(\tau_{u,j}\mid\mathcal H^-_{u,j})
+\log p(G_{u,j}\mid\tau_{u,j},\mathcal H^-_{u,j})
\right]
-\int \Lambda_G(t\mid\mathcal H_t)dt.
$$

它必须额外定义：

- group cardinality $M_{u,j}$；
- group 内 feedback/item 的集合或多重集合分布；
- permutation-invariant set decoder；
- 生成停止条件。

优点：数学对象和 tied timestamp 完全一致。  
缺点：它已经不是对原 SNMPP 的小改造，工程与评估复杂度最高。

### 3.3 方案 B：组到达过程 + 条件 mark 多重集伪似然（正式选定，Timestamp Gate 已验证）

定义四个正值 score $\lambda_k$，以及 timestamp-group arrival intensity：

$$
\lambda_k(t\mid\mathcal H_t)>0,
\qquad
\Lambda(t\mid\mathcal H_t)=\sum_{k=1}^{4}\lambda_k(t\mid\mathcal H_t).
$$

$\Lambda(t)$ 的严格解释是“下一 timestamp group 到达”的强度。给定观测到该组发生、且给定观测 group cardinality $M$，feedback composition 的条件概率参数化为：

$$
q_k(t\mid\mathcal H_t)=\frac{\lambda_k(t\mid\mathcal H_t)}
{\Lambda(t\mid\mathcal H_t)}.
$$

本协议 Gate 中检验的 B 版本为：

$$
\mathcal L_{time}^{(u,j)}
=-\log \Lambda(\tau_{u,j}\mid\mathcal H^-_{u,j})
+\int_{\tau_{u,j-1}}^{\tau_{u,j}}
\Lambda(t\mid\mathcal H_t)dt,
$$

$$
\mathcal L_{feedback}^{(u,j)}
=-\sum_{e\in G_{u,j}}
\log q_{f_e}(\tau_{u,j}\mid\mathcal H^-_{u,j}).
$$

它的含义是：

- 下一组发生时间只计一次；
- 组内事件的 feedback 在共享前序历史下作条件独立评分；
- group 评分完成后整体加入历史；
- 组间无事件区间的积分只计一次。

对于 $M_{u,j}>1$，不把每个 $\lambda_k$ 严格解释为标准连续时间“单个 feedback event intensity”。它是构造 $q_k=\lambda_k/\Lambda$ 的正值分量；$\Lambda$ 才承担 group-arrival intensity 的时间解释。

当前没有建模：

$$
p(M_{u,j}\mid \tau_{u,j},\mathcal H^-_{u,j}).
$$

因此 B 只给出 conditioned on observed cardinality $M$ 的 mark multiset pseudo-likelihood，不能声称模型能够完整生成下一 group 的事件数量。

这不同于旧实现中“每个组内事件都重复贡献一次 `log intensity`，但积分只计一次”的写法。Timestamp Gate 必须明确验证这两个目标的差异，不能混称为同一 likelihood。

当 $|G_{u,j}|=1$ 且真实 feedback 为 $f$ 时：

$$
\mathcal L_{time}+\mathcal L_{feedback}
=\int\Lambda(t)dt-\log\Lambda(\tau)
-\log\frac{\lambda_f(\tau)}{\Lambda(\tau)}
=\int\Lambda(t)dt-\log\lambda_f(\tau),
$$

因此 B 在 singleton 上应严格退化为标准 marked point-process NLL。这是 Timestamp Gate 的可执行数学单元测试，不是经验判断。

方案 B 的已知限制是：没有建模 $p(M_{u,j})$，所以它能对一个已观察 group 中的事件进行条件评分，但不能严格生成完整 group set。论文中必须称为：

> grouped shared-history conditional pseudo-likelihood（组共享历史条件伪似然）

不能称为原论文严格连续时间单事件 likelihood。

### 3.4 理论 grouped pseudo-likelihood 与优化目标必须分开

理论目标是所有 scored group 的 time 项与所有观测 group member 的 conditional mark 项之和：

$$
\mathcal L_{theory}=\sum_{u,j}\mathcal L^{(u,j)}_{time}
+\sum_{u,j}\mathcal L^{(u,j)}_{feedback}.
$$

后续训练为平衡梯度尺度，可以使用 normalized multi-task optimization objective：

$$
\mathcal L_{opt}=\operatorname{mean}_{group}(\mathcal L_{time})
+\operatorname{mean}_{event}(-\log q_f).
$$

后者是归一化多任务优化目标，不等于原始严格 NLL，也不能用其数值直接进行不同归一化协议之间的似然比较。报告时必须分别给出 `time loss/group` 与 `mark loss/event`。

### 3.5 方案 C：singleton-only sanity baseline

只保留 $M_{u,j}=1$ 的 group，使用原 SNMPP 标准似然。

它只用于：

- 验证 B 在 singleton 上是否退化为标准单事件似然；
- 检查 tied timestamp 实现是否引入异常；
- 提供小规模时间预测 sanity baseline。

当前约 48.12% 的 D_SID 事件位于 multi-event group，因此 C 不得作为主实验。

### 3.6 Timestamp Gate 最小实验及批准标准

Gate 只使用小规模、固定、可复现 subset，不训练最终模型。

必须通过以下检查：

1. **组内排列不变性**：随机打乱 $G_{u,j}$ 内事件，loss 完全不变；
2. **共享前序历史**：组内任一事件都看不到本组其他事件；
3. **延迟更新**：整个 group 完成评分后才进入下一组 history；
4. **积分唯一性**：每个相邻 unique timestamp interval 只积分一次；
5. **无零长度伪区间**：同 timestamp 不生成多个 $\Delta t=0$ interval；
6. **singleton 等价性**：B 在全部 singleton 序列上的 time+feedback loss 与标准 SNMPP 在数值误差内一致；
7. **有限性**：intensity、积分、NLL、gradient 均有限；
8. **四类梯度可见**：四种 feedback 对应参数在包含该类事件的 batch 中均有非零有限梯度；
9. **group-size 切片**：NLL 按 group size 切片后不得出现由公式重复计数导致的机械爆炸；
10. **时间预测可定义**：survival、event mass、expected next-group time 均有限且单调一致。

决策规则：

- B 全部通过：批准 B 作为第一版主协议；
- B 在 singleton 等价性、校准或 group-size 稳定性上失败：停止，进入 A 的 cardinality/set 建模设计；
- C 无论指标如何都只保留为 sanity baseline。

本协议已经选择 B 作为唯一主实现候选。本轮 Timestamp Likelihood Gate 的全部硬性测试均已通过，状态更新为 `timestamp_implementation_validated=true`。该批准只说明时间分组、似然记账和历史更新契约正确，不代表 SNMPP 已经训练或具有效果。

### 3.7 Train-only 时间尺度冻结结果

只使用 train split 的 61,535,193 个 unique timestamp groups，得到 60,759,338 个正 group gap，组级 zero gap 为 0。正 gap 的最小值、众数和全体最大公约数均为 5；结合官方字段说明，raw timestamp 按秒解释并量化到 5 秒网格。

第一版冻结：

```text
raw timestamp unit        = second
timestamp quantum         = 5 seconds
model time unit           = hour
model delta               = raw delta / 3600
prediction horizon        = 2,505,600 seconds = 696 hours = 29 days
selection                 = train p99 rounded upward to a whole day
train positive-gap mass   = 99.0103%
```

该 horizon 未使用 validation/test 选择；P95、P99.9 只作为 train-only 候选诊断。

### 3.8 Timestamp Gate 执行结果与边界

固定 train-only 子集包含 765 个目标 group；当前组大小与上一历史组大小各按 `1 / 2-5 / 6-20 / 21-100 / 101-500 / 500+` 抽取每档 64 个样本。审计模型是确定性四反馈 signed-kernel harness，不含优化器，不属于 Minimal SNMPP 训练。

以下检查全部通过：

- group 内排列不变性；
- shared pre-group history；
- group 完成后 delayed update；
- 每个 unique interval 只积分一次；
- 不产生组内零间隔转移；
- singleton 与标准 marked NLL 的最大绝对误差为 0；
- intensity、积分、loss、gradient 全部有限，四类反馈参数均有非零有限梯度；
- 控制其他输入不变、只复制 target multiset 时，time loss 与 intensity 完全不随 cardinality 变化；理论 mark-sum 按 M 相加，而 mark/event 保持稳定。

历史 burst 压力测试未使用 mean/sqrt(M) normalization，也未 clipping。上一 group 大于 500 时无 NaN/Inf，但相对 `101-500` 档，绝对 influence mass、总强度和 gradient norm 的中位数分别约放大 4.90、6.39、4.39 倍，最大 gradient norm 为 17,872.17。这说明 raw history sum 对 extreme burst 存在明显尺度敏感性，但不是方案 B 重复计算时间似然造成的机械增长。

因此：

```text
timestamp_B_approved_for_minimal_snmpp = true
extreme_burst_over_500 = sensitivity/exclusion candidate only
```

本轮不删除 burst、不做 normalization、不训练 Minimal SNMPP。后续若获授权训练，必须独立报告 `>500` 切片的稳定性。

---

## 4. Step 3：最小 SNMPP（time + feedback 输出）

### 4.1 “Feedback-only”术语边界

本阶段的输出只有：

```text
next group time
next feedback distribution
```

但按当前设想，历史事件表示包含其 item SID。因此准确名称应是：

> item-aware history, feedback/time-output SNMPP

后续消融中的“strict feedback-only SNMPP”才会删除历史 SID，只使用 feedback embedding。两者不能混用名称。

### 4.2 历史事件表示

冻结 RQ codebook：

$$
C_\ell\in\mathbb R^{256\times128},\qquad \ell=1,2,3,4.
$$

SID lookup 是 categorical lookup：

$$
h_{SID}(i)=\sum_{\ell=1}^{4}C_\ell[s_{i\ell}].
$$

不能把 `17、82、143` 作为连续数值输入。

由于 residual KMeans 的四层 reconstruction 正是四层 codeword 之和，$h_{SID}(i)$ 有明确的音频重构含义。$C_\ell$ 永久冻结。

令 feedback embedding 为 $E_f(f_n)$，trainable projection 为 $P_{SID}$：

$$
z_n=E_f(f_n)+P_{SID}\,h_{SID}(i_n).
$$

这里要求两项输出维度相同。`P_SID` 可以训练，`C_l` 不得训练。

### 4.3 Item-aware SNMPP intensity

保留原 SNMPP 的 signed interaction 与 delay-aware monotone temporal kernel：

$$
\lambda_k(t\mid\mathcal H_t)=
\sigma\left(
\alpha_k+
\sum_{n:t_n<t}
\psi(z_n,E_f(k))\,
\phi\left(z_n,E_f(k),|t-t_n-d_{f_n,k}|\right)
\right).
$$

其中：

- $\psi$ 输出有符号作用强度：正为 excitation，负为 inhibition；
- $\phi\in[0,1]$，对距离 $|\Delta t-d|$ 单调下降；
- $d_{f_n,k}\ge 0$ 暂时仍是 4×4 feedback-pair delay，不为每个 item 建 delay；
- $\sigma$ 使用固定 positive link，确保 intensity 为正；
- 历史严格使用 $t_n<t$，不是 $t_n\le t$。

这比原论文多了 source item SID context，但目标 intensity 仍然只有四类 feedback。它不是 action-conditioned predictor，也不是推荐模拟器。

### 4.4 积分

每个相邻 group interval 划分为 $Q$ 个等长 segment，每段采一个点：

$$
\int_a^b\Lambda(t\mid\mathcal H_t)dt
\approx
\frac{b-a}{Q}\sum_{q=1}^{Q}\Lambda(\hat t_q\mid\mathcal H_{\hat t_q}).
$$

训练期使用每个 segment 内的分层随机采样；验证/测试使用每段中点，避免评估噪声。Numerical Integration Accuracy Audit 已冻结：

```text
integration_segments_Q = 64
train integration points = one stratified random point per segment
validation/test integration points = deterministic segment midpoint
```

该决定仅改变积分分段数，不改变 grouped pseudo-likelihood、history、split、SID 或 timestamp group 定义。

训练前审计使用 338 个固定 train-only intervals，覆盖 5 个 gap 档与 6 个 previous-group-size 档，30 个交叉单元全部非空。所有 Q 使用完全相同的未训练参数、历史与 interval，以 deterministic midpoint 比较。Q=32 最初只作为临时 reference；其相对 Q=64 仍未收敛，因此最终以全样本 Q=128 为 reference，并用 30 个代表样本的 Q=256 复核 reference 稳定性。

相对 Q=128：Q=4 的 integral relative error 为 P50 0.0239%、P95 28.5585%、P99 42.6946%、最大 49.7692%；Q=64 为 P50 0.00056%、P95 0.4309%、P99 1.5222%、最大 4.5108%。Q=4 在 `previous group >500` 切片的 P95/P99/最大误差为 43.3248%/47.4853%/49.7692%，不稳定；Q=64 将其降为 1.7382%/3.6020%/4.5108%。Q=128 相对 Q=256 的代表样本 P95/P99/最大误差为 0.2468%/0.9761%/1.2565%，足以作为本轮高精度 reference。

审计 CPU 实现中，Q=64 成本约为 Q=4 的 14.02 倍，Q=128 约为 27.69 倍。Q=64 是第一个同时将全样本 P95 压到 0.5% 以下、P99 压到 2% 左右，并将 extreme-burst P95 压到 2% 以下的候选；继续到 Q=128 会再增加约一倍成本，收益主要集中在极少数尾部。因此正式冻结 Q=64，不批准 Q=4/8/16/32。

### 4.5 最小阶段训练目标

Timestamp Gate 已批准 B；后续 Pilot 可使用：

$$
\mathcal L_{minimal}
=\operatorname{mean}_{group}(\mathcal L_{time})
+\operatorname{mean}_{event}(\mathcal L_{feedback}).
$$

分别按 group 和 event 求均值，避免 multi-event group 仅因事件多就重复放大时间损失。该式明确称为 normalized multi-task optimization objective，不称为原始严格 NLL。

这一阶段不增加 SID 输出 loss。

### 4.6 最小阶段工程通过条件

- intensity 始终为正且有限；
- train/validation NLL 可正常计算；
- loss 相对初始化和简单基线下降；
- 四类 feedback 均有有限非零梯度；
- $\psi$、$\phi$、delay 不全部塌成常数或零梯度；
- 同 timestamp 排列不影响任何输出；
- 导出的 4×4 汇总作用只作统计解释，必须同时报告 amplitude 与 seed 稳定性，不能声称因果关系。

若模型再次全部预测 like，或者 time NLL/MAE 不超过简单基线，本分支停止，不增加 SID head。

---

## 5. Step 4：加入 Hierarchical SID prediction

### 5.1 联合概率分解

在组发生时间和 feedback 下，将语义 SID 分解为：

$$
p(SID\mid f,t,\mathcal H^-)
=\prod_{\ell=1}^{4}
p(s_\ell\mid s_{<\ell},f,t,\mathcal H^-).
$$

因此单个 event 的条件 mark 分解为：

$$
p(f,SID\mid t,\mathcal H^-)
=q_f(t\mid\mathcal H^-)
\prod_{\ell=1}^{4}p(s_\ell\mid s_{<\ell},f,t,\mathcal H^-).
$$

### 5.2 从 SNMPP 获得 SID head 的 history context

原 SNMPP 没有 RNN/Transformer hidden state，不能在论文中凭空称其为 hidden representation。第一版显式构造由 influence kernel 导出的向量 context：

$$
c_f(t)=\operatorname{LayerNorm}\left(
\sum_{n:t_n<t}
f_{n\rightarrow f}(t-t_n)\,W_vz_n
\right),
$$

其中：

$$
f_{n\rightarrow f}(\Delta t)=
\psi(z_n,E_f(f))\phi(z_n,E_f(f),|\Delta t-d_{f_n,f}|).
$$

它复用 SNMPP 已学到的 signed temporal influence，而不是另加一个未说明的序列编码器。

SID heads 为：

```text
Head 1: p(s1 | c_f, f, t)
Head 2: p(s2 | c_f, f, t, s1)
Head 3: p(s3 | c_f, f, t, s1, s2)
Head 4: p(s4 | c_f, f, t, s1, s2, s3)
```

每层都是 256 分类。

### 5.3 Train 与 inference

训练：

1. target group 使用严格前序历史；
2. time loss 每个 group 一次；
3. feedback loss 对组内每个 event 计算；
4. SID head 使用真实 feedback；
5. SID 自回归使用 teacher forcing；
6. 整个 group 评分完成后才进入历史。

测试：

1. 从历史预测下一 group time distribution；
2. 得到 expected/median next-group time；
3. 在指定评价协议的 query time 预测 feedback；
4. 使用预测 feedback；
5. 依次生成 $\hat s_1,\hat s_2,\hat s_3,\hat s_4$，不得使用真实 prefix；
6. multi-event group 不人工指定“第一条事件”，所有真实成员共享同一 pre-group prediction distribution。

第一轮必须区分两种评价，不能混报：

- **conditional mark evaluation**：在真实下一 group time 评价 feedback/SID，隔离 mark 能力；
- **end-to-end forecast diagnostic**：在预测时间评价 feedback/SID，观察时间误差传播。

主表第一轮以 conditional mark evaluation 为准；end-to-end 只作诊断。

### 5.4 完整 loss

每项先按自己的自然单位归一化：

$$
\mathcal L=
\overline{\mathcal L}_{time/group}
+\overline{\mathcal L}_{feedback/event}
+\sum_{\ell=1}^{4}\beta_\ell
\overline{\mathcal L}_{SID_\ell/event}.
$$

Baseline 固定：

$$
\beta_1=\beta_2=\beta_3=\beta_4=1.
$$

这只是 baseline，不是最终最优权重。后续只能根据 validation 调整，不能看 test 调权重。

时间损失和 SID 损失保持独立量纲；不定义“时间错一个量化单位等价于 SID 错一层”之类的人为换算。

---

## 6. Step 5：第一轮正式指标

### 6.1 时间指标

按下一 **unique timestamp group** 计算一次，不因 group 内事件数重复：

- normalized joint objective/event：仅作固定归一化协议内的优化诊断；
- time NLL/group：时间部分除以 scored groups，用于检查 B 的 group-time 建模；
- Time MAE（秒）；
- Time Median AE（秒）。

预测 horizon 与 time scale 已由 train period 冻结并写入 manifest：`model_time_unit=hour`，`prediction_horizon=696 hours=29 days`。验证/测试不得重新估计。

从历史截止时间 $t_0$ 开始，下一 group 等待时间的 survival 与 density 定义为：

$$
S(\delta)=\exp\left[-\int_0^\delta\Lambda(t_0+r\mid\mathcal H_{t_0+r})dr\right],
$$

$$
p(\delta)=\Lambda(t_0+\delta\mid\mathcal H_{t_0+\delta})S(\delta).
$$

第一版使用截断网格上的期望等待时间作为点预测；MAE 与 Median AE 分别是测试 group 绝对误差的均值和中位数。必须同时报告预测 horizon 内 event mass，避免把 horizon 截断误当成准确预测。

### 6.2 Feedback 指标

- Accuracy；
- Macro-F1；
- like/dislike/unlike/undislike 各自 Recall。

singleton group 允许报告 feedback Accuracy / Macro-F1、SID TokenAcc、PrefixAcc 和 JointExact。

multi-event group 不允许随机指定唯一 next event，必须使用 group/set-aware 指标：

- Group Recall@K；
- Group Precision@K；
- feedback composition NLL；
- semantic event coverage@K；
- SID prefix coverage@K。

因为同一 group 中多个事件共享预测分布，必须公开这一困难，不能随机排序后获得虚假的单事件标签。

### 6.3 SID 指标

- TokenAcc@1/2/3/4；
- PrefixAcc@1/2/3/4。

其中：

$$
PrefixAcc@2=\mathbb 1[\hat s_1=s_1\ \land\ \hat s_2=s_2].
$$

测试必须真实自回归；teacher-forced token accuracy 只能作为训练诊断，不能进入主结果。

### 6.4 完整 semantic event

报告：

$$
JointExact=\mathbb 1[
\hat f=f,\hat s_1=s_1,\ldots,\hat s_4=s_4].
$$

JointExact 不是唯一判断指标。四层 SID 有碰撞，且后层对量化边界敏感，必须同时解释 Prefix@1/2 和完整 Prefix@4。

### 6.5 最小基线

第一轮至少需要：

- feedback 训练集频率先验；
- 每用户或全局训练集 median next-group interval；
- SID 每层训练频率/条件频率基线；
- singleton standard SNMPP sanity baseline。

没有简单基线，不能仅凭 loss 下降声称模型有效。

---

## 7. Step 6：Collision 与 exact-item resolution 的边界

Phase 2 第一阶段预测：

$$
(time,feedback,semantic\ SID),
$$

不预测 exact item_id。

如果多个 item 共享四层 SID，预测该 SID 即算 semantic path 正确；不能使用 test popularity 将 bucket 自动解析成 item 后再声称 ExactItem。

后续独立阶段才允许研究：

```text
predicted SID
    -> collision bucket
    -> residual or retrieval module
    -> exact item candidate
```

已知 `SID + float32 residual` 对 96.7175% 的源向量身份唯一，但 residual 不是离散 token，也不能区分源 audio embedding 完全相同的 77,709 个重复 surplus item。因此该阶段必须作为 `Fine-grained Item Resolution`，不能偷偷并入第一版 SNMPP 的四层语义输出。

---

## 8. Step 7：第一轮 ablation 顺序

主模型通过后才按以下顺序增加对照：

1. strict feedback-only SNMPP：历史只含 feedback；
2. feedback + flat item/SID representation；
3. feedback + hierarchical SID；
4. feedback + hierarchical SID + residual；
5. learned signed temporal influence；
6. no temporal influence / fixed simple decay。

每个消融只改变一个因素，并共享：

- D_SID；
- global cutoff；
- timestamp protocol；
- train/validation/test targets；
- metrics；
- seeds；
- 评价时的 candidate/semantic universe。

要回答的因果归因仅限模型结构消融：

- SID history 是否改善反馈/时间预测；
- hierarchical SID 是否优于 flat representation；
- signed delayed temporal kernel 是否优于无作用或简单衰减；
- residual 是否只帮助 fine-grained resolution，而不损害 coarse semantics。

---

## 9. 实施状态机与停止条件

```text
协议审阅
  ↓ approved
Gate 2 materialization
  ↓ all conservation checks pass
Timestamp Likelihood Gate: A vs B vs C
  ↓ B or A explicitly approved
Minimal time + feedback SNMPP
  ↓ beats simple baselines and no collapse
Hierarchical SID heads
  ↓ semantic metrics show learnable signal
Semantic-event evaluation
  ↓ only then
Residual / exact-item resolution
  ↓ only then
HPN / BOLA investigation
```

任何阶段允许输出“假设不成立，停止该分支”。

明确停止条件：

- Gate 2 不守恒：停止；
- Timestamp B 不满足 singleton 等价性或组内排列不变性：停止 B，设计 A；
- time/feedback 不超过简单基线或再次 collapse：不增加 SID head；
- hierarchical SID 不优于频率/flat baseline：停止 residual 与 HPN 融合；
- 不允许通过改 split、删困难 group、加 listen 或查看 test 调参挽救结果。

---

## 10. v1.1 当前实现状态

```text
timestamp_protocol_selected = true
timestamp_protocol = B_grouped_shared_history_conditional_pseudolikelihood
timestamp_implementation_validated = true
timestamp_B_approved_for_minimal_snmpp = true
gate2_materialized = true
numerical_integration_validated = true
recommended_integration_Q = 64
minimal_snmpp_pilot_approved = true
minimal_snmpp_training_started = false
```

Gate 2、train-only Timestamp Scale Audit、Timestamp Likelihood Gate 与 Numerical Integration Accuracy Audit 已完成。即使 Pilot 已获准，本轮仍在训练前 STOP，不创建 Minimal SNMPP 训练任务，不加入 Hierarchical SID head。
