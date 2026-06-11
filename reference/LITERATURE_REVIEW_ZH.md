# 中文综述：多动作用户反馈模拟与离线评估

## 1. 先把问题拆清楚

你现在遇到的困难，本质上不是“模型再调一调”就能解决，而是目标定义有点混了。

当前问题至少包含 3 个不同层次：

1. `response simulation`
   预测用户面对下一个 item 或 slate 时，会不会听、听多少、会不会 like/dislike/unlike。
2. `utility scoring`
   把上面的多种反馈映射成一个业务目标分数。
3. `offline evaluation / policy evaluation`
   用历史日志判断一个新策略是否真的更好。

很多实验失败，是因为把这三层硬压进了一个标量 reward 监督里。

更合理的范式是：

- 先学 `p(response | history, item/slate)`
- 再定义或学习 `u(response)`
- 最后做 `E_response[u(response)]` 的策略评估

这就是为什么我前面说 `simulator` 和 `scorer` 不冲突，它们是串联关系，不是二选一。

## 2. 对你最相关的文献分组

### 2.1 用户反馈模拟 / simulator

#### RecSim

- 论文：RecSim: A Configurable Simulation Platform for Recommender Systems
- 链接：https://arxiv.org/abs/1909.04847
- 本地代码：[recsim](/root/autodl-tmp/0408Yambda/reference/code/recsim)
- 作用：
  - 显式建模用户状态、文档状态、状态转移和响应生成
  - 强调 sequential interaction，而不是单步 CTR
- 对你的启发：
  - response 应该被看作“生成出来的行为变量”，而不是一个手工合成 reward 标签

#### RecSim NG

- 论文：RecSim NG: Toward Principled Uncertainty Modeling for Recommender Ecosystems
- 链接：https://arxiv.org/abs/2103.08057
- 本地代码：[recsim_ng](/root/autodl-tmp/0408Yambda/reference/code/recsim_ng)
- 作用：
  - 在 simulator 里更强调概率建模和不确定性
  - 更适合“输出行为分布”而不是单点
- 对你的启发：
  - 你们现在的 user response 很适合走 distributional head，而不是只回归一个 reward

#### KuaiSim

- 论文：KuaiSim: A Comprehensive Simulator for Recommender Systems
- 链接：https://arxiv.org/abs/2309.12645
- 本地 PDF：[kuaisim_neurips2023.pdf](/root/autodl-tmp/0408Yambda/reference/papers/kuaisim_neurips2023.pdf)
- 本地代码：[KuaiSim](/root/autodl-tmp/0408Yambda/reference/code/KuaiSim)
- 作用：
  - 直接覆盖即时多行为反馈、session、retention
  - 很接近短视频推荐真实链路
- 对你的启发：
  - 最像你现在的问题设定
  - 非常适合参考它的 `immediate response model + retention model` 分层思路

#### Generative Adversarial User Model

- 论文：Generative Adversarial User Model for Reinforcement Learning Based Recommendation System
- 链接：https://arxiv.org/abs/1812.10613
- 本地 PDF：[generative_adversarial_user_model_icml2019.pdf](/root/autodl-tmp/0408Yambda/reference/papers/generative_adversarial_user_model_icml2019.pdf)
- 作用：
  - 用生成式用户模型来给 RL 推荐提供更真实的反馈环境
- 对你的启发：
  - 如果你后面发现 hard label 训练出来的反馈模式太“死”，生成式 simulator 会是下一步方向

#### Estimating and Penalizing Induced Preference Shifts

- 论文：Estimating and Penalizing Induced Preference Shifts in Recommender Systems
- 链接：https://arxiv.org/abs/2204.11966
- 作用：
  - 研究推荐系统对用户偏好本身的长期影响
- 对你的启发：
  - 你们如果未来不只看“下一步反馈”，还想看长期偏移，这篇很关键

### 2.2 多反馈融合 / reward fusion

#### xMTF

- 论文：xMTF: A Formula-Free Model for Reinforcement-Learning-Based Multi-Task Fusion in Recommender Systems
- 链接：https://arxiv.org/abs/2504.05669
- 作用：
  - 明确把多任务预测 `MTL` 和多任务融合 `MTF` 分开
  - 讨论如何从多种反馈预测走到单一排序分数
- 对你的启发：
  - 你们现在最大的问题之一，就是把“预测多个反馈”和“合成 reward”揉在一起
  - xMTF 的思路非常契合你当前的重构方向

#### DFAR

- 论文：Dual-interest Factorization-heads Attention for Sequential Recommendation
- 链接：https://arxiv.org/abs/2302.03965
- 本地代码：[WWW2023-DFAR](/root/autodl-tmp/0408Yambda/reference/code/WWW2023-DFAR)
- 作用：
  - 更偏历史反馈编码和兴趣分解
- 对你的启发：
  - 如果你们后面想强化“历史中正负反馈分路编码”，可以参考它的编码思路

### 2.3 播放比例 / 时长偏置

#### D2Q

- 论文：Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation
- 链接：https://arxiv.org/abs/2206.06003
- 本地代码：[Ks-D2Q](/root/autodl-tmp/0408Yambda/reference/code/Ks-D2Q)
- 作用：
  - 研究视频时长本身对 watch-time label 的污染
- 对你的启发：
  - 你们把 `play ratio` 当核心信号时，必须考虑 duration bias

#### CWM

- 论文：Counteracting Duration Bias in Video Recommendation via Counterfactual Watch Time
- 链接：https://arxiv.org/abs/2406.07932
- 本地代码：[CWM](/root/autodl-tmp/0408Yambda/reference/code/CWM)
- 作用：
  - 用反事实 watch time 去解释和修正 duration bias
- 对你的启发：
  - 比单纯分桶回归更进一步，告诉你“播放”标签本身可能已经有偏

#### KuaiRand

- 论文：KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos
- 链接：https://arxiv.org/abs/2208.08696
- 本地 PDF：[kuairand_cikm2022.pdf](/root/autodl-tmp/0408Yambda/reference/papers/kuairand_cikm2022.pdf)
- 本地代码：[KuaiRand](/root/autodl-tmp/0408Yambda/reference/code/KuaiRand)
- 作用：
  - 随机曝光日志，天然更适合做去偏和评估
- 对你的启发：
  - 你们当前如果继续只在强偏日志上学 reward，会很容易把曝光偏差当成用户偏好

### 2.4 离线评估 / slate / OPE

#### Off-Policy Evaluation for Slate Recommendation

- 论文：Off-Policy Evaluation for Slate Recommendation
- 链接：https://arxiv.org/abs/1605.04812
- 本地 PDF：[off_policy_evaluation_for_slate_recommendation_nips2017.pdf](/root/autodl-tmp/0408Yambda/reference/papers/off_policy_evaluation_for_slate_recommendation_nips2017.pdf)
- 作用：
  - 研究有序 item 集合 `slate` 的离线评估
- 对你的启发：
  - 多样性、组合效应、位置依赖都更接近 slate 问题，而不是单 item 回归

#### Distributional Off-Policy Evaluation for Slate Recommendations

- 论文：Distributional Off-Policy Evaluation for Slate Recommendations
- 链接：https://arxiv.org/abs/2308.14165
- 作用：
  - 不只估计期望回报，还估计完整分布
- 对你的启发：
  - 如果你们以后要关心“风险”“稳定性”“尾部坏结果”，它比只看均值更有价值

#### Open Bandit Dataset and Pipeline

- 论文：Open Bandit Dataset and Pipeline
- 链接：https://arxiv.org/abs/2008.07146
- 本地 PDF：[open_bandit_dataset_and_pipeline_neurips2021.pdf](/root/autodl-tmp/0408Yambda/reference/papers/open_bandit_dataset_and_pipeline_neurips2021.pdf)
- 本地代码：[zr-obp](/root/autodl-tmp/0408Yambda/reference/code/zr-obp)
- 作用：
  - 最成熟的 bandit / OPE 实验框架之一
- 对你的启发：
  - 你们后面要做“离线验证一个新打分方式是不是更好”，这是最实用的落地工具

#### SlateQ

- 论文：SlateQ: A Tractable Decomposition for Reinforcement Learning with Recommendation Sets
- 链接：https://arxiv.org/abs/1905.12767
- 本地 PDF：[slateq_ijcai2019.pdf](/root/autodl-tmp/0408Yambda/reference/papers/slateq_ijcai2019.pdf)
- 作用：
  - 关注 slate 层级的长期价值分解
- 对你的启发：
  - 如果你们把“多样性动作”放到策略层而不是 response label 层，会更顺

### 2.5 因果去偏

#### Recommendations as Treatments

- 论文：Recommendations as Treatments: Debiasing Learning and Evaluation
- 链接：https://arxiv.org/abs/1602.05352
- 作用：
  - 经典选择偏差 / 曝光偏差视角
- 对你的启发：
  - 你不能把“看到了并互动了”直接当作无偏监督

#### Unbiased Learning for the Causal Effect of Recommendation

- 论文：Unbiased Learning for the Causal Effect of Recommendation
- 链接：https://arxiv.org/abs/2008.06820
- 本地 PDF：[unbiased_learning_for_causal_effect_of_recommendation_recsys2020.pdf](/root/autodl-tmp/0408Yambda/reference/papers/unbiased_learning_for_causal_effect_of_recommendation_recsys2020.pdf)
- 作用：
  - 进一步把推荐的因果效应和观测相关性分开
- 对你的启发：
  - 如果后面要回答“模型只是学会了历史策略，还是学到了真实偏好变化”，这类方法很关键

## 3. 从这些文献里提炼出的结论

### 结论 1

当前最不该继续加码的方向，是“把多动作反馈先手工聚成一个 reward，再直接回归这个 reward”。

原因：

- 会把标签噪声、曝光偏差、duration bias、动作间异质性全压进一个数字
- 错了之后根本不知道是 `play` 错、`like` 错，还是融合公式错
- 很容易学成均值器

### 结论 2

更合理的目标定义是多头 simulator：

- `listen_head`: 是否发生消费
- `play_head`: `bucket / ordinal / survival`
- `like_head`
- `dislike_head`
- `unlike_head`
- `undislike_head`
- 可选 `stay / skip / exit / return`

这些 head 共同输出反馈分布，然后再走 scorer。

### 结论 3

`play ratio` 不应只做纯回归，至少应优先比较下面两种方式：

- bucket classification
- ordinal regression

原因不是技巧问题，而是它本来就更像“分段行为结果”，同时还受视频时长影响。

### 结论 4

显式反馈不是总是“有效”的，应该建 validity。

例如：

- 没有真正消费到一定程度时，`like` 的缺失未必等于负反馈
- `undislike` 更像状态撤销，而不是和 `like` 同类正反馈
- 一些动作之间有依赖和时序约束

这点非常适合写进你们自己的创新点。

### 结论 5

多样性更像策略层 / slate 层目标，不是单条 user response 标签。

也就是说：

- `user simulator` 负责预测用户对当前曝光内容的细粒度反馈
- `policy scorer` 负责把多样性、新颖性、覆盖度等目标一起并入决策

## 4. 对你当前项目的推荐重构路线

### 路线 A：近期最务实

1. 先把当前 `user response` 改成多头分布预测，不以标量 reward 为主监督。
2. `play` 改成 `bucket / ordinal`。
3. 显式反馈加 `valid mask`。
4. 单独加一个 scorer，把多头输出映射成 reward。
5. 训练和评估时分开报告：
   - head-level calibration
   - derived reward
   - by-type metrics

### 路线 B：中期增强

1. 引入 duration debias，参考 D2Q / CWM。
2. 引入用户状态演化，参考 KuaiSim / RecSim。
3. 评估阶段补 OPE，参考 zr-obp 和 slate OPE。

### 路线 C：如果要追更强研究味道

1. 把 scorer 做成可学习但有符号约束的融合器。
2. 把长期偏好漂移也建进去。
3. 把策略层目标显式扩展到 diversity / novelty / retention。

## 5. 我对当前实验困境的宏观判断

你们不是“模型太弱”，而是“监督对象过早坍缩成了一个标量”。

想要的效果其实应该是：

- 能预测不同类型反馈，而不是只拟合平均 reward
- 能区分负反馈为什么发生
- 能支持后续策略层做多目标权衡
- 能在离线环境下比较新策略，而不是只比较回归误差

如果按照这个标准看，当前最值得做的不是继续堆 loss 权重，而是把任务重新拆开。
