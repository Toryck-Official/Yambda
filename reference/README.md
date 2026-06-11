# Reference Pack for User Response Simulation

这个目录整理的是和你当前课题最相关的文献与代码，核心主题有 5 类：

- 用户反馈模拟 `user response simulation`
- 多反馈融合 `multi-feedback fusion`
- 播放时长 / 完播率去偏 `watch-time / duration debias`
- 离线评估与 slate OPE
- 因果去偏与偏好漂移

你现在最需要抓住的一点是：`simulator` 和 `scorer` 不冲突，应该拆开。

- `simulator` 学的是 `p(response | history, item/slate)`
- `scorer` 学的是 `u(response)` 或 `u(history, response, slate)`
- 真正用于决策的是 `E_response[u(response)]`

这比直接把多动作硬压成一个标量 reward 再回归，通常更稳定，也更容易分析失败原因。

## 目录结构

- `code/`: 相关开源代码
- `papers/`: 已成功下载到本地的 PDF
- [CODE_INDEX.md](/root/autodl-tmp/0408Yambda/reference/CODE_INDEX.md): 本地代码索引和入口
- [LITERATURE_REVIEW_ZH.md](/root/autodl-tmp/0408Yambda/reference/LITERATURE_REVIEW_ZH.md): 中文综述，按问题拆解

## 最值得先看的内容

如果只看一小部分，建议顺序如下：

1. [LITERATURE_REVIEW_ZH.md](/root/autodl-tmp/0408Yambda/reference/LITERATURE_REVIEW_ZH.md)
2. [KuaiSim README](/root/autodl-tmp/0408Yambda/reference/code/KuaiSim/README.md)
3. [CWM README](/root/autodl-tmp/0408Yambda/reference/code/CWM/README.md)
4. [Open Bandit Pipeline README](/root/autodl-tmp/0408Yambda/reference/code/zr-obp/README.md)
5. [RecSim README](/root/autodl-tmp/0408Yambda/reference/code/recsim/README.md)

## 本地已下载 PDF

- [Generative Adversarial User Model for Reinforcement Learning Based Recommendation System](/root/autodl-tmp/0408Yambda/reference/papers/generative_adversarial_user_model_icml2019.pdf)
- [KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos](/root/autodl-tmp/0408Yambda/reference/papers/kuairand_cikm2022.pdf)
- [KuaiSim: A Comprehensive Simulator for Recommender Systems](/root/autodl-tmp/0408Yambda/reference/papers/kuaisim_neurips2023.pdf)
- [Off-Policy Evaluation for Slate Recommendation](/root/autodl-tmp/0408Yambda/reference/papers/off_policy_evaluation_for_slate_recommendation_nips2017.pdf)
- [Open Bandit Dataset and Pipeline](/root/autodl-tmp/0408Yambda/reference/papers/open_bandit_dataset_and_pipeline_neurips2021.pdf)
- [Sim2Rec](/root/autodl-tmp/0408Yambda/reference/papers/sim2rec_icde2023.pdf)
- [SlateQ](/root/autodl-tmp/0408Yambda/reference/papers/slateq_ijcai2019.pdf)
- [Unbiased Learning for the Causal Effect of Recommendation](/root/autodl-tmp/0408Yambda/reference/papers/unbiased_learning_for_causal_effect_of_recommendation_recsys2020.pdf)

## 在线补充阅读

下面这些条目和你的任务非常相关，但当前环境通过代理下载 PDF 时有 SSL EOF，先保留在线链接：

- [xMTF: A Formula-Free Model for Reinforcement-Learning-Based Multi-Task Fusion in Recommender Systems](https://arxiv.org/abs/2504.05669)
- [Counteracting Duration Bias in Video Recommendation via Counterfactual Watch Time](https://arxiv.org/abs/2406.07932)
- [Distributional Off-Policy Evaluation for Slate Recommendations](https://arxiv.org/abs/2308.14165)
- [Recommendations as Treatments: Debiasing Learning and Evaluation](https://arxiv.org/abs/1602.05352)
- [Estimating and Penalizing Induced Preference Shifts in Recommender Systems](https://arxiv.org/abs/2204.11966)
- [RecSim NG: Toward Principled Uncertainty Modeling for Recommender Ecosystems](https://arxiv.org/abs/2103.08057)

## 和你当前项目最直接的结论

- 不要继续把 `user response` 训练成单一 reward 回归器为主，这会把可解释性和可校准性一起丢掉。
- 先学“反馈分布”，再学“反馈到效用的映射”。
- `play ratio` 更像带顺序结构的离散变量，优先考虑 `bucket / ordinal`，而不是纯连续回归。
- `like / dislike / unlike / undislike` 不是互相对称、也不是总是同时有效，最好显式建 `valid mask`。
- 多样性不是用户单条反馈标签本身，而更像 slate 或策略层的效用项，不要硬塞进 response label。
