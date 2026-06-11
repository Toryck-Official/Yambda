# Code Index

下面按“和当前课题的相关度”整理本地代码。

## A. 最相关

### 1. KuaiSim

- 作用：短视频推荐模拟器，直接覆盖“即时多反馈 + session + retention + RL benchmark”。
- 为什么重要：和你现在的问题最像，不是单一 click，而是多反馈、多阶段行为。
- 本地路径：
  - [README.md](/root/autodl-tmp/0408Yambda/reference/code/KuaiSim/README.md)
  - [code/README.md](/root/autodl-tmp/0408Yambda/reference/code/KuaiSim/code/README.md)
  - [run_multibehavior.sh](/root/autodl-tmp/0408Yambda/reference/code/KuaiSim/code/run_multibehavior.sh)
  - [generate_session_data.sh](/root/autodl-tmp/0408Yambda/reference/code/KuaiSim/code/generate_session_data.sh)
- 你最该看的点：
  - 即时用户响应模型 `Immediate User Response Model`
  - 多行为 joint training
  - session/retention 分层建模

### 2. CWM

- 作用：处理 watch-time / played-ratio 的时长偏置问题。
- 为什么重要：你们现在对 `play ratio` 很敏感，而这类标签很容易被视频时长本身污染。
- 本地路径：
  - [README.md](/root/autodl-tmp/0408Yambda/reference/code/CWM/README.md)
  - [src/run.sh](/root/autodl-tmp/0408Yambda/reference/code/CWM/src/run.sh)
  - [src/prepare_data.py](/root/autodl-tmp/0408Yambda/reference/code/CWM/src/prepare_data.py)
  - [src/main.py](/root/autodl-tmp/0408Yambda/reference/code/CWM/src/main.py)
  - [src/train_model2.py](/root/autodl-tmp/0408Yambda/reference/code/CWM/src/train_model2.py)
- 入口：
  - `bash src/run.sh`

### 3. KuaiRand

- 作用：带随机曝光的短视频顺序推荐数据集。
- 为什么重要：它提供了更接近无偏评估的基础，适合做 response model 的校准和比较。
- 本地路径：
  - [README.md](/root/autodl-tmp/0408Yambda/reference/code/KuaiRand/README.md)
- 你最该看的点：
  - random exposure
  - rich multi-feedback
  - sequential logs

### 4. zr-obp

- 作用：离线 bandit / OPE 工具箱，支持 slate 场景。
- 为什么重要：如果后面要评估“根据 simulator 打分的新策略是否值得离线采用”，它是最实用的工具。
- 本地路径：
  - [README.md](/root/autodl-tmp/0408Yambda/reference/code/zr-obp/README.md)
  - [quickstart](/root/autodl-tmp/0408Yambda/reference/code/zr-obp/examples/quickstart/README.md)
  - [synthetic_slate.ipynb](/root/autodl-tmp/0408Yambda/reference/code/zr-obp/examples/quickstart/synthetic_slate.ipynb)
  - [examples/obd/evaluate_off_policy_estimators.py](/root/autodl-tmp/0408Yambda/reference/code/zr-obp/examples/obd/evaluate_off_policy_estimators.py)
- 入口：
  - `examples/quickstart/synthetic_slate.ipynb`
  - `examples/obd/evaluate_off_policy_estimators.py`

## B. 强相关但更偏研究平台

### 5. RecSim

- 作用：经典推荐系统仿真平台。
- 为什么重要：如果你后面决定把当前项目抽象成“状态转移 + 反馈生成 + 策略学习”三段式，它是最标准的参考实现之一。
- 本地路径：
  - [README.md](/root/autodl-tmp/0408Yambda/reference/code/recsim/README.md)
  - [recsim/main.py](/root/autodl-tmp/0408Yambda/reference/code/recsim/recsim/main.py)
  - [setup.py](/root/autodl-tmp/0408Yambda/reference/code/recsim/setup.py)

### 6. RecSim NG

- 作用：Google 的 probabilistic / differentiable simulator。
- 为什么重要：比 RecSim 更强调不确定性建模和显式概率图式，适合做“response distribution”而不是单点预测。
- 当前状态：由于 GitHub 直拉不稳定，我把 PyPI wheel 解包到了本地，可直接读源码。
- 本地路径：
  - [METADATA](/root/autodl-tmp/0408Yambda/reference/code/recsim_ng/recsim_ng-0.1.2.dist-info/METADATA)
  - [demo.py](/root/autodl-tmp/0408Yambda/reference/code/recsim_ng/recsim_ng/applications/demo.py)
  - [ecosystem_simulation_demo.py](/root/autodl-tmp/0408Yambda/reference/code/recsim_ng/recsim_ng/applications/ecosystem_simulation/ecosystem_simulation_demo.py)
  - [interest_evolution_simulation_demo.py](/root/autodl-tmp/0408Yambda/reference/code/recsim_ng/recsim_ng/applications/recsys_partially_observable_rl/interest_evolution_simulation_demo.py)

## C. 补充基线

### 7. Ks-D2Q

- 作用：D2Q 的离线实验代码，对应 duration bias / watch-time deconfounding。
- 为什么重要：如果你们继续保留 `play ratio` 作为核心反馈变量，它是比普通回归更贴题的对照方法。
- 本地路径：
  - [README.md](/root/autodl-tmp/0408Yambda/reference/code/Ks-D2Q/README.md)
- 入口：
  - `python main.py train`
  - `python main.py eval`

### 8. WWW2023-DFAR

- 作用：序列推荐中的 dual-interest / feedback-aware 编码基线。
- 为什么重要：它不直接解决 simulator/OPE，但对“历史反馈如何编码”有参考价值。
- 本地路径：
  - [README.md](/root/autodl-tmp/0408Yambda/reference/code/WWW2023-DFAR/README.md)

## 对你现在最有用的代码阅读顺序

1. [KuaiSim/code/run_multibehavior.sh](/root/autodl-tmp/0408Yambda/reference/code/KuaiSim/code/run_multibehavior.sh)
2. [CWM/src/run.sh](/root/autodl-tmp/0408Yambda/reference/code/CWM/src/run.sh)
3. [zr-obp/examples/quickstart/synthetic_slate.ipynb](/root/autodl-tmp/0408Yambda/reference/code/zr-obp/examples/quickstart/synthetic_slate.ipynb)
4. [recsim/recsim/main.py](/root/autodl-tmp/0408Yambda/reference/code/recsim/recsim/main.py)
5. [recsim_ng/applications/ecosystem_simulation/ecosystem_simulation_demo.py](/root/autodl-tmp/0408Yambda/reference/code/recsim_ng/recsim_ng/applications/ecosystem_simulation/ecosystem_simulation_demo.py)

## 一个务实建议

如果你的目标不是做一个通用 simulator 平台，而是先把当前实验跑通，那么最实用的路线不是直接照搬 RecSim，而是：

1. 参考 KuaiSim，把 `response model` 做成多头概率模型。
2. 参考 CWM / D2Q，把 `play ratio` 单独建模并去偏。
3. 参考 zr-obp，把“策略层评估”从“response 预测”里拆出去。
