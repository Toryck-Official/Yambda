# Group-SNMPP Design & Minimal Validation Gate

## 1. Hypothesis

以 `uid + timestamp` group 作为唯一 temporal occurrence 后，SNMPP 的 signed、delay-aware 历史核能否分别学习下一组到达时间和下一组反馈构成，并在 validation 上达到简单模型或 Full-history GRU 的水平。

## 2. Data contract

- Time matched subset：303 users，30,258 train groups / 4,321 validation groups。
- Mark matched subset：2,000 users，101,241 train groups / 44,365 validation groups。
- group feature 为 4 类反馈各自的 mean SID semantic vector、反馈 composition、presence mask、`log(1+M)` 与 previous gap，共 522 维。
- 空反馈类型使用 zero semantic vector + zero mask；组内 permutation check 最大误差 6.532e-08。
- 没有组内排序、没有 event-level temporal source、没有截断 history、没有删除 burst、没有 listen、没有使用 test。

## 3. Model

Group-Time 每个历史 group 只产生一次 signed delayed influence，并经 positive link 得到 group-arrival hazard。Group-Mark 使用 composition-weighted `source feedback r -> target feedback k` 的 4x4 signed kernel；group size 只进入有界 context gate，不以 M 倍复制 temporal source。

Time 与 Mark 分开训练，Adam、LR=1e-3、3 seeds。正式 Time 训练使用冻结的 Q=64 协议：每个 segment 随机一点；validation 使用 deterministic midpoint。最初的 Q=16 exploratory run 只作为数值诊断，不参与最终模型判定。最大 30 epochs，patience=3。

## 4. Unit tests

positive/finite hazard、finite loss/gradient、Mark/Time group-source permutation invariance 全部通过。

## 5. Time results

| Method | Validation NLL/group | MAE (hour) | Median AE (hour) |
|---|---:|---:|---:|
| History-free log-normal | 2.2894 | 38.6666 | 0.8077 |
| Full-history GRU | 2.1542 | 37.6416 | 2.0480 |
| Group-Time SNMPP (3-seed mean, Q=64) | 2.2710 | 40.1990 | 12.7171 |

- Q=64 validation NLL seeds：2.2674 / 2.2763 / 2.2691。
- Best/stopped epochs：[30, 30, 30] / [30, 30, 30]。
- Validation hazard P99 最大 183.1064/hour，hazard max 203.2584/hour。
- Time psi（三 seed）为 [[1.4142009019851685, 0.9828118681907654, 0.47291791439056396, 0.727230429649353], [1.3833556175231934, 0.9308571815490723, 0.4621284306049347, 0.6785287857055664], [1.4170702695846558, 0.9870706796646118, 0.44012323021888733, 0.6881051659584045]]；是否全部为正：`true`。
- Time 结果不是数值 NaN，但是否通过由 Q=64 与 baseline 的逐 seed 比较决定。

## 6. Mark results

| Method | Validation CE/event |
|---|---:|
| Previous-group composition | 0.7285 |
| Last-group MLP | 0.6927 |
| Last-5 GRU | 0.5959 |
| Full-history GRU | 0.5577 |
| Group-Mark SNMPP | 0.7865 +/- 0.0110 |

- Group-Mark CE seeds：0.7983 / 0.7895 / 0.7718；与 Full-history GRU 的平均差为 +0.2289（越低越好）。
- Macro-F1：0.4918 / 0.4913 / 0.4913。
- Recall mean：like=0.9610，dislike=0.6070，unlike=0.3092，undislike=0.0012。
- argmax 预测分布（三 seed）：[[0.7953567001014313, 0.1636650512791615, 0.04097824861940719, 0.0], [0.7947706525414178, 0.16111799842217964, 0.04399864758255381, 0.00011270145384875464], [0.7981742364476502, 0.15983320184830385, 0.04187986025019723, 0.00011270145384875464]]。模型没有完全退化成单一类别，但几乎不预测 undislike。

## 7. Signed influence and cardinality diagnostics

- Mark 4x4 psi 在 16 个位置中有 14/16 个三 seed 同号；pairwise Pearson 为 0.704 / 0.680 / 0.966。
- 三个 seed 均同时存在正、负 psi；因此 `signed_influence_learned=true` 仅表示参数没有全正退化。
- 因 Mark predictive gate 失败，这些矩阵不能被当成已经验证的用户反馈规律，更不能作因果解释。

三 seed 平均 psi（行是 source，列是 target）：

| source \ target | like | dislike | unlike | undislike |
|---|---:|---:|---:|---:|
| like | +0.1327 | -0.1141 | +0.0914 | -0.1207 |
| dislike | -0.1326 | +0.1711 | -0.1506 | +0.1236 |
| unlike | +0.0129 | -0.1538 | +0.1470 | -0.0938 |
| undislike | -0.4073 | +0.0107 | -0.3748 | +0.1779 |

三 seed 平均 delay（hour，仅作参数诊断）：

| source \ target | like | dislike | unlike | undislike |
|---|---:|---:|---:|---:|
| like | 0.352 | 0.328 | 0.520 | 0.430 |
| dislike | 0.378 | 0.336 | 0.252 | 0.594 |
| unlike | 0.495 | 0.227 | 0.385 | 0.490 |
| undislike | 0.513 | 0.507 | 0.385 | 0.471 |

- Mark validation 历史中确有 >500 大组 source；其每 source 平均 absolute influence 相对 101-500 档的比例为 0.926 / 0.745 / 0.814，未呈现随 M 的机械线性爆炸。
- Time 的 303-user matched subset 没有 >500 source group，因此 Time 的 >500 cardinality 结论不可评估。另有清楚的 long-history accumulation：原 12-epoch checkpoint 的 history>500 slice hazard max 达约 220/hour；这是历史 group 数量累加问题，不是单个 group 内 M 条 event 重复求和。

## 8. Decision

- `group_time_snmpp_passed = false`
- `group_mark_snmpp_passed = false`
- `signed_influence_learned = true`
- `joint_group_snmpp_approved = false`

Group-Time 是否接近 GRU、是否超过 history-free baseline，以 Q=64 final values 为准。Group-Mark 虽能稳定优化并得到正负关系参数，但 CE 明显落后于所有主要 Mark baseline，故不批准 Joint。本轮按协议停止；未启动 Joint、Hierarchical SID、HPN、BOLA，也未查看 test。
