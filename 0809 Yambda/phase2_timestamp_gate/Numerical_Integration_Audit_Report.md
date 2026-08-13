# SNMPP Phase 2 Numerical Integration Accuracy Audit

状态：训练前数值检查完成；Minimal SNMPP Pilot 获准，但本轮未启动训练。

## 1. 审计契约

- 只使用 D_SID train split；未查看 validation/test；
- 338 个固定 interval，5 个 gap 档、6 个 previous-group-size 档，30 个交叉单元全部非空；
- 所有 Q 使用相同模型参数、history、target interval 和 intensity function；
- 参数未训练；积分全部使用 deterministic midpoint rule；
- Q=32 是用户指定的临时 reference。因其与 Q=64 差异仍明显，追加全样本 Q=128 和 30 条代表样本 Q=256 sanity；
- 未删除或归一化 `>500` burst。

## 2. Q 对照结果

下表误差均相对全样本 Q=128；相对误差以比例表示，括号中换算成百分比。

| Q | Integral RelErr P50 | P95 | P99 | Max | Runtime / 338 intervals | Throughput | Cost vs Q4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.0391% | 31.1818% | 49.7759% | 73.8244% | 0.128s | 2643.4/s | 0.56× |
| 4 | 0.0239% | 28.5585% | 42.6946% | 49.7692% | 0.228s | 1484.7/s | 1.00× |
| 8 | 0.0126% | 17.8384% | 25.7133% | 46.7947% | 0.431s | 783.5/s | 1.89× |
| 16 | 0.0061% | 6.7465% | 16.0312% | 29.6832% | 0.819s | 412.8/s | 3.60× |
| 32 | 0.0024% | 1.9873% | 6.3483% | 16.2557% | 1.605s | 210.6/s | 7.05× |
| 64 | 0.00056% | 0.4309% | 1.5222% | 4.5108% | 3.192s | 105.9/s | 14.02× |
| 128 | reference | reference | reference | reference | 6.305s | 53.6/s | 27.69× |

Time-NLL difference 与 integral error 数值相同，因为 target-time intensity 在所有 Q 下完全一致，Q 只改变积分项。Q=4 相对用户指定 Q=32 的 relative integral error 为 P50 0.0199%、P95 25.5881%、P99 41.8282%、最大 45.2224%；因此即使只用 Q=32 作 reference，Q=4 也明确不合格。

## 3. Q=32 reference sanity

Q=32 相对 Q=64 的尾部差异仍明显，因此没有把 Q=32 当最终 reference。追加的 30 条代表样本中，Q=128 相对 Q=256 为：

| 指标 | Relative error |
|---|---:|
| P50 | 0.00014% |
| P95 | 0.2468% |
| P99 | 0.9761% |
| Max | 1.2565% |

Q=128 已基本稳定，可作为本轮工程 reference；Q=256 不用于大规模配置。

## 4. Q=4 压力切片

### Previous group size

| Slice | RelErr P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|
| previous group ≤500 | 0.0189% | 3.6496% | 5.9159% | 8.8612% |
| previous group >500 | 19.6552% | 43.3248% | 47.4853% | 49.7692% |

Q=4 在普通历史上也存在尾部误差，在 extreme burst history 上明显失稳，不能冻结。

### Gap

短 gap（≤P50）中 Q=4 的 P99 仅 0.0266%，但 P90–P95 gap 的 P95/P99 达 39.5543%/43.9094%，P95–P99 gap 的 P95/P99 达 26.3295%/36.9380%。因此总体中位数很小不能证明 Q=4 可靠。

Q=4 在 low/high target intensity 与 low/high absolute influence mass 切片均出现显著尾部误差；问题不是由单一强度二分即可隔离，而主要集中在 interval 长度与大历史组共同造成的快速区间变化。

## 5. 为什么冻结 Q=64

Q=32 在 `previous group >500` 上仍有 P95 7.6109%、P99 13.2342%、最大 16.2557%，不批准。

Q=64 相对 Q=128：

- 全样本 P95 0.4309%、P99 1.5222%、最大 4.5108%；
- previous group ≤500：P95 0.0989%、P99 0.2525%、最大 1.0393%；
- previous group >500：P95 1.7382%、P99 3.6020%、最大 4.5108%；
- P95–P99 gap：P95 1.3163%、P99 2.0482%；
- P99–29天 gap：P95 0.1480%、P99 2.8098%。

Q=128 会把尾部误差进一步降低，但成本从 Q=64 的 14.02×Q4 上升到 27.69×Q4，收益主要位于极少数 extreme-burst 尾部。Q=64 是本轮第一个在普通样本达到亚百分比 P95、同时将 burst P95 压到约 2% 以下的配置，因此是更合理的精度/成本平衡。

## 6. 冻结状态

```text
numerical_integration_validated = true
recommended_integration_Q = 64
minimal_snmpp_pilot_approved = true
minimal_snmpp_training_started = false
```

未来配置：训练期每个 segment 内取一个分层随机点；validation/test 使用 deterministic midpoint。该规则沿用 v1.1 数学协议，本轮只冻结 Q，不改变模型、数据或 likelihood。
