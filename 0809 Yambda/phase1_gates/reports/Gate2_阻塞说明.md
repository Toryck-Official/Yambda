# Gate 2 阻塞说明

日期：2026-08-10

状态：未启动最终 RQKMeans/SID materialization。

Gate 2 的输入契约要求：

```text
real items -> real embedding
metadata items -> Gate 1 confirmed proxy rule
cold items -> unknown/global centroid
```

Gate 1 技术判断为 C，没有确认 metadata proxy 可作为四层 semantic SID 来源。因此当前缺少 450,594 个 metadata-available items 的获准表示，不能完成“全部 2,875,071 items 的实际 collision audit”。

以下行为都将改变已确认协议，当前未执行：

- 强制把失败的 proxy rule 送入 frozen quantizer；
- 把 450,594 metadata items 静默改成 cold_unknown；
- 从 explicit item universe 删除这些 item。

另外，当前运行环境没有可见 GPU，CPU memory limit 为 2 GiB。旧 `ResidualKMeans.fit_encode` 会同时持有 full feature matrix 和 residual copy，仅两个 2,367,341×128 float32 矩阵就超过 2.25 GiB，不含 FAISS workspace，无法运行。200k provisional fit 已实际触发资源终止，100k fit 才成功。

因此 Gate 2 同时存在：

1. 科学阻塞：没有 Gate-1-confirmed proxy rule；
2. 工程阻塞：当前资源不能运行旧式 full-matrix RQKMeans。

需要用户先决定 missing-item representation；随后应提供 GPU，或明确授权实现与旧全局 FAISS fit 不同的 streaming/minibatch RQKMeans。未经确认不改算法。

