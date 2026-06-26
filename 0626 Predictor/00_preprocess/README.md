# 00 预处理

目标是为完整链路准备四类产物：

```text
1. codebook
2. dense item 映射和 SID
3. future predictor 训练样本
4. predictor 用的 embedding store
```

默认脚本：

```bash
./00_preprocess/run_preprocess.sh
```

本机只建议 smoke：

```bash
SAMPLE_SIZE=2000 \
CODEBOOK_SIZE=32 \
SID_LEVELS=3 \
MAX_ITER=3 \
DEVICE=cpu \
MAX_USERS=5 \
MAX_ROWS=1000 \
SPLIT_MODE=row \
./00_preprocess/run_preprocess.sh
```

服务器建议从较小配置开始：

```bash
SAMPLE_SIZE=200000 \
CODEBOOK_SIZE=256 \
SID_LEVELS=4 \
MAX_ITER=30 \
DEVICE=cuda \
MAX_USERS=0 \
MAX_ROWS=0 \
SPLIT_MODE=user \
./00_preprocess/run_preprocess.sh
```

输出默认在：

```text
artifacts/preprocess/
```

注意：

```text
artifacts/ 已被 .gitignore 排除，不会推到 GitHub。
```
