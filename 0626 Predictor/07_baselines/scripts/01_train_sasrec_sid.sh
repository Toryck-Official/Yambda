#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${BASELINE_DIR}/configs/baselines.env"

mkdir -p "$SASREC_OUT_DIR" "$BASELINE_LOG_DIR"

python3 -u "$SASREC_TRAIN_ENTRY" \
  --data_dir "$DATA_DIR" \
  --embed_store "$EMBED_STORE" \
  --out_dir "$SASREC_OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --max_train_rows "$MAX_TRAIN_ROWS" \
  --max_val_rows "$MAX_VAL_ROWS" \
  --d_model "$D_MODEL" \
  --n_layer "$N_LAYER" \
  --n_head "$N_HEAD" \
  --dropout "$DROPOUT" \
  --state_pooling "$STATE_POOLING" \
  --sid_levels "$SID_LEVELS" \
  --sid_vocab_size "$SID_VOCAB_SIZE" \
  --seed "$SEED" \
  --device "$DEVICE" \
  "$@" \
  2>&1 | tee "${BASELINE_LOG_DIR}/sasrec_sid_train.log"
