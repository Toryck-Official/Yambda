#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${BASELINE_DIR}/configs/baselines.env"

mkdir -p "$HSRL_OUT_DIR" "$BASELINE_LOG_DIR"

python3 -u "$HSRL_TRAIN_ENTRY" \
  --train_mode offline_transition \
  --transition_root "$TRANSITION_ROOT" \
  --item_features_npy "$ITEM_FEATURES_NPY" \
  --dense_item2sid_npy "$DENSE_ITEM2SID_NPY" \
  --hpn_checkpoint "$HSRL_HPN_CHECKPOINT" \
  --save_path "$HSRL_SAVE_PREFIX" \
  --save_meta "${HSRL_SAVE_PREFIX}.meta.json" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --max_train_rows "$MAX_TRAIN_ROWS" \
  --max_val_rows "$MAX_VAL_ROWS" \
  --offline_epochs "$EPOCHS" \
  --train_sample_across_files \
  --sasrec_n_layer "$N_LAYER" \
  --sasrec_d_model 64 \
  --sasrec_d_forward 128 \
  --sasrec_n_head 4 \
  --sasrec_dropout "$DROPOUT" \
  "$@" \
  2>&1 | tee "${BASELINE_LOG_DIR}/hsrl_offline_train.log"
