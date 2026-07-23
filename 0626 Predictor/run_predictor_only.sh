#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DATA_DIR="${DATA_DIR:-$ROOT/01_data/processed/predictor_seq_data}"
EMBED_STORE="${EMBED_STORE:-$ROOT/01_data/processed/raw_rqkmeans}"
OUT_DIR="${OUT_DIR:-$ROOT/artifacts/predictor_only}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-1}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-1000000}"
MAX_VAL_ROWS="${MAX_VAL_ROWS:-50000}"
D_MODEL="${D_MODEL:-128}"
N_LAYER="${N_LAYER:-2}"
N_HEAD="${N_HEAD:-4}"
STATE_POOLING="${STATE_POOLING:-last_mean}"
RESPONSE_POS_WEIGHT="${RESPONSE_POS_WEIGHT:-1,10,40,60,200}"
RESPONSE_CLASS_WEIGHT="${RESPONSE_CLASS_WEIGHT:-1,2,4,4,2}"
REGRET_CLASS_WEIGHT="${REGRET_CLASS_WEIGHT:-1,1,20,30}"
RESPONSE_WEIGHT="${RESPONSE_WEIGHT:-1.0}"
PLAY_WEIGHT="${PLAY_WEIGHT:-0.5}"
REWARD_WEIGHT="${REWARD_WEIGHT:-1.0}"
REGRET_WEIGHT="${REGRET_WEIGHT:-1.0}"
FUTURE_RETURN_WEIGHT="${FUTURE_RETURN_WEIGHT:-1.0}"
FUTURE_REGRET_WEIGHT="${FUTURE_REGRET_WEIGHT:-1.0}"
FUTURE_REGRET_POS_WEIGHT="${FUTURE_REGRET_POS_WEIGHT:-50}"
CANDIDATE_CE_WEIGHT="${CANDIDATE_CE_WEIGHT:-1.0}"
CANDIDATE_CE_K="${CANDIDATE_CE_K:-8}"
CANDIDATE_NEGATIVE_MODE="${CANDIDATE_NEGATIVE_MODE:-history_semantic_inbatch}"
SEMANTIC_PREFIX_LEVEL="${SEMANTIC_PREFIX_LEVEL:-1}"

mkdir -p "$OUT_DIR" "$ROOT/artifacts/logs"

echo "[predictor-only] data=$DATA_DIR"
echo "[predictor-only] out=$OUT_DIR"
python3 "$ROOT/03_train/train_predictor.py" \
  --data_dir "$DATA_DIR" \
  --embed_store "$EMBED_STORE" \
  --out_dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --max_train_rows "$MAX_TRAIN_ROWS" \
  --max_val_rows "$MAX_VAL_ROWS" \
  --d_model "$D_MODEL" \
  --n_layer "$N_LAYER" \
  --n_head "$N_HEAD" \
  --state_pooling "$STATE_POOLING" \
  --response_pos_weight "$RESPONSE_POS_WEIGHT" \
  --response_class_weight "$RESPONSE_CLASS_WEIGHT" \
  --regret_class_weight "$REGRET_CLASS_WEIGHT" \
  --response_weight "$RESPONSE_WEIGHT" \
  --play_weight "$PLAY_WEIGHT" \
  --reward_weight "$REWARD_WEIGHT" \
  --regret_weight "$REGRET_WEIGHT" \
  --future_return_weight "$FUTURE_RETURN_WEIGHT" \
  --future_regret_weight "$FUTURE_REGRET_WEIGHT" \
  --future_regret_pos_weight "$FUTURE_REGRET_POS_WEIGHT" \
  --candidate_ce_weight "$CANDIDATE_CE_WEIGHT" \
  --candidate_ce_k "$CANDIDATE_CE_K" \
  --candidate_negative_mode "$CANDIDATE_NEGATIVE_MODE" \
  --semantic_prefix_level "$SEMANTIC_PREFIX_LEVEL" \
  --device "$DEVICE"
