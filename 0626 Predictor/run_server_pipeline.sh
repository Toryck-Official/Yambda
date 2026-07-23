#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

PREPROCESS_ROOT="${PREPROCESS_ROOT:-$ROOT/artifacts/preprocess}"
FUTURE_DATA="${FUTURE_DATA:-$PREPROCESS_ROOT/future_data}"
EMBED_STORE="${EMBED_STORE:-$PREPROCESS_ROOT/embed_store}"
DENSE_ITEM2SID="${DENSE_ITEM2SID:-$PREPROCESS_ROOT/mappings/yambda_dense_item2sid.npy}"
DENSE2ORIG="${DENSE2ORIG:-$PREPROCESS_ROOT/mappings/yambda_dense2orig_item_id.npy}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-3}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-0}"
MAX_VAL_ROWS="${MAX_VAL_ROWS:-100000}"
D_MODEL="${D_MODEL:-128}"
N_LAYER="${N_LAYER:-2}"
N_HEAD="${N_HEAD:-4}"
SID_LEVELS="${SID_LEVELS:-4}"
SID_VOCAB_SIZE="${SID_VOCAB_SIZE:-256}"
CANDIDATE_K="${CANDIDATE_K:-32}"
SAMPLE_M="${SAMPLE_M:-1}"
MAX_EVAL_ROWS="${MAX_EVAL_ROWS:-10000}"
HISTORY_LEN="${HISTORY_LEN:-50}"
RESPONSE_LOSS_WEIGHT="${RESPONSE_LOSS_WEIGHT:-1.0}"
PLAY_LOSS_WEIGHT="${PLAY_LOSS_WEIGHT:-0.1}"
REWARD_LOSS_WEIGHT="${REWARD_LOSS_WEIGHT:-0.1}"
REGRET_LOSS_WEIGHT="${REGRET_LOSS_WEIGHT:-0.1}"
CANDIDATE_LOSS_WEIGHT="${CANDIDATE_LOSS_WEIGHT:-0.1}"

python3 "$ROOT/03_train/train_hpn.py" \
  --data_dir "$FUTURE_DATA" \
  --embed_store "$EMBED_STORE" \
  --out_dir "$ROOT/artifacts/hpn" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --max_train_rows "$MAX_TRAIN_ROWS" \
  --max_val_rows "$MAX_VAL_ROWS" \
  --d_model "$D_MODEL" \
  --n_layer "$N_LAYER" \
  --n_head "$N_HEAD" \
  --max_seq_len "$HISTORY_LEN" \
  --sid_levels "$SID_LEVELS" \
  --sid_vocab_size "$SID_VOCAB_SIZE" \
  --device "$DEVICE"

python3 "$ROOT/03_train/train_predictor.py" \
  --data_dir "$FUTURE_DATA" \
  --embed_store "$EMBED_STORE" \
  --out_dir "$ROOT/artifacts/predictor" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --max_train_rows "$MAX_TRAIN_ROWS" \
  --max_val_rows "$MAX_VAL_ROWS" \
  --d_model "$D_MODEL" \
  --n_layer "$N_LAYER" \
  --n_head "$N_HEAD" \
  --max_seq_len "$HISTORY_LEN" \
  --response_loss_weight "$RESPONSE_LOSS_WEIGHT" \
  --play_loss_weight "$PLAY_LOSS_WEIGHT" \
  --reward_loss_weight "$REWARD_LOSS_WEIGHT" \
  --regret_loss_weight "$REGRET_LOSS_WEIGHT" \
  --device "$DEVICE"

python3 "$ROOT/04_eval/eval_predictor.py" \
  --data_dir "$FUTURE_DATA" \
  --embed_store "$EMBED_STORE" \
  --ckpt "$ROOT/artifacts/predictor/future_predictor.pt" \
  --split val \
  --batch_size "$BATCH_SIZE" \
  --max_rows "$MAX_EVAL_ROWS" \
  --device "$DEVICE"

python3 "$ROOT/03_train/train_value.py" \
  --data_dir "$FUTURE_DATA" \
  --embed_store "$EMBED_STORE" \
  --predictor_ckpt "$ROOT/artifacts/predictor/future_predictor.pt" \
  --hpn_ckpt "$ROOT/artifacts/hpn/hpn.pt" \
  --dense_item2sid_npy "$DENSE_ITEM2SID" \
  --dense2orig_npy "$DENSE2ORIG" \
  --out_dir "$ROOT/artifacts/value" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --max_train_rows "$MAX_TRAIN_ROWS" \
  --max_val_rows "$MAX_VAL_ROWS" \
  --candidate_source hpn \
  --candidate_k "$CANDIDATE_K" \
  --top_sid_paths "$CANDIDATE_K" \
  --sample_m "$SAMPLE_M" \
  --candidate_loss_weight "$CANDIDATE_LOSS_WEIGHT" \
  --device "$DEVICE"

python3 "$ROOT/04_eval/eval_hpn_future_rerank.py" \
  --data_dir "$FUTURE_DATA" \
  --embed_store "$EMBED_STORE" \
  --dense_item2sid_npy "$DENSE_ITEM2SID" \
  --dense2orig_npy "$DENSE2ORIG" \
  --hpn_ckpt "$ROOT/artifacts/hpn/hpn.pt" \
  --predictor_ckpt "$ROOT/artifacts/predictor/future_predictor.pt" \
  --value_ckpt "$ROOT/artifacts/value/future_value.pt" \
  --split val \
  --batch_size 64 \
  --max_rows "$MAX_EVAL_ROWS" \
  --top_sid_paths "$CANDIDATE_K" \
  --max_candidates "$CANDIDATE_K" \
  --device "$DEVICE"
