#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DATA_ROOT="${DATA_ROOT:-$ROOT/01_data/processed}"
PREDICTOR_SEQ_DATA="$DATA_ROOT/predictor_seq_data"
DEFAULT_TRANSITIONS="$DATA_ROOT/regret_current_data"
if [ -d "$PREDICTOR_SEQ_DATA/train" ]; then
  DEFAULT_TRANSITIONS="$PREDICTOR_SEQ_DATA"
elif [ ! -d "$DEFAULT_TRANSITIONS/train" ]; then
  DEFAULT_TRANSITIONS="/root/autodl-tmp/0408Yambda/Regret/artifacts/current/data"
fi
FUTURE_DATA="${FUTURE_DATA:-$DEFAULT_TRANSITIONS}"
EMBED_STORE="${EMBED_STORE:-$DATA_ROOT/raw_rqkmeans}"
DENSE_ITEM2SID="${DENSE_ITEM2SID:-$DATA_ROOT/raw_rqkmeans/dense_item2sid.npy}"
DENSE2ORIG="${DENSE2ORIG:-$DATA_ROOT/raw_rqkmeans/dense2orig_item_id.npy}"

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
HPN_STATE_POOLING="${HPN_STATE_POOLING:-last_mean}"
HPN_CANDIDATE_CE_WEIGHT="${HPN_CANDIDATE_CE_WEIGHT:-1.0}"
HPN_CANDIDATE_CE_K="${HPN_CANDIDATE_CE_K:-32}"
HPN_CANDIDATE_CE_TEMPERATURE="${HPN_CANDIDATE_CE_TEMPERATURE:-1.0}"
HPN_CANDIDATE_NEGATIVE_MODE="${HPN_CANDIDATE_NEGATIVE_MODE:-semantic_hard}"
HPN_CANDIDATE_POOL_K="${HPN_CANDIDATE_POOL_K:-128}"
HPN_HARD_PREFIX_LEVELS="${HPN_HARD_PREFIX_LEVELS:-3,2,1}"
CANDIDATE_K="${CANDIDATE_K:-32}"
MAX_INDEX_ITEMS="${MAX_INDEX_ITEMS:-0}"
CANDIDATE_CHUNK_SIZE="${CANDIDATE_CHUNK_SIZE:-65536}"
SAMPLE_M="${SAMPLE_M:-1}"
BAYES_SAMPLES="${BAYES_SAMPLES:-$SAMPLE_M}"
REWARD_SAMPLE_MODE="${REWARD_SAMPLE_MODE:-sampled_formula}"
PREDICTOR_ENSEMBLE_SIZE="${PREDICTOR_ENSEMBLE_SIZE:-1}"
PREDICTOR_BASE_SEED="${PREDICTOR_BASE_SEED:-2026}"
RESPONSE_POS_WEIGHT="${RESPONSE_POS_WEIGHT:-}"
REGRET_CLASS_WEIGHT="${REGRET_CLASS_WEIGHT:-}"
MAX_EVAL_ROWS="${MAX_EVAL_ROWS:-10000}"

PREDICTOR_SINGLE_DIR="$ROOT/artifacts/predictor"
PREDICTOR_ENSEMBLE_DIR="$ROOT/artifacts/predictor_ensemble"
PREDICTOR_PRIMARY="$PREDICTOR_SINGLE_DIR/future_predictor.pt"
PREDICTOR_MANIFEST=""

PREDICTOR_WEIGHT_ARGS=()
if [ -n "$RESPONSE_POS_WEIGHT" ]; then
  PREDICTOR_WEIGHT_ARGS+=(--response_pos_weight "$RESPONSE_POS_WEIGHT")
fi
if [ -n "$REGRET_CLASS_WEIGHT" ]; then
  PREDICTOR_WEIGHT_ARGS+=(--regret_class_weight "$REGRET_CLASS_WEIGHT")
fi

VALUE_PREDICTOR_ARGS=()
EVAL_PREDICTOR_ARGS=()


echo "[data] FUTURE_DATA=$FUTURE_DATA"
echo "[stage 1/4] train HPN"
python3 "$ROOT/03_train/train_hpn.py" \
  --data_dir "$FUTURE_DATA" \
  --embed_store "$EMBED_STORE" \
  --dense_item2sid_npy "$DENSE_ITEM2SID" \
  --out_dir "$ROOT/artifacts/hpn" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --max_train_rows "$MAX_TRAIN_ROWS" \
  --max_val_rows "$MAX_VAL_ROWS" \
  --d_model "$D_MODEL" \
  --n_layer "$N_LAYER" \
  --n_head "$N_HEAD" \
  --sid_levels "$SID_LEVELS" \
  --sid_vocab_size "$SID_VOCAB_SIZE" \
  --state_pooling "$HPN_STATE_POOLING" \
  --candidate_ce_weight "$HPN_CANDIDATE_CE_WEIGHT" \
  --candidate_ce_k "$HPN_CANDIDATE_CE_K" \
  --candidate_ce_temperature "$HPN_CANDIDATE_CE_TEMPERATURE" \
  --candidate_negative_mode "$HPN_CANDIDATE_NEGATIVE_MODE" \
  --candidate_pool_k "$HPN_CANDIDATE_POOL_K" \
  --hard_prefix_levels "$HPN_HARD_PREFIX_LEVELS" \
  --device "$DEVICE"

if [ "$PREDICTOR_ENSEMBLE_SIZE" -gt 1 ]; then
  echo "[stage 2/4] train future predictor ensemble size=$PREDICTOR_ENSEMBLE_SIZE"
  python3 "$ROOT/03_train/train_predictor_ensemble.py" \
    --data_dir "$FUTURE_DATA" \
    --embed_store "$EMBED_STORE" \
    --out_dir "$PREDICTOR_ENSEMBLE_DIR" \
    --ensemble_size "$PREDICTOR_ENSEMBLE_SIZE" \
    --base_seed "$PREDICTOR_BASE_SEED" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --max_train_rows "$MAX_TRAIN_ROWS" \
    --max_val_rows "$MAX_VAL_ROWS" \
    --d_model "$D_MODEL" \
    --n_layer "$N_LAYER" \
    --n_head "$N_HEAD" \
    --device "$DEVICE" \
    "${PREDICTOR_WEIGHT_ARGS[@]}"
  PREDICTOR_MANIFEST="$PREDICTOR_ENSEMBLE_DIR/manifest.json"
  PREDICTOR_PRIMARY="$PREDICTOR_ENSEMBLE_DIR/member_00/future_predictor.pt"
  VALUE_PREDICTOR_ARGS=(--predictor_manifest "$PREDICTOR_MANIFEST" --predictor_ckpt "$PREDICTOR_PRIMARY")
  EVAL_PREDICTOR_ARGS=(--predictor_manifest "$PREDICTOR_MANIFEST" --predictor_ckpt "$PREDICTOR_PRIMARY")
else
  echo "[stage 2/4] train future predictor"
  python3 "$ROOT/03_train/train_predictor.py" \
    --data_dir "$FUTURE_DATA" \
    --embed_store "$EMBED_STORE" \
    --out_dir "$PREDICTOR_SINGLE_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --max_train_rows "$MAX_TRAIN_ROWS" \
    --max_val_rows "$MAX_VAL_ROWS" \
    --d_model "$D_MODEL" \
    --n_layer "$N_LAYER" \
    --n_head "$N_HEAD" \
    --device "$DEVICE" \
    "${PREDICTOR_WEIGHT_ARGS[@]}"
  VALUE_PREDICTOR_ARGS=(--predictor_ckpt "$PREDICTOR_PRIMARY")
  EVAL_PREDICTOR_ARGS=(--predictor_ckpt "$PREDICTOR_PRIMARY")
fi

echo "[stage 3/4] train value/reranker"
python3 "$ROOT/03_train/train_value.py" \
  --data_dir "$FUTURE_DATA" \
  --embed_store "$EMBED_STORE" \
  "${VALUE_PREDICTOR_ARGS[@]}" \
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
  --max_index_items "$MAX_INDEX_ITEMS" \
  --candidate_chunk_size "$CANDIDATE_CHUNK_SIZE" \
  --sample_m "$BAYES_SAMPLES" \
  --bayes_samples "$BAYES_SAMPLES" \
  --reward_sample_mode "$REWARD_SAMPLE_MODE" \
  --device "$DEVICE"

echo "[stage 4/4] eval HPN + predictor + value rerank"
python3 "$ROOT/04_eval/eval_hpn_future_rerank.py" \
  --data_dir "$FUTURE_DATA" \
  --embed_store "$EMBED_STORE" \
  --dense_item2sid_npy "$DENSE_ITEM2SID" \
  --dense2orig_npy "$DENSE2ORIG" \
  --hpn_ckpt "$ROOT/artifacts/hpn/hpn.pt" \
  "${EVAL_PREDICTOR_ARGS[@]}" \
  --value_ckpt "$ROOT/artifacts/value/future_value.pt" \
  --split val \
  --batch_size 64 \
  --max_rows "$MAX_EVAL_ROWS" \
  --top_sid_paths "$CANDIDATE_K" \
  --max_candidates "$CANDIDATE_K" \
  --max_index_items "$MAX_INDEX_ITEMS" \
  --candidate_chunk_size "$CANDIDATE_CHUNK_SIZE" \
  --bayes_samples "$BAYES_SAMPLES" \
  --reward_sample_mode "$REWARD_SAMPLE_MODE" \
  --device "$DEVICE"
