#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

CKPT=${CKPT:-artifacts/predictor_semantic_state_1m/future_predictor.pt}
VALUE_CKPT=${VALUE_CKPT:-artifacts/value/future_value.pt}

OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} \
MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} \
python3 04_eval/eval_hpn_future_rerank.py \
  --data_dir 01_data/processed/predictor_seq_data \
  --embed_store 01_data/processed/raw_rqkmeans \
  --dense_item2sid_npy 01_data/processed/raw_rqkmeans/dense_item2sid.npy \
  --dense2orig_npy 01_data/processed/raw_rqkmeans/dense2orig_item_id.npy \
  --hpn_ckpt artifacts/hpn/hpn.pt \
  --predictor_ckpt "$CKPT" \
  --value_ckpt "$VALUE_CKPT" \
  --split ${SPLIT:-val} \
  --batch_size ${BATCH_SIZE:-64} \
  --max_rows ${MAX_ROWS:-20000} \
  --top_sid_paths ${TOP_SID_PATHS:-32} \
  --branch_k ${BRANCH_K:-16} \
  --max_candidates ${MAX_CANDIDATES:-32} \
  --max_index_items ${MAX_INDEX_ITEMS:-0} \
  --candidate_chunk_size ${CANDIDATE_CHUNK_SIZE:-65536} \
  --hpn_score_weight ${HPN_SCORE_WEIGHT:-1.0} \
  --future_score_weight ${FUTURE_SCORE_WEIGHT:-0.0} \
  --candidate_logit_weight ${CANDIDATE_LOGIT_WEIGHT:-1.0} \
  --semantic_prefix_weight ${SEMANTIC_PREFIX_WEIGHT:-1.0} \
  --semantic_level_weights ${SEMANTIC_LEVEL_WEIGHTS:-1,0.7,0.5,0.3} \
  --device ${DEVICE:-cuda}
