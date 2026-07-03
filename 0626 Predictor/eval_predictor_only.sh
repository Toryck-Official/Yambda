#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DATA_DIR="${DATA_DIR:-$ROOT/01_data/processed/predictor_seq_data}"
EMBED_STORE="${EMBED_STORE:-$ROOT/01_data/processed/raw_rqkmeans}"
CKPT="${CKPT:-$ROOT/artifacts/predictor_only/future_predictor.pt}"
SPLIT="${SPLIT:-val}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_ROWS="${MAX_ROWS:-50000}"
CANDIDATE_K="${CANDIDATE_K:-8}"
CANDIDATE_NEGATIVE_MODE="${CANDIDATE_NEGATIVE_MODE:-history_semantic_inbatch}"
SEMANTIC_PREFIX_LEVEL="${SEMANTIC_PREFIX_LEVEL:-1}"

mkdir -p "$ROOT/artifacts/logs"

echo "[predictor-only eval] head metrics"
python3 "$ROOT/04_eval/eval_predictor.py" \
  --data_dir "$DATA_DIR" \
  --embed_store "$EMBED_STORE" \
  --ckpt "$CKPT" \
  --split "$SPLIT" \
  --batch_size "$BATCH_SIZE" \
  --max_rows "$MAX_ROWS" \
  --device "$DEVICE"

echo "[predictor-only eval] oracle candidate contrast"
python3 "$ROOT/04_eval/eval_predictor_oracle_contrast.py" \
  --data_dir "$DATA_DIR" \
  --embed_store "$EMBED_STORE" \
  --ckpt "$CKPT" \
  --split "$SPLIT" \
  --batch_size "$BATCH_SIZE" \
  --max_rows "$MAX_ROWS" \
  --candidate_k "$CANDIDATE_K" \
  --candidate_negative_mode "$CANDIDATE_NEGATIVE_MODE" \
  --semantic_prefix_level "$SEMANTIC_PREFIX_LEVEL" \
  --device "$DEVICE"
