#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PREPROCESS_ROOT="${PREPROCESS_ROOT:-$ROOT/artifacts/preprocess}"

python3 "$ROOT/01_data/build_future_data.py" \
  --transition_root "$PREPROCESS_ROOT/session_run" \
  --out_dir "$ROOT/artifacts/smoke_future_data" \
  --dense2orig_npy "$PREPROCESS_ROOT/mappings/yambda_dense2orig_item_id.npy" \
  --dense_item2sid_npy "$PREPROCESS_ROOT/mappings/yambda_dense_item2sid.npy" \
  --history_len 50 \
  --future_horizon 3 \
  --max_rows 240 \
  --shard_rows 200 \
  --write_needed_items

python3 "$ROOT/01_data/build_embed_store.py" \
  --out_dir "$ROOT/artifacts/smoke_embed_store" \
  --needed_item_ids "$ROOT/artifacts/smoke_future_data/needed_item_ids.npy"

python3 "$ROOT/03_train/train_predictor.py" \
  --data_dir "$ROOT/artifacts/smoke_future_data" \
  --embed_store "$ROOT/artifacts/smoke_embed_store" \
  --out_dir "$ROOT/artifacts/smoke_predictor" \
  --epochs 1 \
  --batch_size 32 \
  --max_train_rows 160 \
  --max_val_rows 24 \
  --d_model 64 \
  --n_layer 1 \
  --n_head 4 \
  --max_seq_len 50 \
  --dropout 0.1 \
  --device cpu

python3 "$ROOT/03_train/train_value.py" \
  --data_dir "$ROOT/artifacts/smoke_future_data" \
  --embed_store "$ROOT/artifacts/smoke_embed_store" \
  --predictor_ckpt "$ROOT/artifacts/smoke_predictor/future_predictor.pt" \
  --out_dir "$ROOT/artifacts/smoke_value" \
  --epochs 1 \
  --batch_size 32 \
  --max_train_rows 160 \
  --max_val_rows 24 \
  --candidate_k 4 \
  --gamma 0.9 \
  --device cpu

python3 "$ROOT/04_eval/eval_predictor.py" \
  --data_dir "$ROOT/artifacts/smoke_future_data" \
  --embed_store "$ROOT/artifacts/smoke_embed_store" \
  --ckpt "$ROOT/artifacts/smoke_predictor/future_predictor.pt" \
  --split val \
  --batch_size 32 \
  --max_rows 24 \
  --device cpu

python3 "$ROOT/04_eval/eval_rerank.py" \
  --data_dir "$ROOT/artifacts/smoke_future_data" \
  --embed_store "$ROOT/artifacts/smoke_embed_store" \
  --predictor_ckpt "$ROOT/artifacts/smoke_predictor/future_predictor.pt" \
  --value_ckpt "$ROOT/artifacts/smoke_value/future_value.pt" \
  --split val \
  --batch_size 32 \
  --candidate_k 4 \
  --max_rows 24 \
  --gamma 0.9 \
  --device cpu
