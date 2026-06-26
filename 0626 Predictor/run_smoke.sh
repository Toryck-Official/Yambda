#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

python3 "$ROOT/01_data/build_future_data.py" \
  --out_dir "$ROOT/artifacts/smoke_future_data" \
  --history_len 50 \
  --future_horizon 3 \
  --gamma 0.9 \
  --max_users 3 \
  --max_rows 240 \
  --shard_rows 200 \
  --write_needed_items \
  --split_mode row

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
