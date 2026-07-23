#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"

EMBEDDINGS_PARQUET="${EMBEDDINGS_PARQUET:-/Users/Toryck/Coding/DATASET/Yambda/embeddings.parquet}"
MULTI_EVENT_PARQUET="${MULTI_EVENT_PARQUET:-/Users/Toryck/Coding/DATASET/Yambda/sequential/50m/multi_event.parquet}"
OUT_ROOT="${OUT_ROOT:-$ROOT/artifacts/preprocess}"

SAMPLE_SIZE="${SAMPLE_SIZE:-200000}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
SID_LEVELS="${SID_LEVELS:-4}"
MAX_ITER="${MAX_ITER:-30}"
DEVICE="${DEVICE:-cuda}"

HISTORY_LEN="${HISTORY_LEN:-50}"
FUTURE_HORIZON="${FUTURE_HORIZON:-5}"
GAMMA="${GAMMA:-0.9}"
MAX_USERS="${MAX_USERS:-0}"
MAX_ROWS="${MAX_ROWS:-0}"
SESSION_GAP_SECONDS="${SESSION_GAP_SECONDS:-3600}"
MAX_SESSION_SPAN_SECONDS="${MAX_SESSION_SPAN_SECONDS:-21600}"
MAX_RUN_EVENTS="${MAX_RUN_EVENTS:-100}"
TIMESTAMP_UNIT_SECONDS="${TIMESTAMP_UNIT_SECONDS:-5}"

mkdir -p "$OUT_ROOT/codebook" "$OUT_ROOT/mappings" "$OUT_ROOT/session_run" "$OUT_ROOT/future_data" "$OUT_ROOT/embed_store"

python3 "$REPO_ROOT/01_build_codebook.py" \
  --embeddings_parquet "$EMBEDDINGS_PARQUET" \
  --embedding_column normalized_embed \
  --sample_size "$SAMPLE_SIZE" \
  --n_levels "$SID_LEVELS" \
  --codebook_size "$CODEBOOK_SIZE" \
  --max_iter "$MAX_ITER" \
  --device "$DEVICE" \
  --output_npz "$OUT_ROOT/codebook/yambda_rq_codebook.npz" \
  --output_meta "$OUT_ROOT/codebook/yambda_rq_codebook.meta.json"

python3 "$REPO_ROOT/02_build_item_sid.py" \
  --embeddings_parquet "$EMBEDDINGS_PARQUET" \
  --codebook_npz "$OUT_ROOT/codebook/yambda_rq_codebook.npz" \
  --embedding_column normalized_embed \
  --output_dir "$OUT_ROOT/mappings" \
  --output_prefix yambda

python3 "$REPO_ROOT/Regret/scripts/02_split_transitions.py" \
  --multi_event_parquet "$MULTI_EVENT_PARQUET" \
  --orig2dense_npy "$OUT_ROOT/mappings/yambda_orig2dense_item_id.npy" \
  --out_root "$OUT_ROOT/session_run" \
  --history_len "$HISTORY_LEN" \
  --future_horizon "$FUTURE_HORIZON" \
  --gamma "$GAMMA" \
  --trajectory_mode session_run \
  --session_gap_seconds "$SESSION_GAP_SECONDS" \
  --max_session_span_seconds "$MAX_SESSION_SPAN_SECONDS" \
  --max_run_events "$MAX_RUN_EVENTS" \
  --timestamp_unit_seconds "$TIMESTAMP_UNIT_SECONDS" \
  --anchor_policy first_visible \
  --max_users "$MAX_USERS" \
  --reward_version v2

python3 "$ROOT/01_data/build_future_data.py" \
  --transition_root "$OUT_ROOT/session_run" \
  --out_dir "$OUT_ROOT/future_data" \
  --dense2orig_npy "$OUT_ROOT/mappings/yambda_dense2orig_item_id.npy" \
  --dense_item2sid_npy "$OUT_ROOT/mappings/yambda_dense_item2sid.npy" \
  --history_len "$HISTORY_LEN" \
  --future_horizon "$FUTURE_HORIZON" \
  --max_rows "$MAX_ROWS" \
  --shard_rows 50000 \
  --write_needed_items

python3 "$ROOT/01_data/build_embed_store.py" \
  --embeddings "$EMBEDDINGS_PARQUET" \
  --out_dir "$OUT_ROOT/embed_store" \
  --column normalized_embed \
  --needed_item_ids "$OUT_ROOT/future_data/needed_item_ids.npy"

echo "[done] preprocess artifacts written to $OUT_ROOT"
