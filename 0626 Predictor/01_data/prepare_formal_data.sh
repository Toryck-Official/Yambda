#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/01_data/processed}"
LOG_DIR="${LOG_DIR:-$ROOT/01_data/logs}"

SOURCE_TRANSITIONS="${SOURCE_TRANSITIONS:-/root/autodl-tmp/0408Yambda/Regret/artifacts/current/data}"
SOURCE_MAPPING="${SOURCE_MAPPING:-/root/autodl-tmp/0408Yambda/Regret/artifacts/mappings/raw_rqkmeans}"

COPY_TRANSITIONS="${COPY_TRANSITIONS:-0}"
BUILD_FUTURE_DATA="${BUILD_FUTURE_DATA:-0}"
TRANSITION_ROOT="${TRANSITION_ROOT:-$SOURCE_TRANSITIONS}"
MAPPING_ROOT="$DATA_ROOT/raw_rqkmeans"
FUTURE_DATA="$DATA_ROOT/future_data"

HISTORY_LEN="${HISTORY_LEN:-50}"
FUTURE_HORIZON="${FUTURE_HORIZON:-5}"
GAMMA="${GAMMA:-0.9}"
REWARD_COLUMN="${REWARD_COLUMN:-reward_scaled}"
SHARD_ROWS="${SHARD_ROWS:-100000}"
MAX_ROWS="${MAX_ROWS:-0}"
SPLITS="${SPLITS:-train,val,test}"

mkdir -p "$DATA_ROOT" "$LOG_DIR" "$MAPPING_ROOT"

echo "[1/3] copying RQKMeans mapping/features to $MAPPING_ROOT"
for name in dense_item2sid.npy dense_item_features.npy dense2orig_item_id.npy orig2dense_item_id.npy mapping.meta.json; do
  if [ ! -f "$MAPPING_ROOT/$name" ]; then
    cp -a "$SOURCE_MAPPING/$name" "$MAPPING_ROOT/$name"
  fi
done

if [ "$COPY_TRANSITIONS" = "1" ]; then
  LOCAL_TRANSITIONS="$DATA_ROOT/regret_current_data"
  echo "[2/3] copying Regret transitions to $LOCAL_TRANSITIONS"
  if [ ! -d "$LOCAL_TRANSITIONS" ]; then
    cp -a "$SOURCE_TRANSITIONS" "$LOCAL_TRANSITIONS"
  fi
  TRANSITION_ROOT="$LOCAL_TRANSITIONS"
else
  echo "[2/3] using source transitions in place: $TRANSITION_ROOT"
fi

cat > "$DATA_ROOT/formal_data_manifest.json" <<EOF
{
  "transition_root": "$TRANSITION_ROOT",
  "mapping_root": "$MAPPING_ROOT",
  "embed_store": "$MAPPING_ROOT",
  "reader": "FutureIterableDataset direct Regret-transition mode",
  "history_len": $HISTORY_LEN,
  "future_horizon": $FUTURE_HORIZON,
  "gamma": $GAMMA,
  "reward_column": "$REWARD_COLUMN",
  "copy_transitions": "$COPY_TRANSITIONS",
  "build_future_data": "$BUILD_FUTURE_DATA"
}
EOF

if [ "$BUILD_FUTURE_DATA" = "1" ]; then
  echo "[3/3] converting transitions to predictor future_data at $FUTURE_DATA"
  python3 "$ROOT/01_data/convert_regret_transitions.py" \
    --transition_root "$TRANSITION_ROOT" \
    --mapping_root "$MAPPING_ROOT" \
    --out_dir "$FUTURE_DATA" \
    --splits "$SPLITS" \
    --history_len "$HISTORY_LEN" \
    --future_horizon "$FUTURE_HORIZON" \
    --gamma "$GAMMA" \
    --reward_column "$REWARD_COLUMN" \
    --shard_rows "$SHARD_ROWS" \
    --max_rows "$MAX_ROWS" 2>&1 | tee "$LOG_DIR/formal_preprocess.log"
else
  echo "[3/3] skipping full future_data materialization; training will read Regret transitions directly"
  echo "[done] manifest: $DATA_ROOT/formal_data_manifest.json" | tee "$LOG_DIR/formal_preprocess.log"
fi

echo "[done] data root: $DATA_ROOT"
echo "[done] train data: $TRANSITION_ROOT"
echo "[done] embed_store: $MAPPING_ROOT"
