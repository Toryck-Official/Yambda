#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${BASELINE_DIR}/configs/baselines.env"

MODEL_TYPE="${MODEL_TYPE:-sasrec_sid}"
mkdir -p "$BASELINE_EVAL_DIR" "$BASELINE_LOG_DIR"

case "$MODEL_TYPE" in
  sasrec_sid)
    CKPT="${CKPT:-${SASREC_OUT_DIR}/sasrec_sid.pt}"
    ;;
  hpn_sid)
    CKPT="${CKPT:-${HPN_OUT_DIR}/hpn.pt}"
    ;;
  hsrl_sid)
    CKPT="${CKPT:-${HSRL_SAVE_PREFIX}_actor}"
    ;;
  *)
    echo "Unsupported MODEL_TYPE=$MODEL_TYPE" >&2
    exit 2
    ;;
esac

SAVE_META="${SAVE_META:-${BASELINE_EVAL_DIR}/${MODEL_TYPE}_${EVAL_SPLIT}_${EVAL_MAX_ROWS}.meta.json}"
LOG_PATH="${LOG_PATH:-${BASELINE_LOG_DIR}/${MODEL_TYPE}_${EVAL_SPLIT}_${EVAL_MAX_ROWS}_eval.log}"

python3 -u "${BASELINE_DIR}/eval_sid_ranking.py" \
  --model_type "$MODEL_TYPE" \
  --checkpoint "$CKPT" \
  --data_dir "$DATA_DIR" \
  --embed_store "$EMBED_STORE" \
  --dense_item2sid_npy "$DENSE_ITEM2SID_NPY" \
  --split "$EVAL_SPLIT" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --max_rows "$EVAL_MAX_ROWS" \
  --candidate_k "$EVAL_CANDIDATE_K" \
  --semantic_prefix "$EVAL_SEMANTIC_PREFIX" \
  --k_list "$EVAL_K_LIST" \
  --hsrl_project_root "$HSRL_PROJECT_ROOT" \
  --hsrl_bootstrap_root "$HSRL_BOOTSTRAP_ROOT" \
  --save_meta "$SAVE_META" \
  --seed "$SEED" \
  --device "$DEVICE" \
  "$@" \
  2>&1 | tee "$LOG_PATH"

