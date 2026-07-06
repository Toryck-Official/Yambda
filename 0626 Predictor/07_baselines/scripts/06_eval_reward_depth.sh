#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${BASELINE_DIR}/configs/baselines.env"

MODEL_TYPE="${MODEL_TYPE:-sasrec_sid}"
case "$MODEL_TYPE" in
  sasrec_sid)
    CKPT="${CKPT:-${SASREC_OUT_DIR}/sasrec_sid.pt}"
    ;;
  hpn_sid)
    CKPT="${CKPT:-${HPN_OUT_DIR}/hpn.pt}"
    ;;
  hsrl_offline|hsrl_sid)
    MODEL_TYPE="hsrl_offline"
    CKPT="${CKPT:-${HSRL_SAVE_PREFIX}_actor}"
    ;;
  *)
    echo "Unsupported MODEL_TYPE=$MODEL_TYPE" >&2
    exit 2
    ;;
esac

mkdir -p "$BASELINE_EVAL_DIR" "$BASELINE_LOG_DIR"
SAVE_META="${SAVE_META:-${BASELINE_EVAL_DIR}/${MODEL_TYPE}_reward_depth_${ROLLOUT_SPLIT}_${ROLLOUT_EPISODES}.meta.json}"
LOG_PATH="${LOG_PATH:-${BASELINE_LOG_DIR}/${MODEL_TYPE}_reward_depth_${ROLLOUT_SPLIT}_${ROLLOUT_EPISODES}.log}"

cmd=(
  python3 -u "${BASELINE_DIR}/eval_reward_depth.py"
  --model_type "$MODEL_TYPE"
  --checkpoint "$CKPT"
  --transition_root "$TRANSITION_ROOT"
  --split "$ROLLOUT_SPLIT"
  --item_features_npy "$ITEM_FEATURES_NPY"
  --dense_item2sid_npy "$DENSE_ITEM2SID_NPY"
  --simulator_checkpoint "$ROLLOUT_SIMULATOR_CHECKPOINT"
  --num_episodes "$ROLLOUT_EPISODES"
  --batch_size "$ROLLOUT_BATCH_SIZE"
  --read_batch_size "$ROLLOUT_READ_BATCH_SIZE"
  --max_steps "$ROLLOUT_MAX_STEPS"
  --decode_top_k "$ROLLOUT_DECODE_TOP_K"
  --action_mode "$ROLLOUT_ACTION_MODE"
  --action_temperature "$ROLLOUT_ACTION_TEMPERATURE"
  --eval_reward_mode "$ROLLOUT_EVAL_REWARD_MODE"
  --failure_signal_scope "$ROLLOUT_FAILURE_SIGNAL_SCOPE"
  --negative_patience "$ROLLOUT_NEGATIVE_PATIENCE"
  --simulator_negative_prob_scale "$ROLLOUT_SIMULATOR_NEGATIVE_PROB_SCALE"
  --hsrl_project_root "$HSRL_PROJECT_ROOT"
  --hsrl_bootstrap_root "$HSRL_BOOTSTRAP_ROOT"
  --save_meta "$SAVE_META"
  --seed "$SEED"
  --device "$DEVICE"
)

if [[ "$ROLLOUT_SAMPLE_RESPONSE" == "0" || "$ROLLOUT_SAMPLE_RESPONSE" == "false" ]]; then
  cmd+=(--no-sample_response)
else
  cmd+=(--sample_response)
fi

if [[ "$ROLLOUT_STRUCTURED_SIMULATOR_RESPONSE" == "0" || "$ROLLOUT_STRUCTURED_SIMULATOR_RESPONSE" == "false" ]]; then
  cmd+=(--no-structured_simulator_response)
else
  cmd+=(--structured_simulator_response)
fi

if [[ -n "$ROLLOUT_REWARD_DONE_THRESHOLD" ]]; then
  cmd+=(--reward_done_threshold "$ROLLOUT_REWARD_DONE_THRESHOLD")
fi

"${cmd[@]}" "$@" 2>&1 | tee "$LOG_PATH"
