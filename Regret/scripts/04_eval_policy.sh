#!/usr/bin/env bash
# 用途：评估策略。比较 base 生成和 RAPI 介入后的 simulator rollout reward、depth、negative rate。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
USER_TRANSITION_ROOT="${TRANSITION_ROOT:-}"
USER_SIMULATOR_CHECKPOINT="${SIMULATOR_CHECKPOINT:-}"
USER_POLICY_ACTOR="${POLICY_ACTOR:-}"
USER_ACTOR_CHECKPOINT="${ACTOR_CHECKPOINT:-}"
USER_MAIN_ID="${MAIN_ID:-}"
source "${REGRET_ROOT}/configs/main.env"
[[ -n "$USER_TRANSITION_ROOT" ]] && TRANSITION_ROOT="$USER_TRANSITION_ROOT"
[[ -n "$USER_SIMULATOR_CHECKPOINT" ]] && SIMULATOR_CHECKPOINT="$USER_SIMULATOR_CHECKPOINT"
[[ -n "$USER_POLICY_ACTOR" ]] && POLICY_ACTOR="$USER_POLICY_ACTOR"
[[ -n "$USER_ACTOR_CHECKPOINT" ]] && ACTOR_CHECKPOINT="$USER_ACTOR_CHECKPOINT"
[[ -n "$USER_MAIN_ID" ]] && MAIN_ID="$USER_MAIN_ID"
cd "$REGRET_ROOT"
mkdir -p artifacts/evals artifacts/logs

SPLIT="${SPLIT:-test}"
EPISODES="${ROLLOUT_EPISODES:-1000}"
ACTOR_CHECKPOINT="${ACTOR_CHECKPOINT:-$POLICY_ACTOR}"
SAVE_META="${SAVE_META:-${REGRET_ROOT}/artifacts/evals/${MAIN_ID}_eta010_${SPLIT}_${EPISODES}.meta.json}"
LOG_PATH="${LOG_PATH:-${REGRET_ROOT}/artifacts/logs/${MAIN_ID}_eta010_${SPLIT}_${EPISODES}.log}"

if [[ -z "${OMP_NUM_THREADS:-}" || "${OMP_NUM_THREADS}" -lt 1 ]]; then
  export OMP_NUM_THREADS=1
fi

revision_gate_args=()
if [[ "${GATE_REVISION_BY_HISTORY:-1}" == "1" ]]; then
  revision_gate_args+=(--gate_revision_by_history)
else
  revision_gate_args+=(--no-gate_revision_by_history)
fi

structured_sim_args=()
if [[ "${STRUCTURED_SIMULATOR_RESPONSE:-1}" == "1" ]]; then
  structured_sim_args+=(--structured_simulator_response)
else
  structured_sim_args+=(--no-structured_simulator_response)
fi

precomputed_memory_args=()
if [[ "${EVAL_USE_PRECOMPUTED_REGRET_MEMORY:-1}" == "1" ]]; then
  precomputed_memory_args+=(--use_precomputed_regret_memory)
else
  precomputed_memory_args+=(--no-use_precomputed_regret_memory)
fi

history_memory_fallback_args=()
if [[ "${EVAL_HISTORY_MEMORY_FALLBACK:-1}" == "1" ]]; then
  history_memory_fallback_args+=(--history_memory_fallback)
else
  history_memory_fallback_args+=(--no-history_memory_fallback)
fi

rapi_candidate_args=()
if [[ "${RAPI_CANDIDATE_RERANK:-1}" == "1" ]]; then
  rapi_candidate_args+=(--rapi_candidate_rerank)
else
  rapi_candidate_args+=(--no-rapi_candidate_rerank)
fi

python3 -u scripts/09_eval_simulator_rollout.py \
  --transition_root "$TRANSITION_ROOT" \
  --split "$SPLIT" \
  --item_features_npy "$DENSE_ITEM_FEATURES_NPY" \
  --dense_item2sid_npy "$DENSE_ITEM2SID_NPY" \
  --actor_checkpoint "$ACTOR_CHECKPOINT" \
  --simulator_checkpoint "$SIMULATOR_CHECKPOINT" \
  --save_meta "$SAVE_META" \
  --device "${DEVICE:-cuda}" \
  --max_seq_len "${SID_MAX_SEQ_LEN:-50}" \
  --num_episodes "$EPISODES" \
  --batch_size "${ROLLOUT_BATCH_SIZE:-64}" \
  --read_batch_size "${ROLLOUT_READ_BATCH_SIZE:-2048}" \
  --max_steps "$ROLLOUT_MAX_STEPS" \
  --sample_response \
  --negative_patience "${ROLLOUT_NEGATIVE_PATIENCE:-5}" \
  --eval_reward_mode "${EVAL_REWARD_MODE:-paper_effective}" \
  --reward_w_listen "$REWARD_W_LISTEN" \
  --reward_w_like "$REWARD_W_LIKE" \
  --reward_w_dislike "$REWARD_W_DISLIKE" \
  --rrca_unlike_weight "$RRCA_UNLIKE_WEIGHT" \
  --rrca_undislike_weight "$RRCA_UNDISLIKE_WEIGHT" \
  --decode_top_k "$ROLLOUT_DECODE_TOP_K" \
  --action_mode "${ROLLOUT_ACTION_MODE:-sample}" \
  --action_temperature "${ROLLOUT_ACTION_TEMPERATURE:-1.0}" \
  "${structured_sim_args[@]}" \
  --simulator_negative_prob_scale "$SIMULATOR_NEGATIVE_PROB_SCALE" \
  --failure_signal_scope "$FAILURE_SIGNAL_SCOPE" \
  --sara_eta "$SARA_ETA" \
  --sara_layer_weights "${SARA_LAYER_WEIGHTS:-0.05,0.25,0.70}" \
  "${rapi_candidate_args[@]}" \
  --rapi_candidate_eta "${RAPI_CANDIDATE_ETA:-1.0}" \
  --regret_pool_size "${REGRET_POOL_SIZE:-20}" \
  --regret_gamma "${REGRET_GAMMA:-0.9}" \
  --regret_phi_scale "${REGRET_PHI_SCALE:-1.0}" \
  --regret_phi_clip "${REGRET_PHI_CLIP:-2.0}" \
  --regret_reward_threshold "${REGRET_REWARD_THRESHOLD:-0.0}" \
  --memory_signal_scope "$MEMORY_SIGNAL_SCOPE" \
  "${precomputed_memory_args[@]}" \
  "${history_memory_fallback_args[@]}" \
  "${revision_gate_args[@]}" \
  "$@" \
  2>&1 | tee "$LOG_PATH"
