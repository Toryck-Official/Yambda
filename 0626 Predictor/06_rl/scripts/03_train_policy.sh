#!/usr/bin/env bash
# 用途：训练主线策略。流程是 HPN 生成语义 item，simulator 给反馈，RRCA 修正有效 reward，再更新 actor/critic。
# 默认输出 artifacts/models/policy_hsac_main_*；如需另存，设置 POLICY_SAVE_PREFIX。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REGRET_ROOT}/configs/main.env"
cd "$REGRET_ROOT"
mkdir -p artifacts/logs artifacts/models

learn_weight_args=()
if [[ "${LEARN_REWARD_WEIGHTS:-0}" == "1" ]]; then
  learn_weight_args+=(--learn_reward_weights)
fi

rapi_args=()
if [[ "${TRAIN_USE_RAPI:-1}" == "1" ]]; then
  rapi_args+=(--use_rapi)
else
  rapi_args+=(--no-use_rapi)
fi

rapi_candidate_args=()
if [[ "${RAPI_CANDIDATE_RERANK:-1}" == "1" ]]; then
  rapi_candidate_args+=(--rapi_candidate_rerank)
else
  rapi_candidate_args+=(--no-rapi_candidate_rerank)
fi

future_predictor_args=()
if [[ "${USE_FUTURE_PREDICTOR:-0}" == "1" ]]; then
  future_predictor_args+=(
    --use_future_predictor
    --predictor_checkpoint "$PREDICTOR_CHECKPOINT"
    --predictor_beta "$PREDICTOR_BETA"
    --predictor_regret_eta "$PREDICTOR_REGRET_ETA"
    --predictor_candidate_k "$PREDICTOR_CANDIDATE_K"
    --predictor_score_mode "$PREDICTOR_SCORE_MODE"
    --predictor_score_norm "$PREDICTOR_SCORE_NORM"
  )
  if [[ "${CACHE_ITEM_FEATURES_ON_GPU:-0}" == "1" ]]; then
    future_predictor_args+=(--cache_item_features_on_gpu)
  else
    future_predictor_args+=(--no-cache_item_features_on_gpu)
  fi
else
  future_predictor_args+=(--no-use_future_predictor)
fi

model_predictor_args=()
if [[ "${USE_MODEL_BASED_PREDICTOR:-0}" == "1" ]]; then
  model_predictor_args+=(
    --use_model_based_predictor
    --future_value_checkpoint "$FUTURE_VALUE_CHECKPOINT"
    --model_predictor_checkpoint "${MODEL_PREDICTOR_CHECKPOINT:-}"
    --model_q_weight "$MODEL_Q_WEIGHT"
    --model_q_gamma "$MODEL_Q_GAMMA"
    --model_q_clip "$MODEL_Q_CLIP"
    --model_q_mode "$MODEL_Q_MODE"
  )
  if [[ "${CACHE_ITEM_FEATURES_ON_GPU:-0}" == "1" ]]; then
    model_predictor_args+=(--cache_item_features_on_gpu)
  else
    model_predictor_args+=(--no-cache_item_features_on_gpu)
  fi
else
  model_predictor_args+=(--no-use_model_based_predictor)
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

reward_done_args=()
if [[ -n "${HSAC_REWARD_DONE_THRESHOLD:-}" ]]; then
  reward_done_args+=(--reward_done_threshold "$HSAC_REWARD_DONE_THRESHOLD")
fi

sample_across_files_args=()
if [[ "${HSAC_SAMPLE_ACROSS_FILES:-1}" == "1" ]]; then
  sample_across_files_args+=(--sample_across_files)
else
  sample_across_files_args+=(--no-sample_across_files)
fi

python3 -u scripts/10_train_hsac_simulator_rollout.py \
  --transition_root "$TRANSITION_ROOT" \
  --split "${SPLIT:-train}" \
  --item_features_npy "$DENSE_ITEM_FEATURES_NPY" \
  --dense_item2sid_npy "$DENSE_ITEM2SID_NPY" \
  --actor_init_checkpoint "$ACTOR_INIT_CHECKPOINT" \
  --simulator_checkpoint "$SIMULATOR_CHECKPOINT" \
  --save_prefix "$POLICY_SAVE_PREFIX" \
  --save_meta "$POLICY_SAVE_META" \
  --device "${DEVICE:-cuda}" \
  --episodes "$HSAC_EPISODES" \
  --batch_size "$HSAC_BATCH_SIZE" \
  --read_batch_size "$HSAC_READ_BATCH_SIZE" \
  "${sample_across_files_args[@]}" \
  --shuffle_buffer_size "${HSAC_SHUFFLE_BUFFER_SIZE:-0}" \
  --epochs "$HSAC_EPOCHS" \
  --max_steps "$HSAC_MAX_STEPS" \
  --actor_lr "$HSAC_ACTOR_LR" \
  --critic_lr "$HSAC_CRITIC_LR" \
  --gamma "$HSAC_GAMMA" \
  --target_tau "${HSAC_TARGET_TAU:-0.05}" \
  --entropy_weight "${HSAC_ENTROPY_WEIGHT:-0.001}" \
  --critic_type "$HSAC_CRITIC_TYPE" \
  --offline_aux_weight "$HSAC_OFFLINE_AUX_WEIGHT" \
  --offline_aux_reward_threshold "$HSAC_OFFLINE_AUX_REWARD_THRESHOLD" \
  --offline_aux_temperature "$HSAC_OFFLINE_AUX_TEMPERATURE" \
  --offline_aux_bc_weight "$HSAC_OFFLINE_AUX_BC_WEIGHT" \
  --offline_aux_avoid_weight "$HSAC_OFFLINE_AUX_AVOID_WEIGHT" \
  --offline_aux_entropy_weight "$HSAC_OFFLINE_AUX_ENTROPY_WEIGHT" \
  --decode_top_k "$HSAC_DECODE_TOP_K" \
  --action_mode "${HSAC_ACTION_MODE:-sample}" \
  --action_temperature "${HSAC_ACTION_TEMPERATURE:-1.0}" \
  "${future_predictor_args[@]}" \
  "${model_predictor_args[@]}" \
  "${structured_sim_args[@]}" \
  --simulator_negative_prob_scale "$SIMULATOR_NEGATIVE_PROB_SCALE" \
  --failure_signal_scope "$FAILURE_SIGNAL_SCOPE" \
  --negative_patience "$HSAC_NEGATIVE_PATIENCE" \
  "${reward_done_args[@]}" \
  "${rapi_args[@]}" \
  --regret_pool_size "${REGRET_POOL_SIZE:-20}" \
  --regret_gamma "${REGRET_GAMMA:-0.9}" \
  --regret_phi_scale "${REGRET_PHI_SCALE:-1.0}" \
  --regret_phi_clip "${REGRET_PHI_CLIP:-2.0}" \
  --memory_signal_scope "$MEMORY_SIGNAL_SCOPE" \
  --sara_eta "$HSAC_SARA_ETA" \
  "${rapi_candidate_args[@]}" \
  --rapi_candidate_eta "${RAPI_CANDIDATE_ETA:-1.0}" \
  --base_reward_mode "$HSAC_BASE_REWARD_MODE" \
  --reward_w_listen "$REWARD_W_LISTEN" \
  --reward_w_like "$REWARD_W_LIKE" \
  --reward_w_dislike "$REWARD_W_DISLIKE" \
  --rrca_unlike_weight "$RRCA_UNLIKE_WEIGHT" \
  --rrca_undislike_weight "$RRCA_UNDISLIKE_WEIGHT" \
  --rrca_apply_to "$RRCA_APPLY_TO" \
  --rrca_signal_scope "$RRCA_SIGNAL_SCOPE" \
  --rrca_callback_target "$RRCA_CALLBACK_TARGET" \
  --learn_reward_init "$LEARN_REWARD_INIT" \
  --reward_weight_lr "$REWARD_WEIGHT_LR" \
  --reward_weight_loss_weight "$REWARD_WEIGHT_LOSS_WEIGHT" \
  "${learn_weight_args[@]}" \
  "${revision_gate_args[@]}" \
  "$@" \
  2>&1 | tee "${REGRET_ROOT}/artifacts/logs/${MAIN_ID}_policy_train.log"
