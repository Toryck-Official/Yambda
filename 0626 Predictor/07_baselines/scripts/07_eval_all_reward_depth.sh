#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${BASELINE_DIR}/configs/baselines.env"

for model in $BASELINE_ROLLOUT_MODELS; do
  echo "[run] reward/depth eval MODEL_TYPE=${model} episodes=${ROLLOUT_EPISODES} split=${ROLLOUT_SPLIT}"
  MODEL_TYPE="$model" "${SCRIPT_DIR}/06_eval_reward_depth.sh" "$@"
done
