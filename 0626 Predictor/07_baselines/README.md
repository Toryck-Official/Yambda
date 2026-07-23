# Baseline Comparison

This folder keeps baseline comparison separate from the predictor / RAPI pipeline in `06_rl`.

The main baseline protocol is reward/depth rollout:

- train each baseline with its own offline training script;
- evaluate all baselines in the same learned `RegretUserResponseEnv`;
- keep future predictor and RAPI disabled;
- report reward/depth as the main result.

Offline SID ranking is kept as an auxiliary sanity check. It is useful for checking whether a model can rank the logged next item, but it is not the main experiment metric.

## Scope

| Method | Training data | Training simulator | Predictor | RAPI | Main metric |
|---|---|---|---|---|---|
| `sasrec_sid` | `predictor_seq_data` | no | no | no | reward/depth |
| `hpn_sid` | `predictor_seq_data` | no | no | no | reward/depth |
| `hsrl_offline` | `regret_current_data` | no | no | no | reward/depth |

Online HAC is intentionally not part of the first baseline table because it reintroduces an environment during training. It can be added later as a separate "environment-trained" comparison.

## Runbook

Train SASRec-SID:

```bash
cd "/root/autodl-tmp/0626/0626 Predictor"
screen -dmS bl_sasrec bash -lc 'cd "/root/autodl-tmp/0626/0626 Predictor" && ./07_baselines/scripts/01_train_sasrec_sid.sh'
tail -f "/root/autodl-tmp/0626/0626 Predictor/07_baselines/artifacts/logs/sasrec_sid_train.log"
```

Train HPN-SID:

```bash
cd "/root/autodl-tmp/0626/0626 Predictor"
screen -dmS bl_hpn bash -lc 'cd "/root/autodl-tmp/0626/0626 Predictor" && ./07_baselines/scripts/02_train_hpn_sid.sh'
tail -f "/root/autodl-tmp/0626/0626 Predictor/07_baselines/artifacts/logs/hpn_sid_train.log"
```

Train HSRL offline:

```bash
cd "/root/autodl-tmp/0626/0626 Predictor"
screen -dmS bl_hsrl bash -lc 'cd "/root/autodl-tmp/0626/0626 Predictor" && ./07_baselines/scripts/03_train_hsrl_offline.sh'
tail -f "/root/autodl-tmp/0626/0626 Predictor/07_baselines/artifacts/logs/hsrl_offline_train.log"
```

The HSRL training entrypoints copied from the original `/root/autodl-tmp/0626/HSRL` project live in `07_baselines/hsrl_upstream/`. The active offline baseline uses `hsrl_upstream/06_train_yambda_sid.py --train_mode offline_transition`.

Evaluate one model with auxiliary SID ranking:

```bash
MODEL_TYPE=sasrec_sid ./07_baselines/scripts/04_eval_sid_ranking.sh
MODEL_TYPE=hpn_sid ./07_baselines/scripts/04_eval_sid_ranking.sh
MODEL_TYPE=hsrl_sid ./07_baselines/scripts/04_eval_sid_ranking.sh
```

Collect ranking JSON files into one table:

```bash
python3 ./07_baselines/scripts/05_collect_ranking_results.py
```

Evaluate one model with the main reward/depth rollout:

```bash
MODEL_TYPE=sasrec_sid ROLLOUT_EPISODES=200 ./07_baselines/scripts/06_eval_reward_depth.sh
MODEL_TYPE=hpn_sid ROLLOUT_EPISODES=200 ./07_baselines/scripts/06_eval_reward_depth.sh
MODEL_TYPE=hsrl_offline ROLLOUT_EPISODES=200 ./07_baselines/scripts/06_eval_reward_depth.sh
```

Evaluate all main baselines:

```bash
ROLLOUT_EPISODES=200 ./07_baselines/scripts/07_eval_all_reward_depth.sh
ROLLOUT_EPISODES=1000 ./07_baselines/scripts/07_eval_all_reward_depth.sh
```

Collect reward/depth JSON files into one table:

```bash
python3 ./07_baselines/scripts/08_collect_reward_depth.py
```

## Notes

- `sasrec_sid` is a plain SASRec-style state encoder with independent SID heads.
- `hpn_sid` uses the existing `FutureHPNPolicy`, including hierarchical residual SID credit.
- `hsrl_offline` uses HSRL `SIDPolicy_credit` with reward-gated offline updates.
- SASRec and HPN training entrypoints are local files: `train_sasrec_sid.py` and `train_hpn_sid.py`.
- The main result table should compare `avg_cum_reward`, `avg_step`, `reward_per_step`, `negative_rate`, and `failure_rate`.
- All methods must use the same rollout settings before writing conclusions.
