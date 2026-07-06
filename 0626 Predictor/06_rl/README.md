# 06_rl

This directory contains the independent RL recommendation mainline copied from
`0408Yambda/Regret` and adapted to this project.

## Scope

`06_rl` is responsible for:

- splitting raw Yambda events into session/consecutive-item RL transitions;
- training the user-response simulator;
- training the HSAC/HPN policy with RRCA and RAPI;
- evaluating simulator rollouts;
- sweeping RAPI eta.

It intentionally does not contain predictor-only training code. Predictor,
value, and offline rerank experiments stay in:

```text
02_model/
03_train/
04_eval/
```

## Main Scripts

```text
scripts/01_split_data.sh
  Build RL transitions from raw Yambda events.

scripts/02_train_simulator.sh
  Train the user-response simulator.

scripts/03_train_policy.sh
  Train the HSAC/HPN policy with simulator rollout, RRCA, and RAPI.

scripts/04_eval_policy.sh
  Evaluate base vs RAPI rollout reward and interaction depth.

scripts/05_sweep_eta.sh
  Sweep RAPI intervention strength.
```

The implementation package copied from Regret lives in:

```text
regret_core/
```

## Default Data Contract

The default config reads local project data:

```text
../01_data/processed/raw_rqkmeans
../01_data/processed/regret_current_data
```

Prepare those files with:

```bash
cd "/root/autodl-tmp/0626/0626 Predictor"
COPY_TRANSITIONS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./01_data/prepare_formal_data.sh
```

## Artifact Contract

RL outputs stay local to this module:

```text
06_rl/artifacts/
```

This directory is ignored by git. The copied initial actor checkpoint is:

```text
06_rl/artifacts/init/regret_sid_session_run_v2_actor
```

No symlinks are required.
