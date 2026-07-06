# HSRL Upstream Training Entrypoints

This folder keeps local copies of the original HSRL training entrypoints used by the baseline comparison.

Copied from `/root/autodl-tmp/0626/HSRL`:

- `04_train_hpn_warmstart.py`
- `05_train_user_response.py`
- `06_train_yambda_sid.py`

For the current baseline table, `06_train_yambda_sid.py --train_mode offline_transition` is the active entrypoint. The copied scripts still reuse the original HSRL core package from `${HSRL_PROJECT_ROOT}/hsrl_core` and the workspace adapter from `/root/autodl-tmp/0626/adapter`; only the runnable stage entrypoints are localized here to make the baseline runbook easier to manage.
