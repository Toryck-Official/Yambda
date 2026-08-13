#!/usr/bin/env python3
"""Complete the fixed Group-Time optimization run to a defensible stopping point.

This is not a hyper-parameter search: it preserves the original data, model,
optimizer, LR, Q=16 training rule and three seeds, changing only the maximum
epoch cap from 12 (where all runs were still improving) to 30.  Final
validation likelihood is additionally recomputed with deterministic Q=64.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from group_snmpp_design_gate import run_gate as gate


def q64_validation(model: gate.GroupTimeSNMPP, data: gate.RichData, device: torch.device) -> float:
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in gate.loader(gate.GroupDataset(data), 2026, False):
            gpu = {k: v.to(device) for k, v in batch.items() if k not in {"index", "previous_size"}}
            values.append(model.nll(gpu, q=64).cpu().numpy())
    return float(np.concatenate(values).mean())


def main() -> None:
    gate.MAX_EPOCHS = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, validation, manifest = gate.prepare(gate.TIME_SELECTION, "time_matched_303_user_extended_convergence")
    runs = []
    for seed in gate.SEEDS:
        model, metrics, _ = gate.fit_time(train, validation, seed, device)
        metrics["validation_Q64_nll_per_group"] = q64_validation(model, validation, device)
        run_dir = gate.WORK / "runs" / f"time_extended_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), run_dir / "group_time_snmpp_best.pt")
        gate.save_json(run_dir / "metrics.json", gate.clean_metric_payload(metrics))
        runs.append(metrics)
    payload = {
        "purpose": "extend fixed Group-Time run because all original runs hit epoch 12 while still improving",
        "protocol_changes": {"max_epochs": {"old": 12, "new": 30}},
        "protocol_unchanged": {
            "data": True, "selection": True, "model": True, "optimizer": "Adam",
            "learning_rate": gate.LEARNING_RATE, "training_integration_Q": gate.INTEGRATION_Q,
            "seeds": gate.SEEDS, "history_truncation": False, "burst_removed": False,
            "test_used": False,
        },
        "manifest": manifest,
        "runs": gate.clean_metric_payload(runs),
        "summary": {
            "Q16_validation_nll_values": [r["validation"]["nll_per_group"] for r in runs],
            "Q16_validation_nll_mean": float(np.mean([r["validation"]["nll_per_group"] for r in runs])),
            "Q64_validation_nll_values": [r["validation_Q64_nll_per_group"] for r in runs],
            "Q64_validation_nll_mean": float(np.mean([r["validation_Q64_nll_per_group"] for r in runs])),
            "validation_mae_hours_mean": float(np.mean([r["validation"]["mae_hours"] for r in runs])),
            "validation_median_ae_hours_mean": float(np.mean([r["validation"]["median_ae_hours"] for r in runs])),
            "best_epochs": [r["best_epoch"] for r in runs],
            "stopped_epochs": [r["stopped_epoch"] for r in runs],
            "lambda_p99_max_across_seeds": float(max(r["validation"]["lambda_p99"] for r in runs)),
            "lambda_max_across_seeds": float(max(r["validation"]["lambda_max"] for r in runs)),
        },
    }
    gate.save_json(gate.WORK / "time_extended_metrics.json", payload)
    print(json.dumps({"TIME_EXTENSION_COMPLETE": True, **payload["summary"]}), flush=True)


if __name__ == "__main__":
    main()
