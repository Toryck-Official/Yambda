#!/usr/bin/env python3
"""Protocol-faithful Group-Time SNMPP training with Q=64 integration.

Training uses one random point in each of 64 equal interval segments;
validation uses deterministic segment midpoints.  No model/data change is made.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from group_snmpp_design_gate import run_gate as gate  # noqa: E402


Q = 64
MAX_EPOCHS = 30


def random_stratified_nll(model: gate.GroupTimeSNMPP, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    amplitude, _ = model.source_amplitude(batch)
    gap = batch["gap"]
    target_hazard, _ = model.hazard(amplitude, batch["age0"], gap[:, None])
    segment = torch.arange(Q, device=gap.device, dtype=gap.dtype)[None]
    points = gap[:, None] * (segment + torch.rand((len(gap), Q), device=gap.device, dtype=gap.dtype)) / Q
    grid_hazard, _ = model.hazard(amplitude, batch["age0"], points)
    integral = gap * grid_hazard.mean(dim=1)
    return -torch.log(target_hazard[:, 0]) + integral


@torch.no_grad()
def evaluate(model: gate.GroupTimeSNMPP, data: gate.RichData, device: torch.device) -> tuple[dict, dict]:
    model.eval()
    records = []
    for batch in gate.loader(gate.GroupDataset(data), 2026, False):
        indices = batch["index"].numpy()
        previous_size = batch["previous_size"].numpy()
        gpu = {k: v.to(device) for k, v in batch.items() if k not in {"index", "previous_size"}}
        nll, detail = model.nll(gpu, q=Q, return_diagnostics=True)
        median, mass = model.predictive_median(gpu)
        records.append((indices, nll.cpu().numpy(), median.cpu().numpy(), mass.cpu().numpy(),
                        detail["target_hazard"].cpu().numpy(), detail["target_signed"].cpu().numpy(),
                        detail["absolute_amplitude"].cpu().numpy(), previous_size))
    order = np.concatenate([row[0] for row in records])
    sorting = np.argsort(order)
    arrays = [np.concatenate([row[column] for row in records])[sorting] for column in range(1, 8)]
    nll, median, mass, hazard, signed, absolute, previous_size = arrays
    error = np.abs(median - data.gaps)
    metrics = {
        "nll_per_group": float(nll.mean()),
        "mae_hours": float(error.mean()),
        "median_ae_hours": float(np.median(error)),
        "horizon_event_mass_mean": float(mass.mean()),
        "horizon_event_mass_p10": float(np.percentile(mass, 10)),
        "lambda_median": float(np.median(hazard)),
        "lambda_p95": float(np.percentile(hazard, 95)),
        "lambda_p99": float(np.percentile(hazard, 99)),
        "lambda_max": float(hazard.max()),
        "finite": bool(all(np.isfinite(value).all() for value in arrays)),
    }
    rows = {"nll": nll, "median": median, "error": error, "mass": mass, "hazard": hazard,
            "signed": signed, "absolute_amplitude": absolute, "previous_size": previous_size}
    return metrics, rows


def fit(train: gate.RichData, validation: gate.RichData, seed: int, device: torch.device):
    gate.seed_all(seed)
    model = gate.GroupTimeSNMPP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=gate.LEARNING_RATE)
    train_dataset, validation_dataset = gate.GroupDataset(train), gate.GroupDataset(validation)
    train_loader = gate.loader(train_dataset, seed, True)
    best_state, best_nll, best_epoch, stale = None, float("inf"), 0, 0
    curves = []
    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.perf_counter()
        model.train(); numerator = denominator = 0.0; nonfinite = 0
        for batch in train_loader:
            gpu = {k: v.to(device) for k, v in batch.items() if k not in {"index", "previous_size"}}
            optimizer.zero_grad(set_to_none=True)
            loss = random_stratified_nll(model, gpu).mean()
            if not torch.isfinite(loss):
                nonfinite += 1
                continue
            loss.backward(); optimizer.step()
            numerator += float(loss.detach()) * len(gpu["gap"]); denominator += len(gpu["gap"])
        validation_metrics, _ = evaluate(model, validation, device)
        value = validation_metrics["nll_per_group"]
        improved = value < best_nll - gate.MIN_DELTA
        if improved:
            best_state = copy.deepcopy(model.state_dict()); best_nll = value; best_epoch = epoch; stale = 0
        else:
            stale += 1
        row = {"epoch": epoch, "train_random_Q64_nll": numerator / denominator,
               "validation_deterministic_Q64": validation_metrics, "improved": improved,
               "nonfinite_steps": nonfinite, "seconds": time.perf_counter() - started}
        curves.append(row)
        print(json.dumps({"task": "group_time_Q64", "seed": seed, "epoch": epoch,
                          "train": row["train_random_Q64_nll"], "validation": value,
                          "best_epoch": best_epoch, "stale": stale, "seconds": row["seconds"]}), flush=True)
        if stale >= gate.PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("no finite Group-Time Q64 checkpoint")
    model.load_state_dict(best_state)
    train_metrics, _ = evaluate(model, train, device)
    validation_metrics, validation_rows = evaluate(model, validation, device)
    result = {
        "seed": seed, "best_epoch": best_epoch, "stopped_epoch": curves[-1]["epoch"],
        "train": train_metrics, "validation": validation_metrics,
        "interaction": {
            "psi": model.psi.detach().cpu().tolist(),
            "decay": (torch.nn.functional.softplus(model.raw_decay) + 1e-4).detach().cpu().tolist(),
            "delay_hours": torch.nn.functional.softplus(model.raw_delay).detach().cpu().tolist(),
            "base_raw": float(model.base.detach()),
        },
        "curves": curves,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, result, validation_rows


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, validation, manifest = gate.prepare(gate.TIME_SELECTION, "time_matched_303_user_Q64_protocol")
    runs, rows = [], []
    for seed in gate.SEEDS:
        model, metrics, validation_rows = fit(train, validation, seed, device)
        run_dir = gate.WORK / "runs" / f"time_q64_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), run_dir / "group_time_snmpp_best.pt")
        gate.save_json(run_dir / "metrics.json", gate.clean_metric_payload(metrics))
        runs.append(metrics); rows.append(validation_rows)
    payload = {
        "protocol": {
            "training_integration": "Q=64 stratified random, one point per segment",
            "validation_integration": "Q=64 deterministic midpoint",
            "max_epochs": MAX_EPOCHS, "patience": gate.PATIENCE, "learning_rate": gate.LEARNING_RATE,
            "optimizer": "Adam", "model_changed": False, "data_changed": False,
            "history_truncation": False, "burst_removed": False, "test_used": False,
        },
        "manifest": manifest,
        "runs": gate.clean_metric_payload(runs),
        "summary": {
            "validation_nll_values": [run["validation"]["nll_per_group"] for run in runs],
            "validation_nll_mean": float(np.mean([run["validation"]["nll_per_group"] for run in runs])),
            "validation_mae_hours_values": [run["validation"]["mae_hours"] for run in runs],
            "validation_mae_hours_mean": float(np.mean([run["validation"]["mae_hours"] for run in runs])),
            "validation_median_ae_hours_values": [run["validation"]["median_ae_hours"] for run in runs],
            "validation_median_ae_hours_mean": float(np.mean([run["validation"]["median_ae_hours"] for run in runs])),
            "best_epochs": [run["best_epoch"] for run in runs],
            "stopped_epochs": [run["stopped_epoch"] for run in runs],
            "lambda_p99_max_across_seeds": float(max(run["validation"]["lambda_p99"] for run in runs)),
            "lambda_max_across_seeds": float(max(run["validation"]["lambda_max"] for run in runs)),
        },
        "validation_slices": gate.clean_metric_payload(gate.time_slices(validation, rows)),
    }
    gate.save_json(gate.WORK / "time_q64_metrics.json", payload)
    print(json.dumps({"TIME_Q64_COMPLETE": True, **payload["summary"]}), flush=True)


if __name__ == "__main__":
    main()
