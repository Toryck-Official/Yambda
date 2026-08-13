#!/usr/bin/env python3
"""Post-hoc numerical and source-cardinality audits for frozen checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from group_snmpp_design_gate.run_gate import (  # noqa: E402
    WORK, SEEDS, TIME_SELECTION, MARK_SELECTION, GroupDataset,
    GroupTimeSNMPP, GroupMarkSNMPP, loader, prepare, split_features,
    sequence_mask, save_json,
)


SOURCE_SIZE_BINS = [(1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"),
                    (21, 100, "21-100"), (101, 500, "101-500"),
                    (501, np.iinfo(np.int64).max, ">500")]


def new_bin_rows() -> dict:
    return {label: {"source_occurrences": 0, "absolute_sum": 0.0, "positive_sum": 0.0,
                    "negative_sum": 0.0, "max_absolute": 0.0}
            for _, _, label in SOURCE_SIZE_BINS}


def update_bins(rows: dict, source_size: np.ndarray, absolute: np.ndarray,
                positive: np.ndarray, negative: np.ndarray, valid: np.ndarray) -> None:
    for lower, upper, label in SOURCE_SIZE_BINS:
        mask = valid & (source_size >= lower) & (source_size <= upper)
        if not mask.any():
            continue
        row = rows[label]
        row["source_occurrences"] += int(mask.sum())
        row["absolute_sum"] += float(absolute[mask].sum())
        row["positive_sum"] += float(positive[mask].sum())
        row["negative_sum"] += float(negative[mask].sum())
        row["max_absolute"] = max(row["max_absolute"], float(absolute[mask].max()))


def finish_bins(rows: dict) -> dict:
    for row in rows.values():
        count = max(row["source_occurrences"], 1)
        row["absolute_mean_per_source_occurrence"] = row["absolute_sum"] / count
        row["positive_mean_per_source_occurrence"] = row["positive_sum"] / count
        row["negative_mean_per_source_occurrence"] = row["negative_sum"] / count
    return rows


@torch.no_grad()
def time_audit(device: torch.device) -> dict:
    _, validation, _ = prepare(TIME_SELECTION, "time_posthoc_Q64")
    dataset = GroupDataset(validation)
    results = []
    for seed in SEEDS:
        model = GroupTimeSNMPP().to(device)
        state = torch.load(WORK / f"runs/time_seed_{seed}/group_time_snmpp_best.pt", map_location=device, weights_only=True)
        model.load_state_dict(state); model.eval()
        nll16, nll64, bins = [], [], new_bin_rows()
        for batch in loader(dataset, 2026, False):
            gpu = {k: v.to(device) for k, v in batch.items() if k not in {"index", "previous_size"}}
            nll16.append(model.nll(gpu, q=16).cpu().numpy())
            nll64.append(model.nll(gpu, q=64).cpu().numpy())
            amplitude, _ = model.source_amplitude(gpu)
            _, _, _, log_size, _ = split_features(gpu["sequence"])
            valid = sequence_mask(gpu["lengths"], gpu["sequence"].shape[1])
            absolute = amplitude.abs().sum(dim=2)
            positive = amplitude.clamp_min(0).sum(dim=2)
            negative = amplitude.clamp_max(0).sum(dim=2)
            update_bins(bins, torch.expm1(log_size).round().cpu().numpy(), absolute.cpu().numpy(),
                        positive.cpu().numpy(), negative.cpu().numpy(), valid.cpu().numpy())
        q16, q64 = np.concatenate(nll16), np.concatenate(nll64)
        difference = q16 - q64
        results.append({
            "seed": seed,
            "validation_targets": int(len(q64)),
            "Q16_nll_per_group": float(q16.mean()),
            "Q64_nll_per_group": float(q64.mean()),
            "Q16_minus_Q64_mean": float(difference.mean()),
            "absolute_difference_mean": float(np.abs(difference).mean()),
            "absolute_difference_p95": float(np.percentile(np.abs(difference), 95)),
            "absolute_difference_p99": float(np.percentile(np.abs(difference), 99)),
            "absolute_difference_max": float(np.abs(difference).max()),
            "source_group_size_influence": finish_bins(bins),
        })
    return {"evaluation_reference": "deterministic midpoint Q=64 on complete matched validation", "seeds": results}


@torch.no_grad()
def mark_audit(device: torch.device) -> dict:
    _, validation, _ = prepare(MARK_SELECTION, "mark_posthoc_source_size")
    dataset = GroupDataset(validation)
    results = []
    for seed in SEEDS:
        model = GroupMarkSNMPP().to(device)
        state = torch.load(WORK / f"runs/mark_seed_{seed}/group_mark_snmpp_best.pt", map_location=device, weights_only=True)
        model.load_state_dict(state); model.eval(); bins = new_bin_rows()
        for batch in loader(dataset, 2026, False):
            gpu = {k: v.to(device) for k, v in batch.items() if k not in {"index", "previous_size"}}
            _, contribution = model(gpu, True)
            _, _, _, log_size, _ = split_features(gpu["sequence"])
            valid = sequence_mask(gpu["lengths"], gpu["sequence"].shape[1])
            absolute = contribution.abs().sum(dim=(2, 3))
            positive = contribution.clamp_min(0).sum(dim=(2, 3))
            negative = contribution.clamp_max(0).sum(dim=(2, 3))
            update_bins(bins, torch.expm1(log_size).round().cpu().numpy(), absolute.cpu().numpy(),
                        positive.cpu().numpy(), negative.cpu().numpy(), valid.cpu().numpy())
        results.append({"seed": seed, "source_group_size_influence": finish_bins(bins)})
    return {"seeds": results}


def matrix_stability() -> dict:
    matrices, delays = [], []
    for seed in SEEDS:
        metrics = json.loads((WORK / f"runs/mark_seed_{seed}/metrics.json").read_text())
        matrices.append(np.asarray(metrics["interaction"]["psi"], dtype=np.float64))
        delays.append(np.asarray(metrics["interaction"]["delay_hours"], dtype=np.float64))
    matrices, delays = np.asarray(matrices), np.asarray(delays)
    pairs = []
    for left in range(3):
        for right in range(left + 1, 3):
            pairs.append({
                "seeds": [SEEDS[left], SEEDS[right]],
                "psi_pearson": float(np.corrcoef(matrices[left].ravel(), matrices[right].ravel())[0, 1]),
                "psi_sign_agreement": float(np.mean(np.sign(matrices[left]) == np.sign(matrices[right]))),
                "delay_pearson": float(np.corrcoef(delays[left].ravel(), delays[right].ravel())[0, 1]),
                "delay_MAE_hours": float(np.mean(np.abs(delays[left] - delays[right]))),
            })
    unanimous = np.all(np.sign(matrices) == np.sign(matrices[:1]), axis=0)
    return {
        "feedback_order": ["like", "dislike", "unlike", "undislike"],
        "psi_mean": matrices.mean(axis=0).tolist(),
        "psi_std": matrices.std(axis=0).tolist(),
        "delay_mean_hours": delays.mean(axis=0).tolist(),
        "delay_std_hours": delays.std(axis=0).tolist(),
        "unanimous_sign_mask": unanimous.tolist(),
        "unanimous_sign_fraction": float(unanimous.mean()),
        "pairwise": pairs,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = {"time": time_audit(device), "mark": mark_audit(device), "interaction_stability": matrix_stability()}
    save_json(WORK / "posthoc_numerical_cardinality_audit.json", payload)
    print(json.dumps({"POSTHOC_COMPLETE": True,
                      "Q64_time_nll": [r["Q64_nll_per_group"] for r in payload["time"]["seeds"]],
                      "psi_unanimous_sign_fraction": payload["interaction_stability"]["unanimous_sign_fraction"]}))


if __name__ == "__main__":
    main()
