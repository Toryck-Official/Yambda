#!/usr/bin/env python3
"""Run the authorized Tiny Overfit sanity and decide whether Stage B may start."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimal_snmpp_pilot.data import FullHistoryTargetDataset, make_loader
from minimal_snmpp_pilot.model import MinimalSNMPP
from minimal_snmpp_pilot.training_utils import (
    evaluate_feedback_and_loss,
    initialize_constant_rate,
    integration_noise,
    parameter_change,
    parameter_snapshot,
    save_json,
    seed_everything,
    train_epoch,
)


def main() -> None:
    work = ROOT / "minimal_snmpp_pilot"
    config = json.loads((work / "pilot_config.json").read_text())
    seed = int(config["stage_a"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FullHistoryTargetDataset(work / "data/pilot_sequences.npz", "tiny_train_targets")
    train_loader = make_loader(dataset, batch_size=int(config["stage_a"]["batch_size"]), seed=seed, shuffle=True)
    eval_loader = make_loader(dataset, batch_size=int(config["stage_a"]["batch_size"]), seed=seed, shuffle=False)
    model = MinimalSNMPP(
        ROOT / "phase2_gate2_sid/materialized_v1_1/frozen_sid/codebooks.npy",
        feedback_embedding_dim=int(config["model"]["feedback_embedding_dim"]),
        hidden_dims=tuple(config["model"]["interaction_hidden_dims"]),
        integration_q=int(config["time"]["integration_Q"]),
        initial_delay_hours=float(config["model"]["initial_delay_hours"]),
    ).to(device)
    manifest = json.loads((work / "data/subset_manifest.json").read_text())
    initialize_constant_rate(model, float(manifest["baselines"]["constant_exponential_rate_per_hour"]))
    initial = parameter_snapshot(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["stage_a"]["learning_rate"]))

    initial_eval = evaluate_feedback_and_loss(model, eval_loader, device)
    fixed_batch = next(iter(eval_loader))
    noise_initial = integration_noise(model, fixed_batch, device)
    print(json.dumps({"stage": "initial", "loss": initial_eval["optimization_loss"], "macro_f1": initial_eval["macro_f1"], "noise": noise_initial}, ensure_ascii=False), flush=True)
    curves = []
    for epoch in range(1, int(config["stage_a"]["max_epochs"]) + 1):
        train = train_epoch(model, train_loader, optimizer, device, float(config["stage_a"]["gradient_clip_norm"]))
        evaluation = evaluate_feedback_and_loss(model, eval_loader, device)
        record = {"epoch": epoch, "train": train, "evaluation": {k: v for k, v in evaluation.items() if k != "rows"}}
        curves.append(record)
        print(json.dumps({"epoch": epoch, "train_loss": train["optimization_loss"], "eval_loss": evaluation["optimization_loss"], "macro_f1": evaluation["macro_f1"], "recall": evaluation["per_class_recall"], "sec": train["seconds"]}, ensure_ascii=False), flush=True)
        if train["nonfinite_steps"]:
            break

    final_eval = evaluate_feedback_and_loss(model, eval_loader, device)
    noise_final = integration_noise(model, fixed_batch, device)
    changes = parameter_change(initial, model)
    trainable_changes = {k: v for k, v in changes.items() if v > 0}
    loss_drop = (initial_eval["optimization_loss"] - final_eval["optimization_loss"]) / max(abs(initial_eval["optimization_loss"]), 1e-12)
    all_recall_nonzero = all(v > 0 for v in final_eval["per_class_recall"].values())
    no_collapse = len({int(np.argmax(row)) for row in np.asarray(final_eval["confusion_matrix"]) if np.sum(row)}) > 1
    finite = math.isfinite(final_eval["optimization_loss"]) and all(r["train"]["nonfinite_steps"] == 0 for r in curves)
    learned_components = {
        "psi": any(v > 1e-6 for k, v in changes.items() if k.startswith("interaction_network")),
        "phi": any(v > 1e-6 for k, v in changes.items() if k.startswith("temporal_network")),
        "delay": changes.get("raw_delays", 0.0) > 1e-6,
        "feedback_embedding": changes.get("feedback_embedding.weight", 0.0) > 1e-6,
    }
    median_step_change = float(np.median([r["train"]["step_loss_change_median_abs"] for r in curves])) if curves else 0.0
    noise_acceptable = noise_final["std"] < max(median_step_change, 1e-8)
    passed = bool(loss_drop > 0.20 and all_recall_nonzero and no_collapse and finite and all(learned_components.values()) and noise_acceptable)
    result = {
        "stage": "A_tiny_overfit",
        "passed": passed,
        "criteria": {
            "relative_loss_drop_gt_20pct": loss_drop > 0.20,
            "all_four_recalls_nonzero": all_recall_nonzero,
            "not_single_class_prediction": no_collapse,
            "finite": finite,
            "psi_phi_delay_feedback_all_changed": all(learned_components.values()),
            "integration_noise_below_typical_step_change": noise_acceptable,
        },
        "relative_loss_drop": loss_drop,
        "initial": {k: v for k, v in initial_eval.items() if k != "rows"},
        "final": {k: v for k, v in final_eval.items() if k != "rows"},
        "integration_noise_initial": noise_initial,
        "integration_noise_final": noise_final,
        "typical_step_loss_change": median_step_change,
        "parameter_change_l2": trainable_changes,
        "learned_components": learned_components,
        "delay_matrix_hours": model.delays().detach().cpu().tolist(),
        "parameter_norms": {n: float(torch.linalg.vector_norm(p.detach())) for n, p in model.named_parameters()},
        "curves": curves,
        "data": {"target_groups": len(dataset), "target_events": int(sum(row["target_events"] for row in final_eval["rows"])), "history_length_percentiles": {str(q): float(np.percentile(dataset.history_event_counts, q)) for q in (0, 50, 90, 95, 99, 100)}},
    }
    output = work / "runs/stage_a_seed2026"
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, "stage_a_passed": passed}, output / "checkpoint_final.pt")
    save_json(output / "training_curve.json", curves)
    save_json(output / "stage_a_metrics.json", result)
    save_json(work / "stage_a_result.json", result)
    print(json.dumps({"STAGE_A_COMPLETE": True, "passed": passed, "relative_loss_drop": loss_drop, "final_macro_f1": final_eval["macro_f1"], "final_recalls": final_eval["per_class_recall"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
