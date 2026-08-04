#!/usr/bin/env python3
"""Diagnose whether rare marks are suppressed at decision time (argmax calibration).

This is a read-only diagnostic on an existing checkpoint.  It recomputes the
observed-time and next-group type decisions as ``argmax(lambda_k * w_k)`` with
``w_k = prior_k ** (-gamma)`` for a sweep of gamma, leaving the likelihood and
the reported probabilities unchanged.  gamma=0 reproduces the standard argmax.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    if os.environ.get(variable) in {None, "", "0"}:
        os.environ[variable] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snmpp.config import load_experiment_config  # noqa: E402
from snmpp.data import GroupedWindowDataset, make_data_loader  # noqa: E402
from snmpp.evaluation import (  # noqa: E402
    EVENT_TYPE_NAMES,
    _balanced_accuracy,
    _collate_prediction_targets,
    _collect_prediction_targets,
    _macro_f1,
    TargetReservoir,
)
from snmpp.training import load_model_checkpoint  # noqa: E402
from snmpp.utils import atomic_write_json, resolve_device, seed_everything  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gamma-list",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0,1.5,2.0",
        help="Comma-separated decision calibration exponents.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    seed_everything(
        config.training.seed,
        config.training.deterministic_algorithms,
    )
    device = resolve_device(config.training.device)
    from snmpp.utils import sha256_file

    expected = sha256_file(config.data.dataset_dir / "manifest.json")
    model, checkpoint = load_model_checkpoint(
        args.checkpoint,
        config.model,
        device,
        expected_data_manifest_sha256=expected,
    )
    dataset = GroupedWindowDataset(
        config.data.dataset_dir,
        args.split,
        config.data.target_groups_per_window,
        config.data.history_groups,
        max_windows=(
            config.data.max_validation_windows
            if args.split == "validation"
            else config.data.max_test_windows
        ),
        subsample_seed=(
            config.training.seed + 1 if args.split == "validation" else config.training.seed + 2
        ),
    )
    train_dataset = GroupedWindowDataset(
        config.data.dataset_dir,
        "train",
        config.data.target_groups_per_window,
        config.data.history_groups,
        max_windows=1,
        subsample_seed=config.training.seed,
    )
    loader = make_data_loader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        seed=config.training.seed,
        num_workers=config.data.num_workers,
    )
    priors = train_dataset.event_counts / train_dataset.event_counts.sum()
    gamma_values = [float(value) for value in args.gamma_list.split(",")]
    weights_by_gamma = {
        gamma: np.power(np.clip(priors, 1e-8, None), -gamma) for gamma in gamma_values
    }
    model.eval()
    reservoir = TargetReservoir(
        config.evaluation.max_prediction_targets,
        config.training.seed + (21 if args.split == "validation" else 22),
    )
    observed = {
        gamma: {
            "confusion": np.zeros((4, 4), dtype=np.float64),
            "actual_counts": np.zeros(4, dtype=np.float64),
            "correct": 0.0,
            "events": 0.0,
        }
        for gamma in gamma_values
    }
    with torch.no_grad():
        for cpu_batch in loader:
            _collect_prediction_targets(cpu_batch, reservoir)
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            output = model.log_likelihood(batch, deterministic_integral=True)
            intensities = output.event_intensities
            score_mask = batch["score_mask"] & batch["group_mask"]
            counts = batch["counts"]
            for gamma in gamma_values:
                weight = torch.as_tensor(
                    weights_by_gamma[gamma], dtype=intensities.dtype, device=intensities.device
                )
                scores = intensities * weight[None, None, :]
                predictions = scores.argmax(dim=-1)
                predicted_counts = counts.gather(dim=-1, index=predictions[..., None]).squeeze(-1)
                observed[gamma]["correct"] += float((predicted_counts * score_mask).sum().item())
                observed[gamma]["events"] += float((counts.sum(dim=-1) * score_mask).sum().item())
                rows = torch.nonzero(score_mask, as_tuple=False)
                for row_index, group_index in rows.tolist():
                    predicted = int(predictions[row_index, group_index].item())
                    actual = counts[row_index, group_index].detach().cpu().numpy()
                    observed[gamma]["confusion"][:, predicted] += actual
                    observed[gamma]["actual_counts"] += actual
        horizon = train_dataset.mean_positive_delta * config.evaluation.prediction_horizon_mean_multiplier
        next_group = {
            gamma: {
                "confusion": np.zeros((4, 4), dtype=np.float64),
                "actual_counts": np.zeros(4, dtype=np.float64),
                "correct": 0.0,
                "events": 0.0,
            }
            for gamma in gamma_values
        }
        for start in range(0, len(reservoir.items), config.evaluation.prediction_batch_size):
            targets = reservoir.items[start : start + config.evaluation.prediction_batch_size]
            (
                history_times,
                history_counts,
                history_mask,
                _actual_deltas,
                target_counts,
            ) = _collate_prediction_targets(targets, device)
            prediction = model.predict_next_event(
                history_times,
                history_counts,
                history_mask,
                horizon=horizon,
                grid_size=config.evaluation.prediction_grid_size,
            )
            mark_probs = prediction["mark_probabilities"]
            for gamma in gamma_values:
                weight = torch.as_tensor(
                    weights_by_gamma[gamma], dtype=mark_probs.dtype, device=mark_probs.device
                )
                scores = mark_probs * weight[None, :]
                predicted_type = scores.argmax(dim=-1)
                correct_counts = target_counts.gather(1, predicted_type[:, None]).squeeze(1)
                next_group[gamma]["correct"] += float(correct_counts.sum().item())
                next_group[gamma]["events"] += float(target_counts.sum().item())
                for row in range(len(targets)):
                    actual = target_counts[row].detach().cpu().numpy()
                    predicted = int(predicted_type[row].item())
                    next_group[gamma]["confusion"][:, predicted] += actual
                    next_group[gamma]["actual_counts"] += actual

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "split": args.split,
        "training_prior": {
            EVENT_TYPE_NAMES[index]: float(value) for index, value in enumerate(priors)
        },
        "calibration_rule": "argmax(lambda_k * prior_k**(-gamma)); gamma=0 is standard argmax",
        "note": "probabilities and likelihood are unchanged; only the decision threshold is swept",
    }
    per_gamma: dict[str, object] = {}
    for gamma in gamma_values:
        observed_cell = observed[gamma]
        next_cell = next_group[gamma]
        majority_confusion = np.zeros_like(observed_cell["confusion"])
        majority_confusion[:, int(priors.argmax())] = observed_cell["actual_counts"]
        per_gamma[str(gamma)] = {
            "type_at_observed_time": {
                "accuracy": observed_cell["correct"] / observed_cell["events"],
                "balanced_accuracy": _balanced_accuracy(observed_cell["confusion"]),
                "macro_f1": _macro_f1(observed_cell["confusion"]),
                "actual_event_counts": observed_cell["actual_counts"].astype(int).tolist(),
                "predicted_event_counts": observed_cell["confusion"].sum(axis=0).astype(int).tolist(),
                "confusion_actual_rows_predicted_columns": (
                    observed_cell["confusion"].astype(int).tolist()
                ),
                "majority_reference": {
                    "balanced_accuracy": _balanced_accuracy(majority_confusion),
                    "macro_f1": _macro_f1(majority_confusion),
                },
            },
            "next_group_prediction": {
                "count_weighted_type_accuracy": next_cell["correct"] / next_cell["events"],
                "actual_event_counts": next_cell["actual_counts"].astype(int).tolist(),
                "predicted_event_counts": next_cell["confusion"].sum(axis=0).astype(int).tolist(),
                "confusion_actual_rows_predicted_columns": next_cell["confusion"].astype(int).tolist(),
            },
        }
    result["per_gamma"] = per_gamma
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    for gamma in gamma_values:
        cell = per_gamma[str(gamma)]
        print(
            f"gamma={gamma}: acc={cell['type_at_observed_time']['accuracy']:.4f} "
            f"bal_acc={cell['type_at_observed_time']['balanced_accuracy']:.4f} "
            f"macro_f1={cell['type_at_observed_time']['macro_f1']:.4f} "
            f"pred={cell['type_at_observed_time']['predicted_event_counts']} "
            f"next_acc={cell['next_group_prediction']['count_weighted_type_accuracy']:.4f} "
            f"next_pred={cell['next_group_prediction']['predicted_event_counts']}"
        )


if __name__ == "__main__":
    main()
