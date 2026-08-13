"""Training and evaluation utilities for the frozen Minimal SNMPP protocol."""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from minimal_snmpp_pilot.data import move_batch


FEEDBACK_NAMES = ("like", "dislike", "unlike", "undislike")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def initialize_constant_rate(model: torch.nn.Module, total_rate_per_hour: float) -> None:
    """Initialize four equal baseline intensities to a train-only constant rate."""
    per_mark = total_rate_per_hour / 4.0
    with torch.no_grad():
        model.baseline_logits.fill_(inverse_softplus(per_mark))


def macro_metrics(confusion: np.ndarray) -> dict[str, Any]:
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    recall = np.divide(true_positive, support, out=np.zeros(4), where=support > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros(4), where=predicted > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(4),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": float(true_positive.sum() / max(confusion.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "per_class_recall": {name: float(recall[i]) for i, name in enumerate(FEEDBACK_NAMES)},
        "per_class_precision": {name: float(precision[i]) for i, name in enumerate(FEEDBACK_NAMES)},
        "confusion_matrix": confusion.astype(int).tolist(),
    }


def parameter_snapshot(model: torch.nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def parameter_change(initial: dict[str, Tensor], model: torch.nn.Module) -> dict[str, float]:
    output = {}
    for name, value in model.named_parameters():
        before = initial[name]
        output[name] = float(torch.linalg.vector_norm(value.detach().cpu() - before))
    return output


def tensor_summary(values: list[np.ndarray] | np.ndarray) -> dict[str, float]:
    array = np.concatenate(values) if isinstance(values, list) else np.asarray(values)
    if array.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def evaluate_feedback_and_loss(
    model, loader, device: torch.device, *, compute_time_predictions: bool = False
) -> dict[str, Any]:
    model.eval()
    time_sum = feedback_sum = events = groups = 0.0
    confusion = np.zeros((4, 4), dtype=np.int64)
    predicted_probabilities: list[np.ndarray] = []
    intensities: list[np.ndarray] = []
    totals: list[np.ndarray] = []
    signed: list[np.ndarray] = []
    absolute: list[np.ndarray] = []
    positive: list[np.ndarray] = []
    negative: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model.loss(batch, deterministic_integral=True)
            time_prediction = (
                model.time_distribution(batch) if compute_time_predictions else None
            )
            batch_groups = len(batch["target_time"])
            time_sum += float(output.time_loss_by_group.sum())
            feedback_sum += float(output.feedback_loss_sum)
            groups += batch_groups
            events += float(output.num_events)
            probabilities = output.target_feedback_probabilities.cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            counts = batch["target_feedback_counts"].long().cpu().numpy()
            for row, prediction in enumerate(predictions):
                confusion[:, prediction] += counts[row]
            predicted_probabilities.append(probabilities)
            intensities.append(output.target_lambdas.cpu().numpy().reshape(-1))
            totals.append(output.target_total_intensity.cpu().numpy())
            signed.append(output.signed_influence_by_target.cpu().numpy().reshape(-1))
            absolute.append(output.absolute_influence_mass_by_target.cpu().numpy().reshape(-1))
            positive.append(output.positive_influence_mass_by_target.cpu().numpy().reshape(-1))
            negative.append(output.negative_influence_mass_by_target.cpu().numpy().reshape(-1))
            for row in range(batch_groups):
                rows.append({
                    "target_group_size": int(batch["target_group_size"][row]),
                    "previous_group_size": int(batch["previous_group_size"][row]),
                    "history_event_count": int(batch["history_event_count"][row]),
                    "time_loss": float(output.time_loss_by_group[row]),
                    "feedback_loss_sum": float(-(batch["target_feedback_counts"][row] * torch.log(output.target_feedback_probabilities[row])).sum()),
                    "target_events": int(batch["target_group_size"][row]),
                    "total_intensity": float(output.target_total_intensity[row]),
                    "prediction": int(predictions[row]),
                    "counts": counts[row].tolist(),
                    "actual_gap_hours": float(batch["gap_seconds"][row]) / 3600.0,
                    "predicted_gap_hours": (
                        float(time_prediction["expected_delta_hours"][row])
                        if time_prediction is not None else None
                    ),
                    "horizon_event_mass": (
                        float(time_prediction["event_mass_within_horizon"][row])
                        if time_prediction is not None else None
                    ),
                })
    result = {
        "optimization_loss": time_sum / groups + feedback_sum / events,
        "time_loss_per_group": time_sum / groups,
        "feedback_loss_per_event": feedback_sum / events,
        **macro_metrics(confusion),
        "predicted_feedback_distribution": np.mean(np.concatenate(predicted_probabilities), axis=0).tolist(),
        "lambda_k": tensor_summary(intensities),
        "total_lambda": tensor_summary(totals),
        "signed_influence": tensor_summary(signed),
        "absolute_influence_mass": tensor_summary(absolute),
        "positive_influence_mass": tensor_summary(positive),
        "negative_influence_mass": tensor_summary(negative),
        "rows": rows,
    }
    return result


def train_epoch(model, loader, optimizer, device: torch.device, clip_norm: float) -> dict[str, float]:
    model.train()
    time_sum = feedback_sum = groups = events = 0.0
    grad_norms: list[float] = []
    clip_count = 0
    nonfinite = 0
    step_losses: list[float] = []
    started = time.perf_counter()
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model.loss(batch, deterministic_integral=False)
        if not torch.isfinite(output.optimization_loss):
            nonfinite += 1
            continue
        output.optimization_loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm))
        if not math.isfinite(norm):
            nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        clip_count += int(norm > clip_norm)
        grad_norms.append(norm)
        optimizer.step()
        groups += float(output.num_groups)
        events += float(output.num_events)
        time_sum += float(output.time_loss_by_group.detach().sum())
        feedback_sum += float(output.feedback_loss_sum.detach())
        step_losses.append(float(output.optimization_loss.detach()))
    elapsed = time.perf_counter() - started
    return {
        "optimization_loss": time_sum / max(groups, 1) + feedback_sum / max(events, 1),
        "time_loss_per_group": time_sum / max(groups, 1),
        "feedback_loss_per_event": feedback_sum / max(events, 1),
        "gradient_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
        "gradient_norm_max": float(np.max(grad_norms)) if grad_norms else float("nan"),
        "gradient_clip_frequency": clip_count / max(len(grad_norms), 1),
        "nonfinite_steps": nonfinite,
        "step_loss_change_median_abs": float(np.median(np.abs(np.diff(step_losses)))) if len(step_losses) > 1 else 0.0,
        "seconds": elapsed,
        "groups_per_second": groups / max(elapsed, 1e-9),
    }


def integration_noise(model, batch, device: torch.device, repeats: int = 5) -> dict[str, float]:
    model.eval()
    batch = move_batch(batch, device)
    values = []
    with torch.no_grad():
        for _ in range(repeats):
            values.append(float(model.loss(batch, deterministic_integral=False).time_loss))
    mean = float(np.mean(values)); std = float(np.std(values))
    return {"values": values, "mean": mean, "std": std, "cv": std / max(abs(mean), 1e-12)}


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True) + "\n")
