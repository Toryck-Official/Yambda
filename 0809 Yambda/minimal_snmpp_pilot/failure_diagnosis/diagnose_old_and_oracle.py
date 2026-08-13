#!/usr/bin/env python3
"""Old coupled-model gradient conflict and shared-group oracle diagnostics."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from minimal_snmpp_pilot.data import FullHistoryTargetDataset, collate, move_batch
from minimal_snmpp_pilot.model import MinimalSNMPP
from minimal_snmpp_pilot.training_utils import initialize_constant_rate, seed_everything

WORK = ROOT / "minimal_snmpp_pilot"
OUT = WORK / "failure_diagnosis"
OUT.mkdir(parents=True, exist_ok=True)
DATA = WORK / "data/pilot_sequences.npz"
NAMES = ("like", "dislike", "unlike", "undislike")


def write(name: str, value: dict) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def fixed_batch(dataset: FullHistoryTargetDataset) -> tuple[dict[str, torch.Tensor], list[int]]:
    counts = dataset.group_feedback_counts[dataset.targets]
    eligible_length = (dataset.history_event_counts >= 100) & (dataset.history_event_counts <= 500)
    positions: list[int] = []
    for feedback in range(4):
        found = np.flatnonzero(eligible_length & (counts[:, feedback] > 0))
        if len(found) < 2:
            raise RuntimeError(f"not enough fixed-batch examples for feedback {feedback}")
        positions.extend(found[:2].tolist())
    return collate([dataset[index] for index in positions]), positions


GROUPS = {
    "feedback_embedding": ("feedback_embedding.",),
    "P_SID": ("sid_projection.",),
    "psi": ("interaction_network.",),
    "phi": ("temporal_network.",),
    "delay": ("raw_delays",),
}


def vector_for(model: torch.nn.Module, prefixes: tuple[str, ...]) -> torch.Tensor:
    chunks = []
    for name, parameter in model.named_parameters():
        if any(name == prefix or name.startswith(prefix) for prefix in prefixes):
            gradient = parameter.grad
            chunks.append(
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None else gradient.detach().reshape(-1)
            )
    if not chunks:
        raise KeyError(prefixes)
    return torch.cat(chunks)


def gradients(model: MinimalSNMPP, batch: dict[str, torch.Tensor]) -> dict:
    results: dict[str, dict[str, float]] = {}
    vectors: dict[str, dict[str, torch.Tensor]] = {"time": {}, "feedback": {}}
    for objective in ("time", "feedback"):
        model.zero_grad(set_to_none=True)
        output = model.loss(batch, deterministic_integral=True)
        loss = output.time_loss if objective == "time" else output.feedback_loss
        loss.backward()
        for group, prefixes in GROUPS.items():
            vectors[objective][group] = vector_for(model, prefixes)
        vectors[objective]["shared_temporal_parameters"] = torch.cat(
            [vectors[objective][group] for group in GROUPS]
        )
    negative = 0
    for group in (*GROUPS, "shared_temporal_parameters"):
        gt = vectors["time"][group]
        gf = vectors["feedback"][group]
        nt = float(torch.linalg.vector_norm(gt))
        nf = float(torch.linalg.vector_norm(gf))
        dot = float(torch.dot(gt, gf))
        cosine = dot / (nt * nf) if nt > 0 and nf > 0 else float("nan")
        negative += int(math.isfinite(cosine) and cosine < 0 and group != "shared_temporal_parameters")
        results[group] = {
            "gradient_norm_time": nt,
            "gradient_norm_feedback": nf,
            "cosine_similarity": cosine,
            "combined_gradient_norm": float(torch.linalg.vector_norm(gt + gf)),
            "dot_product": dot,
            "finite": bool(torch.isfinite(gt).all() and torch.isfinite(gf).all()),
        }
    results["negative_cosine_parameter_group_count"] = negative
    return results


def gradient_diagnostic(dataset: FullHistoryTargetDataset) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_cpu, positions = fixed_batch(dataset)
    batch = move_batch(batch_cpu, device)
    manifest = json.loads((WORK / "data/subset_manifest.json").read_text())
    output = {
        "batch_contract": {
            "positions": positions,
            "target_group_ids": batch_cpu["target_group_id"].tolist(),
            "target_feedback_counts": batch_cpu["target_feedback_counts"].int().tolist(),
            "history_event_counts": batch_cpu["history_event_count"].tolist(),
            "same_batch_for_all_objectives_and_states": True,
        },
        "states": {},
    }
    for state in ("initialization", "failed_checkpoint"):
        seed_everything(2026)
        model = MinimalSNMPP(
            ROOT / "phase2_gate2_sid/materialized_v1_1/frozen_sid/codebooks.npy"
        ).to(device)
        initialize_constant_rate(
            model, float(manifest["baselines"]["constant_exponential_rate_per_hour"])
        )
        if state == "failed_checkpoint":
            saved = torch.load(
                WORK / "runs/stage_a_seed2026/checkpoint_final.pt",
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(saved["model"])
        output["states"][state] = gradients(model, batch)
    return output


def oracle_floor(dataset: FullHistoryTargetDataset) -> dict:
    counts = dataset.group_feedback_counts[dataset.targets].astype(np.float64)
    size = counts.sum(axis=1)
    probabilities = np.divide(
        counts, size[:, None], out=np.zeros_like(counts), where=size[:, None] > 0
    )
    event_loss_by_group = -(counts * np.log(np.clip(probabilities, 1e-300, 1.0))).sum(axis=1)
    composition_entropy = event_loss_by_group / size

    def summarize(mask: np.ndarray) -> dict:
        selected_events = float(size[mask].sum())
        return {
            "groups": int(mask.sum()),
            "events": int(selected_events),
            "oracle_feedback_ce_per_event": float(event_loss_by_group[mask].sum() / selected_events),
            "mean_group_composition_entropy": float(composition_entropy[mask].mean()),
        }

    bins = (
        ("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20),
        ("21-100", 21, 100), ("101-500", 101, 500), (">500", 501, np.inf),
    )
    class_counts = counts.sum(axis=0)
    proportions = class_counts / class_counts.sum()
    old = json.loads((WORK / "stage_a_result.json").read_text())
    majority = int(np.argmax(proportions))
    predicted = np.asarray(old["final"]["confusion_matrix"]).sum(axis=0)
    result = {
        "all_groups": summarize(np.ones(len(size), dtype=bool)),
        "singleton": summarize(size == 1),
        "multi_event": summarize(size > 1),
        "by_group_size": {
            label: summarize((size >= low) & (size <= high))
            for label, low, high in bins if np.any((size >= low) & (size <= high))
        },
        "class_counts": {NAMES[i]: int(class_counts[i]) for i in range(4)},
        "class_proportions": {NAMES[i]: float(proportions[i]) for i in range(4)},
        "majority_class": NAMES[majority],
        "majority_class_accuracy": float(proportions[majority]),
        "failed_model_predicted_class_proportions": {
            NAMES[i]: float(predicted[i] / predicted.sum()) for i in range(4)
        },
        "failed_model_feedback_loss_per_event": old["final"]["feedback_loss_per_event"],
        "failed_model_feedback_excess_loss": (
            old["final"]["feedback_loss_per_event"]
            - float(event_loss_by_group.sum() / size.sum())
        ),
        "unlike_collapse_is_majority_collapse": bool(
            majority == 2 and predicted[2] / predicted.sum() > 0.99
        ),
    }
    return result


def main() -> None:
    dataset = FullHistoryTargetDataset(DATA, "tiny_train_targets")
    if len(dataset) != 6413:
        raise RuntimeError(f"frozen Tiny mismatch: {len(dataset)}")
    target_events = int(dataset.group_feedback_counts[dataset.targets].sum())
    if target_events != 38363:
        raise RuntimeError(f"frozen Tiny target events mismatch: {target_events}")
    conflict = gradient_diagnostic(dataset)
    oracle = oracle_floor(dataset)
    write("gradient_conflict_metrics.json", conflict)
    write("feedback_oracle_floor.json", oracle)
    print(json.dumps({"gradient_conflict": conflict, "oracle": oracle}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
