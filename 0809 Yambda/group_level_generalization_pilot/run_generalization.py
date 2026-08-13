#!/usr/bin/env python3
"""Moderate train/validation generalization pilot for ordinary group models."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from group_level_learnability_gate.run_gate import (  # noqa: E402
    CLASS_NAMES,
    CODEBOOK,
    PREVIOUS_SIZE_BINS,
    SOURCE_MANIFEST,
    SOURCE_NPZ,
    MarkModel,
    PreparedData,
    SequenceDataset,
    TimeModel,
    _features_for_group_range,
    classification_metrics,
    clean_metric_payload,
    lognormal_nll,
    make_loader,
    mark_loss,
    save_json,
    seed_all,
    sha256,
    time_metrics,
)


WORK = Path(__file__).resolve().parent
SEEDS = [2026, 2027, 2028]
TARGET_GOAL = 30_000
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def stable_user_order(users: np.ndarray) -> np.ndarray:
    values = users.astype(np.uint64)
    keys = values * np.uint64(11400714819323198485) + np.uint64(2026)
    return users[np.argsort(keys, kind="stable")]


def _user_start(group_uid: np.ndarray) -> np.ndarray:
    starts = np.flatnonzero(np.r_[True, group_uid[1:] != group_uid[:-1]])
    marker = np.zeros(len(group_uid), dtype=np.int64)
    marker[starts] = starts
    return np.maximum.accumulate(marker)


def _target_arrays(
    targets: np.ndarray,
    *,
    group_uid: np.ndarray,
    group_timestamp: np.ndarray,
    group_global_id: np.ndarray,
    group_counts: np.ndarray,
    start_for_group: np.ndarray,
    global_to_local: np.ndarray,
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    contract: dict,
) -> PreparedData:
    starts = global_to_local[start_for_group[targets]].astype(np.int64)
    ends = (global_to_local[targets - 1] + 1).astype(np.int64)
    if np.any(starts < 0) or np.any(ends <= starts):
        raise RuntimeError("invalid strict chronological sequence mapping")
    target_gap = (group_timestamp[targets].astype(np.float64) - group_timestamp[targets - 1]) / 3600.0
    # For a user's second group there is no earlier observed gap.  Use the frozen
    # complete-train median only for that baseline covariate; never index the
    # previous user's final timestamp.
    previous_gap = np.full(len(targets), 2845.0 / 3600.0, dtype=np.float64)
    has_observed_previous_gap = (targets - 1) > start_for_group[targets]
    observed = targets[has_observed_previous_gap]
    previous_gap[has_observed_previous_gap] = (
        group_timestamp[observed - 1].astype(np.float64)
        - group_timestamp[observed - 2].astype(np.float64)
    ) / 3600.0
    if np.any(target_gap <= 0) or np.any(previous_gap <= 0):
        raise RuntimeError("non-positive group gap")
    return PreparedData(
        features=features,
        sequence_starts=starts,
        sequence_ends=ends,
        previous_features=features[ends - 1].copy(),
        target_counts=group_counts[targets].astype(np.float32),
        target_gap_hours=target_gap.astype(np.float32),
        previous_gap_hours=previous_gap.astype(np.float32),
        previous_sizes=group_counts[targets - 1].sum(axis=1).astype(np.int64),
        target_group_ids=group_global_id[targets].astype(np.uint64),
        target_uids=group_uid[targets].astype(np.uint32),
        feature_mean=feature_mean,
        feature_std=feature_std,
        contract=contract,
    )


def prepare() -> tuple[PreparedData, PreparedData, dict]:
    archive = np.load(SOURCE_NPZ, allow_pickle=False)
    uid = archive["group_uid"]
    timestamp = archive["group_timestamp"]
    global_id = archive["group_global_id"]
    offsets = archive["group_event_offsets"]
    counts = archive["group_feedback_counts"]
    event_sid = archive["event_sid"]
    codebooks = np.load(CODEBOOK).astype(np.float32)
    all_train = archive["pilot_train_targets"].astype(np.int64)
    all_validation = archive["pilot_validation_targets"].astype(np.int64)
    pilot_users = archive["pilot_users"].astype(np.uint32)

    unique_train_users, train_counts = np.unique(uid[all_train], return_counts=True)
    count_by_user = dict(zip(unique_train_users.tolist(), train_counts.tolist()))
    selected: list[int] = []
    accumulated = 0
    for user in stable_user_order(pilot_users):
        amount = int(count_by_user.get(int(user), 0))
        if amount == 0:
            continue
        selected.append(int(user))
        accumulated += amount
        if accumulated >= TARGET_GOAL:
            break
    selected_users = np.asarray(sorted(selected), dtype=np.uint32)
    locations = np.searchsorted(selected_users, uid)
    selected_group = np.zeros(len(uid), dtype=bool)
    valid = locations < len(selected_users)
    selected_group[valid] = selected_users[locations[valid]] == uid[valid]
    train_targets = all_train[selected_group[all_train]]
    validation_targets = all_validation[selected_group[all_validation]]
    if len(train_targets) < TARGET_GOAL or len(validation_targets) == 0:
        raise RuntimeError("moderate pilot selection failed")

    start_for_group = _user_start(uid)
    # Materialize one group-feature range per selected user through its final validation target,
    # or through its final train target when that user has no validation target.
    max_train: dict[int, int] = {}
    max_validation: dict[int, int] = {}
    for target in train_targets:
        max_train[int(uid[target])] = max(int(target), max_train.get(int(uid[target]), -1))
    for target in validation_targets:
        max_validation[int(uid[target])] = max(int(target), max_validation.get(int(uid[target]), -1))

    ranges: list[tuple[int, int, int]] = []
    for user in selected_users:
        user_int = int(user)
        train_end = max_train[user_int]
        full_end = max(max_validation.get(user_int, train_end), train_end)
        ranges.append((int(start_for_group[train_end]), full_end, train_end))
    ranges.sort()
    total_groups = sum(end - start for start, end, _ in ranges)
    features = np.empty((total_groups, 134), dtype=np.float32)
    global_to_local = np.full(len(uid), -1, dtype=np.int64)
    scaler_mask = np.zeros(total_groups, dtype=bool)
    cursor = 0
    for number, (start, end, train_end) in enumerate(ranges, 1):
        block = _features_for_group_range(
            start, end, event_sid=event_sid, event_offsets=offsets,
            feedback_counts=counts, timestamps=timestamp, codebooks=codebooks,
        )
        features[cursor:cursor + len(block)] = block
        global_to_local[start:end] = np.arange(cursor, cursor + len(block))
        scaler_mask[cursor:cursor + (train_end - start)] = True
        cursor += len(block)
        if number % 100 == 0:
            print(json.dumps({"prepared_users": number, "groups": cursor}), flush=True)
    if not scaler_mask.any():
        raise RuntimeError("no train history groups for scaler")
    mean = features[scaler_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features[scaler_mask].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    features -= mean
    features /= std
    if not np.isfinite(features).all():
        raise RuntimeError("non-finite standardized feature")

    base_contract = {
        "source": str(SOURCE_NPZ),
        "source_sha256": sha256(SOURCE_NPZ),
        "selected_user_rule": "prefix of seed-2026 stable hash order over existing pilot users until >=30k train targets",
        "selection_uses_train_only": True,
        "selected_users": int(len(selected_users)),
        "train_targets": int(len(train_targets)),
        "validation_targets": int(len(validation_targets)),
        "train_users": int(len(np.unique(uid[train_targets]))),
        "validation_users": int(len(np.unique(uid[validation_targets]))),
        "train_target_events": int(counts[train_targets].sum()),
        "validation_target_events": int(counts[validation_targets].sum()),
        "feature_groups_materialized_in_memory": int(total_groups),
        "scaler_fit": "strict train-history groups only",
        "validation_history": "all groups strictly before each validation target, including prior validation groups",
        "within_group_order_used": False,
        "listen_used": False,
        "test_used": False,
        "burst_removed": False,
        "feature_definition": "mean frozen SID semantics + feedback composition + log1p group size + log1p previous gap hours",
    }
    train = _target_arrays(
        train_targets, group_uid=uid, group_timestamp=timestamp, group_global_id=global_id,
        group_counts=counts, start_for_group=start_for_group, global_to_local=global_to_local,
        features=features, feature_mean=mean, feature_std=std,
        contract={**base_contract, "split": "train"},
    )
    validation = _target_arrays(
        validation_targets, group_uid=uid, group_timestamp=timestamp, group_global_id=global_id,
        group_counts=counts, start_for_group=start_for_group, global_to_local=global_to_local,
        features=features, feature_mean=mean, feature_std=std,
        contract={**base_contract, "split": "validation"},
    )

    selection_path = WORK / "selection_ids.npz"
    np.savez_compressed(
        selection_path, selected_users=selected_users,
        train_target_local_group_indices=train_targets,
        validation_target_local_group_indices=validation_targets,
        train_target_global_group_ids=global_id[train_targets],
        validation_target_global_group_ids=global_id[validation_targets],
    )
    manifest = {
        **base_contract,
        "target_goal": TARGET_GOAL,
        "seed": 2026,
        "selected_user_ids": selected_users.tolist(),
        "selection_artifact": str(selection_path),
        "selection_artifact_sha256": sha256(selection_path),
        "train_history_length_percentiles": {str(q): float(np.percentile(train.sequence_ends - train.sequence_starts, q)) for q in (0, 50, 90, 95, 99, 100)},
        "validation_history_length_percentiles": {str(q): float(np.percentile(validation.sequence_ends - validation.sequence_starts, q)) for q in (0, 50, 90, 95, 99, 100)},
    }
    return train, validation, manifest


@torch.no_grad()
def predict(model: torch.nn.Module, dataset: SequenceDataset, device: torch.device, task: str) -> dict:
    loader = make_loader(dataset, seed=2026, shuffle=False, batch_size=BATCH_SIZE)
    model.eval()
    records = []
    for batch in loader:
        indices = batch["index"].numpy()
        gpu = {key: value.to(device) for key, value in batch.items() if key not in {"index", "previous_size"}}
        if task == "mark":
            records.append((indices, torch.softmax(model(gpu), dim=1).cpu().numpy()))
        else:
            mu, sigma = model(gpu)
            records.append((indices, mu.cpu().numpy(), sigma.cpu().numpy()))
    order = np.concatenate([row[0] for row in records])
    sorting = np.argsort(order)
    outputs = [np.concatenate([row[col] for row in records])[sorting] for col in range(1, len(records[0]))]
    if task == "mark":
        return {"metrics": classification_metrics(outputs[0], dataset.data.target_counts), "probabilities": outputs[0]}
    result = time_metrics(outputs[0], outputs[1], dataset.data.target_gap_hours)
    return {"metrics": {key: value for key, value in result.items() if key != "rows"}, "mu": outputs[0], "sigma": outputs[1]}


def fit_model(
    model: torch.nn.Module,
    train_data: PreparedData,
    validation_data: PreparedData,
    *,
    task: str,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    seed_all(seed)
    model.to(device)
    train_dataset, validation_dataset = SequenceDataset(train_data), SequenceDataset(validation_data)
    loader = make_loader(train_dataset, seed=seed, shuffle=True, batch_size=BATCH_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_state, best_train_loss = None, float("inf")
    curves = []
    clipping_steps = total_steps = nonfinite_steps = 0
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        model.train()
        numerator = denominator = 0.0
        for batch in loader:
            gpu = {key: value.to(device) for key, value in batch.items() if key not in {"index", "previous_size"}}
            optimizer.zero_grad(set_to_none=True)
            if task == "mark":
                loss = mark_loss(model(gpu), gpu["counts"])
                weight = float(gpu["counts"].sum())
            else:
                mu, sigma = model(gpu)
                loss = lognormal_nll(gpu["gap"], mu, sigma).mean()
                weight = len(gpu["gap"])
            if not torch.isfinite(loss):
                nonfinite_steps += 1
                continue
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            clipping_steps += int(float(norm) > 5.0)
            total_steps += 1
            optimizer.step()
            numerator += float(loss.detach()) * weight
            denominator += weight
        train_output = predict(model, train_dataset, device, task)
        validation_output = predict(model, validation_dataset, device, task)
        train_selection = train_output["metrics"]["ce_per_event" if task == "mark" else "nll_per_group"]
        if train_selection < best_train_loss:
            best_train_loss = train_selection
            best_state = copy.deepcopy(model.state_dict())
        record = {
            "epoch": epoch,
            "train_batch_objective": numerator / max(denominator, 1),
            "train": clean_metric_payload(train_output["metrics"]),
            "validation": clean_metric_payload(validation_output["metrics"]),
            "seconds": time.perf_counter() - started,
        }
        curves.append(record)
        print(json.dumps({"task": task, "seed": seed, "epoch": epoch,
                          "train": train_selection,
                          "validation": validation_output["metrics"]["ce_per_event" if task == "mark" else "nll_per_group"],
                          "seconds": record["seconds"]}), flush=True)
    if best_state is None:
        raise RuntimeError("no finite train checkpoint")
    # Checkpoint selection is based on train objective only; validation does not tune the checkpoint.
    model.load_state_dict(best_state)
    train_final = predict(model, train_dataset, device, task)
    validation_final = predict(model, validation_dataset, device, task)
    diagnosis = {
        "seed": seed,
        "task": task,
        "checkpoint_selection": "minimum train objective across fixed 20 epochs",
        "train": clean_metric_payload(train_final["metrics"]),
        "validation": clean_metric_payload(validation_final["metrics"]),
        "train_to_validation_gap": {
            key: float(validation_final["metrics"][key] - train_final["metrics"][key])
            for key in (["ce_per_event", "macro_f1"] if task == "mark" else ["nll_per_group", "mae_hours", "median_ae_hours"])
        },
        "gradient_clip_frequency": clipping_steps / max(total_steps, 1),
        "nonfinite_steps": nonfinite_steps,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "curves": curves,
    }
    if task == "mark":
        diagnosis["validation_probabilities"] = validation_final["probabilities"]
    else:
        diagnosis["validation_mu"] = validation_final["mu"]
        diagnosis["validation_sigma"] = validation_final["sigma"]
    return model, diagnosis


def mark_baseline_metrics(data: PreparedData, global_probability: np.ndarray) -> dict:
    count = len(data.target_counts)
    global_prediction = np.repeat(global_probability[None], count, axis=0)
    previous_composition = data.previous_features[:, 128:132] * data.feature_std[128:132] + data.feature_mean[128:132]
    size = data.previous_sizes[:, None].astype(np.float64)
    previous_prediction = (previous_composition * size + global_probability) / (size + 1.0)
    return {
        "global_full_train_feedback_prior": classification_metrics(global_prediction, data.target_counts),
        "previous_group_composition_global_Dirichlet_strength_1": classification_metrics(previous_prediction, data.target_counts),
    }


def _evaluate_lognormal(mu: np.ndarray, sigma: float | np.ndarray, gaps: np.ndarray) -> dict:
    result = time_metrics(np.asarray(mu), np.broadcast_to(sigma, len(gaps)), gaps)
    result.pop("rows")
    return result


def fit_time_baselines(train: PreparedData, validation: PreparedData, global_median_hours: float) -> dict:
    train_gap = train.target_gap_hours.astype(np.float64)
    validation_gap = validation.target_gap_hours.astype(np.float64)
    log_train = np.log(train_gap)
    mu, sigma = float(log_train.mean()), max(float(log_train.std()), 0.05)
    residual = log_train - np.log(train.previous_gap_hours.astype(np.float64))
    residual_mu, residual_sigma = float(residual.mean()), max(float(residual.std()), 0.05)

    def point(prediction: np.ndarray, truth: np.ndarray) -> dict:
        error = np.abs(prediction - truth)
        return {"mae_hours": float(error.mean()), "median_ae_hours": float(np.median(error)), "finite": bool(np.isfinite(error).all())}

    return {
        "global_full_train_median_gap": {
            "definition": "2845-second median computed from the complete D_SID train split",
            "train": point(np.full(len(train_gap), global_median_hours), train_gap),
            "validation": point(np.full(len(validation_gap), global_median_hours), validation_gap),
        },
        "history_free_lognormal_fit_on_moderate_train": {
            "mu": mu, "sigma": sigma,
            "train": _evaluate_lognormal(np.full(len(train_gap), mu), sigma, train_gap),
            "validation": _evaluate_lognormal(np.full(len(validation_gap), mu), sigma, validation_gap),
        },
        "previous_gap_point": {
            "train": point(train.previous_gap_hours.astype(np.float64), train_gap),
            "validation": point(validation.previous_gap_hours.astype(np.float64), validation_gap),
        },
        "previous_gap_lognormal_residual_fit_on_moderate_train": {
            "residual_mu": residual_mu, "residual_sigma": residual_sigma,
            "train": _evaluate_lognormal(np.log(train.previous_gap_hours) + residual_mu, residual_sigma, train_gap),
            "validation": _evaluate_lognormal(np.log(validation.previous_gap_hours) + residual_mu, residual_sigma, validation_gap),
        },
    }


def validation_slices(data: PreparedData, probability: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> dict:
    result = {}
    for lower, upper, label in PREVIOUS_SIZE_BINS:
        mask = (data.previous_sizes >= lower) & (data.previous_sizes <= upper)
        if not mask.any():
            result[label] = {"groups": 0}
            continue
        time_result = time_metrics(mu[mask], sigma[mask], data.target_gap_hours[mask])
        time_result.pop("rows")
        result[label] = {
            "groups": int(mask.sum()),
            "events": int(data.target_counts[mask].sum()),
            "mark": classification_metrics(probability[mask], data.target_counts[mask]),
            "time": time_result,
            "outputs_finite": bool(np.isfinite(probability[mask]).all() and np.isfinite(mu[mask]).all() and np.isfinite(sigma[mask]).all()),
        }
    return result


def aggregate(runs: list[dict], task: str, model_name: str) -> dict:
    keys = ["ce_per_event", "macro_f1", "accuracy"] if task == "mark" else ["nll_per_group", "mae_hours", "median_ae_hours"]
    result = {"model": model_name}
    for split in ["train", "validation"]:
        result[split] = {}
        for key in keys:
            values = [run[split][key] for run in runs]
            result[split][key] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
    result["validation_collapsed_seeds"] = int(sum(run["validation"].get("collapsed", False) for run in runs))
    result["nonfinite_steps"] = int(sum(run["nonfinite_steps"] for run in runs))
    return result


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, validation, manifest = prepare()
    save_json(WORK / "pilot_manifest.json", manifest)
    np.savez_compressed(WORK / "feature_standardization.npz", mean=train.feature_mean, std=train.feature_std)
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    global_probability = np.asarray(source_manifest["baselines"]["global_train_feedback_probabilities"], dtype=np.float64)
    baselines = {
        "definition_audit": {
            "global_feedback_prior_source": "complete D_SID train split, not Tiny and not moderate Pilot",
            "global_feedback_counts": source_manifest["baselines"]["global_train_feedback_counts"],
            "global_feedback_probability": global_probability.tolist(),
            "global_median_gap_source": "complete D_SID train split",
            "global_median_gap_seconds": source_manifest["baselines"]["global_train_gap_median_seconds"],
        },
        "mark": {
            "train": mark_baseline_metrics(train, global_probability),
            "validation": mark_baseline_metrics(validation, global_probability),
        },
        "time": fit_time_baselines(train, validation, float(source_manifest["baselines"]["global_train_gap_median_seconds"]) / 3600.0),
    }

    last_runs, mark_runs, time_runs = [], [], []
    seed2026_raw = None
    for seed in SEEDS:
        seed_dir = WORK / f"runs/seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_all(seed)
        last_model, last = fit_model(MarkModel(False), train, validation, task="mark", seed=seed, device=device)
        torch.save(last_model.state_dict(), seed_dir / "last_group_mark.pt")
        save_json(seed_dir / "last_group_mark_metrics.json", clean_metric_payload(last))
        last_runs.append(clean_metric_payload({key: value for key, value in last.items() if key not in {"validation_probabilities", "curves"}}))

        seed_all(seed)
        mark_model, mark = fit_model(MarkModel(True), train, validation, task="mark", seed=seed, device=device)
        torch.save(mark_model.state_dict(), seed_dir / "full_history_mark.pt")
        save_json(seed_dir / "full_history_mark_metrics.json", clean_metric_payload(mark))
        mark_runs.append(clean_metric_payload({key: value for key, value in mark.items() if key not in {"validation_probabilities", "curves"}}))

        seed_all(seed)
        time_model, time_run = fit_model(TimeModel(True), train, validation, task="time", seed=seed, device=device)
        torch.save(time_model.state_dict(), seed_dir / "full_history_time.pt")
        save_json(seed_dir / "full_history_time_metrics.json", clean_metric_payload(time_run))
        time_runs.append(clean_metric_payload({key: value for key, value in time_run.items() if key not in {"validation_mu", "validation_sigma", "curves"}}))
        if seed == 2026:
            seed2026_raw = (mark["validation_probabilities"], time_run["validation_mu"], time_run["validation_sigma"])

    last_summary = aggregate(last_runs, "mark", "last_group_only_MLP")
    mark_summary = aggregate(mark_runs, "mark", "full_history_GRU")
    time_summary = aggregate(time_runs, "time", "full_history_GRU_lognormal")

    validation_mark_baselines = baselines["mark"]["validation"]
    strongest_simple_ce = min(
        validation_mark_baselines["global_full_train_feedback_prior"]["ce_per_event"],
        validation_mark_baselines["previous_group_composition_global_Dirichlet_strength_1"]["ce_per_event"],
        last_summary["validation"]["ce_per_event"]["mean"],
    )
    strongest_simple_f1 = max(
        validation_mark_baselines["global_full_train_feedback_prior"]["macro_f1"],
        validation_mark_baselines["previous_group_composition_global_Dirichlet_strength_1"]["macro_f1"],
        last_summary["validation"]["macro_f1"]["mean"],
    )
    mark_ce_gain = (strongest_simple_ce - mark_summary["validation"]["ce_per_event"]["mean"]) / strongest_simple_ce
    mark_f1_gain = mark_summary["validation"]["macro_f1"]["mean"] - strongest_simple_f1
    mark_generalizes = bool(mark_ce_gain >= 0.01 and mark_f1_gain >= 0.01 and mark_summary["validation_collapsed_seeds"] == 0)

    history_free = baselines["time"]["history_free_lognormal_fit_on_moderate_train"]["validation"]
    previous_parametric = baselines["time"]["previous_gap_lognormal_residual_fit_on_moderate_train"]["validation"]
    strongest_nll = min(history_free["nll_per_group"], previous_parametric["nll_per_group"])
    point_maes = [
        baselines["time"]["global_full_train_median_gap"]["validation"]["mae_hours"],
        baselines["time"]["previous_gap_point"]["validation"]["mae_hours"],
        history_free["mae_hours"], previous_parametric["mae_hours"],
    ]
    strongest_mae = min(point_maes)
    time_nll_gain = (strongest_nll - time_summary["validation"]["nll_per_group"]["mean"]) / max(abs(strongest_nll), 1e-12)
    time_mae_gain = (strongest_mae - time_summary["validation"]["mae_hours"]["mean"]) / strongest_mae
    time_generalizes = bool(time_nll_gain >= 0.01 and time_mae_gain >= 0.01)

    if seed2026_raw is None:
        raise RuntimeError("missing seed-2026 outputs")
    slices = validation_slices(validation, *seed2026_raw)
    all_slices_finite = all(row.get("outputs_finite", True) for row in slices.values())
    stable = bool(mark_summary["nonfinite_steps"] == 0 and time_summary["nonfinite_steps"] == 0 and all_slices_finite)
    history_supported = bool(mark_generalizes or time_generalizes)
    status = {
        "group_level_mark_generalizes": mark_generalizes,
        "group_level_time_generalizes": time_generalizes,
        "full_history_generalization_supported": history_supported,
        "group_snmpp_design_gate_approved": bool(history_supported and stable),
        "ordinary_group_models_numerically_stable": stable,
        "test_used": False,
        "group_snmpp_designed_or_trained": False,
        "decision_thresholds": {
            "mark": "validation CE >=1% better and Macro-F1 >=0.01 higher than strongest simple baseline, with zero collapsed seeds",
            "time": "validation NLL and MAE each >=1% better than strongest respective baseline",
            "approval": "at least one full-history task generalizes and all runs/slices are finite",
        },
        "mark_validation_relative_CE_gain": mark_ce_gain,
        "mark_validation_absolute_MacroF1_gain": mark_f1_gain,
        "time_validation_relative_NLL_gain": time_nll_gain,
        "time_validation_relative_MAE_gain": time_mae_gain,
        "stop_reason": "Group-Level Generalization Pilot complete; Group-SNMPP Design Gate not started.",
    }
    metrics = {
        "experiment": "Group-Level Generalization Pilot",
        "device": str(device),
        "config": {"seeds": SEEDS, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                   "learning_rate": LEARNING_RATE, "optimizer": "Adam", "hidden_size": 64,
                   "gradient_clip_norm": 5.0, "architecture_search": False,
                   "checkpoint_selection": "train objective only; validation never selects checkpoint"},
        "manifest": manifest,
        "baselines": baselines,
        "models": {"last_group_mark": last_summary, "full_history_mark": mark_summary, "full_history_time": time_summary},
        "per_seed": {"last_group_mark": last_runs, "full_history_mark": mark_runs, "full_history_time": time_runs},
        "validation_previous_group_size_slices_seed2026": clean_metric_payload(slices),
        "status": status,
    }
    save_json(WORK / "generalization_metrics.json", clean_metric_payload(metrics))
    save_json(WORK / "status.json", status)
    write_report(metrics, status)
    print(json.dumps({"GENERALIZATION_PILOT_COMPLETE": True, **status}, ensure_ascii=False), flush=True)


def write_report(metrics: dict, status: dict) -> None:
    baseline = metrics["baselines"]
    models = metrics["models"]
    mark = models["full_history_mark"]
    last = models["last_group_mark"]
    time_model = models["full_history_time"]
    lines = [
        "# Group-Level Generalization Pilot", "",
        "## Contract", "",
        f"- train targets: {metrics['manifest']['train_targets']:,}",
        f"- validation targets: {metrics['manifest']['validation_targets']:,}",
        f"- selected users: {metrics['manifest']['selected_users']:,}",
        "- user selection uses train-only stable hashing; no test data was read", 
        "- validation history is rolling and contains only groups strictly before each target", "",
        "## Baseline audit", "",
        "Global feedback prior means the category distribution of the complete D_SID train split. It is not the Tiny or moderate-Pilot target distribution.",
        f"Probabilities: {baseline['definition_audit']['global_feedback_probability']}", "",
        "## Mark", "",
        f"- full-history train CE: {mark['train']['ce_per_event']['mean']:.6f}",
        f"- full-history validation CE: {mark['validation']['ce_per_event']['mean']:.6f}",
        f"- full-history validation Macro-F1: {mark['validation']['macro_f1']['mean']:.6f}",
        f"- last-group validation CE: {last['validation']['ce_per_event']['mean']:.6f}",
        f"- last-group validation Macro-F1: {last['validation']['macro_f1']['mean']:.6f}",
        f"- validation CE gain vs strongest simple baseline: {status['mark_validation_relative_CE_gain']:.2%}",
        f"- validation Macro-F1 absolute gain: {status['mark_validation_absolute_MacroF1_gain']:.6f}", "",
        "## Time", "",
        f"- full-history train NLL: {time_model['train']['nll_per_group']['mean']:.6f}",
        f"- full-history validation NLL: {time_model['validation']['nll_per_group']['mean']:.6f}",
        f"- full-history validation MAE: {time_model['validation']['mae_hours']['mean']:.6f} hours",
        f"- full-history validation Median AE: {time_model['validation']['median_ae_hours']['mean']:.6f} hours",
        f"- validation NLL gain vs strongest parametric baseline: {status['time_validation_relative_NLL_gain']:.2%}",
        f"- validation MAE gain vs strongest point baseline: {status['time_validation_relative_MAE_gain']:.2%}", "",
        "## Decision", "",
        f"- group_level_mark_generalizes: `{str(status['group_level_mark_generalizes']).lower()}`",
        f"- group_level_time_generalizes: `{str(status['group_level_time_generalizes']).lower()}`",
        f"- full_history_generalization_supported: `{str(status['full_history_generalization_supported']).lower()}`",
        f"- group_snmpp_design_gate_approved: `{str(status['group_snmpp_design_gate_approved']).lower()}`", "",
        "This pilot stops here. It does not design or train Group-SNMPP.",
    ]
    (WORK / "Group_Level_Generalization_Pilot_Report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
