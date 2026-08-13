#!/usr/bin/env python3
"""Mark Generalization Confirmation Gate.

This gate evaluates only next timestamp-group feedback composition.  It keeps
the frozen D_SID chronological and group contracts, never reads test data, and
does not contain SNMPP, time prediction, SID prediction, HPN, or BOLA code.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from group_level_learnability_gate.run_gate import (  # noqa: E402
    CLASS_NAMES,
    CODEBOOK,
    SOURCE_MANIFEST,
    SOURCE_NPZ,
    MarkModel,
    PreparedData,
    SequenceDataset,
    _features_for_group_range,
    classification_metrics,
    clean_metric_payload,
    make_loader,
    mark_loss,
    save_json,
    seed_all,
    sha256,
)
from group_level_generalization_pilot.run_generalization import (  # noqa: E402
    _target_arrays,
    _user_start,
    stable_user_order,
)


WORK = Path(__file__).resolve().parent
SEEDS = [2026, 2027, 2028]
SELECTED_USERS = 2_000
MAX_TRAIN_TARGETS_PER_USER = 64
MAX_EPOCHS = 12
PATIENCE = 3
MIN_DELTA = 1e-4
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
HISTORY_BINS = [
    (1, 1, "1"),
    (2, 5, "2-5"),
    (6, 20, "6-20"),
    (21, 50, "21-50"),
    (51, 100, "51-100"),
    (101, 500, "101-500"),
    (501, np.iinfo(np.int64).max, ">500"),
]


def evenly_spaced_targets(indices: np.ndarray, cap: int) -> np.ndarray:
    """Select target positions without inspecting their labels."""
    if len(indices) <= cap:
        return indices
    positions = np.unique(np.rint(np.linspace(0, len(indices) - 1, cap)).astype(np.int64))
    return indices[positions]


def prepare() -> tuple[PreparedData, PreparedData, dict]:
    archive = np.load(SOURCE_NPZ, allow_pickle=False)
    uid = archive["group_uid"]
    timestamp = archive["group_timestamp"]
    global_id = archive["group_global_id"]
    offsets = archive["group_event_offsets"]
    counts = archive["group_feedback_counts"]
    event_sid = archive["event_sid"]
    codebooks = np.load(CODEBOOK).astype(np.float32)
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    train_cutoff = int(source_manifest["cutoffs"]["train_inclusive"])
    validation_cutoff = int(source_manifest["cutoffs"]["validation_inclusive"])

    starts = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1]])
    ends = np.r_[starts[1:], len(uid)]
    eligible = []
    for start, end in zip(starts, ends):
        user_timestamps = timestamp[start:end]
        n_train = int(np.searchsorted(user_timestamps, train_cutoff, side="right"))
        n_through_validation = int(np.searchsorted(user_timestamps, validation_cutoff, side="right"))
        n_validation = n_through_validation - n_train
        if n_train >= 6:
            eligible.append((int(uid[start]), int(start), int(end), n_train, n_validation))
    eligible_array = np.asarray(eligible, dtype=np.int64)
    ordered_users = stable_user_order(eligible_array[:, 0].astype(np.uint32))
    chosen_order = ordered_users[:SELECTED_USERS]
    chosen = set(map(int, chosen_order))
    selected_rows = [row for row in eligible if row[0] in chosen]
    selected_rows.sort(key=lambda row: row[1])
    selected_users = np.asarray(sorted(chosen), dtype=np.uint32)
    if len(selected_rows) != SELECTED_USERS:
        raise RuntimeError("failed to select the frozen 2,000-user subset")

    train_targets_parts: list[np.ndarray] = []
    validation_targets_parts: list[np.ndarray] = []
    for _, start, _, n_train, n_validation in selected_rows:
        all_user_train_targets = np.arange(start + 1, start + n_train, dtype=np.int64)
        train_targets_parts.append(evenly_spaced_targets(all_user_train_targets, MAX_TRAIN_TARGETS_PER_USER))
        if n_validation:
            validation_targets_parts.append(
                np.arange(start + n_train, start + n_train + n_validation, dtype=np.int64)
            )
    train_targets = np.concatenate(train_targets_parts)
    validation_targets = np.concatenate(validation_targets_parts)
    if len(validation_targets) == 0:
        raise RuntimeError("selected users have no validation targets")

    start_for_group = _user_start(uid)
    ranges: list[tuple[int, int, int]] = []
    for _, start, _, n_train, n_validation in selected_rows:
        # end is exclusive.  A target itself is never in its own history.
        final_target = start + n_train + n_validation - 1 if n_validation else start + n_train - 1
        last_train_target = start + n_train - 1
        ranges.append((start, final_target, last_train_target))
    total_groups = sum(end - start for start, end, _ in ranges)
    features = np.empty((total_groups, 134), dtype=np.float32)
    global_to_local = np.full(len(uid), -1, dtype=np.int64)
    scaler_mask = np.zeros(total_groups, dtype=bool)
    cursor = 0
    started = time.perf_counter()
    for number, (start, end, last_train_target) in enumerate(ranges, 1):
        block = _features_for_group_range(
            start,
            end,
            event_sid=event_sid,
            event_offsets=offsets,
            feedback_counts=counts,
            timestamps=timestamp,
            codebooks=codebooks,
        )
        features[cursor : cursor + len(block)] = block
        global_to_local[start:end] = np.arange(cursor, cursor + len(block))
        scaler_mask[cursor : cursor + (last_train_target - start)] = True
        cursor += len(block)
        if number % 250 == 0:
            print(json.dumps({"prepared_users": number, "groups": cursor}), flush=True)
    mean = features[scaler_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features[scaler_mask].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    features -= mean
    features /= std
    if not np.isfinite(features).all():
        raise RuntimeError("non-finite standardized feature")

    base_contract = {
        "dataset": "D_SID real-audio subset",
        "source": str(SOURCE_NPZ),
        "source_sha256": sha256(SOURCE_NPZ),
        "train_cutoff_inclusive": train_cutoff,
        "validation_cutoff_inclusive": validation_cutoff,
        "selection_population": "7,343 users already materialized through validation for prior train-only Tiny/Pilot selection",
        "selection_population_limitation": "materialized union is train-only selected but partly enriched by the prior Tiny stratification; it is not a fresh scan of all 854,649 D_SID users",
        "selected_user_rule": "first 2,000 eligible users in seed-2026 stable hash order; eligibility is >=6 train-period groups",
        "train_target_rule": "all chronological train targets when <=64, otherwise 64 evenly spaced positions; feedback labels are never inspected",
        "validation_target_rule": "all chronological validation targets for the frozen selected users",
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
        "validation_history": "all groups strictly before target, including preceding validation groups",
        "within_group_order_used": False,
        "listen_used": False,
        "test_used": False,
        "burst_removed": False,
        "feature_definition": "mean frozen SID semantics + feedback composition + log1p group size + log1p previous gap hours",
        "preparation_seconds": time.perf_counter() - started,
    }
    train = _target_arrays(
        train_targets,
        group_uid=uid,
        group_timestamp=timestamp,
        group_global_id=global_id,
        group_counts=counts,
        start_for_group=start_for_group,
        global_to_local=global_to_local,
        features=features,
        feature_mean=mean,
        feature_std=std,
        contract={**base_contract, "split": "train"},
    )
    validation = _target_arrays(
        validation_targets,
        group_uid=uid,
        group_timestamp=timestamp,
        group_global_id=global_id,
        group_counts=counts,
        start_for_group=start_for_group,
        global_to_local=global_to_local,
        features=features,
        feature_mean=mean,
        feature_std=std,
        contract={**base_contract, "split": "validation"},
    )

    selection_path = WORK / "selection_ids.npz"
    np.savez_compressed(
        selection_path,
        selected_users=selected_users,
        stable_hash_user_order=chosen_order,
        train_target_local_group_indices=train_targets,
        validation_target_local_group_indices=validation_targets,
        train_target_global_group_ids=global_id[train_targets],
        validation_target_global_group_ids=global_id[validation_targets],
    )
    manifest = {
        **base_contract,
        "seed": 2026,
        "selected_user_ids": selected_users.tolist(),
        "selection_artifact": str(selection_path),
        "selection_artifact_sha256": sha256(selection_path),
        "train_history_length_percentiles": {
            str(q): float(np.percentile(train.sequence_ends - train.sequence_starts, q))
            for q in (0, 50, 75, 90, 95, 99, 100)
        },
        "validation_history_length_percentiles": {
            str(q): float(np.percentile(validation.sequence_ends - validation.sequence_starts, q))
            for q in (0, 50, 75, 90, 95, 99, 100)
        },
    }
    return train, validation, manifest


class LimitedHistoryDataset(Dataset):
    """Same targets/features, with an optional most-recent group limit."""

    def __init__(self, data: PreparedData, limit: int | None) -> None:
        self.data = data
        self.limit = limit
        raw = data.sequence_ends - data.sequence_starts
        self.lengths = raw if limit is None else np.minimum(raw, limit)

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> dict:
        end = int(self.data.sequence_ends[index])
        start = int(self.data.sequence_starts[index])
        if self.limit is not None:
            start = max(start, end - self.limit)
        return {
            "sequence": torch.from_numpy(self.data.features[start:end]),
            "previous": torch.from_numpy(self.data.previous_features[index]),
            "counts": torch.from_numpy(self.data.target_counts[index]),
            "gap": torch.tensor(self.data.target_gap_hours[index]),
            "previous_size": torch.tensor(self.data.previous_sizes[index]),
            "index": torch.tensor(index),
        }


@torch.no_grad()
def predict(model: torch.nn.Module, dataset: Dataset, device: torch.device) -> tuple[dict, np.ndarray]:
    loader = make_loader(dataset, seed=2026, shuffle=False, batch_size=BATCH_SIZE)
    model.eval()
    records = []
    for batch in loader:
        indices = batch["index"].numpy()
        gpu = {key: value.to(device) for key, value in batch.items() if key not in {"index", "previous_size"}}
        probability = torch.softmax(model(gpu), dim=1).cpu().numpy()
        records.append((indices, probability))
    order = np.concatenate([row[0] for row in records])
    probability = np.concatenate([row[1] for row in records])[np.argsort(order)]
    return classification_metrics(probability, dataset.data.target_counts), probability


def fit_model(
    name: str,
    model: torch.nn.Module,
    train_data: PreparedData,
    validation_data: PreparedData,
    *,
    history_limit: int | None,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict, np.ndarray]:
    seed_all(seed)
    model.to(device)
    train_dataset = LimitedHistoryDataset(train_data, history_limit)
    validation_dataset = LimitedHistoryDataset(validation_data, history_limit)
    train_loader = make_loader(train_dataset, seed=seed, shuffle=True, batch_size=BATCH_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_state = None
    best_validation_ce = float("inf")
    best_epoch = 0
    stale_epochs = 0
    curves = []
    nonfinite_steps = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.perf_counter()
        model.train()
        numerator = denominator = 0.0
        for batch in train_loader:
            gpu = {key: value.to(device) for key, value in batch.items() if key not in {"index", "previous_size"}}
            optimizer.zero_grad(set_to_none=True)
            loss = mark_loss(model(gpu), gpu["counts"])
            if not torch.isfinite(loss):
                nonfinite_steps += 1
                continue
            loss.backward()
            optimizer.step()
            weight = float(gpu["counts"].sum())
            numerator += float(loss.detach()) * weight
            denominator += weight
        validation_metrics, _ = predict(model, validation_dataset, device)
        validation_ce = float(validation_metrics["ce_per_event"])
        improved = validation_ce < best_validation_ce - MIN_DELTA
        if improved:
            best_validation_ce = validation_ce
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_batch_ce_per_event": numerator / max(denominator, 1.0),
            "validation": clean_metric_payload(validation_metrics),
            "improved": improved,
            "seconds": time.perf_counter() - started,
        }
        curves.append(row)
        print(json.dumps({
            "model": name,
            "seed": seed,
            "epoch": epoch,
            "train_batch_ce": row["train_batch_ce_per_event"],
            "validation_ce": validation_ce,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "seconds": row["seconds"],
        }), flush=True)
        if stale_epochs >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f"{name} seed {seed} produced no checkpoint")
    model.load_state_dict(best_state)
    train_metrics, _ = predict(model, train_dataset, device)
    validation_metrics, validation_probability = predict(model, validation_dataset, device)
    result = {
        "name": name,
        "seed": seed,
        "history_limit": history_limit,
        "early_stopping_metric": "validation CE/event",
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "max_epochs": MAX_EPOCHS,
        "best_epoch": best_epoch,
        "stopped_epoch": curves[-1]["epoch"],
        "train": clean_metric_payload(train_metrics),
        "validation": clean_metric_payload(validation_metrics),
        "train_validation_ce_gap": validation_metrics["ce_per_event"] - train_metrics["ce_per_event"],
        "train_validation_macro_f1_gap": validation_metrics["macro_f1"] - train_metrics["macro_f1"],
        "nonfinite_steps": nonfinite_steps,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "curves": curves,
    }
    return model, result, validation_probability


def baseline_metrics(data: PreparedData, global_probability: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    global_prediction = np.repeat(global_probability[None], len(data.target_counts), axis=0)
    previous_composition = data.previous_features[:, 128:132] * data.feature_std[128:132] + data.feature_mean[128:132]
    previous_size = data.previous_sizes[:, None].astype(np.float64)
    previous_prediction = (previous_composition * previous_size + global_probability) / (previous_size + 1.0)
    return ({
        "global_full_train_feedback_prior": classification_metrics(global_prediction, data.target_counts),
        "previous_group_composition_global_Dirichlet_strength_1": classification_metrics(previous_prediction, data.target_counts),
    }, {"global": global_prediction, "previous_composition": previous_prediction})


def history_slices(data: PreparedData, predictions: dict[str, np.ndarray]) -> dict:
    lengths = data.sequence_ends - data.sequence_starts
    output = {}
    for lower, upper, label in HISTORY_BINS:
        mask = (lengths >= lower) & (lengths <= upper)
        if not mask.any():
            output[label] = {"targets": 0}
            continue
        output[label] = {
            "targets": int(mask.sum()),
            "events": int(data.target_counts[mask].sum()),
            "history_length_mean": float(lengths[mask].mean()),
            "models": {
                name: classification_metrics(probability[mask], data.target_counts[mask])
                for name, probability in predictions.items()
            },
        }
    return output


def summarize_runs(runs: list[dict]) -> dict:
    result = {"seeds": [row["seed"] for row in runs]}
    for split in ("train", "validation"):
        result[split] = {}
        for metric in ("ce_per_event", "macro_f1", "accuracy"):
            values = [row[split][metric] for row in runs]
            result[split][metric] = {"values": values, "mean": float(np.mean(values)), "std": float(np.std(values))}
    result["best_epochs"] = [row["best_epoch"] for row in runs]
    result["stopped_epochs"] = [row["stopped_epoch"] for row in runs]
    result["validation_per_class_recall"] = {
        klass: {
            "values": [row["validation"]["per_class_recall"][klass] for row in runs],
            "mean": float(np.mean([row["validation"]["per_class_recall"][klass] for row in runs])),
        }
        for klass in CLASS_NAMES
    }
    result["validation_predicted_argmax_distribution"] = {
        klass: {
            "values": [row["validation"]["predicted_argmax_distribution"][index] for row in runs],
            "mean": float(np.mean([row["validation"]["predicted_argmax_distribution"][index] for row in runs])),
        }
        for index, klass in enumerate(CLASS_NAMES)
    }
    result["collapsed_seeds"] = int(sum(row["validation"]["collapsed"] for row in runs))
    return result


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, validation, manifest = prepare()
    save_json(WORK / "pilot_manifest.json", manifest)
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    global_probability = np.asarray(source_manifest["baselines"]["global_train_feedback_probabilities"], dtype=np.float64)
    train_baselines, _ = baseline_metrics(train, global_probability)
    validation_baselines, validation_baseline_predictions = baseline_metrics(validation, global_probability)
    baselines = {
        "definition": "global prior uses the complete D_SID train split, never this Pilot subset",
        "global_train_feedback_counts": source_manifest["baselines"]["global_train_feedback_counts"],
        "global_train_feedback_probabilities": global_probability.tolist(),
        "train": clean_metric_payload(train_baselines),
        "validation": clean_metric_payload(validation_baselines),
    }

    run_sets: dict[str, list[dict]] = {"last_group_mlp": [], "last5_gru": [], "full_history_gru": []}
    slice_rows = []
    for seed in SEEDS:
        seed_dir = WORK / "runs" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_predictions = dict(validation_baseline_predictions)
        specifications = [
            ("last_group_mlp", MarkModel(False), 1),
            ("last5_gru", MarkModel(True), 5),
            ("full_history_gru", MarkModel(True), None),
        ]
        for name, model, history_limit in specifications:
            trained, metrics, probability = fit_model(
                name,
                model,
                train,
                validation,
                history_limit=history_limit,
                seed=seed,
                device=device,
            )
            torch.save(trained.state_dict(), seed_dir / f"{name}_best.pt")
            save_json(seed_dir / f"{name}_metrics.json", clean_metric_payload(metrics))
            run_sets[name].append(metrics)
            seed_predictions[name] = probability
        slices = history_slices(validation, seed_predictions)
        save_json(seed_dir / "history_length_slices.json", clean_metric_payload(slices))
        slice_rows.append(slices)

    summaries = {name: summarize_runs(rows) for name, rows in run_sets.items()}
    previous_ce = validation_baselines["previous_group_composition_global_Dirichlet_strength_1"]["ce_per_event"]
    full_values = summaries["full_history_gru"]["validation"]["ce_per_event"]["values"]
    last_values = summaries["last_group_mlp"]["validation"]["ce_per_event"]["values"]
    last5_values = summaries["last5_gru"]["validation"]["ce_per_event"]["values"]
    beats_previous_each_seed = all(value < previous_ce for value in full_values)
    beats_last_each_seed = all(full < last for full, last in zip(full_values, last_values))
    beats_last5_each_seed = all(full < last5 for full, last5 in zip(full_values, last5_values))
    confirmed = bool(beats_previous_each_seed and beats_last_each_seed)

    full_mean_ce = summaries["full_history_gru"]["validation"]["ce_per_event"]["mean"]
    previous_relative_gain = (previous_ce - full_mean_ce) / previous_ce
    last_relative_gain = (
        summaries["last_group_mlp"]["validation"]["ce_per_event"]["mean"] - full_mean_ce
    ) / summaries["last_group_mlp"]["validation"]["ce_per_event"]["mean"]
    last5_relative_gain = (
        summaries["last5_gru"]["validation"]["ce_per_event"]["mean"] - full_mean_ce
    ) / summaries["last5_gru"]["validation"]["ce_per_event"]["mean"]
    full_undislike_recall = summaries["full_history_gru"]["validation_per_class_recall"]["undislike"]["mean"]
    full_like_prediction_share = summaries["full_history_gru"]["validation_predicted_argmax_distribution"]["like"]["mean"]

    status = {
        "mark_long_history_generalization_confirmed": confirmed,
        "group_snmpp_design_approved": confirmed,
        "full_history_beats_previous_composition_all_3_seeds": beats_previous_each_seed,
        "full_history_beats_last_group_mlp_all_3_seeds": beats_last_each_seed,
        "full_history_beats_last5_gru_all_3_seeds_diagnostic": beats_last5_each_seed,
        "validation_CE_relative_gain_vs_previous_composition": previous_relative_gain,
        "validation_CE_relative_gain_vs_last_group_MLP": last_relative_gain,
        "validation_CE_relative_gain_vs_last5_GRU": last5_relative_gain,
        "severe_undislike_failure_remains": bool(full_undislike_recall < 0.05),
        "full_history_mean_undislike_recall": full_undislike_recall,
        "full_history_mean_like_argmax_share": full_like_prediction_share,
        "confirmation_scope": "the fixed 2,000-user subset selected from the already materialized 7,343-user train-only pool",
        "population_scope_limitation": "the materialized pool is partly enriched by prior Tiny train stratification and is not a fresh random sample of all D_SID users",
        "test_used": False,
        "time_model_repeated": False,
        "group_snmpp_designed_or_trained": False,
        "decision_rule": "approve iff Full-history GRU validation CE is lower than previous-group composition and last-group MLP for every seed",
        "stop_reason": "Mark Generalization Confirmation Gate complete; Group-SNMPP was not implemented.",
    }
    metrics = {
        "experiment": "Mark Generalization Confirmation Gate",
        "device": str(device),
        "config": {
            "seeds": SEEDS,
            "selected_users": SELECTED_USERS,
            "max_train_targets_per_user": MAX_TRAIN_TARGETS_PER_USER,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "early_stopping_metric": "validation CE/event",
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "hidden_size": 64,
            "architecture_search": False,
        },
        "manifest": manifest,
        "baselines": baselines,
        "models": summaries,
        "per_seed": {name: clean_metric_payload(rows) for name, rows in run_sets.items()},
        "history_length_slices_per_seed": clean_metric_payload(slice_rows),
        "status": status,
    }
    save_json(WORK / "mark_generalization_metrics.json", clean_metric_payload(metrics))
    save_json(WORK / "status.json", status)
    write_report(metrics)
    print(json.dumps({"MARK_GENERALIZATION_CONFIRMATION_COMPLETE": True, **status}), flush=True)


def write_report(metrics: dict) -> None:
    status = metrics["status"]
    models = metrics["models"]
    baselines = metrics["baselines"]["validation"]
    full = models["full_history_gru"]
    last = models["last_group_mlp"]
    last5 = models["last5_gru"]
    previous = baselines["previous_group_composition_global_Dirichlet_strength_1"]
    lines = [
        "# Mark Generalization Confirmation Gate",
        "",
        "## Protocol",
        "",
        f"- selected users: {metrics['manifest']['selected_users']:,}",
        f"- train targets: {metrics['manifest']['train_targets']:,}",
        f"- validation targets: {metrics['manifest']['validation_targets']:,}",
        "- user selection uses train-period eligibility and stable hashing only",
        "- validation contains all validation-period targets of selected users",
        "- early stopping was predeclared: validation CE/event, patience 3, max 12 epochs",
        "- no test, Time, SNMPP, SID prediction, HPN, BOLA, listen, or burst filtering",
        "",
        "## Validation CE/event",
        "",
        f"- previous-group composition: {previous['ce_per_event']:.6f}",
        f"- last-group MLP: {last['validation']['ce_per_event']['mean']:.6f} ± {last['validation']['ce_per_event']['std']:.6f}",
        f"- last-5 GRU: {last5['validation']['ce_per_event']['mean']:.6f} ± {last5['validation']['ce_per_event']['std']:.6f}",
        f"- full-history GRU: {full['validation']['ce_per_event']['mean']:.6f} ± {full['validation']['ce_per_event']['std']:.6f}",
        f"- full-history best epochs: {full['best_epochs']}",
        "",
        "## Decision",
        "",
        f"- beats previous composition in all seeds: `{str(status['full_history_beats_previous_composition_all_3_seeds']).lower()}`",
        f"- beats last-group MLP in all seeds: `{str(status['full_history_beats_last_group_mlp_all_3_seeds']).lower()}`",
        f"- beats last-5 GRU in all seeds (diagnostic): `{str(status['full_history_beats_last5_gru_all_3_seeds_diagnostic']).lower()}`",
        f"- mark_long_history_generalization_confirmed: `{str(status['mark_long_history_generalization_confirmed']).lower()}`",
        f"- group_snmpp_design_approved: `{str(status['group_snmpp_design_approved']).lower()}`",
        "",
        "This gate stops here and does not implement Group-SNMPP.",
    ]
    (WORK / "Mark_Generalization_Confirmation_Report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
