#!/usr/bin/env python3
"""Group-Level Learnability Gate on the frozen 6,413-group Tiny subset.

This deliberately contains no SNMPP interaction/kernel, SID prediction head,
listen event, HPN, or BOLA component.  Group representations are permutation
invariant and histories end strictly before the target timestamp group.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = Path(__file__).resolve().parents[1]
WORK = Path(__file__).resolve().parent
SOURCE_NPZ = ROOT / "minimal_snmpp_pilot/data/pilot_sequences.npz"
SOURCE_MANIFEST = ROOT / "minimal_snmpp_pilot/data/subset_manifest.json"
CODEBOOK = ROOT / "phase2_gate2_sid/materialized_v1_1/frozen_sid/codebooks.npy"
N_CLASSES = 4
CLASS_NAMES = ["like", "dislike", "unlike", "undislike"]
SEEDS = [2026, 2027, 2028]
PREVIOUS_SIZE_BINS = [(1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"),
                      (21, 100, "21-100"), (101, 500, "101-500"),
                      (501, np.iinfo(np.int64).max, ">500")]


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


@dataclass
class PreparedData:
    features: np.ndarray
    sequence_starts: np.ndarray
    sequence_ends: np.ndarray
    previous_features: np.ndarray
    target_counts: np.ndarray
    target_gap_hours: np.ndarray
    previous_gap_hours: np.ndarray
    previous_sizes: np.ndarray
    target_group_ids: np.ndarray
    target_uids: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    contract: dict


def _features_for_group_range(
    start: int,
    end: int,
    *,
    event_sid: np.ndarray,
    event_offsets: np.ndarray,
    feedback_counts: np.ndarray,
    timestamps: np.ndarray,
    codebooks: np.ndarray,
) -> np.ndarray:
    """Return [mean SID semantics, feedback composition, log size, log gap]."""
    event_start = int(event_offsets[start])
    event_end = int(event_offsets[end])
    sid = event_sid[event_start:event_end]
    semantic = (
        codebooks[0, sid[:, 0]] + codebooks[1, sid[:, 1]]
        + codebooks[2, sid[:, 2]] + codebooks[3, sid[:, 3]]
    ).astype(np.float32, copy=False)
    local_offsets = event_offsets[start:end].astype(np.int64) - event_start
    semantic_sum = np.add.reduceat(semantic, local_offsets, axis=0)
    sizes = feedback_counts[start:end].sum(axis=1).astype(np.float32)
    semantic_mean = semantic_sum / sizes[:, None]
    composition = feedback_counts[start:end].astype(np.float32) / sizes[:, None]
    log_size = np.log1p(sizes)[:, None]
    gaps = np.zeros(end - start, dtype=np.float32)
    if end - start > 1:
        gaps[1:] = np.diff(timestamps[start:end].astype(np.float64)) / 3600.0
    log_gap = np.log1p(gaps)[:, None]
    result = np.concatenate([semantic_mean, composition, log_size, log_gap], axis=1)
    if result.shape[1] != 134 or not np.isfinite(result).all():
        raise RuntimeError("invalid group representation")
    return result


def prepare_data() -> PreparedData:
    archive = np.load(SOURCE_NPZ, allow_pickle=False)
    codebooks = np.load(CODEBOOK).astype(np.float32)
    if codebooks.shape != (4, 256, 128):
        raise RuntimeError(f"unexpected codebook shape {codebooks.shape}")
    event_sid = archive["event_sid"]
    offsets = archive["group_event_offsets"]
    counts = archive["group_feedback_counts"]
    uids = archive["group_uid"]
    timestamps = archive["group_timestamp"]
    group_ids = archive["group_global_id"]
    targets = archive["tiny_train_targets"].astype(np.int64)

    user_starts = np.flatnonzero(np.r_[True, uids[1:] != uids[:-1]])
    marker = np.zeros(len(uids), dtype=np.int64)
    marker[user_starts] = user_starts
    start_for_group = np.maximum.accumulate(marker)

    # One union prefix per selected user; no target group enters its own history.
    ranges: dict[int, tuple[int, int]] = {}
    for target in targets:
        uid = int(uids[target])
        start = int(start_for_group[target])
        prior_end = ranges.get(uid, (start, start))[1]
        ranges[uid] = (start, max(prior_end, int(target)))
    ordered_ranges = sorted(ranges.values())
    total_groups = sum(end - start for start, end in ordered_ranges)
    features = np.empty((total_groups, 134), dtype=np.float32)
    global_to_local = np.full(len(uids), -1, dtype=np.int64)
    cursor = 0
    started = time.perf_counter()
    for number, (start, end) in enumerate(ordered_ranges, 1):
        block = _features_for_group_range(
            start, end, event_sid=event_sid, event_offsets=offsets,
            feedback_counts=counts, timestamps=timestamps, codebooks=codebooks,
        )
        features[cursor:cursor + len(block)] = block
        global_to_local[start:end] = np.arange(cursor, cursor + len(block))
        cursor += len(block)
        if number % 1000 == 0:
            print(json.dumps({"prepare_ranges": number, "groups": cursor}), flush=True)

    starts = np.empty(len(targets), dtype=np.int64)
    ends = np.empty(len(targets), dtype=np.int64)
    for row, target in enumerate(targets):
        global_start = int(start_for_group[target])
        starts[row] = int(global_to_local[global_start])
        ends[row] = int(global_to_local[target - 1]) + 1
        if starts[row] < 0 or ends[row] <= starts[row]:
            raise RuntimeError("invalid strict pre-target history range")

    # Standardization is fitted exclusively on strict pre-target train histories.
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    features -= mean
    features /= std
    previous_features = features[ends - 1].copy()

    target_gap = (timestamps[targets].astype(np.float64) - timestamps[targets - 1]) / 3600.0
    previous_gap = (
        timestamps[targets - 1].astype(np.float64) - timestamps[targets - 2].astype(np.float64)
    ) / 3600.0
    if np.any(target_gap <= 0) or np.any(previous_gap <= 0):
        raise RuntimeError("group-level gaps must be strictly positive")

    # Explicit permutation-invariance sanity on groups with multiple events.
    multi = np.flatnonzero(counts.sum(axis=1) > 1)
    rng = np.random.default_rng(2026)
    checked = multi[rng.choice(len(multi), size=min(100, len(multi)), replace=False)]
    maximum_difference = 0.0
    for group in checked:
        begin, finish = int(offsets[group]), int(offsets[group + 1])
        sid = event_sid[begin:finish]
        original = sum(codebooks[level, sid[:, level]].sum(axis=0) for level in range(4)) / len(sid)
        permutation = rng.permutation(len(sid))
        shuffled = sum(codebooks[level, sid[permutation, level]].sum(axis=0) for level in range(4)) / len(sid)
        maximum_difference = max(maximum_difference, float(np.max(np.abs(original - shuffled))))
    if maximum_difference > 2e-5:
        raise RuntimeError(f"group representation is not permutation invariant: {maximum_difference}")

    contract = {
        "source_npz": str(SOURCE_NPZ),
        "source_npz_sha256": sha256(SOURCE_NPZ),
        "codebook": str(CODEBOOK),
        "codebook_sha256": sha256(CODEBOOK),
        "target_groups": int(len(targets)),
        "target_events": int(counts[targets].sum()),
        "target_users": int(len(np.unique(uids[targets]))),
        "union_history_groups": int(total_groups),
        "history_group_length_percentiles": {
            str(q): float(np.percentile(ends - starts, q)) for q in (0, 50, 90, 95, 99, 100)
        },
        "permutation_check_groups": int(len(checked)),
        "permutation_check_max_abs_difference": maximum_difference,
        "strict_pre_target_history": True,
        "within_group_order_used": False,
        "listen_used": False,
        "test_used": False,
        "feature_definition": "mean_frozen_SID_semantics_128 + feedback_composition_4 + log1p_group_size + log1p_previous_gap_hours",
        "preparation_seconds": time.perf_counter() - started,
    }
    return PreparedData(
        features=features,
        sequence_starts=starts,
        sequence_ends=ends,
        previous_features=previous_features,
        target_counts=counts[targets].astype(np.float32),
        target_gap_hours=target_gap.astype(np.float32),
        previous_gap_hours=previous_gap.astype(np.float32),
        previous_sizes=counts[targets - 1].sum(axis=1).astype(np.int64),
        target_group_ids=group_ids[targets].astype(np.uint64),
        target_uids=uids[targets].astype(np.uint32),
        feature_mean=mean,
        feature_std=std,
        contract=contract,
    )


class SequenceDataset(Dataset):
    def __init__(self, data: PreparedData) -> None:
        self.data = data
        self.lengths = data.sequence_ends - data.sequence_starts

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> dict:
        start, end = int(self.data.sequence_starts[index]), int(self.data.sequence_ends[index])
        return {
            "sequence": torch.from_numpy(self.data.features[start:end]),
            "previous": torch.from_numpy(self.data.previous_features[index]),
            "counts": torch.from_numpy(self.data.target_counts[index]),
            "gap": torch.tensor(self.data.target_gap_hours[index]),
            "previous_size": torch.tensor(self.data.previous_sizes[index]),
            "index": torch.tensor(index),
        }


class LengthBucketSampler(Sampler[list[int]]):
    def __init__(self, lengths: np.ndarray, batch_size: int, seed: int, shuffle: bool) -> None:
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0
        order = np.argsort(lengths, kind="stable")
        self.batches = [order[i:i + batch_size].tolist() for i in range(0, len(order), batch_size)]

    def __iter__(self) -> Iterable[list[int]]:
        order = np.arange(len(self.batches))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        for index in order:
            yield self.batches[int(index)]

    def __len__(self) -> int:
        return len(self.batches)


def collate(rows: list[dict]) -> dict[str, torch.Tensor]:
    lengths = torch.tensor([len(row["sequence"]) for row in rows], dtype=torch.long)
    maximum = int(lengths.max())
    sequence = torch.zeros(len(rows), maximum, 134)
    for index, row in enumerate(rows):
        sequence[index, : lengths[index]] = row["sequence"]
    return {
        "sequence": sequence,
        "lengths": lengths,
        "previous": torch.stack([row["previous"] for row in rows]),
        "counts": torch.stack([row["counts"] for row in rows]),
        "gap": torch.stack([row["gap"] for row in rows]),
        "previous_size": torch.stack([row["previous_size"] for row in rows]),
        "index": torch.stack([row["index"] for row in rows]),
    }


def make_loader(dataset: SequenceDataset, seed: int, shuffle: bool, batch_size: int = 64) -> DataLoader:
    sampler = LengthBucketSampler(dataset.lengths, batch_size=batch_size, seed=seed, shuffle=shuffle)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=0)


class HistoryEncoder(nn.Module):
    def __init__(self, input_dim: int = 134, hidden: int = 64) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(self.input_norm(sequence))
        packed = pack_padded_sequence(projected, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return hidden[-1]


class MarkModel(nn.Module):
    def __init__(self, recurrent: bool, hidden: int = 64) -> None:
        super().__init__()
        self.recurrent = recurrent
        if recurrent:
            self.encoder = HistoryEncoder(hidden=hidden)
            self.head = nn.Linear(hidden, N_CLASSES)
        else:
            self.encoder = nn.Sequential(nn.LayerNorm(134), nn.Linear(134, hidden), nn.Tanh())
            self.head = nn.Linear(hidden, N_CLASSES)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state = self.encoder(batch["sequence"], batch["lengths"]) if self.recurrent else self.encoder(batch["previous"])
        return self.head(state)


class TimeModel(nn.Module):
    def __init__(self, recurrent: bool = True, hidden: int = 64) -> None:
        super().__init__()
        self.recurrent = recurrent
        if recurrent:
            self.encoder = HistoryEncoder(hidden=hidden)
        else:
            self.encoder = nn.Sequential(nn.LayerNorm(134), nn.Linear(134, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encoder(batch["sequence"], batch["lengths"]) if self.recurrent else self.encoder(batch["previous"])
        raw = self.head(state)
        sigma = torch.nn.functional.softplus(raw[:, 1]) + 0.05
        return raw[:, 0], sigma


class JointModel(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = HistoryEncoder(hidden=hidden)
        self.mark_head = nn.Linear(hidden, N_CLASSES)
        self.time_head = nn.Linear(hidden, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.encoder(batch["sequence"], batch["lengths"])
        time_raw = self.time_head(state)
        return self.mark_head(state), time_raw[:, 0], torch.nn.functional.softplus(time_raw[:, 1]) + 0.05


def lognormal_nll(gap: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    log_gap = torch.log(gap.clamp_min(1e-8))
    return torch.log(gap.clamp_min(1e-8)) + torch.log(sigma) + 0.5 * math.log(2 * math.pi) + 0.5 * ((log_gap - mu) / sigma) ** 2


def mark_loss(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return -(counts * torch.log_softmax(logits, dim=1)).sum() / counts.sum()


def classification_metrics(probabilities: np.ndarray, counts: np.ndarray) -> dict:
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for row, prediction in enumerate(predicted):
        confusion[:, prediction] += counts[row].astype(np.int64)
    recalls, precisions, f1s = [], [], []
    for klass in range(N_CLASSES):
        tp = confusion[klass, klass]
        recall = tp / max(confusion[klass].sum(), 1)
        precision = tp / max(confusion[:, klass].sum(), 1)
        f1 = 2 * recall * precision / max(recall + precision, 1e-15)
        recalls.append(float(recall)); precisions.append(float(precision)); f1s.append(float(f1))
    total_events = counts.sum()
    ce = float(-(counts * np.log(np.clip(probabilities, 1e-12, 1))).sum() / total_events)
    weighted_prob = (probabilities * counts.sum(axis=1, keepdims=True)).sum(axis=0) / total_events
    return {
        "ce_per_event": ce,
        "accuracy": float(np.trace(confusion) / max(confusion.sum(), 1)),
        "macro_f1": float(np.mean(f1s)),
        "per_class_recall": dict(zip(CLASS_NAMES, recalls)),
        "per_class_precision": dict(zip(CLASS_NAMES, precisions)),
        "predicted_argmax_distribution": (np.bincount(predicted, minlength=4) / len(predicted)).tolist(),
        "event_weighted_predicted_probability": weighted_prob.tolist(),
        "confusion_matrix": confusion.tolist(),
        "collapsed": bool(np.count_nonzero(np.bincount(predicted, minlength=4)) == 1),
    }


def time_metrics(mu: np.ndarray, sigma: np.ndarray, gaps: np.ndarray) -> dict:
    log_gap = np.log(np.clip(gaps, 1e-8, None))
    nll_rows = np.log(gaps) + np.log(sigma) + 0.5 * np.log(2 * np.pi) + 0.5 * ((log_gap - mu) / sigma) ** 2
    prediction = np.exp(np.clip(mu, -20, 20))  # predictive median, optimal for absolute error
    absolute = np.abs(prediction - gaps)
    return {
        "nll_per_group": float(nll_rows.mean()),
        "mae_hours": float(absolute.mean()),
        "median_ae_hours": float(np.median(absolute)),
        "prediction_median_hours": float(np.median(prediction)),
        "prediction_p95_hours": float(np.percentile(prediction, 95)),
        "finite": bool(np.isfinite(nll_rows).all() and np.isfinite(prediction).all()),
        "rows": {"nll": nll_rows, "absolute_error": absolute, "prediction": prediction},
    }


@torch.no_grad()
def predict_model(model: nn.Module, loader: DataLoader, device: torch.device, task: str) -> dict:
    model.eval()
    rows: list[tuple[np.ndarray, ...]] = []
    hidden_norms: list[np.ndarray] = []
    for batch in loader:
        indices = batch["index"].numpy()
        gpu = {key: value.to(device) for key, value in batch.items() if key not in {"index", "previous_size"}}
        if task == "mark":
            logits = model(gpu)
            rows.append((indices, torch.softmax(logits, 1).cpu().numpy()))
        elif task == "time":
            mu, sigma = model(gpu)
            rows.append((indices, mu.cpu().numpy(), sigma.cpu().numpy()))
        else:
            logits, mu, sigma = model(gpu)
            rows.append((indices, torch.softmax(logits, 1).cpu().numpy(), mu.cpu().numpy(), sigma.cpu().numpy()))
    order = np.concatenate([row[0] for row in rows])
    sorting = np.argsort(order)
    outputs = [np.concatenate([row[column] for row in rows], axis=0)[sorting] for column in range(1, len(rows[0]))]
    return {"outputs": outputs, "hidden_norms": hidden_norms}


def train_model(
    model: nn.Module,
    dataset: SequenceDataset,
    *,
    task: str,
    seed: int,
    device: torch.device,
    epochs: int = 20,
    learning_rate: float = 1e-3,
) -> tuple[nn.Module, list[dict], dict]:
    seed_all(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = make_loader(dataset, seed=seed, shuffle=True)
    eval_loader = make_loader(dataset, seed=seed, shuffle=False)
    best_state, best_loss = None, float("inf")
    curves = []
    clipping_steps = 0
    total_steps = 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        weighted_loss, denominator = 0.0, 0.0
        nonfinite = 0
        for batch in train_loader:
            gpu = {key: value.to(device) for key, value in batch.items() if key not in {"index", "previous_size"}}
            optimizer.zero_grad(set_to_none=True)
            if task == "mark":
                loss = mark_loss(model(gpu), gpu["counts"])
                weight = float(gpu["counts"].sum())
            elif task == "time":
                mu, sigma = model(gpu)
                loss = lognormal_nll(gpu["gap"], mu, sigma).mean()
                weight = len(gpu["gap"])
            else:
                logits, mu, sigma = model(gpu)
                loss = mark_loss(logits, gpu["counts"]) + lognormal_nll(gpu["gap"], mu, sigma).mean()
                weight = len(gpu["gap"])
            if not torch.isfinite(loss):
                nonfinite += 1
                continue
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            clipping_steps += int(float(norm) > 5.0)
            total_steps += 1
            optimizer.step()
            weighted_loss += float(loss.detach()) * weight
            denominator += weight
        prediction = predict_model(model, eval_loader, device, task)["outputs"]
        if task == "mark":
            evaluation = classification_metrics(prediction[0], dataset.data.target_counts)
            selection_loss = evaluation["ce_per_event"]
        elif task == "time":
            evaluation = time_metrics(prediction[0], prediction[1], dataset.data.target_gap_hours)
            evaluation.pop("rows")
            selection_loss = evaluation["nll_per_group"]
        else:
            mark_eval = classification_metrics(prediction[0], dataset.data.target_counts)
            time_eval = time_metrics(prediction[1], prediction[2], dataset.data.target_gap_hours)
            time_eval.pop("rows")
            evaluation = {"mark": mark_eval, "time": time_eval}
            selection_loss = mark_eval["ce_per_event"] + time_eval["nll_per_group"]
        record = {
            "epoch": epoch,
            "train_batch_objective": weighted_loss / max(denominator, 1),
            "evaluation": evaluation,
            "seconds": time.perf_counter() - started,
            "nonfinite_steps": nonfinite,
        }
        curves.append(record)
        print(json.dumps({"task": task, "seed": seed, "epoch": epoch, "loss": selection_loss, "seconds": record["seconds"]}), flush=True)
        if selection_loss < best_loss and math.isfinite(selection_loss):
            best_loss = selection_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"{task} produced no finite checkpoint")
    model.load_state_dict(best_state)
    final_prediction = predict_model(model, eval_loader, device, task)["outputs"]
    diagnostics = {
        "best_selection_loss": best_loss,
        "gradient_clip_frequency": clipping_steps / max(total_steps, 1),
        "nonfinite_steps": int(sum(row["nonfinite_steps"] for row in curves)),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    if task == "mark":
        diagnostics["metrics"] = classification_metrics(final_prediction[0], dataset.data.target_counts)
        diagnostics["probabilities"] = final_prediction[0]
    elif task == "time":
        diagnostics["metrics"] = time_metrics(final_prediction[0], final_prediction[1], dataset.data.target_gap_hours)
        diagnostics["mu"], diagnostics["sigma"] = final_prediction
    else:
        diagnostics["metrics"] = {
            "mark": classification_metrics(final_prediction[0], dataset.data.target_counts),
            "time": time_metrics(final_prediction[1], final_prediction[2], dataset.data.target_gap_hours),
        }
    return model, curves, diagnostics


def mark_baselines(data: PreparedData, global_probability: np.ndarray) -> dict:
    n = len(data.target_counts)
    global_predictions = np.repeat(global_probability[None], n, axis=0)
    # One event-equivalent Dirichlet shrinkage prevents undefined zero probability.
    previous_raw = data.previous_features[:, 128:132] * data.feature_std[128:132] + data.feature_mean[128:132]
    previous_sizes = data.previous_sizes[:, None].astype(np.float64)
    previous_predictions = (previous_raw * previous_sizes + global_probability) / (previous_sizes + 1.0)
    target_prop = data.target_counts / data.target_counts.sum(axis=1, keepdims=True)
    oracle = classification_metrics(target_prop, data.target_counts)
    return {
        "global_train_feedback_prior": classification_metrics(global_predictions, data.target_counts),
        "previous_group_composition_with_global_Dirichlet_strength_1": classification_metrics(previous_predictions, data.target_counts),
        "shared_group_oracle": oracle,
    }


def constant_time_metrics(prediction: np.ndarray, gaps: np.ndarray) -> dict:
    error = np.abs(prediction - gaps)
    return {"mae_hours": float(error.mean()), "median_ae_hours": float(np.median(error)),
            "prediction_median_hours": float(np.median(prediction)), "finite": bool(np.isfinite(error).all())}


def time_baselines(data: PreparedData, global_median_hours: float) -> dict:
    gaps = data.target_gap_hours.astype(np.float64)
    log_gap = np.log(gaps)
    mu, sigma = float(log_gap.mean()), max(float(log_gap.std()), 0.05)
    history_free = time_metrics(np.full(len(gaps), mu), np.full(len(gaps), sigma), gaps)
    history_free.pop("rows")
    residual = log_gap - np.log(data.previous_gap_hours.astype(np.float64))
    residual_mu, residual_sigma = float(residual.mean()), max(float(residual.std()), 0.05)
    previous_mu = np.log(data.previous_gap_hours) + residual_mu
    previous_parametric = time_metrics(previous_mu, np.full(len(gaps), residual_sigma), gaps)
    previous_parametric.pop("rows")
    global_point = constant_time_metrics(np.full(len(gaps), global_median_hours), gaps)
    previous_point = constant_time_metrics(data.previous_gap_hours.astype(np.float64), gaps)
    return {
        "global_train_median_gap": global_point,
        "history_free_lognormal_fit_on_tiny_train": {**history_free, "mu": mu, "sigma": sigma},
        "previous_gap_point": previous_point,
        "previous_gap_lognormal_residual_fit_on_tiny_train": {**previous_parametric, "residual_mu": residual_mu, "residual_sigma": residual_sigma},
    }


def clean_metric_payload(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: clean_metric_payload(item) for key, item in value.items() if key != "rows"}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def slice_metrics(data: PreparedData, probabilities: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> dict:
    result = {}
    for lower, upper, label in PREVIOUS_SIZE_BINS:
        mask = (data.previous_sizes >= lower) & (data.previous_sizes <= upper)
        if not mask.any():
            result[label] = {"groups": 0}
            continue
        mark = classification_metrics(probabilities[mask], data.target_counts[mask])
        time_result = time_metrics(mu[mask], sigma[mask], data.target_gap_hours[mask])
        time_result.pop("rows")
        result[label] = {
            "groups": int(mask.sum()), "events": int(data.target_counts[mask].sum()),
            "mark": mark, "time": time_result,
            "outputs_finite": bool(np.isfinite(probabilities[mask]).all() and np.isfinite(mu[mask]).all() and np.isfinite(sigma[mask]).all()),
        }
    return result


def mean_std(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}


def summarize_runs(runs: list[dict], task: str) -> dict:
    if task == "mark":
        return {
            "ce_per_event": mean_std([run["metrics"]["ce_per_event"] for run in runs]),
            "macro_f1": mean_std([run["metrics"]["macro_f1"] for run in runs]),
            "accuracy": mean_std([run["metrics"]["accuracy"] for run in runs]),
            "collapsed_seeds": int(sum(run["metrics"]["collapsed"] for run in runs)),
            "nonfinite_steps": int(sum(run["nonfinite_steps"] for run in runs)),
        }
    return {
        "nll_per_group": mean_std([run["metrics"]["nll_per_group"] for run in runs]),
        "mae_hours": mean_std([run["metrics"]["mae_hours"] for run in runs]),
        "median_ae_hours": mean_std([run["metrics"]["median_ae_hours"] for run in runs]),
        "nonfinite_steps": int(sum(run["nonfinite_steps"] for run in runs)),
    }


def write_report(metrics: dict, status: dict) -> None:
    mark = metrics["summary"]["recurrent_mark"]
    time_summary = metrics["summary"]["recurrent_time"]
    baselines = metrics["baselines"]
    lines = [
        "# Group-Level Learnability Gate Report", "",
        "## Protocol", "",
        "The gate reuses exactly the frozen 6,413 train target groups. Evaluation is on the same Tiny train subset and therefore establishes optimization/learnability only, not validation or test generalization. No within-group order, listen, SNMPP, SID target, HPN, or BOLA is used.", "",
        "## Data", "",
        f"- target groups: {metrics['data_contract']['target_groups']:,}",
        f"- target events: {metrics['data_contract']['target_events']:,}",
        f"- users: {metrics['data_contract']['target_users']:,}",
        f"- strict-history group union: {metrics['data_contract']['union_history_groups']:,}",
        "- representation: mean frozen SID semantics + feedback composition + log group size + previous log gap", "",
        "## Mark-only result", "",
        f"- recurrent CE/event: {mark['ce_per_event']['mean']:.6f} +/- {mark['ce_per_event']['std']:.6f}",
        f"- recurrent Macro-F1: {mark['macro_f1']['mean']:.6f} +/- {mark['macro_f1']['std']:.6f}",
        f"- last-group MLP CE/event: {metrics['summary']['last_group_mark']['ce_per_event']['mean']:.6f}",
        f"- global prior CE/event: {baselines['mark']['global_train_feedback_prior']['ce_per_event']:.6f}",
        f"- previous composition CE/event: {baselines['mark']['previous_group_composition_with_global_Dirichlet_strength_1']['ce_per_event']:.6f}",
        f"- collapsed recurrent seeds: {mark['collapsed_seeds']} / {len(SEEDS)}", "",
        "## Time-only result", "",
        f"- recurrent NLL/group: {time_summary['nll_per_group']['mean']:.6f} +/- {time_summary['nll_per_group']['std']:.6f}",
        f"- recurrent MAE: {time_summary['mae_hours']['mean']:.6f} hours", 
        f"- recurrent Median AE: {time_summary['median_ae_hours']['mean']:.6f} hours",
        f"- history-free lognormal NLL: {baselines['time']['history_free_lognormal_fit_on_tiny_train']['nll_per_group']:.6f}",
        f"- previous-gap residual lognormal NLL: {baselines['time']['previous_gap_lognormal_residual_fit_on_tiny_train']['nll_per_group']:.6f}", "",
        "## Decision", "",
        f"- mark learnable: `{str(status['group_level_mark_learnable']).lower()}`",
        f"- time learnable: `{str(status['group_level_time_learnable']).lower()}`",
        f"- history signal supported: `{str(status['history_signal_supported']).lower()}`",
        f"- group SNMPP design approved: `{str(status['group_snmpp_design_approved']).lower()}`", "",
        "The gate stops here. It does not design or train Group-SNMPP.",
    ]
    (WORK / "Group_Level_Learnability_Gate_Report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = prepare_data()
    save_json(WORK / "data_contract.json", data.contract)
    np.savez_compressed(WORK / "feature_standardization.npz", mean=data.feature_mean, std=data.feature_std)
    dataset = SequenceDataset(data)
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    global_probability = np.asarray(manifest["baselines"]["global_train_feedback_probabilities"], dtype=np.float64)
    baselines = {
        "mark": mark_baselines(data, global_probability),
        "time": time_baselines(data, float(manifest["baselines"]["global_train_gap_median_seconds"]) / 3600.0),
    }

    all_runs: dict[str, list[dict]] = {"last_group_mark": [], "recurrent_mark": [], "recurrent_time": []}
    raw_outputs = {}
    for seed in SEEDS:
        seed_all(seed)
        last_model, last_curve, last_diag = train_model(MarkModel(False), dataset, task="mark", seed=seed, device=device)
        seed_dir = WORK / f"runs/seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(last_model.state_dict(), seed_dir / "last_group_mark.pt")
        save_json(seed_dir / "last_group_mark_curve.json", clean_metric_payload(last_curve))
        all_runs["last_group_mark"].append(clean_metric_payload(last_diag))

        seed_all(seed)
        mark_model, mark_curve, mark_diag = train_model(MarkModel(True), dataset, task="mark", seed=seed, device=device)
        torch.save(mark_model.state_dict(), seed_dir / "recurrent_mark.pt")
        save_json(seed_dir / "recurrent_mark_curve.json", clean_metric_payload(mark_curve))
        all_runs["recurrent_mark"].append(clean_metric_payload(mark_diag))

        seed_all(seed)
        time_model, time_curve, time_diag = train_model(TimeModel(True), dataset, task="time", seed=seed, device=device)
        torch.save(time_model.state_dict(), seed_dir / "recurrent_time.pt")
        save_json(seed_dir / "recurrent_time_curve.json", clean_metric_payload(time_curve))
        all_runs["recurrent_time"].append(clean_metric_payload(time_diag))

        if seed == SEEDS[0]:
            raw_outputs = {
                "mark_probability": mark_diag["probabilities"],
                "time_mu": time_diag["mu"], "time_sigma": time_diag["sigma"],
            }

    summary = {key: summarize_runs(value, "time" if key == "recurrent_time" else "mark") for key, value in all_runs.items()}
    strongest_simple_mark_ce = min(
        baselines["mark"]["global_train_feedback_prior"]["ce_per_event"],
        baselines["mark"]["previous_group_composition_with_global_Dirichlet_strength_1"]["ce_per_event"],
        summary["last_group_mark"]["ce_per_event"]["mean"],
    )
    recurrent_mark_ce = summary["recurrent_mark"]["ce_per_event"]["mean"]
    mark_relative_gain = (strongest_simple_mark_ce - recurrent_mark_ce) / strongest_simple_mark_ce
    mark_learnable = bool(mark_relative_gain >= 0.01 and summary["recurrent_mark"]["collapsed_seeds"] == 0)

    strongest_time_nll = min(
        baselines["time"]["history_free_lognormal_fit_on_tiny_train"]["nll_per_group"],
        baselines["time"]["previous_gap_lognormal_residual_fit_on_tiny_train"]["nll_per_group"],
    )
    recurrent_time_nll = summary["recurrent_time"]["nll_per_group"]["mean"]
    time_nll_gain = (strongest_time_nll - recurrent_time_nll) / max(abs(strongest_time_nll), 1e-12)
    baseline_mae = min(
        baselines["time"]["global_train_median_gap"]["mae_hours"],
        baselines["time"]["previous_gap_point"]["mae_hours"],
        baselines["time"]["history_free_lognormal_fit_on_tiny_train"]["mae_hours"],
        baselines["time"]["previous_gap_lognormal_residual_fit_on_tiny_train"]["mae_hours"],
    )
    time_mae_gain = (baseline_mae - summary["recurrent_time"]["mae_hours"]["mean"]) / max(baseline_mae, 1e-12)
    time_learnable = bool(time_nll_gain >= 0.01 and time_mae_gain >= 0.01)
    history_supported = bool(mark_learnable or time_learnable)

    joint = None
    if history_supported:
        joint_runs = []
        for seed in SEEDS:
            seed_all(seed)
            model, curve, diagnosis = train_model(JointModel(), dataset, task="joint", seed=seed, device=device)
            seed_dir = WORK / f"runs/seed_{seed}"
            torch.save(model.state_dict(), seed_dir / "joint.pt")
            save_json(seed_dir / "joint_curve.json", clean_metric_payload(curve))
            joint_runs.append(clean_metric_payload(diagnosis))
        joint = joint_runs

    slices = slice_metrics(data, raw_outputs["mark_probability"], raw_outputs["time_mu"], raw_outputs["time_sigma"])
    burst = slices[">500"]
    normal = {label: value for label, value in slices.items() if label != ">500"}
    burst_stable = bool(burst["groups"] > 0 and burst["outputs_finite"] and
                        math.isfinite(burst["mark"]["ce_per_event"]) and math.isfinite(burst["time"]["nll_per_group"]))
    stable_training = bool(
        summary["recurrent_mark"]["nonfinite_steps"] == 0
        and summary["recurrent_time"]["nonfinite_steps"] == 0 and burst_stable
    )
    status = {
        "group_level_mark_learnable": mark_learnable,
        "group_level_time_learnable": time_learnable,
        "history_signal_supported": history_supported,
        "group_snmpp_design_approved": bool(history_supported and stable_training),
        "ordinary_group_model_stably_trainable": stable_training,
        "joint_model_executed": joint is not None,
        "stage_after_gate_started": False,
        "decision_thresholds": {
            "mark": "recurrent CE >=1% below strongest simple baseline and no seed collapses",
            "time": "recurrent NLL and MAE each >=1% below strongest respective baseline",
            "approval": "at least one task learnable and all recurrent/burst outputs finite",
        },
        "mark_relative_CE_gain_vs_strongest_simple": mark_relative_gain,
        "time_relative_NLL_gain_vs_strongest_simple": time_nll_gain,
        "time_relative_MAE_gain_vs_strongest_simple": time_mae_gain,
        "stop_reason": "Group-Level Learnability Gate complete; Group-SNMPP design/training is outside this round.",
    }
    metrics = {
        "experiment": "Group-Level Learnability Gate",
        "device": str(device),
        "seeds": SEEDS,
        "config": {"hidden_size": 64, "learning_rate": 1e-3, "optimizer": "Adam", "epochs": 20,
                   "batch_size": 64, "gradient_clip_norm": 5.0, "time_distribution": "log-normal",
                   "time_point_prediction": "predictive median exp(mu)"},
        "data_contract": data.contract,
        "baselines": clean_metric_payload(baselines),
        "summary": summary,
        "runs": all_runs,
        "joint_runs": joint,
        "previous_group_size_slices_seed2026": clean_metric_payload(slices),
        "burst_stability": burst_stable,
    }
    save_json(WORK / "group_level_metrics.json", clean_metric_payload(metrics))
    save_json(WORK / "status.json", status)
    write_report(metrics, status)
    print(json.dumps({"GATE_COMPLETE": True, **status}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
