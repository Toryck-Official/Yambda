#!/usr/bin/env python3
"""Group-SNMPP Design & Minimal Validation Gate.

The temporal source object is always one uid+timestamp group.  No operation in
this file imposes an order inside a group or sums its events as independent
temporal sources.  Time and mark models are trained separately.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from group_level_learnability_gate.run_gate import (  # noqa: E402
    CLASS_NAMES,
    CODEBOOK,
    SOURCE_MANIFEST,
    SOURCE_NPZ,
    classification_metrics,
    clean_metric_payload,
    save_json,
    seed_all,
    sha256,
)


WORK = Path(__file__).resolve().parent
TIME_SELECTION = ROOT / "group_level_generalization_pilot/selection_ids.npz"
TIME_BASELINES = ROOT / "group_level_generalization_pilot/generalization_metrics.json"
MARK_SELECTION = ROOT / "mark_generalization_confirmation_gate/selection_ids.npz"
MARK_BASELINES = ROOT / "mark_generalization_confirmation_gate/mark_generalization_metrics.json"
SEEDS = [2026, 2027, 2028]
MAX_EPOCHS = 12
PATIENCE = 3
MIN_DELTA = 1e-4
LEARNING_RATE = 1e-3
MAX_BATCH = 128
MAX_PADDED_GROUPS = 65_536
INTEGRATION_Q = 16
HORIZON_HOURS = 696.0
N_CLASSES = 4
SEMANTIC_DIM = 128
FEATURE_DIM = 4 * SEMANTIC_DIM + 4 + 4 + 2
EPS = 1e-8
HISTORY_BINS = [(1, 20, "1-20"), (21, 50, "21-50"), (51, 100, "51-100"),
                (101, 500, "101-500"), (501, np.iinfo(np.int64).max, ">500")]
PREVIOUS_SIZE_BINS = [(1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"),
                      (21, 100, "21-100"), (101, 500, "101-500"),
                      (501, np.iinfo(np.int64).max, ">500")]


@dataclass
class RichData:
    features: np.ndarray
    timestamps_hours: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    counts: np.ndarray
    gaps: np.ndarray
    previous_sizes: np.ndarray
    group_ids: np.ndarray
    uids: np.ndarray
    contract: dict


def user_start(group_uid: np.ndarray) -> np.ndarray:
    starts = np.flatnonzero(np.r_[True, group_uid[1:] != group_uid[:-1]])
    marker = np.zeros(len(group_uid), dtype=np.int64)
    marker[starts] = starts
    return np.maximum.accumulate(marker)


def _group_features(
    start: int,
    end: int,
    *,
    offsets: np.ndarray,
    feedback: np.ndarray,
    event_sid: np.ndarray,
    counts: np.ndarray,
    timestamp: np.ndarray,
    codebooks: np.ndarray,
) -> np.ndarray:
    """Per-feedback semantic means + composition/mask/cardinality/gap."""
    begin, finish = int(offsets[start]), int(offsets[end])
    sid = event_sid[begin:finish]
    event_semantic = (
        codebooks[0, sid[:, 0]] + codebooks[1, sid[:, 1]]
        + codebooks[2, sid[:, 2]] + codebooks[3, sid[:, 3]]
    ).astype(np.float32, copy=False)
    sizes = np.diff(offsets[start:end + 1]).astype(np.int64)
    local_group = np.repeat(np.arange(end - start, dtype=np.int64), sizes)
    relation = feedback[begin:finish].astype(np.int64)
    flat = local_group * 4 + relation
    semantic_sum = np.zeros(((end - start) * 4, SEMANTIC_DIM), dtype=np.float32)
    np.add.at(semantic_sum, flat, event_semantic)
    relation_counts = counts[start:end].astype(np.float32)
    semantic_mean = semantic_sum.reshape(end - start, 4, SEMANTIC_DIM)
    denominator = np.maximum(relation_counts, 1.0)[..., None]
    semantic_mean /= denominator
    mask = (relation_counts > 0).astype(np.float32)
    semantic_mean *= mask[..., None]
    group_size = relation_counts.sum(axis=1)
    composition = relation_counts / group_size[:, None]
    gap = np.zeros(end - start, dtype=np.float32)
    if end - start > 1:
        gap[1:] = np.diff(timestamp[start:end].astype(np.float64)) / 3600.0
    result = np.concatenate([
        semantic_mean.reshape(end - start, -1),
        composition,
        mask,
        np.log1p(group_size)[:, None],
        np.log1p(gap)[:, None],
    ], axis=1).astype(np.float32)
    if result.shape != (end - start, FEATURE_DIM) or not np.isfinite(result).all():
        raise RuntimeError("invalid group feature construction")
    return result


def prepare(selection_path: Path, split_name: str) -> tuple[RichData, RichData, dict]:
    source = np.load(SOURCE_NPZ, allow_pickle=False)
    selection = np.load(selection_path, allow_pickle=False)
    group_uid = source["group_uid"]
    group_timestamp = source["group_timestamp"]
    group_ids = source["group_global_id"]
    offsets = source["group_event_offsets"]
    counts = source["group_feedback_counts"]
    feedback = source["event_feedback"]
    event_sid = source["event_sid"]
    codebooks = np.load(CODEBOOK).astype(np.float32)
    train_targets = selection["train_target_local_group_indices"].astype(np.int64)
    validation_targets = selection["validation_target_local_group_indices"].astype(np.int64)
    starts_for_group = user_start(group_uid)

    selected_users = np.unique(group_uid[np.r_[train_targets, validation_targets]])
    max_train: dict[int, int] = {}
    max_validation: dict[int, int] = {}
    for target in train_targets:
        max_train[int(group_uid[target])] = max(int(target), max_train.get(int(group_uid[target]), -1))
    for target in validation_targets:
        max_validation[int(group_uid[target])] = max(int(target), max_validation.get(int(group_uid[target]), -1))
    ranges = []
    for uid in selected_users:
        uid_int = int(uid)
        train_end = max_train[uid_int]
        full_end = max(max_validation.get(uid_int, train_end), train_end)
        ranges.append((int(starts_for_group[train_end]), full_end, train_end))
    ranges.sort()
    total_groups = sum(end - start for start, end, _ in ranges)
    features = np.empty((total_groups, FEATURE_DIM), dtype=np.float32)
    local_timestamps = np.empty(total_groups, dtype=np.float64)
    global_to_local = np.full(len(group_uid), -1, dtype=np.int64)
    cursor = 0
    started = time.perf_counter()
    for number, (start, end, _) in enumerate(ranges, 1):
        block = _group_features(
            start, end, offsets=offsets, feedback=feedback, event_sid=event_sid,
            counts=counts, timestamp=group_timestamp, codebooks=codebooks,
        )
        features[cursor:cursor + len(block)] = block
        local_timestamps[cursor:cursor + len(block)] = group_timestamp[start:end] / 3600.0
        global_to_local[start:end] = np.arange(cursor, cursor + len(block))
        cursor += len(block)
        if number % 250 == 0:
            print(json.dumps({"split_contract": split_name, "prepared_users": number, "groups": cursor}), flush=True)

    def target_data(targets: np.ndarray, split: str) -> RichData:
        starts = global_to_local[starts_for_group[targets]]
        ends = global_to_local[targets - 1] + 1
        if np.any(starts < 0) or np.any(ends <= starts):
            raise RuntimeError("strict history mapping failed")
        gaps = (group_timestamp[targets].astype(np.float64) - group_timestamp[targets - 1]) / 3600.0
        if np.any(gaps <= 0):
            raise RuntimeError("non-positive group gap")
        return RichData(
            features=features,
            timestamps_hours=local_timestamps,
            starts=starts.astype(np.int64),
            ends=ends.astype(np.int64),
            counts=counts[targets].astype(np.float32),
            gaps=gaps.astype(np.float32),
            previous_sizes=counts[targets - 1].sum(axis=1).astype(np.int64),
            group_ids=group_ids[targets].astype(np.uint64),
            uids=group_uid[targets].astype(np.uint32),
            contract={"name": split_name, "split": split},
        )

    train, validation = target_data(train_targets, "train"), target_data(validation_targets, "validation")

    # Feature-level permutation invariance: group representation is fully defined
    # by per-relation sums/counts.  Reordering the raw events leaves it unchanged.
    rng = np.random.default_rng(2026)
    multi = np.flatnonzero(counts.sum(axis=1) > 1)
    checked = multi[rng.choice(len(multi), min(100, len(multi)), replace=False)]
    permutation_max_error = 0.0
    for group in checked:
        begin, finish = int(offsets[group]), int(offsets[group + 1])
        order = rng.permutation(finish - begin)
        sid = event_sid[begin:finish]
        fb = feedback[begin:finish]
        semantic = sum(codebooks[level, sid[:, level]] for level in range(4))
        original = np.zeros((4, 128), np.float32)
        shuffled = np.zeros((4, 128), np.float32)
        np.add.at(original, fb, semantic)
        np.add.at(shuffled, fb[order], semantic[order])
        denominator = np.maximum(np.bincount(fb, minlength=4), 1)[:, None]
        permutation_max_error = max(permutation_max_error, float(np.max(np.abs(original / denominator - shuffled / denominator))))
    if permutation_max_error > 2e-5:
        raise RuntimeError("group feature is not permutation invariant")

    manifest = {
        "name": split_name,
        "source_npz": str(SOURCE_NPZ),
        "source_sha256": sha256(SOURCE_NPZ),
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "users": int(len(selected_users)),
        "train_targets": int(len(train_targets)),
        "train_events": int(train.counts.sum()),
        "validation_targets": int(len(validation_targets)),
        "validation_events": int(validation.counts.sum()),
        "materialized_history_groups": int(total_groups),
        "feature_dim": FEATURE_DIM,
        "feature_definition": "4x per-feedback mean frozen SID semantics + composition + presence mask + log1p group size + log1p previous gap",
        "empty_feedback_representation": "zero semantic vector plus zero presence mask",
        "within_group_order_used": False,
        "event_level_temporal_source_used": False,
        "burst_removed": False,
        "listen_used": False,
        "test_used": False,
        "permutation_check_groups": int(len(checked)),
        "permutation_check_max_error": permutation_max_error,
        "train_history_length_percentiles": {str(q): float(np.percentile(train.ends - train.starts, q)) for q in (0, 50, 90, 95, 99, 100)},
        "validation_history_length_percentiles": {str(q): float(np.percentile(validation.ends - validation.starts, q)) for q in (0, 50, 90, 95, 99, 100)},
        "preparation_seconds": time.perf_counter() - started,
    }
    return train, validation, manifest


class GroupDataset(Dataset):
    def __init__(self, data: RichData, limit: int | None = None) -> None:
        self.data, self.limit = data, limit
        raw = data.ends - data.starts
        self.lengths = raw if limit is None else np.minimum(raw, limit)

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> dict:
        end = int(self.data.ends[index])
        start = int(self.data.starts[index])
        if self.limit is not None:
            start = max(start, end - self.limit)
        timestamps = self.data.timestamps_hours[start:end]
        age0 = timestamps[-1] - timestamps
        return {
            "sequence": torch.from_numpy(self.data.features[start:end]),
            "age0": torch.from_numpy(age0.astype(np.float32)),
            "counts": torch.from_numpy(self.data.counts[index]),
            "gap": torch.tensor(self.data.gaps[index]),
            "previous_size": torch.tensor(self.data.previous_sizes[index]),
            "index": torch.tensor(index),
        }


class TokenBucketSampler(Sampler[list[int]]):
    def __init__(self, lengths: np.ndarray, seed: int, shuffle: bool) -> None:
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0
        order = np.argsort(lengths, kind="stable")
        batches, current, maximum = [], [], 0
        for index in order:
            length = int(lengths[index])
            candidate_max = max(maximum, length)
            if current and (len(current) >= MAX_BATCH or candidate_max * (len(current) + 1) > MAX_PADDED_GROUPS):
                batches.append(current); current, maximum = [], 0
            current.append(int(index)); maximum = max(maximum, length)
        if current:
            batches.append(current)
        self.batches = batches

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
    sequence = torch.zeros(len(rows), maximum, FEATURE_DIM)
    age0 = torch.zeros(len(rows), maximum)
    for index, row in enumerate(rows):
        length = int(lengths[index])
        sequence[index, :length] = row["sequence"]
        age0[index, :length] = row["age0"]
    return {
        "sequence": sequence,
        "age0": age0,
        "lengths": lengths,
        "counts": torch.stack([row["counts"] for row in rows]),
        "gap": torch.stack([row["gap"] for row in rows]),
        "previous_size": torch.stack([row["previous_size"] for row in rows]),
        "index": torch.stack([row["index"] for row in rows]),
    }


def loader(dataset: GroupDataset, seed: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_sampler=TokenBucketSampler(dataset.lengths, seed, shuffle), collate_fn=collate, num_workers=0)


def split_features(sequence: torch.Tensor) -> tuple[torch.Tensor, ...]:
    mu = sequence[..., :512].reshape(*sequence.shape[:-1], 4, 128)
    composition = sequence[..., 512:516]
    presence = sequence[..., 516:520]
    log_size = sequence[..., 520]
    log_previous_gap = sequence[..., 521]
    return mu, composition, presence, log_size, log_previous_gap


def sequence_mask(lengths: torch.Tensor, maximum: int) -> torch.Tensor:
    return torch.arange(maximum, device=lengths.device)[None] < lengths[:, None]


class GroupMarkSNMPP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.zeros(4))
        self.semantic_weight = nn.Parameter(torch.randn(4, 4, 128) * 0.02)
        self.semantic_bias = nn.Parameter(torch.zeros(4, 4))
        self.size_weight = nn.Parameter(torch.zeros(4, 4))
        self.gap_weight = nn.Parameter(torch.zeros(4, 4))
        self.psi = nn.Parameter(torch.randn(4, 4) * 0.03)
        self.raw_decay = nn.Parameter(torch.full((4, 4), -2.0))
        self.raw_delay = nn.Parameter(torch.full((4, 4), -0.5))

    def interaction_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.psi, torch.nn.functional.softplus(self.raw_decay) + 1e-4, torch.nn.functional.softplus(self.raw_delay)

    def forward(self, batch: dict[str, torch.Tensor], return_diagnostics: bool = False):
        sequence, lengths = batch["sequence"], batch["lengths"]
        mu, composition, presence, log_size, log_previous_gap = split_features(sequence)
        mask = sequence_mask(lengths, sequence.shape[1]).to(sequence.dtype)
        semantic_score = torch.einsum("blrh,rkh->blrk", mu, self.semantic_weight) / math.sqrt(128.0)
        semantic_score = semantic_score + self.semantic_bias[None, None]
        semantic_gate = 2.0 * torch.sigmoid(semantic_score)
        context_gate = 2.0 * torch.sigmoid(
            log_size[..., None, None] * self.size_weight[None, None]
            + log_previous_gap[..., None, None] * self.gap_weight[None, None]
        )
        psi, decay, delay = self.interaction_parameters()
        age = batch["age0"] + batch["gap"][:, None]
        phi = torch.exp(-decay[None, None] * torch.abs(age[..., None, None] - delay[None, None]))
        contribution = (
            mask[..., None, None] * presence[..., :, None] * composition[..., :, None]
            * semantic_gate * context_gate * psi[None, None] * phi
        )
        logits = self.base + contribution.sum(dim=(1, 2))
        if return_diagnostics:
            return logits, contribution
        return logits


class GroupTimeSNMPP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # softplus(-3.8) ~= 0.022/hour, close to the frozen train rate.
        self.base = nn.Parameter(torch.tensor(-3.8))
        self.semantic_weight = nn.Parameter(torch.randn(4, 128) * 0.02)
        self.semantic_bias = nn.Parameter(torch.zeros(4))
        self.size_weight = nn.Parameter(torch.zeros(4))
        self.gap_weight = nn.Parameter(torch.zeros(4))
        self.psi = nn.Parameter(torch.randn(4) * 0.03)
        self.raw_decay = nn.Parameter(torch.full((4,), -2.0))
        self.raw_delay = nn.Parameter(torch.full((4,), -0.5))

    def source_amplitude(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, lengths = batch["sequence"], batch["lengths"]
        mu, composition, presence, log_size, log_previous_gap = split_features(sequence)
        mask = sequence_mask(lengths, sequence.shape[1]).to(sequence.dtype)
        semantic = torch.einsum("blrh,rh->blr", mu, self.semantic_weight) / math.sqrt(128.0) + self.semantic_bias
        semantic_gate = 2.0 * torch.sigmoid(semantic)
        context_gate = 2.0 * torch.sigmoid(log_size[..., None] * self.size_weight + log_previous_gap[..., None] * self.gap_weight)
        amplitude = mask[..., None] * presence * composition * semantic_gate * context_gate * self.psi
        return amplitude, mask

    def hazard(self, amplitude: torch.Tensor, age0: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        decay = torch.nn.functional.softplus(self.raw_decay) + 1e-4
        delay = torch.nn.functional.softplus(self.raw_delay)
        # x: [B,Q] -> contribution [B,Q,L,R]
        age = age0[:, None, :, None] + x[:, :, None, None]
        phi = torch.exp(-decay[None, None, None] * torch.abs(age - delay[None, None, None]))
        signed = (amplitude[:, None] * phi).sum(dim=(2, 3))
        hazard = torch.nn.functional.softplus(self.base + signed) + EPS
        return hazard, signed

    def nll(self, batch: dict[str, torch.Tensor], q: int = INTEGRATION_Q, return_diagnostics: bool = False):
        amplitude, _ = self.source_amplitude(batch)
        gap = batch["gap"]
        target_hazard, target_signed = self.hazard(amplitude, batch["age0"], gap[:, None])
        midpoint = (torch.arange(q, device=gap.device, dtype=gap.dtype) + 0.5) / q
        points = gap[:, None] * midpoint[None]
        grid_hazard, grid_signed = self.hazard(amplitude, batch["age0"], points)
        integral = gap * grid_hazard.mean(dim=1)
        rows = -torch.log(target_hazard[:, 0]) + integral
        if return_diagnostics:
            return rows, {
                "target_hazard": target_hazard[:, 0],
                "integral": integral,
                "target_signed": target_signed[:, 0],
                "absolute_amplitude": amplitude.abs().sum(dim=(1, 2)),
                "positive_amplitude": amplitude.clamp_min(0).sum(dim=(1, 2)),
                "negative_amplitude": amplitude.clamp_max(0).sum(dim=(1, 2)),
                "grid_signed_abs_max": grid_signed.abs().amax(dim=1),
            }
        return rows

    @torch.no_grad()
    def predictive_median(self, batch: dict[str, torch.Tensor], grid_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        amplitude, _ = self.source_amplitude(batch)
        device, dtype = amplitude.device, amplitude.dtype
        positive = torch.exp(torch.linspace(math.log(1e-3), math.log(HORIZON_HOURS), grid_size - 1, device=device, dtype=dtype))
        grid = torch.cat([torch.zeros(1, device=device, dtype=dtype), positive])
        x = grid[None].expand(len(batch["gap"]), -1)
        hazard, _ = self.hazard(amplitude, batch["age0"], x)
        widths = grid[1:] - grid[:-1]
        cumulative = torch.zeros_like(hazard)
        cumulative[:, 1:] = torch.cumsum(0.5 * (hazard[:, 1:] + hazard[:, :-1]) * widths[None], dim=1)
        reached = cumulative >= math.log(2.0)
        first = reached.float().argmax(dim=1)
        never = ~reached.any(dim=1)
        first = torch.where(never, torch.full_like(first, grid_size - 1), first)
        previous = (first - 1).clamp_min(0)
        row = torch.arange(len(first), device=device)
        c0, c1 = cumulative[row, previous], cumulative[row, first]
        x0, x1 = grid[previous], grid[first]
        fraction = ((math.log(2.0) - c0) / (c1 - c0).clamp_min(1e-8)).clamp(0, 1)
        median = x0 + fraction * (x1 - x0)
        horizon_mass = 1.0 - torch.exp(-cumulative[:, -1])
        return median, horizon_mass


def mark_loss(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return -(counts * torch.log_softmax(logits, dim=1)).sum() / counts.sum()


@torch.no_grad()
def evaluate_mark(model: GroupMarkSNMPP, dataset: GroupDataset, device: torch.device, diagnostics: bool = False) -> dict:
    model.eval(); records, signed = [], []
    for batch in loader(dataset, 2026, False):
        indices = batch["index"].numpy()
        gpu = {k: v.to(device) for k, v in batch.items() if k not in {"index", "previous_size"}}
        logits, contribution = model(gpu, True)
        records.append((indices, torch.softmax(logits, dim=1).cpu().numpy()))
        if diagnostics:
            signed.append({
                "positive": float(contribution.clamp_min(0).sum()),
                "negative": float(contribution.clamp_max(0).sum()),
                "absolute": float(contribution.abs().sum()),
                "events": float(gpu["counts"].sum()),
            })
    order = np.concatenate([r[0] for r in records]); probability = np.concatenate([r[1] for r in records])[np.argsort(order)]
    result = {"metrics": classification_metrics(probability, dataset.data.counts), "probability": probability}
    if diagnostics:
        result["influence"] = {key: float(sum(row[key] for row in signed)) for key in signed[0]}
    return result


@torch.no_grad()
def evaluate_time(model: GroupTimeSNMPP, dataset: GroupDataset, device: torch.device, diagnostics: bool = False) -> dict:
    model.eval(); rows = []
    for batch in loader(dataset, 2026, False):
        indices = batch["index"].numpy()
        previous_size = batch["previous_size"].numpy()
        gpu = {k: v.to(device) for k, v in batch.items() if k not in {"index", "previous_size"}}
        nll, detail = model.nll(gpu, return_diagnostics=True)
        median, mass = model.predictive_median(gpu)
        rows.append((indices, nll.cpu().numpy(), median.cpu().numpy(), mass.cpu().numpy(),
                     detail["target_hazard"].cpu().numpy(), detail["target_signed"].cpu().numpy(),
                     detail["absolute_amplitude"].cpu().numpy(), previous_size))
    order = np.concatenate([r[0] for r in rows]); sorting = np.argsort(order)
    arrays = [np.concatenate([r[c] for r in rows])[sorting] for c in range(1, 8)]
    nll, median, mass, hazard, signed, absolute, previous_size = arrays
    error = np.abs(median - dataset.data.gaps)
    result = {
        "metrics": {
            "nll_per_group": float(nll.mean()),
            "mae_hours": float(error.mean()),
            "median_ae_hours": float(np.median(error)),
            "horizon_event_mass_mean": float(mass.mean()),
            "horizon_event_mass_p10": float(np.percentile(mass, 10)),
            "lambda_median": float(np.median(hazard)),
            "lambda_p95": float(np.percentile(hazard, 95)),
            "lambda_p99": float(np.percentile(hazard, 99)),
            "lambda_max": float(hazard.max()),
            "finite": bool(all(np.isfinite(x).all() for x in arrays)),
        },
        "rows": {"nll": nll, "median": median, "error": error, "mass": mass, "hazard": hazard,
                 "signed": signed, "absolute_amplitude": absolute, "previous_size": previous_size},
    }
    return result


def fit_mark(train: RichData, validation: RichData, seed: int, device: torch.device) -> tuple[GroupMarkSNMPP, dict, np.ndarray]:
    seed_all(seed); model = GroupMarkSNMPP().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_ds, val_ds = GroupDataset(train), GroupDataset(validation)
    train_loader = loader(train_ds, seed, True)
    best_state, best_ce, best_epoch, stale = None, float("inf"), 0, 0
    curves = []
    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.perf_counter(); model.train(); numerator = denominator = 0.0; nonfinite = 0
        for batch in train_loader:
            gpu = {k: v.to(device) for k, v in batch.items() if k not in {"index", "previous_size"}}
            optimizer.zero_grad(set_to_none=True); loss = mark_loss(model(gpu), gpu["counts"])
            if not torch.isfinite(loss): nonfinite += 1; continue
            loss.backward(); optimizer.step()
            weight = float(gpu["counts"].sum()); numerator += float(loss.detach()) * weight; denominator += weight
        validation_result = evaluate_mark(model, val_ds, device)
        ce = validation_result["metrics"]["ce_per_event"]
        improved = ce < best_ce - MIN_DELTA
        if improved: best_state = copy.deepcopy(model.state_dict()); best_ce = ce; best_epoch = epoch; stale = 0
        else: stale += 1
        curves.append({"epoch": epoch, "train_batch_ce": numerator / denominator, "validation": clean_metric_payload(validation_result["metrics"]),
                       "improved": improved, "nonfinite_steps": nonfinite, "seconds": time.perf_counter() - started})
        print(json.dumps({"task": "group_mark", "seed": seed, "epoch": epoch, "train": numerator/denominator,
                          "validation": ce, "best_epoch": best_epoch, "stale": stale, "seconds": curves[-1]["seconds"]}), flush=True)
        if stale >= PATIENCE: break
    if best_state is None: raise RuntimeError("no Group-Mark checkpoint")
    model.load_state_dict(best_state)
    train_result = evaluate_mark(model, train_ds, device)
    validation_result = evaluate_mark(model, val_ds, device, diagnostics=True)
    psi, decay, delay = model.interaction_parameters()
    result = {
        "seed": seed, "best_epoch": best_epoch, "stopped_epoch": curves[-1]["epoch"],
        "train": clean_metric_payload(train_result["metrics"]), "validation": clean_metric_payload(validation_result["metrics"]),
        "interaction": {"psi": psi.detach().cpu().tolist(), "decay": decay.detach().cpu().tolist(), "delay_hours": delay.detach().cpu().tolist()},
        "validation_influence": validation_result["influence"], "curves": curves,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    return model, result, validation_result["probability"]


def fit_time(train: RichData, validation: RichData, seed: int, device: torch.device) -> tuple[GroupTimeSNMPP, dict, dict]:
    seed_all(seed); model = GroupTimeSNMPP().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_ds, val_ds = GroupDataset(train), GroupDataset(validation)
    train_loader = loader(train_ds, seed, True)
    best_state, best_nll, best_epoch, stale = None, float("inf"), 0, 0
    curves = []
    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.perf_counter(); model.train(); numerator = denominator = 0.0; nonfinite = 0
        for batch in train_loader:
            gpu = {k: v.to(device) for k, v in batch.items() if k not in {"index", "previous_size"}}
            optimizer.zero_grad(set_to_none=True); loss = model.nll(gpu).mean()
            if not torch.isfinite(loss): nonfinite += 1; continue
            loss.backward(); optimizer.step(); numerator += float(loss.detach()) * len(gpu["gap"]); denominator += len(gpu["gap"])
        validation_result = evaluate_time(model, val_ds, device)
        nll = validation_result["metrics"]["nll_per_group"]
        improved = nll < best_nll - MIN_DELTA
        if improved: best_state = copy.deepcopy(model.state_dict()); best_nll = nll; best_epoch = epoch; stale = 0
        else: stale += 1
        curves.append({"epoch": epoch, "train_batch_nll": numerator/denominator, "validation": validation_result["metrics"],
                       "improved": improved, "nonfinite_steps": nonfinite, "seconds": time.perf_counter()-started})
        print(json.dumps({"task": "group_time", "seed": seed, "epoch": epoch, "train": numerator/denominator,
                          "validation": nll, "best_epoch": best_epoch, "stale": stale, "seconds": curves[-1]["seconds"]}), flush=True)
        if stale >= PATIENCE: break
    if best_state is None: raise RuntimeError("no Group-Time checkpoint")
    model.load_state_dict(best_state)
    train_result = evaluate_time(model, train_ds, device)
    validation_result = evaluate_time(model, val_ds, device, diagnostics=True)
    result = {
        "seed": seed, "best_epoch": best_epoch, "stopped_epoch": curves[-1]["epoch"],
        "train": train_result["metrics"], "validation": validation_result["metrics"],
        "interaction": {
            "psi": model.psi.detach().cpu().tolist(),
            "decay": (torch.nn.functional.softplus(model.raw_decay)+1e-4).detach().cpu().tolist(),
            "delay_hours": torch.nn.functional.softplus(model.raw_delay).detach().cpu().tolist(),
            "base_raw": float(model.base.detach()),
        },
        "curves": curves, "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    return model, result, validation_result["rows"]


def mark_slices(data: RichData, predictions: list[np.ndarray]) -> dict:
    lengths = data.ends - data.starts; result = {}
    for lower, upper, label in HISTORY_BINS:
        mask = (lengths >= lower) & (lengths <= upper)
        result[label] = {"targets": int(mask.sum())}
        if mask.any():
            result[label]["seeds"] = [classification_metrics(p[mask], data.counts[mask]) for p in predictions]
    return clean_metric_payload(result)


def time_slices(data: RichData, rows_by_seed: list[dict]) -> dict:
    lengths = data.ends - data.starts; result = {"history_length": {}, "previous_group_size": {}}
    for lower, upper, label in HISTORY_BINS:
        mask = (lengths >= lower) & (lengths <= upper)
        result["history_length"][label] = _time_slice_rows(mask, rows_by_seed)
    for lower, upper, label in PREVIOUS_SIZE_BINS:
        mask = (data.previous_sizes >= lower) & (data.previous_sizes <= upper)
        result["previous_group_size"][label] = _time_slice_rows(mask, rows_by_seed)
    return result


def _time_slice_rows(mask: np.ndarray, rows_by_seed: list[dict]) -> dict:
    if not mask.any(): return {"targets": 0}
    seeds = []
    for row in rows_by_seed:
        seeds.append({
            "nll_per_group": float(row["nll"][mask].mean()),
            "mae_hours": float(row["error"][mask].mean()),
            "median_ae_hours": float(np.median(row["error"][mask])),
            "lambda_p99": float(np.percentile(row["hazard"][mask], 99)),
            "lambda_max": float(row["hazard"][mask].max()),
            "absolute_amplitude_mean": float(row["absolute_amplitude"][mask].mean()),
        })
    return {"targets": int(mask.sum()), "seeds": seeds}


def summarize(runs: list[dict], task: str) -> dict:
    metrics = ["ce_per_event", "macro_f1", "accuracy"] if task == "mark" else ["nll_per_group", "mae_hours", "median_ae_hours"]
    result = {"seeds": [r["seed"] for r in runs], "best_epochs": [r["best_epoch"] for r in runs], "stopped_epochs": [r["stopped_epoch"] for r in runs]}
    for split in ("train", "validation"):
        result[split] = {}
        for metric in metrics:
            values = [r[split][metric] for r in runs]
            result[split][metric] = {"values": values, "mean": float(np.mean(values)), "std": float(np.std(values))}
    return result


def integration_audit(model: GroupTimeSNMPP, validation: RichData, device: torch.device) -> dict:
    dataset = GroupDataset(validation)
    rng = np.random.default_rng(2026)
    chosen = rng.choice(len(dataset), min(256, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, chosen.tolist())
    # A fixed ordinary loader is sufficient for the small numerical audit.
    audit_loader = DataLoader(subset, batch_size=32, shuffle=False, collate_fn=collate)
    values = {16: [], 32: [], 64: []}
    model.eval()
    with torch.no_grad():
        for batch in audit_loader:
            gpu = {k: v.to(device) for k, v in batch.items() if k not in {"index", "previous_size"}}
            for q in values:
                values[q].append(model.nll(gpu, q=q).cpu().numpy())
    joined = {q: np.concatenate(rows) for q, rows in values.items()}
    reference = joined[64]
    return {str(q): {
        "mean_absolute_nll_difference_vs_Q64": float(np.mean(np.abs(joined[q]-reference))),
        "p95_absolute_nll_difference_vs_Q64": float(np.percentile(np.abs(joined[q]-reference), 95)),
        "max_absolute_nll_difference_vs_Q64": float(np.max(np.abs(joined[q]-reference))),
    } for q in (16, 32, 64)}


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Unit-level invariants are checked before any training.
    unit_tests = run_unit_tests(device)
    save_json(WORK / "unit_tests.json", unit_tests)
    if not all(unit_tests.values()): raise RuntimeError("Group-SNMPP unit test failure")

    time_train, time_validation, time_manifest = prepare(TIME_SELECTION, "time_matched_303_user_pilot")
    save_json(WORK / "time_manifest.json", time_manifest)
    time_runs, time_rows = [], []
    for seed in SEEDS:
        model, metrics, rows = fit_time(time_train, time_validation, seed, device)
        run_dir = WORK / "runs" / f"time_seed_{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), run_dir / "group_time_snmpp_best.pt"); save_json(run_dir / "metrics.json", clean_metric_payload(metrics))
        time_runs.append(metrics); time_rows.append(rows)
    time_summary = summarize(time_runs, "time")
    time_slice_metrics = time_slices(time_validation, time_rows)
    time_integration = integration_audit(model, time_validation, device)
    save_json(WORK / "time_slices.json", clean_metric_payload(time_slice_metrics))
    del time_train

    mark_train, mark_validation, mark_manifest = prepare(MARK_SELECTION, "mark_matched_2000_user_confirmation")
    save_json(WORK / "mark_manifest.json", mark_manifest)
    mark_runs, mark_probabilities = [], []
    for seed in SEEDS:
        model, metrics, probability = fit_mark(mark_train, mark_validation, seed, device)
        run_dir = WORK / "runs" / f"mark_seed_{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), run_dir / "group_mark_snmpp_best.pt"); save_json(run_dir / "metrics.json", clean_metric_payload(metrics))
        mark_runs.append(metrics); mark_probabilities.append(probability)
    mark_summary = summarize(mark_runs, "mark")
    mark_slice_metrics = mark_slices(mark_validation, mark_probabilities)
    save_json(WORK / "mark_slices.json", clean_metric_payload(mark_slice_metrics))

    baseline_time = json.loads(TIME_BASELINES.read_text())
    baseline_mark = json.loads(MARK_BASELINES.read_text())
    decisions = make_decisions(time_runs, mark_runs, time_summary, mark_summary, baseline_time, baseline_mark, time_slice_metrics)
    payload = {
        "experiment": "Group-SNMPP Design & Minimal Validation Gate",
        "device": str(device),
        "config": {"seeds": SEEDS, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA,
                   "learning_rate": LEARNING_RATE, "integration_Q": INTEGRATION_Q, "horizon_hours": HORIZON_HOURS,
                   "architecture_search": False, "class_weighting": False, "history_truncation": False},
        "unit_tests": unit_tests,
        "time": {"manifest": time_manifest, "group_snmpp": time_summary, "runs": clean_metric_payload(time_runs),
                 "matched_baselines": baseline_time["models"]["full_history_time"], "simple_baselines": baseline_time["baselines"]["time"],
                 "slices": clean_metric_payload(time_slice_metrics), "integration_audit": time_integration},
        "mark": {"manifest": mark_manifest, "group_snmpp": mark_summary, "runs": clean_metric_payload(mark_runs),
                 "matched_baselines": baseline_mark["models"], "simple_baselines": baseline_mark["baselines"],
                 "slices": clean_metric_payload(mark_slice_metrics)},
        "status": decisions,
    }
    save_json(WORK / "group_snmpp_metrics.json", clean_metric_payload(payload)); save_json(WORK / "status.json", decisions)
    write_report(payload)
    print(json.dumps({"GROUP_SNMPP_GATE_COMPLETE": True, **decisions}), flush=True)


def run_unit_tests(device: torch.device) -> dict:
    seed_all(2026); batch = {
        "sequence": torch.randn(3, 5, FEATURE_DIM, device=device),
        "age0": torch.tensor([[4.,3.,2.,1.,0.],[4.,3.,2.,1.,0.],[4.,3.,2.,1.,0.]], device=device),
        "lengths": torch.tensor([5,4,3], device=device), "gap": torch.tensor([1.,2.,3.], device=device),
        "counts": torch.tensor([[1.,0,0,0],[1.,1.,0,0],[0,0,2.,1]], device=device),
    }
    # Force valid composition/mask fields and explicitly zero empty semantics.
    batch["sequence"][..., 512:516] = torch.softmax(batch["sequence"][..., 512:516], -1)
    batch["sequence"][..., 516:520] = 1.0
    batch["sequence"][..., 520:] = batch["sequence"][..., 520:].abs()
    mark = GroupMarkSNMPP().to(device); time_model = GroupTimeSNMPP().to(device)
    logits = mark(batch); nll = time_model.nll(batch)
    permutation = torch.tensor([4,2,0,3,1], device=device)
    permuted = {k: v.clone() for k, v in batch.items()}
    # Only the first full-length row is a valid arbitrary permutation test; ages
    # move with group features, preserving source identity.
    permuted["sequence"][0] = batch["sequence"][0, permutation]
    permuted["age0"][0] = batch["age0"][0, permutation]
    logits_p = mark(permuted); nll_p = time_model.nll(permuted)
    (mark_loss(logits, batch["counts"]) + nll.mean()).backward()
    return {
        "mark_output_finite": bool(torch.isfinite(logits).all()),
        "time_nll_finite": bool(torch.isfinite(nll).all()),
        "time_hazard_positive": bool((time_model.nll(batch, return_diagnostics=True)[1]["target_hazard"] > 0).all()),
        "group_source_permutation_invariant_mark": bool(torch.allclose(logits[0], logits_p[0], atol=2e-6)),
        "group_source_permutation_invariant_time": bool(torch.allclose(nll[0], nll_p[0], atol=2e-6)),
        "signed_parameters_have_finite_gradients": bool(all(p.grad is None or torch.isfinite(p.grad).all() for p in list(mark.parameters())+list(time_model.parameters()))),
    }


def make_decisions(time_runs, mark_runs, time_summary, mark_summary, baseline_time, baseline_mark, time_slices):
    time_gru = baseline_time["models"]["full_history_time"]["validation"]["nll_per_group"]["mean"]
    time_simple = baseline_time["baselines"]["time"]["history_free_lognormal_fit_on_moderate_train"]["validation"]["nll_per_group"]
    time_values = time_summary["validation"]["nll_per_group"]["values"]
    time_pass = bool(all(v < time_simple for v in time_values) and all(np.isfinite(v) for v in time_values))
    mark_previous = baseline_mark["baselines"]["validation"]["previous_group_composition_global_Dirichlet_strength_1"]["ce_per_event"]
    mark_last = baseline_mark["models"]["last_group_mlp"]["validation"]["ce_per_event"]["mean"]
    mark_full = baseline_mark["models"]["full_history_gru"]["validation"]["ce_per_event"]["mean"]
    mark_values = mark_summary["validation"]["ce_per_event"]["values"]
    mark_pass = bool(all(v < min(mark_previous, mark_last) for v in mark_values) and all(np.isfinite(v) for v in mark_values))
    matrices = np.asarray([r["interaction"]["psi"] for r in mark_runs])
    sign_agreement = np.mean(np.all(np.sign(matrices) == np.sign(matrices[:1]), axis=0))
    both_signs = all((m > 0).any() and (m < 0).any() for m in matrices)
    nontrivial = all(np.max(np.abs(m)) > 1e-3 for m in matrices)
    signed = bool(both_signs and nontrivial and sign_agreement >= 0.5)
    max_lambda = max(r["validation"]["lambda_max"] for r in time_runs)
    finite_burst = True
    burst_rows = time_slices["previous_group_size"].get(">500", {"targets": 0})
    if burst_rows.get("targets", 0):
        finite_burst = all(np.isfinite(s["lambda_max"]) and s["lambda_max"] < 1e4 for s in burst_rows["seeds"])
    return {
        "group_time_snmpp_passed": time_pass,
        "group_mark_snmpp_passed": mark_pass,
        "signed_influence_learned": signed,
        "joint_group_snmpp_approved": bool(time_pass and mark_pass and signed and finite_burst),
        "group_time_validation_nll_mean": time_summary["validation"]["nll_per_group"]["mean"],
        "matched_GRU_time_validation_nll": time_gru,
        "group_mark_validation_CE_mean": mark_summary["validation"]["ce_per_event"]["mean"],
        "matched_full_history_GRU_mark_validation_CE": mark_full,
        "mark_gap_vs_full_history_GRU": mark_summary["validation"]["ce_per_event"]["mean"] - mark_full,
        "mark_psi_sign_agreement_fraction": float(sign_agreement),
        "both_positive_and_negative_psi_all_seeds": both_signs,
        "validation_lambda_max_all_seeds": float(max_lambda),
        "previous_burst_slice_finite": finite_burst,
        "test_used": False,
        "joint_model_started": False,
        "hierarchical_sid_head_used": False,
        "stop_reason": "Group-SNMPP Design & Minimal Validation Gate complete; Joint was not started.",
    }


def write_report(payload: dict) -> None:
    status = payload["status"]; t = payload["time"]["group_snmpp"]; m = payload["mark"]["group_snmpp"]
    lines = [
        "# Group-SNMPP Design & Minimal Validation Gate", "",
        "## Result", "",
        f"- group_time_snmpp_passed: `{str(status['group_time_snmpp_passed']).lower()}`",
        f"- group_mark_snmpp_passed: `{str(status['group_mark_snmpp_passed']).lower()}`",
        f"- signed_influence_learned: `{str(status['signed_influence_learned']).lower()}`",
        f"- joint_group_snmpp_approved: `{str(status['joint_group_snmpp_approved']).lower()}`", "",
        "## Time", "",
        f"- Group-Time validation NLL/group: {t['validation']['nll_per_group']['mean']:.6f}",
        f"- matched GRU validation NLL/group: {status['matched_GRU_time_validation_nll']:.6f}",
        f"- Group-Time validation MAE: {t['validation']['mae_hours']['mean']:.6f} hours", "",
        "## Mark", "",
        f"- Group-Mark validation CE/event: {m['validation']['ce_per_event']['mean']:.6f}",
        f"- matched Full-history GRU CE/event: {status['matched_full_history_GRU_mark_validation_CE']:.6f}",
        f"- CE gap (Group-Mark - GRU): {status['mark_gap_vs_full_history_GRU']:.6f}",
        f"- psi sign agreement: {status['mark_psi_sign_agreement_fraction']:.2%}", "",
        "No test data, Joint model, hierarchical SID head, HPN, or BOLA was used. This gate stops here.",
    ]
    (WORK / "Group_SNMPP_Design_Minimal_Validation_Report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
