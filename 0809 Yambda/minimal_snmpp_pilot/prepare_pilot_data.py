#!/usr/bin/env python3
"""Prepare fixed Tiny/Pilot full-history subsets from materialized D_SID."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
EVENT_DIR = ROOT / "phase2_gate2_sid/materialized_v1_1/events"
GATE2_MANIFEST = ROOT / "phase2_gate2_sid/gate2_manifest.json"
OUT_DIR = ROOT / "minimal_snmpp_pilot/data"
OUT_NPZ = OUT_DIR / "pilot_sequences.npz"
OUT_MANIFEST = OUT_DIR / "subset_manifest.json"
SEED = 2026
TINY_PER_FEEDBACK = 1500
PILOT_TARGET_GOAL = 100_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def evenly(values: np.ndarray, count: int) -> np.ndarray:
    if values.size <= count:
        return values
    return values[np.linspace(0, values.size - 1, count, dtype=np.int64)]


def stable_user_order(users: np.ndarray) -> np.ndarray:
    values = users.astype(np.uint64)
    keys = (values * np.uint64(11400714819323198485) + np.uint64(SEED))
    return users[np.argsort(keys, kind="stable")]


def run() -> None:
    started = time.time()
    gate2 = json.loads(GATE2_MANIFEST.read_text())
    n_groups = int(gate2["before_join"]["groups"])
    train_cutoff = int(gate2["cutoffs"]["train_inclusive"])
    validation_cutoff = int(gate2["cutoffs"]["validation_inclusive"])

    group_size = np.zeros(n_groups, dtype=np.uint32)
    group_uid = np.zeros(n_groups, dtype=np.uint32)
    group_ts = np.zeros(n_groups, dtype=np.uint32)
    group_feedback = np.zeros((n_groups, 4), dtype=np.uint16)

    files = sorted(EVENT_DIR.glob("feedback_type=*.parquet"))
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=1_000_000,
            columns=["group_id", "uid", "timestamp", "feedback_type"],
        ):
            gid = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            uid = batch.column(1).to_numpy(zero_copy_only=False)
            timestamp = batch.column(2).to_numpy(zero_copy_only=False)
            feedback = batch.column(3).to_numpy(zero_copy_only=False)
            starts = np.r_[True, gid[1:] != gid[:-1]]
            positions = np.flatnonzero(starts)
            unique_gid = gid[positions]
            counts = np.diff(np.r_[positions, gid.size]).astype(np.uint32)
            group_size[unique_gid] += counts
            group_uid[unique_gid] = uid[positions]
            group_ts[unique_gid] = timestamp[positions]
            mark = int(feedback[0])
            if np.any(feedback != mark):
                raise RuntimeError(f"feedback shard is not homogeneous: {path}")
            group_feedback[unique_gid, mark] += counts.astype(np.uint16)

    if np.any(group_size == 0):
        raise RuntimeError("missing group during D_SID scan")
    if not np.array_equal(group_feedback.sum(axis=1, dtype=np.uint32), group_size):
        raise RuntimeError("group feedback counts do not conserve group size")

    gid = np.arange(n_groups, dtype=np.int64)
    same_previous_user = np.zeros(n_groups, dtype=bool)
    same_previous_user[1:] = group_uid[1:] == group_uid[:-1]
    positive_gap = np.zeros(n_groups, dtype=bool)
    positive_gap[1:] = group_ts[1:] > group_ts[:-1]
    target_eligible = same_previous_user & positive_gap
    train_target = target_eligible & (group_ts <= train_cutoff)
    validation_target = target_eligible & (group_ts > train_cutoff) & (
        group_ts <= validation_cutoff
    )

    # Group position is derived only from user-contiguous group membership.
    user_group_start = np.flatnonzero(np.r_[True, group_uid[1:] != group_uid[:-1]])
    start_marker = np.zeros(n_groups, dtype=np.int64)
    start_marker[user_group_start] = user_group_start
    last_start = np.maximum.accumulate(start_marker)
    group_position = gid - last_start
    tiny_eligible = train_target & (group_position >= 5)

    tiny_targets: set[int] = set()
    for feedback in range(4):
        candidates = np.flatnonzero(tiny_eligible & (group_feedback[:, feedback] > 0))
        tiny_targets.update(int(x) for x in evenly(candidates, TINY_PER_FEEDBACK))
    for low, high in ((1, 1), (2, 5), (6, 20), (21, 100), (101, 500)):
        candidates = np.flatnonzero(tiny_eligible & (group_size >= low) & (group_size <= high))
        tiny_targets.update(int(x) for x in evenly(candidates, 80))
    previous_size = np.zeros(n_groups, dtype=np.uint32)
    previous_size[1:] = group_size[:-1]
    burst_candidates = np.flatnonzero(tiny_eligible & (previous_size > 500))
    tiny_targets.update(int(x) for x in evenly(burst_candidates, 16))
    tiny_targets_array = np.asarray(sorted(tiny_targets), dtype=np.int64)
    tiny_users = np.unique(group_uid[tiny_targets_array])

    max_uid = int(group_uid.max())
    train_group_counts = np.bincount(
        group_uid[group_ts <= train_cutoff], minlength=max_uid + 1
    ).astype(np.uint32)
    pilot_candidates = np.flatnonzero(train_group_counts >= 6).astype(np.uint32)
    ordered_users = stable_user_order(pilot_candidates)
    pilot_users_list = []
    accumulated = 0
    for user in ordered_users:
        pilot_users_list.append(int(user))
        accumulated += int(train_group_counts[user]) - 1
        if accumulated >= PILOT_TARGET_GOAL:
            break
    pilot_users = np.asarray(sorted(pilot_users_list), dtype=np.uint32)

    pilot_user_mask = np.zeros(max_uid + 1, dtype=bool)
    pilot_user_mask[pilot_users] = True
    pilot_train_targets = np.flatnonzero(train_target & pilot_user_mask[group_uid])
    pilot_validation_targets = np.flatnonzero(validation_target & pilot_user_mask[group_uid])

    selected_users = np.unique(np.concatenate([tiny_users, pilot_users]))
    selected_events: dict[str, list[np.ndarray]] = {
        name: []
        for name in ("uid", "timestamp", "group_id", "feedback", "sid_1", "sid_2", "sid_3", "sid_4")
    }
    for path in files:
        parquet = pq.ParquetFile(path)
        columns = [
            "uid",
            "timestamp",
            "group_id",
            "feedback_type",
            "sid_1",
            "sid_2",
            "sid_3",
            "sid_4",
        ]
        for batch in parquet.iter_batches(batch_size=1_000_000, columns=columns):
            uid = batch.column(0).to_numpy(zero_copy_only=False)
            timestamp = batch.column(1).to_numpy(zero_copy_only=False)
            locations = np.searchsorted(selected_users, uid)
            valid = locations < selected_users.size
            selected = np.zeros_like(valid)
            selected[valid] = selected_users[locations[valid]] == uid[valid]
            selected &= timestamp <= validation_cutoff
            if not np.any(selected):
                continue
            for name, column in zip(selected_events, range(len(columns)), strict=True):
                selected_events[name].append(
                    batch.column(column).to_numpy(zero_copy_only=False)[selected]
                )

    arrays = {name: np.concatenate(parts) for name, parts in selected_events.items()}
    order = np.lexsort(
        (
            arrays["sid_4"],
            arrays["sid_3"],
            arrays["sid_2"],
            arrays["sid_1"],
            arrays["feedback"],
            arrays["group_id"],
        )
    )
    arrays = {name: value[order] for name, value in arrays.items()}
    group_starts = np.flatnonzero(np.r_[True, arrays["group_id"][1:] != arrays["group_id"][:-1]])
    local_group_ids = arrays["group_id"][group_starts].astype(np.uint64)
    group_offsets = np.r_[group_starts, len(arrays["group_id"])].astype(np.int64)
    local_group_uid = arrays["uid"][group_starts].astype(np.uint32)
    local_group_ts = arrays["timestamp"][group_starts].astype(np.uint32)
    local_group_feedback = np.zeros((len(group_starts), 4), dtype=np.uint16)
    for index in range(len(group_starts)):
        values = arrays["feedback"][group_offsets[index] : group_offsets[index + 1]]
        local_group_feedback[index] = np.bincount(values, minlength=4).astype(np.uint16)

    def map_targets(global_ids: np.ndarray) -> np.ndarray:
        positions = np.searchsorted(local_group_ids, global_ids.astype(np.uint64))
        valid = positions < local_group_ids.size
        if not np.all(valid) or not np.array_equal(local_group_ids[positions], global_ids):
            raise RuntimeError("selected target group missing from materialized subset")
        return positions.astype(np.int64)

    tiny_local = map_targets(tiny_targets_array)
    pilot_train_local = map_targets(pilot_train_targets)
    pilot_validation_local = map_targets(pilot_validation_targets)

    # Global train-only baselines use all D_SID train groups/events.
    train_gap_mask = train_target
    train_gaps = group_ts[train_gap_mask].astype(np.int64) - group_ts[np.flatnonzero(train_gap_mask) - 1].astype(np.int64)
    global_feedback_counts = group_feedback[group_ts <= train_cutoff].sum(axis=0, dtype=np.uint64)
    baseline = {
        "global_train_feedback_counts": [int(x) for x in global_feedback_counts],
        "global_train_feedback_probabilities": [
            float(x / global_feedback_counts.sum()) for x in global_feedback_counts
        ],
        "global_train_gap_mean_seconds": float(train_gaps.mean()),
        "global_train_gap_median_seconds": float(np.median(train_gaps)),
        "constant_exponential_rate_per_hour": float(3600.0 / train_gaps.mean()),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        event_timestamp=arrays["timestamp"].astype(np.uint32),
        event_feedback=arrays["feedback"].astype(np.uint8),
        event_sid=np.column_stack(
            [arrays[f"sid_{level}"].astype(np.uint8) for level in range(1, 5)]
        ),
        group_global_id=local_group_ids,
        group_uid=local_group_uid,
        group_timestamp=local_group_ts,
        group_event_offsets=group_offsets,
        group_feedback_counts=local_group_feedback,
        tiny_train_targets=tiny_local,
        pilot_train_targets=pilot_train_local,
        pilot_validation_targets=pilot_validation_local,
        tiny_users=tiny_users,
        pilot_users=pilot_users,
    )
    manifest = {
        "status": "complete_fixed_full_history_pilot_subsets",
        "seed": SEED,
        "data_contract": {
            "dataset": "D_SID",
            "listen_used": False,
            "missing_audio_used": False,
            "test_used": False,
            "history_truncated": False,
            "within_group_order_used": False,
            "canonical_event_storage_order": "group_id,feedback,sid solely for permutation-stable storage",
            "user_selection_uses_train_period_only": True,
        },
        "cutoffs": {
            "train_inclusive": train_cutoff,
            "validation_inclusive": validation_cutoff,
        },
        "selection": {
            "tiny": "train target stratification by feedback/group size plus 16 previous-burst cases",
            "pilot": "stable hash order over users with at least six train groups until 100k train targets",
            "pilot_target_goal": PILOT_TARGET_GOAL,
        },
        "counts": {
            "selected_users_union": int(selected_users.size),
            "materialized_events_through_validation": int(len(arrays["feedback"])),
            "materialized_groups_through_validation": int(len(local_group_ids)),
            "tiny_users": int(tiny_users.size),
            "tiny_train_targets": int(tiny_local.size),
            "pilot_users": int(pilot_users.size),
            "pilot_train_targets": int(pilot_train_local.size),
            "pilot_validation_targets": int(pilot_validation_local.size),
            "tiny_target_feedback_counts": [
                int(x) for x in local_group_feedback[tiny_local].sum(axis=0, dtype=np.uint64)
            ],
            "tiny_previous_burst_targets": int(
                sum(
                    local_group_feedback[index - 1].sum() > 500
                    for index in tiny_local
                )
            ),
        },
        "baselines": baseline,
        "artifacts": {
            "npz": str(OUT_NPZ),
            "npz_sha256": None,
        },
        "elapsed_seconds": time.time() - started,
    }
    manifest["artifacts"]["npz_sha256"] = sha256(OUT_NPZ)
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"counts": manifest["counts"], "baselines": baseline}, indent=2))


if __name__ == "__main__":
    run()
