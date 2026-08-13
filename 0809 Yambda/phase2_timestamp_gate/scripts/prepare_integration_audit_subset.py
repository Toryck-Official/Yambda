#!/usr/bin/env python3
"""Create a deterministic train-only subset for integration-Q accuracy audit.

Sampling is stratified by target interval length and immediately preceding
timestamp-group cardinality.  Within-group events are stored as canonical
feedback multisets; no row order is interpreted as chronology.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
EVENT_DIR = ROOT / "phase2_gate2_sid/materialized_v1_1/events"
GATE2 = ROOT / "phase2_gate2_sid/gate2_manifest.json"
TIME_AUDIT = ROOT / "phase2_timestamp_gate/artifacts/train_time_scale_audit.json"
OUT = ROOT / "phase2_timestamp_gate/artifacts/integration_audit_subset.json"

GAP_NAMES = ("le_p50", "p50_p90", "p90_p95", "p95_p99", "gt_p99_le_horizon")
SIZE_NAMES = ("1", "2_5", "6_20", "21_100", "101_500", "gt_500")
PER_CELL = 12


def evenly_spaced(values: np.ndarray, count: int) -> np.ndarray:
    if values.size <= count:
        return values
    return values[np.linspace(0, values.size - 1, count, dtype=np.int64)]


def run() -> None:
    started = time.time()
    gate2 = json.loads(GATE2.read_text())
    time_audit = json.loads(TIME_AUDIT.read_text())
    n_groups = int(gate2["before_join"]["groups"])
    train_cutoff = int(gate2["cutoffs"]["train_inclusive"])
    horizon = int(time_audit["frozen_train_only_decision"]["prediction_horizon_seconds"])
    gap_stats = time_audit["raw_timestamp_gap"]
    boundaries = (
        int(gap_stats["p50"]),
        int(gap_stats["p90"]),
        int(gap_stats["p95"]),
        int(gap_stats["p99"]),
        horizon,
    )

    group_size = np.zeros(n_groups, dtype=np.uint32)
    group_uid = np.zeros(n_groups, dtype=np.uint32)
    group_ts = np.zeros(n_groups, dtype=np.uint32)

    for path in sorted(EVENT_DIR.glob("feedback_type=*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=1_000_000, columns=["group_id", "uid", "timestamp"]
        ):
            gid = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            uid = batch.column(1).to_numpy(zero_copy_only=False)
            timestamp = batch.column(2).to_numpy(zero_copy_only=False)
            starts = np.r_[True, gid[1:] != gid[:-1]]
            indices = np.flatnonzero(starts)
            unique_gid = gid[indices]
            counts = np.diff(np.r_[indices, gid.size]).astype(np.uint32)
            group_size[unique_gid] += counts
            group_uid[unique_gid] = uid[indices]
            group_ts[unique_gid] = timestamp[indices]

    if np.any(group_size == 0):
        raise RuntimeError("non-dense or missing Gate-2 groups")
    if int(group_size.astype(np.uint64).sum()) != int(gate2["before_join"]["events"]):
        raise RuntimeError("event conservation failed")

    previous_same_user = np.zeros(n_groups, dtype=bool)
    previous_same_user[1:] = group_uid[1:] == group_uid[:-1]
    gaps = np.zeros(n_groups, dtype=np.uint32)
    valid_difference = previous_same_user & (group_ts > np.r_[np.uint32(0), group_ts[:-1]])
    positions = np.flatnonzero(valid_difference)
    gaps[positions] = group_ts[positions] - group_ts[positions - 1]
    eligible = valid_difference & (group_ts <= train_cutoff) & (gaps <= horizon)

    gap_code = np.full(n_groups, 255, dtype=np.uint8)
    p50, p90, p95, p99, h = boundaries
    gap_code[eligible & (gaps <= p50)] = 0
    gap_code[eligible & (gaps > p50) & (gaps <= p90)] = 1
    gap_code[eligible & (gaps > p90) & (gaps <= p95)] = 2
    gap_code[eligible & (gaps > p95) & (gaps <= p99)] = 3
    gap_code[eligible & (gaps > p99) & (gaps <= h)] = 4

    previous_size = np.zeros(n_groups, dtype=np.uint32)
    previous_size[1:] = group_size[:-1]
    size_code = np.full(n_groups, 255, dtype=np.uint8)
    size_code[eligible & (previous_size == 1)] = 0
    size_code[eligible & (previous_size >= 2) & (previous_size <= 5)] = 1
    size_code[eligible & (previous_size >= 6) & (previous_size <= 20)] = 2
    size_code[eligible & (previous_size >= 21) & (previous_size <= 100)] = 3
    size_code[eligible & (previous_size >= 101) & (previous_size <= 500)] = 4
    size_code[eligible & (previous_size > 500)] = 5

    selected_by_cell: dict[str, list[int]] = {}
    candidate_counts: dict[str, int] = {}
    selected_targets: set[int] = set()
    for gap_index, gap_name in enumerate(GAP_NAMES):
        for size_index, size_name in enumerate(SIZE_NAMES):
            key = f"{gap_name}__prev_{size_name}"
            candidates = np.flatnonzero((gap_code == gap_index) & (size_code == size_index))
            chosen = evenly_spaced(candidates, PER_CELL)
            candidate_counts[key] = int(candidates.size)
            selected_by_cell[key] = [int(x) for x in chosen]
            selected_targets.update(int(x) for x in chosen)

    selected_groups = sorted(selected_targets | {x - 1 for x in selected_targets})
    selected = np.asarray(selected_groups, dtype=np.int64)
    feedback_by_gid: dict[int, list[int]] = {int(g): [] for g in selected}
    for path in sorted(EVENT_DIR.glob("feedback_type=*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1_000_000, columns=["group_id", "feedback_type"]):
            gid = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            feedback = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            location = np.searchsorted(selected, gid)
            valid = location < selected.size
            matched = np.zeros_like(valid)
            matched[valid] = selected[location[valid]] == gid[valid]
            for group, mark in zip(gid[matched], feedback[matched], strict=False):
                feedback_by_gid[int(group)].append(int(mark))

    records = []
    for gid in sorted(selected_targets):
        previous = gid - 1
        target_marks = sorted(feedback_by_gid[gid])
        previous_marks = sorted(feedback_by_gid[previous])
        if len(target_marks) != int(group_size[gid]):
            raise RuntimeError(f"target group {gid} cardinality mismatch")
        if len(previous_marks) != int(group_size[previous]):
            raise RuntimeError(f"previous group {previous} cardinality mismatch")
        records.append(
            {
                "target_group_id": gid,
                "uid": int(group_uid[gid]),
                "target_timestamp": int(group_ts[gid]),
                "target_feedback_multiset": target_marks,
                "target_group_size": int(group_size[gid]),
                "previous_group_id": previous,
                "previous_timestamp": int(group_ts[previous]),
                "previous_feedback_multiset": previous_marks,
                "previous_group_size": int(group_size[previous]),
                "gap_seconds": int(gaps[gid]),
                "gap_bin": GAP_NAMES[int(gap_code[gid])],
                "previous_group_size_bin": SIZE_NAMES[int(size_code[gid])],
            }
        )

    result = {
        "status": "complete_train_only_integration_audit_subset",
        "data_contract": {
            "dataset": "D_SID",
            "split": "train only",
            "train_cutoff_inclusive": train_cutoff,
            "validation_or_test_used": False,
            "within_group_order_used": False,
            "history_scope": "immediately preceding complete timestamp group",
            "selection": "deterministic evenly spaced group ids per gap x previous-size cell",
            "per_cell_requested": PER_CELL,
        },
        "gap_boundaries_seconds": {
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "horizon": h,
        },
        "candidate_counts_by_cell": candidate_counts,
        "selected_group_ids_by_cell": selected_by_cell,
        "counts": {
            "records": len(records),
            "materialized_groups": len(selected_groups),
            "nonempty_cells": sum(bool(x) for x in selected_by_cell.values()),
            "total_cells": len(selected_by_cell),
        },
        "records": records,
        "elapsed_seconds": time.time() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"], indent=2))
    for gap_name in GAP_NAMES:
        print(gap_name, [len(selected_by_cell[f"{gap_name}__prev_{s}"]) for s in SIZE_NAMES])


if __name__ == "__main__":
    run()
