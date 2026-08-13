#!/usr/bin/env python3
"""Build a deterministic, train-only subset for the timestamp likelihood gate.

This script never invents an order inside a timestamp group.  It first rebuilds
group-level metadata from the four feedback-partitioned Gate-2 event tables,
then samples target groups by (a) target cardinality and (b) immediately
preceding group cardinality.  Only feedback multiplicities are materialized;
the original within-group row order is deliberately discarded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
EVENT_DIR = ROOT / "phase2_gate2_sid/materialized_v1_1/events"
MANIFEST = ROOT / "phase2_gate2_sid/gate2_manifest.json"
OUT = ROOT / "phase2_timestamp_gate/artifacts/gate_subset.json"

BIN_SPECS = (
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-20", 6, 20),
    ("21-100", 21, 100),
    ("101-500", 101, 500),
    ("500+", 501, np.iinfo(np.uint32).max),
)
PER_BIN = 64


def choose_evenly(ids: np.ndarray, n: int) -> np.ndarray:
    """Deterministically cover the full candidate-id range."""
    if ids.size <= n:
        return ids
    positions = np.linspace(0, ids.size - 1, num=n, dtype=np.int64)
    return ids[positions]


def run() -> None:
    started = time.time()
    manifest = json.loads(MANIFEST.read_text())
    n_groups = int(manifest["before_join"]["groups"])
    cutoff = int(manifest["cutoffs"]["train_inclusive"])

    group_size = np.zeros(n_groups, dtype=np.uint32)
    group_uid = np.zeros(n_groups, dtype=np.uint32)
    group_ts = np.zeros(n_groups, dtype=np.uint32)

    files = sorted(EVENT_DIR.glob("feedback_type=*.parquet"))
    if len(files) != 4:
        raise RuntimeError(f"expected four event shards, found {len(files)}")

    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(
            batch_size=1_000_000, columns=["group_id", "uid", "timestamp"]
        ):
            gid = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            uid = batch.column(1).to_numpy(zero_copy_only=False)
            ts = batch.column(2).to_numpy(zero_copy_only=False)
            starts = np.r_[True, gid[1:] != gid[:-1]]
            start_idx = np.flatnonzero(starts)
            unique_gid = gid[start_idx]
            counts = np.diff(np.r_[start_idx, gid.size]).astype(np.uint32)
            group_size[unique_gid] += counts
            group_uid[unique_gid] = uid[start_idx]
            group_ts[unique_gid] = ts[start_idx]

    if int((group_size == 0).sum()) != 0:
        raise RuntimeError("Gate-2 group ids are not dense or some groups were lost")
    if int(group_size.astype(np.uint64).sum()) != int(manifest["before_join"]["events"]):
        raise RuntimeError("event conservation failed while rebuilding group metadata")

    gid_all = np.arange(n_groups, dtype=np.int64)
    has_previous = np.zeros(n_groups, dtype=bool)
    has_previous[1:] = group_uid[1:] == group_uid[:-1]
    has_previous &= gid_all > 0
    has_previous &= group_ts > np.r_[np.uint32(0), group_ts[:-1]]
    train_target = (group_ts <= cutoff) & has_previous

    selected_target: dict[str, list[int]] = {}
    selected_previous: dict[str, list[int]] = {}
    selected_ids: set[int] = set()

    for name, lo, hi in BIN_SPECS:
        mask_target = train_target & (group_size >= lo) & (group_size <= hi)
        ids = np.flatnonzero(mask_target)
        chosen = choose_evenly(ids, PER_BIN)
        selected_target[name] = [int(x) for x in chosen]
        selected_ids.update(int(x) for x in chosen)
        selected_ids.update(int(x - 1) for x in chosen)

        previous_size = np.zeros_like(group_size)
        previous_size[1:] = group_size[:-1]
        mask_previous = train_target & (previous_size >= lo) & (previous_size <= hi)
        ids = np.flatnonzero(mask_previous)
        chosen = choose_evenly(ids, PER_BIN)
        selected_previous[name] = [int(x) for x in chosen]
        selected_ids.update(int(x) for x in chosen)
        selected_ids.update(int(x - 1) for x in chosen)

    selected = np.asarray(sorted(selected_ids), dtype=np.int64)
    feedback_by_gid: dict[int, list[int]] = {int(g): [] for g in selected}

    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1_000_000, columns=["group_id", "feedback_type"]):
            gid = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            fb = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            pos = np.searchsorted(selected, gid)
            valid = pos < selected.size
            matched = np.zeros_like(valid)
            matched[valid] = selected[pos[valid]] == gid[valid]
            for g, f in zip(gid[matched], fb[matched], strict=False):
                feedback_by_gid[int(g)].append(int(f))

    for gid in selected:
        observed = feedback_by_gid[int(gid)]
        if len(observed) != int(group_size[gid]):
            raise RuntimeError(
                f"feedback collection mismatch for group {gid}: "
                f"{len(observed)} != {int(group_size[gid])}"
            )
        observed.sort()  # multiset canonicalization, not chronology

    union_targets = sorted(
        set(x for values in selected_target.values() for x in values)
        | set(x for values in selected_previous.values() for x in values)
    )
    records = []
    for gid in union_targets:
        prev = gid - 1
        records.append(
            {
                "target_group_id": gid,
                "uid": int(group_uid[gid]),
                "target_timestamp": int(group_ts[gid]),
                "target_group_size": int(group_size[gid]),
                "target_feedback_multiset": feedback_by_gid[gid],
                "previous_group_id": prev,
                "previous_timestamp": int(group_ts[prev]),
                "previous_group_size": int(group_size[prev]),
                "previous_feedback_multiset": feedback_by_gid[prev],
            }
        )

    result = {
        "status": "complete_fixed_train_only_gate_subset",
        "selection": {
            "method": "deterministic evenly-spaced group ids within each bin",
            "per_bin_requested": PER_BIN,
            "train_cutoff_inclusive": cutoff,
            "validation_or_test_used": False,
            "within_group_order_used": False,
            "feedback_representation": "sorted multiset solely for canonical storage",
        },
        "target_group_size_samples": selected_target,
        "previous_group_size_samples": selected_previous,
        "records": records,
        "counts": {
            "records": len(records),
            "materialized_groups": len(selected),
            "source_groups": n_groups,
        },
        "elapsed_seconds": time.time() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    run()
