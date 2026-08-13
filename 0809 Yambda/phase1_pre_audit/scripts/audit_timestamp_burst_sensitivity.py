#!/usr/bin/env python3
"""Read-only timestamp-burst sensitivity audit for Yambda explicit events.

No event is removed.  Candidate thresholds are derived after the scan from the
empirical timestamp-group distribution.  A strict revision pair is considered
"touched" by a threshold when either its origin group or revision group would
be excluded; the script does not invent a re-pairing after hypothetical removal.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE0_SCRIPTS = PROJECT_ROOT / "phase0_5b_audit" / "scripts"
sys.path.insert(0, str(PHASE0_SCRIPTS))

from audit_explicit_5b import EVENT_NAMES, EVENT_SPECS, FlatUserStream  # noqa: E402


MAX_GROUP_SIZE = 10_000
DEFAULT_FLAT_DIR = PROJECT_ROOT / "phase0_5b_audit" / "work" / "flat_explicit_5b"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "phase1_pre_audit"
    / "artifacts"
    / "timestamp_burst_sensitivity.json"
)
DEFAULT_PROGRESS = (
    PROJECT_ROOT / "phase1_pre_audit" / "artifacts" / "timestamp_burst_progress.json"
)
QUANTILES = (
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
    ("p99_9", 0.999),
    ("p99_99", 0.9999),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-dir", type=Path, default=DEFAULT_FLAT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--progress-every-users", type=int, default=10_000)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def quantile_from_histogram(histogram: np.ndarray, q: float) -> int:
    total = int(histogram.sum(dtype=np.uint64))
    if total <= 0:
        raise ValueError("empty histogram")
    rank = max(1, math.ceil(q * total))
    cumulative = np.cumsum(histogram, dtype=np.uint64)
    return int(np.searchsorted(cumulative, rank, side="left"))


def tail_sum(array: np.ndarray, threshold: int) -> int:
    return int(array[threshold + 1 :].sum(dtype=np.uint64))


def threshold_report(
    threshold: int,
    group_hist: np.ndarray,
    event_hist: np.ndarray,
    user_max_hist: np.ndarray,
    feedback_hist: np.ndarray,
    like_pair_hist: np.ndarray,
    dislike_pair_hist: np.ndarray,
) -> dict:
    total_groups = int(group_hist.sum(dtype=np.uint64))
    total_events = int(event_hist.sum(dtype=np.uint64))
    total_users = int(user_max_hist.sum(dtype=np.uint64))
    feedback_before = feedback_hist.sum(axis=0, dtype=np.uint64)
    feedback_excluded = feedback_hist[threshold + 1 :].sum(axis=0, dtype=np.uint64)
    feedback_after = feedback_before - feedback_excluded
    excluded_groups = tail_sum(group_hist, threshold)
    excluded_events = tail_sum(event_hist, threshold)
    affected_users = tail_sum(user_max_hist, threshold)
    like_pairs_total = int(like_pair_hist.sum(dtype=np.uint64))
    dislike_pairs_total = int(dislike_pair_hist.sum(dtype=np.uint64))
    like_pairs_touched = tail_sum(like_pair_hist, threshold)
    dislike_pairs_touched = tail_sum(dislike_pair_hist, threshold)
    return {
        "rule": f"exclude timestamp groups with size > {threshold}",
        "threshold_inclusive_keep_max": threshold,
        "excluded": {
            "groups": excluded_groups,
            "group_fraction": excluded_groups / total_groups,
            "events": excluded_events,
            "event_fraction": excluded_events / total_events,
            "users_with_at_least_one_excluded_group": affected_users,
            "user_fraction": affected_users / total_users,
            "feedback_events": {
                name: int(feedback_excluded[index])
                for index, name in enumerate(EVENT_NAMES)
            },
        },
        "retained": {
            "groups": total_groups - excluded_groups,
            "events": total_events - excluded_events,
            "feedback_events": {
                name: int(feedback_after[index])
                for index, name in enumerate(EVENT_NAMES)
            },
        },
        "strict_revision_pairs_touched_at_either_endpoint": {
            "like_to_unlike": like_pairs_touched,
            "like_to_unlike_fraction": like_pairs_touched / like_pairs_total,
            "dislike_to_undislike": dislike_pairs_touched,
            "dislike_to_undislike_fraction": (
                dislike_pairs_touched / dislike_pairs_total
            ),
            "definition": (
                "existing strict pair has origin or revision in an excluded group; "
                "no hypothetical post-filter re-pairing"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    streams = {
        event_name: FlatUserStream(args.flat_dir / event_name)
        for _, event_name, _ in EVENT_SPECS
    }
    group_hist = np.zeros(MAX_GROUP_SIZE + 1, dtype=np.uint64)
    event_hist = np.zeros(MAX_GROUP_SIZE + 1, dtype=np.uint64)
    user_max_hist = np.zeros(MAX_GROUP_SIZE + 1, dtype=np.uint64)
    feedback_hist = np.zeros((MAX_GROUP_SIZE + 1, len(EVENT_NAMES)), dtype=np.uint64)
    like_pair_endpoint_max_hist = np.zeros(MAX_GROUP_SIZE + 1, dtype=np.uint64)
    dislike_pair_endpoint_max_hist = np.zeros(MAX_GROUP_SIZE + 1, dtype=np.uint64)
    users = 0
    events = 0
    pair_counts = Counter()

    while True:
        active_streams = [stream for stream in streams.values() if stream.current is not None]
        if not active_streams:
            break
        if args.max_users and users >= args.max_users:
            break
        uid = min(int(stream.current.uid) for stream in active_streams if stream.current)
        timestamp_parts: list[np.ndarray] = []
        item_parts: list[np.ndarray] = []
        type_parts: list[np.ndarray] = []
        for event_id, event_name, _ in EVENT_SPECS:
            stream = streams[event_name]
            if stream.current is None or stream.current.uid != uid:
                continue
            row = stream.pop()
            count = int(row.timestamp.size)
            timestamp_parts.append(row.timestamp)
            item_parts.append(row.item_id)
            type_parts.append(np.full(count, event_id, dtype=np.uint8))

        timestamp = np.concatenate(timestamp_parts).astype(np.uint32, copy=False)
        item_id = np.concatenate(item_parts).astype(np.uint32, copy=False)
        event_type = np.concatenate(type_parts).astype(np.uint8, copy=False)
        order = np.lexsort((event_type, item_id, timestamp))
        timestamp = timestamp[order]
        item_id = item_id[order]
        event_type = event_type[order]
        count = int(timestamp.size)
        events += count
        users += 1

        group_start_mask = np.empty(count, dtype=bool)
        group_start_mask[0] = True
        group_start_mask[1:] = timestamp[1:] != timestamp[:-1]
        group_starts = np.flatnonzero(group_start_mask)
        group_ends = np.r_[group_starts[1:], count]
        group_sizes = (group_ends - group_starts).astype(np.int64, copy=False)
        if int(group_sizes.max()) > MAX_GROUP_SIZE:
            raise ValueError(f"group size exceeds audited maximum for uid={uid}")
        group_hist += np.bincount(group_sizes, minlength=MAX_GROUP_SIZE + 1).astype(
            np.uint64, copy=False
        )
        event_hist += np.bincount(
            group_sizes, weights=group_sizes, minlength=MAX_GROUP_SIZE + 1
        ).astype(np.uint64, copy=False)
        user_max_hist[int(group_sizes.max())] += 1
        for mark in range(len(EVENT_NAMES)):
            mark_per_group = np.add.reduceat(
                (event_type == mark).astype(np.uint32), group_starts
            )
            feedback_hist[:, mark] += np.bincount(
                group_sizes,
                weights=mark_per_group,
                minlength=MAX_GROUP_SIZE + 1,
            ).astype(np.uint64, copy=False)

        pair_start_mask = np.empty(count, dtype=bool)
        pair_start_mask[0] = True
        pair_start_mask[1:] = (
            (timestamp[1:] != timestamp[:-1]) | (item_id[1:] != item_id[:-1])
        )
        pair_starts = np.flatnonzero(pair_start_mask)
        pair_ends = np.r_[pair_starts[1:], count]
        pair_group_indices = (
            np.searchsorted(group_starts, pair_starts, side="right") - 1
        )
        pair_group_sizes = group_sizes[pair_group_indices]
        active_like: dict[int, tuple[int, int]] = {}
        active_dislike: dict[int, tuple[int, int]] = {}
        for start, end, current_group_size in zip(
            pair_starts.tolist(),
            pair_ends.tolist(),
            pair_group_sizes.tolist(),
            strict=True,
        ):
            item = int(item_id[start])
            first_mark = int(event_type[start])
            if first_mark != int(event_type[end - 1]):
                pair_counts["ambiguous_same_item_time_groups"] += 1
                continue
            current_time = int(timestamp[start])
            if first_mark == 0:
                if item not in active_like:
                    active_like[item] = (current_time, int(current_group_size))
            elif first_mark == 1:
                if item not in active_dislike:
                    active_dislike[item] = (current_time, int(current_group_size))
            elif first_mark == 2:
                origin = active_like.pop(item, None)
                if origin is not None:
                    if current_time <= origin[0]:
                        raise RuntimeError("non-positive strict like->unlike delay")
                    endpoint_max = max(origin[1], int(current_group_size))
                    like_pair_endpoint_max_hist[endpoint_max] += 1
            else:
                origin = active_dislike.pop(item, None)
                if origin is not None:
                    if current_time <= origin[0]:
                        raise RuntimeError("non-positive strict dislike->undislike delay")
                    endpoint_max = max(origin[1], int(current_group_size))
                    dislike_pair_endpoint_max_hist[endpoint_max] += 1

        if users % args.progress_every_users == 0:
            progress = {
                "status": "running",
                "users": users,
                "events": events,
                "last_uid": uid,
                "elapsed_seconds": time.time() - started,
                "max_users": args.max_users,
            }
            atomic_json(args.progress, progress)
            print(progress, flush=True)

    quantiles = {
        label: quantile_from_histogram(group_hist, q) for label, q in QUANTILES
    }
    candidate_thresholds = {
        label: quantiles[label] for label in ("p99", "p99_9", "p99_99")
    }
    threshold_effects = {
        label: threshold_report(
            threshold,
            group_hist,
            event_hist,
            user_max_hist,
            feedback_hist,
            like_pair_endpoint_max_hist,
            dislike_pair_endpoint_max_hist,
        )
        for label, threshold in candidate_thresholds.items()
    }
    output = {
        "status": "complete_read_only_audit",
        "data_contract": {
            "events": list(EVENT_NAMES),
            "group_key": ["uid", "timestamp"],
            "threshold_rule": "exclude group only when group_size > threshold",
            "cleaning_or_filtering_performed": False,
            "same_timestamp_order_invented": False,
            "source": str(args.flat_dir.resolve()),
        },
        "counts": {
            "users": users,
            "events": events,
            "timestamp_groups": int(group_hist.sum(dtype=np.uint64)),
            "feedback_events": {
                name: int(feedback_hist[:, index].sum(dtype=np.uint64))
                for index, name in enumerate(EVENT_NAMES)
            },
            "strict_revision_pairs": {
                "like_to_unlike": int(
                    like_pair_endpoint_max_hist.sum(dtype=np.uint64)
                ),
                "dislike_to_undislike": int(
                    dislike_pair_endpoint_max_hist.sum(dtype=np.uint64)
                ),
            },
            "ambiguous_same_item_time_groups": int(
                pair_counts["ambiguous_same_item_time_groups"]
            ),
        },
        "group_size_quantiles": quantiles,
        "group_size_max": int(np.flatnonzero(group_hist)[-1]),
        "candidate_thresholds_are_distribution_derived": candidate_thresholds,
        "threshold_effects": threshold_effects,
        "histograms": {
            "group_count_by_size": group_hist.tolist(),
            "event_count_by_group_size": event_hist.tolist(),
            "user_count_by_max_group_size": user_max_hist.tolist(),
            "feedback_count_by_group_size": {
                name: feedback_hist[:, index].tolist()
                for index, name in enumerate(EVENT_NAMES)
            },
            "strict_like_to_unlike_count_by_max_endpoint_group_size": (
                like_pair_endpoint_max_hist.tolist()
            ),
            "strict_dislike_to_undislike_count_by_max_endpoint_group_size": (
                dislike_pair_endpoint_max_hist.tolist()
            ),
        },
        "interpretation_boundary": {
            "D_all": "all groups retained",
            "D_human_like": "not yet defined; threshold requires user confirmation",
            "batch_event_flag": "not yet materialized; this audit provides thresholds",
            "strict_revision_effect": (
                "counts existing strict pairs touched at either endpoint; after a "
                "threshold is selected, Phase 1 must recompute strict chains"
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.output, output)
    atomic_json(
        args.progress,
        {
            "status": "complete",
            "users": users,
            "events": events,
            "elapsed_seconds": output["elapsed_seconds"],
            "output": str(args.output.resolve()),
        },
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "counts": output["counts"],
                "group_size_quantiles": quantiles,
                "threshold_effects": threshold_effects,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
