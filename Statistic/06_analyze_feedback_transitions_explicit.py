#!/usr/bin/env python3
"""Analyze explicit feedback transitions from Yambda feedback parquet files."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


STAT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STAT_ROOT.parent
DATA_ROOT = PROJECT_ROOT.parent / "0330Yambda/data/sequential-50m"

FEEDBACK_FILES = {
    "like": "likes.parquet",
    "dislike": "dislikes.parquet",
    "unlike": "unlikes.parquet",
    "undislike": "undislikes.parquet",
}
FEEDBACK_TYPES = ("like", "dislike", "unlike", "undislike")

GAP_BINS = [
    (0, 10, "<=10s"),
    (10, 60, "10s-1m"),
    (60, 300, "1m-5m"),
    (300, 3600, "5m-1h"),
    (3600, 86400, "1h-1d"),
    (86400, 604800, "1d-7d"),
    (604800, math.inf, ">7d"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze explicit Yambda feedback transitions")
    parser.add_argument("--data_root", default=str(DATA_ROOT))
    parser.add_argument("--timestamp_unit_seconds", type=float, default=5.0)
    parser.add_argument(
        "--out_json",
        default=str(STAT_ROOT / "feedback_transition_state_machine.json"),
    )
    return parser.parse_args()


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ["min", "p25", "p50", "p75", "p90", "p95", "p99", "max"]}
    values = sorted(values)
    n = len(values)

    def at(q: float) -> float:
        idx = min(max(int(round(q * (n - 1))), 0), n - 1)
        return float(values[idx])

    return {
        "min": float(values[0]),
        "p25": at(0.25),
        "p50": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": float(values[-1]),
    }


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "std": None, **quantiles(values)}
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) * (x - mean) for x in values) / n
    return {"n": n, "mean": float(mean), "std": float(math.sqrt(var)), **quantiles(values)}


def binned_gap(seconds: float) -> str:
    for lower, upper, label in GAP_BINS:
        if label == "<=10s" and 0 <= seconds <= upper:
            return label
        if seconds > lower and seconds <= upper:
            return label
    return ">7d"


def load_feedback_events(data_root: Path) -> tuple[dict[int, dict[int, list[tuple[int, str]]]], Counter[str]]:
    by_user_item: dict[int, dict[int, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    explicit_counts: Counter[str] = Counter()
    for event_type, filename in FEEDBACK_FILES.items():
        table = pq.read_table(data_root / filename, columns=["uid", "timestamp", "item_id"])
        pyd = table.to_pydict()
        for uid, timestamps, item_ids in zip(pyd["uid"], pyd["timestamp"], pyd["item_id"], strict=True):
            if timestamps is None or item_ids is None:
                continue
            for timestamp, item_id in zip(timestamps, item_ids, strict=True):
                by_user_item[int(uid)][int(item_id)].append((int(timestamp), event_type))
                explicit_counts[event_type] += 1
    return by_user_item, explicit_counts


def analyze(args: argparse.Namespace) -> dict:
    by_user_item, explicit_counts = load_feedback_events(Path(args.data_root))
    transition_counts: Counter[str] = Counter()
    transition_gaps: dict[str, list[float]] = defaultdict(list)
    transition_gap_bins: dict[str, Counter[str]] = defaultdict(Counter)
    user_item_pairs_with_feedback = 0
    user_item_pairs_with_transition = 0
    exact_duplicate_events_removed: Counter[str] = Counter()
    ambiguous_same_timestamp_groups_excluded = 0
    ambiguous_same_timestamp_events_excluded: Counter[str] = Counter()
    invalid_state_events_excluded: Counter[str] = Counter()
    accepted_events: Counter[str] = Counter()

    def is_valid(event_type: str, state: str) -> bool:
        if event_type == "like":
            return state != "liked"
        if event_type == "unlike":
            return state == "liked"
        if event_type == "dislike":
            return state != "disliked"
        if event_type == "undislike":
            return state == "disliked"
        return False

    def next_state(event_type: str) -> str:
        if event_type == "like":
            return "liked"
        if event_type == "dislike":
            return "disliked"
        return "neutral"

    for item_histories in by_user_item.values():
        user_item_pairs_with_feedback += len(item_histories)
        for item_events in item_histories.values():
            if not item_events:
                continue
            by_timestamp: dict[int, Counter[str]] = defaultdict(Counter)
            for timestamp, event_type in item_events:
                by_timestamp[timestamp][event_type] += 1

            cleaned_events: list[tuple[int, str]] = []
            for timestamp in sorted(by_timestamp):
                type_counts = by_timestamp[timestamp]
                for event_type, count in type_counts.items():
                    if count > 1:
                        exact_duplicate_events_removed[event_type] += count - 1
                if len(type_counts) > 1:
                    ambiguous_same_timestamp_groups_excluded += 1
                    for event_type, count in type_counts.items():
                        ambiguous_same_timestamp_events_excluded[event_type] += count
                    continue
                event_type = next(iter(type_counts))
                cleaned_events.append((timestamp, event_type))

            if len(cleaned_events) < 2:
                continue
            state = "neutral"
            last_event_type: str | None = None
            last_timestamp: int | None = None
            item_has_transition = False
            for timestamp, event_type in cleaned_events:
                if not is_valid(event_type, state):
                    invalid_state_events_excluded[event_type] += 1
                    continue
                accepted_events[event_type] += 1
                if last_event_type is not None and last_timestamp is not None:
                    gap = (timestamp - last_timestamp) * float(args.timestamp_unit_seconds)
                    if gap >= 0:
                        key = f"{last_event_type}_to_{event_type}"
                        transition_counts[key] += 1
                        transition_gaps[key].append(float(gap))
                        transition_gap_bins[key][binned_gap(gap)] += 1
                        item_has_transition = True
                state = next_state(event_type)
                last_event_type = event_type
                last_timestamp = timestamp
            if item_has_transition:
                user_item_pairs_with_transition += 1

    total_transitions = sum(transition_counts.values())
    matrix = {
        src: {dst: int(transition_counts.get(f"{src}_to_{dst}", 0)) for dst in FEEDBACK_TYPES}
        for src in FEEDBACK_TYPES
    }
    row_share = {}
    for src in FEEDBACK_TYPES:
        row_total = sum(matrix[src].values())
        row_share[src] = {
            dst: float(matrix[src][dst] / row_total) if row_total else 0.0
            for dst in FEEDBACK_TYPES
        }

    transition_summary = {}
    for key, count in sorted(transition_counts.items(), key=lambda item: (-item[1], item[0])):
        bins = transition_gap_bins[key]
        transition_summary[key] = {
            "count": int(count),
            "share_of_all_transitions": float(count / max(total_transitions, 1)),
            "gap_seconds": summarize(transition_gaps[key]),
            "gap_bins": dict(bins),
            "gap_bin_share": {bin_key: float(bin_value / max(count, 1)) for bin_key, bin_value in bins.items()},
        }

    return {
        "args": vars(args),
        "users_with_explicit_feedback": int(len(by_user_item)),
        "explicit_events": int(sum(explicit_counts.values())),
        "explicit_counts": dict(explicit_counts),
        "accepted_events_after_filter": dict(accepted_events),
        "user_item_pairs_with_feedback": int(user_item_pairs_with_feedback),
        "user_item_pairs_with_transition": int(user_item_pairs_with_transition),
        "total_adjacent_feedback_transitions": int(total_transitions),
        "filter_rules": [
            "Group events by (user_id, item_id).",
            "Within each user-item history, remove exact duplicate events with identical timestamp and event type.",
            "If multiple different explicit feedback types occur at the same timestamp for the same user-item, exclude that timestamp group as ambiguous because event order cannot be recovered.",
            "Apply a single preference state machine with states neutral/liked/disliked.",
            "Accept like only when state is not liked; accept dislike only when state is not disliked.",
            "Accept unlike only when state is liked; accept undislike only when state is disliked.",
            "Count transitions only between adjacent accepted state-machine events.",
        ],
        "filter_diagnostics": {
            "exact_duplicate_events_removed": dict(exact_duplicate_events_removed),
            "ambiguous_same_timestamp_groups_excluded": int(ambiguous_same_timestamp_groups_excluded),
            "ambiguous_same_timestamp_events_excluded": dict(ambiguous_same_timestamp_events_excluded),
            "invalid_state_events_excluded": dict(invalid_state_events_excluded),
        },
        "transition_matrix_counts": matrix,
        "transition_matrix_row_share": row_share,
        "transition_summary": transition_summary,
    }


def main() -> None:
    args = parse_args()
    result = analyze(args)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[done] diagnostics saved to {out_path}")


if __name__ == "__main__":
    main()
