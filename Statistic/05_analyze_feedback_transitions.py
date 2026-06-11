#!/usr/bin/env python3
"""Analyze explicit feedback state transitions in Yambda.

The transition unit is an adjacent explicit feedback change for the same
user-item pair after sorting each user's event stream by timestamp.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
from tqdm import tqdm


STAT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STAT_ROOT.parent

FEEDBACK_TYPES = ("like", "dislike", "unlike", "undislike")
FEEDBACK_SET = set(FEEDBACK_TYPES)

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
    parser = argparse.ArgumentParser(description="Analyze explicit feedback transitions")
    parser.add_argument(
        "--multi_event_parquet",
        default=str(PROJECT_ROOT.parent / "0330Yambda/data/sequential-50m/multi_event.parquet"),
    )
    parser.add_argument("--timestamp_unit_seconds", type=float, default=5.0)
    parser.add_argument("--max_users", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--out_json",
        default=str(STAT_ROOT / "feedback_transition_full.json"),
    )
    return parser.parse_args()


def iter_user_rows(parquet_path: Path, batch_size: int) -> Iterable[dict]:
    pf = pq.ParquetFile(parquet_path)
    columns = ["uid", "timestamp", "item_id", "event_type"]
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        pyd = batch.to_pydict()
        for idx in range(len(pyd["uid"])):
            yield {key: pyd[key][idx] for key in columns}


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


def analyze(args: argparse.Namespace) -> dict:
    users_processed = 0
    total_events = 0
    explicit_events = 0
    explicit_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    transition_gaps: dict[str, list[float]] = defaultdict(list)
    transition_gap_bins: dict[str, Counter[str]] = defaultdict(Counter)
    repeated_same_counts: Counter[str] = Counter()
    user_item_pairs_with_feedback = 0
    user_item_pairs_with_transition = 0

    rows = iter_user_rows(Path(args.multi_event_parquet), args.batch_size)
    pbar = None if args.quiet else tqdm(rows, desc="[feedback-transitions]", unit="users", ncols=90)
    iterable = rows if args.quiet else pbar
    for row in iterable:
        if args.max_users and users_processed >= args.max_users:
            break
        users_processed += 1
        n_events = len(row["timestamp"])
        total_events += n_events

        last_by_item: dict[int, tuple[str, int]] = {}
        seen_items: set[int] = set()
        transition_items: set[int] = set()
        explicit_positions = [
            idx for idx in range(n_events)
            if str(row["event_type"][idx]) in FEEDBACK_SET
        ]
        order = sorted(explicit_positions, key=lambda idx: (int(row["timestamp"][idx]), idx))

        for pos in order:
            event_type = str(row["event_type"][pos])
            explicit_events += 1
            explicit_counts[event_type] += 1
            timestamp = int(row["timestamp"][pos])
            item_id = int(row["item_id"][pos])
            seen_items.add(item_id)

            prior = last_by_item.get(item_id)
            if prior is not None:
                prev_type, prev_ts = prior
                source_counts[prev_type] += 1
                gap = (timestamp - prev_ts) * float(args.timestamp_unit_seconds)
                if gap >= 0:
                    key = f"{prev_type}_to_{event_type}"
                    transition_counts[key] += 1
                    transition_gaps[key].append(float(gap))
                    transition_gap_bins[key][binned_gap(gap)] += 1
                    transition_items.add(item_id)
                    if prev_type == event_type:
                        repeated_same_counts[event_type] += 1

            last_by_item[item_id] = (event_type, timestamp)

        user_item_pairs_with_feedback += len(seen_items)
        user_item_pairs_with_transition += len(transition_items)
        if pbar is not None and users_processed % 200 == 0:
            pbar.set_postfix_str(
                f"events={total_events:,} explicit={explicit_events:,} transitions={sum(transition_counts.values()):,}",
                refresh=False,
            )
    if pbar is not None:
        pbar.close()

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
        gaps = transition_gaps[key]
        bins = transition_gap_bins[key]
        transition_summary[key] = {
            "count": int(count),
            "share_of_all_transitions": float(count / max(sum(transition_counts.values()), 1)),
            "gap_seconds": summarize(gaps),
            "gap_bins": dict(bins),
            "gap_bin_share": {bin_key: float(bin_value / max(count, 1)) for bin_key, bin_value in bins.items()},
        }

    return {
        "args": vars(args),
        "users_processed": int(users_processed),
        "total_events": int(total_events),
        "explicit_events": int(explicit_events),
        "explicit_counts": dict(explicit_counts),
        "user_item_pairs_with_feedback": int(user_item_pairs_with_feedback),
        "user_item_pairs_with_transition": int(user_item_pairs_with_transition),
        "total_adjacent_feedback_transitions": int(sum(transition_counts.values())),
        "transition_matrix_counts": matrix,
        "transition_matrix_row_share": row_share,
        "transition_summary": transition_summary,
        "repeated_same_counts": dict(repeated_same_counts),
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
