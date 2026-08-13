#!/usr/bin/env python3
"""Audit train-period collaborative evidence without selecting a split cutoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "phase1_gate1c" / "configs" / "audit.json"
DEFAULT_OUTPUT = ROOT / "phase1_gate1c" / "artifacts" / "collaborative_coverage_curve.json"
DEFAULT_WORK = ROOT / "phase1_gate1c" / "work"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    return p.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def distribution(values: np.ndarray) -> dict:
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": int(values.min()),
        "mean": float(values.mean()),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": int(values.max()),
    }


def threshold_counts(values: np.ndarray, thresholds: list[int]) -> dict:
    return {f"at_least_{threshold}": int(np.count_nonzero(values >= threshold)) for threshold in thresholds}


def timestamp_histogram(records: dict, resolution: int) -> tuple[np.ndarray, dict]:
    maximum = max(int(record["timestamp_max"]) for record in records.values())
    hist = np.zeros(maximum // resolution + 1, dtype=np.uint64)
    by_mark = {}
    for mark, record in records.items():
        timestamps = np.load(record["output_files"]["timestamp.npy"], mmap_mode="r")
        mark_hist = np.zeros_like(hist)
        for start in range(0, len(timestamps), 5_000_000):
            chunk = np.asarray(timestamps[start : start + 5_000_000], dtype=np.uint32)
            mark_hist += np.bincount(chunk // resolution, minlength=len(hist)).astype(np.uint64)
        hist += mark_hist
        by_mark[mark] = mark_hist
    return hist, by_mark


def candidate_cutoffs(hist: np.ndarray, fractions: list[float], resolution: int) -> list[dict]:
    cumulative = np.cumsum(hist, dtype=np.uint64)
    total = int(cumulative[-1])
    rows = []
    for fraction in fractions:
        target = int(np.ceil(total * fraction))
        index = int(np.searchsorted(cumulative, target, side="left"))
        train_events = int(cumulative[index])
        rows.append({
            "candidate_fraction": float(fraction),
            "cutoff_timestamp": index * resolution,
            "train_events_including_full_cutoff_timestamp": train_events,
            "actual_train_event_fraction": train_events / total,
            "same_timestamp_not_split": True,
        })
    return rows


def expand_missing_events(
    records: dict,
    missing_lookup: np.ndarray,
    expected: int,
    work_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    work_dir.mkdir(parents=True, exist_ok=True)
    item_out = np.lib.format.open_memmap(work_dir / "missing_item_position.uint32.npy", mode="w+", dtype=np.uint32, shape=(expected,))
    uid_out = np.lib.format.open_memmap(work_dir / "missing_uid.uint32.npy", mode="w+", dtype=np.uint32, shape=(expected,))
    time_out = np.lib.format.open_memmap(work_dir / "missing_timestamp.uint32.npy", mode="w+", dtype=np.uint32, shape=(expected,))
    cursor = 0
    by_mark = {}
    for mark, record in records.items():
        files = record["output_files"]
        users = np.load(files["uid.npy"], mmap_mode="r")
        offsets = np.load(files["offsets.npy"], mmap_mode="r")
        items = np.load(files["item_id.npy"], mmap_mode="r")
        timestamps = np.load(files["timestamp.npy"], mmap_mode="r")
        mark_start = cursor
        for user_start in range(0, len(users), 4096):
            user_end = min(user_start + 4096, len(users))
            event_start = int(offsets[user_start])
            event_end = int(offsets[user_end])
            chunk_items = np.asarray(items[event_start:event_end], dtype=np.uint32)
            position = missing_lookup[chunk_items].astype(np.int64) - 1
            keep = position >= 0
            count = int(keep.sum())
            if not count:
                continue
            lengths = np.diff(np.asarray(offsets[user_start : user_end + 1], dtype=np.uint64)).astype(np.int64)
            expanded_uid = np.repeat(np.asarray(users[user_start:user_end], dtype=np.uint32), lengths)
            item_out[cursor : cursor + count] = position[keep].astype(np.uint32)
            uid_out[cursor : cursor + count] = expanded_uid[keep]
            time_out[cursor : cursor + count] = timestamps[event_start:event_end][keep]
            cursor += count
        by_mark[mark] = cursor - mark_start
        print(f"expanded {mark}: {by_mark[mark]:,} missing-item events", flush=True)
    if cursor != expected:
        raise RuntimeError(f"missing event conservation failed: {cursor} != {expected}")
    item_out.flush(); uid_out.flush(); time_out.flush()
    return item_out, uid_out, time_out, by_mark


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    paths = {key: Path(value) for key, value in config["paths"].items()}
    manifest = json.loads(paths["flat_manifest"].read_text())
    records = manifest["records"]
    fractions = [float(x) for x in config["candidate_global_train_event_fractions"]]
    resolution = int(config["timestamp_resolution_seconds"])
    print("[1/5] global timestamp histogram", flush=True)
    hist, mark_hist = timestamp_histogram(records, resolution)
    cutoffs = candidate_cutoffs(hist, fractions, resolution)
    cutoff_values = np.array([row["cutoff_timestamp"] for row in cutoffs], dtype=np.uint32)

    with np.load(paths["item_frequency"], allow_pickle=False) as z:
        explicit_item = z["item_id"].astype(np.uint32, copy=False)
        full_event_count = z["total_count"].astype(np.uint32, copy=False)
    orig2dense = np.load(paths["orig2dense"], mmap_mode="r")
    real = np.asarray(orig2dense[explicit_item] > 0)
    missing = ~real
    explicit_lookup = np.zeros(len(orig2dense), dtype=np.int32)
    explicit_lookup[explicit_item] = np.arange(1, len(explicit_item) + 1, dtype=np.int32)
    missing_item = explicit_item[missing]
    missing_lookup = np.zeros(len(orig2dense), dtype=np.int32)
    missing_lookup[missing_item] = np.arange(1, len(missing_item) + 1, dtype=np.int32)
    expected_missing_events = int(full_event_count[missing].sum(dtype=np.uint64))

    # One event is assigned to exactly one time interval, then cumulative counts
    # produce every candidate cutoff without rescanning an event four times.
    print("[2/5] item-period and mark-period counts", flush=True)
    periods = len(cutoff_values) + 1
    item_period = np.zeros((periods, len(explicit_item)), dtype=np.uint32)
    mark_period: dict[str, list[int]] = {}
    for mark, record in records.items():
        items = np.load(record["output_files"]["item_id.npy"], mmap_mode="r")
        timestamps = np.load(record["output_files"]["timestamp.npy"], mmap_mode="r")
        mark_counts = np.zeros(periods, dtype=np.uint64)
        for start in range(0, len(items), 2_000_000):
            end = min(start + 2_000_000, len(items))
            chunk_items = np.asarray(items[start:end], dtype=np.uint32)
            position = explicit_lookup[chunk_items].astype(np.int64) - 1
            if np.any(position < 0):
                raise RuntimeError("explicit event item missing from explicit universe")
            period = np.searchsorted(cutoff_values, np.asarray(timestamps[start:end], dtype=np.uint32), side="left")
            mark_counts += np.bincount(period, minlength=periods).astype(np.uint64)
            for value in range(periods):
                mask = period == value
                if np.any(mask):
                    np.add.at(item_period[value], position[mask], np.uint32(1))
        mark_period[mark] = [int(x) for x in mark_counts]
    item_cumulative = np.cumsum(item_period, axis=0, dtype=np.uint32)

    print("[3/5] missing-item event triples", flush=True)
    missing_pos, missing_uid, missing_time, missing_by_mark = expand_missing_events(
        records, missing_lookup, expected_missing_events, args.work_dir
    )
    missing_mark_train_counts: list[dict[str, int]] = [dict() for _ in cutoffs]
    missing_mark_cursor = 0
    for mark, count in missing_by_mark.items():
        mark_times = np.asarray(
            missing_time[missing_mark_cursor : missing_mark_cursor + count],
            dtype=np.uint32,
        )
        for cutoff_index, cutoff in enumerate(cutoffs):
            missing_mark_train_counts[cutoff_index][mark] = int(
                np.count_nonzero(mark_times <= cutoff["cutoff_timestamp"])
            )
        missing_mark_cursor += count
    if missing_mark_cursor != expected_missing_events:
        raise RuntimeError("missing feedback slice conservation failed")
    print("[4/5] exact missing item-user first timestamps", flush=True)
    # Sort by item, user, timestamp so the first row of each pair is its earliest
    # observed explicit event. Only missing-item events are sorted (13.4M), not
    # all 136M events.
    order = np.lexsort((np.asarray(missing_time), np.asarray(missing_uid), np.asarray(missing_pos)))
    sorted_item = np.asarray(missing_pos[order], dtype=np.uint32)
    sorted_uid = np.asarray(missing_uid[order], dtype=np.uint32)
    sorted_time = np.asarray(missing_time[order], dtype=np.uint32)
    first = np.ones(len(order), dtype=bool)
    first[1:] = (sorted_item[1:] != sorted_item[:-1]) | (sorted_uid[1:] != sorted_uid[:-1])
    pair_item = sorted_item[first]
    pair_first_time = sorted_time[first]
    first_item_time = np.full(len(missing_item), np.iinfo(np.uint32).max, dtype=np.uint32)
    np.minimum.at(first_item_time, np.asarray(missing_pos), np.asarray(missing_time))

    print("[5/5] candidate coverage reports", flush=True)
    curves = []
    event_thresholds = [int(x) for x in config["event_count_thresholds"]]
    user_thresholds = [int(x) for x in config["unique_user_thresholds"]]
    for index, cutoff in enumerate(cutoffs):
        train_count = item_cumulative[index]
        real_count = train_count[real]
        missing_count = train_count[missing]
        unique_users = np.bincount(
            pair_item[pair_first_time <= cutoff["cutoff_timestamp"]],
            minlength=len(missing_item),
        ).astype(np.uint32)
        missing_evidence = missing_count > 0
        real_evidence = real_count > 0
        feedback_counts = {
            mark: int(sum(period_counts[: index + 1]))
            for mark, period_counts in mark_period.items()
        }
        curves.append({
            **cutoff,
            "train_feedback_counts": feedback_counts,
            "real_audio_items": {
                "total": int(real.sum()),
                "with_train_collaborative_evidence": int(real_evidence.sum()),
                "without_train_collaborative_evidence": int((~real_evidence).sum()),
                "train_event_count_distribution_among_evidence": distribution(real_count[real_evidence]),
                "event_threshold_counts": threshold_counts(real_count, event_thresholds),
            },
            "missing_audio_items": {
                "total": int(missing.sum()),
                "with_train_collaborative_evidence": int(missing_evidence.sum()),
                "without_train_collaborative_evidence": int((~missing_evidence).sum()),
                "first_explicit_event_after_train_cutoff": int(np.count_nonzero(first_item_time > cutoff["cutoff_timestamp"])),
                "train_events": int(missing_count.sum(dtype=np.uint64)),
                "train_feedback_counts": missing_mark_train_counts[index],
                "train_event_count_distribution_among_evidence": distribution(missing_count[missing_evidence]),
                "event_threshold_counts": threshold_counts(missing_count, event_thresholds),
                "train_unique_user_count_distribution_among_evidence": distribution(unique_users[unique_users > 0]),
                "unique_user_threshold_counts": threshold_counts(unique_users, user_thresholds),
                "events_but_only_one_unique_train_user": int(np.count_nonzero(missing_evidence & (unique_users == 1))),
                "event_and_unique_user_conservation": bool(np.array_equal(missing_evidence, unique_users > 0)),
            },
        })
    report = {
        "status": "complete_gate1c_read_only_candidate_cutoff_coverage_audit",
        "data_contract": {
            "source": str(paths["flat_manifest"]),
            "explicit_only": True,
            "listen_used": False,
            "raw_events_before_phase1_dedup_or_burst_decision": True,
            "global_chronological": True,
            "same_timestamp_not_split": True,
            "candidate_fractions_are_not_approved_protocol": True,
            "collaborative_model_trained": False,
            "gate2_started": False,
        },
        "counts": {
            "explicit_events": int(hist.sum(dtype=np.uint64)),
            "explicit_items": int(len(explicit_item)),
            "real_audio_items": int(real.sum()),
            "missing_audio_items": int(missing.sum()),
            "missing_audio_events": expected_missing_events,
            "missing_item_user_pairs_full_period": int(len(pair_item)),
            "missing_events_by_feedback": missing_by_mark,
        },
        "candidate_curves": curves,
        "interpretation_boundary": {
            "formal_train_cutoff_confirmed": False,
            "why_no_training": "global chronological split ratio/cutoff and Phase 1 duplicate/burst cleaning policy remain unconfirmed",
            "future_graph_edges_used": False,
            "recommended_metrics_if_approved": ["Cosine", "MSE", "PrefixAcc@1", "PrefixAcc@2", "PrefixAcc@3", "PrefixAcc@4", "NN@10", "NN@50"],
            "primary_semantic_metrics": ["PrefixAcc@1", "PrefixAcc@2", "NN@10", "NN@50"],
        },
    }
    atomic_json(args.output, report)
    print(json.dumps({"counts": report["counts"], "candidate_curves": curves, "output": str(args.output)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
