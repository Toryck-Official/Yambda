#!/usr/bin/env python3
"""Audit bias induced by retaining only real-audio items in deduplicated D_all."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DALL = ROOT / "phase1_canonical" / "artifacts" / "dall_manifest.json"
SUPPORT = ROOT / "phase1_gates" / "artifacts" / "metadata_loo_support.npz"
OUTPUT = ROOT / "sid_feasibility_gate" / "artifacts" / "real_audio_subset_bias.json"
PROGRESS = ROOT / "sid_feasibility_gate" / "artifacts" / "real_audio_subset_bias.progress.json"
MARKS = ("like", "dislike", "unlike", "undislike")
MAX_TIMESTAMP_GROUP = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dall", type=Path, default=DALL)
    parser.add_argument("--support", type=Path, default=SUPPORT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--progress", type=Path, default=PROGRESS)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10_000)
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def describe(values: np.ndarray, include_zero: bool = True) -> dict:
    array = values if include_zero else values[values > 0]
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "min": int(array.min()),
        "mean": float(array.mean()),
        "median": int(np.quantile(array, .5, method="higher")),
        "p75": int(np.quantile(array, .75, method="higher")),
        "p90": int(np.quantile(array, .9, method="higher")),
        "p95": int(np.quantile(array, .95, method="higher")),
        "p99": int(np.quantile(array, .99, method="higher")),
        "max": int(array.max()),
        "quantile_rule": "numpy higher empirical quantile",
    }


def describe_histogram(histogram: np.ndarray) -> dict:
    total = int(histogram.sum(dtype=np.uint64))
    result = {"count": total}
    cumulative = np.cumsum(histogram, dtype=np.uint64)
    nonzero = np.flatnonzero(histogram)
    result["min"] = int(nonzero[0])
    result["mean"] = float(np.dot(np.arange(len(histogram), dtype=np.float64), histogram) / total)
    for name, q in (("median", .5), ("p75", .75), ("p90", .9), ("p95", .95), ("p99", .99), ("p99_9", .999)):
        result[name] = int(np.searchsorted(cumulative, math.ceil(q * total), side="left"))
    result["max"] = int(nonzero[-1])
    result["quantile_rule"] = "nearest observed histogram value at ceil(q*n)"
    return result


def main() -> None:
    args = parse_args()
    started = time.time()
    manifest = json.loads(args.dall.read_text())
    with np.load(args.support, allow_pickle=False) as support:
        explicit_item = support["item_id"]
        real_flag = support["real_embedding"]
    real_lookup = np.zeros(int(explicit_item.max()) + 1, dtype=bool)
    real_lookup[explicit_item] = real_flag
    streams = []
    for mark in MARKS:
        view = manifest["split_views"][mark]
        streams.append({
            "uid": np.load(view["uid"], mmap_mode="r"),
            "offsets": np.load(view["offsets"], mmap_mode="r"),
            "timestamp": np.load(view["shared_timestamp"], mmap_mode="r"),
            "item": np.load(view["shared_item_id"], mmap_mode="r"),
            "row": 0,
        })
    # Yambda user IDs are one-based and the observed maximum is 1,000,000.
    sequence_all = np.zeros(1_000_001, dtype=np.uint32)
    sequence_real = np.zeros(1_000_001, dtype=np.uint32)
    feedback_all = np.zeros(4, dtype=np.uint64)
    feedback_real = np.zeros(4, dtype=np.uint64)
    group_hist_all = np.zeros(MAX_TIMESTAMP_GROUP + 1, dtype=np.uint64)
    group_hist_real = np.zeros(MAX_TIMESTAMP_GROUP + 1, dtype=np.uint64)
    group_change = {
        "groups_original": 0, "groups_retained": 0, "groups_removed_entirely": 0,
        "groups_unchanged": 0, "groups_shrunk": 0,
        "original_multi_groups": 0, "original_multi_to_singleton": 0,
        "events_in_original_multi_groups": 0, "events_in_real_multi_groups": 0,
    }
    strict = {
        "like_to_unlike": {"all": 0, "real_item": 0},
        "dislike_to_undislike": {"all": 0, "real_item": 0},
        "ambiguous_same_item_timestamp_groups": 0,
    }
    users = 0
    while True:
        active = [stream for stream in streams if stream["row"] < len(stream["uid"])]
        if not active or (args.max_users and users >= args.max_users):
            break
        uid = min(int(stream["uid"][stream["row"]]) for stream in active)
        timestamp_parts = []
        item_parts = []
        type_parts = []
        for mark, stream in enumerate(streams):
            row = stream["row"]
            if row >= len(stream["uid"]) or int(stream["uid"][row]) != uid:
                continue
            start, end = int(stream["offsets"][row]), int(stream["offsets"][row + 1])
            timestamp_parts.append(np.asarray(stream["timestamp"][start:end], dtype=np.uint32))
            item_parts.append(np.asarray(stream["item"][start:end], dtype=np.uint32))
            type_parts.append(np.full(end - start, mark, dtype=np.uint8))
            stream["row"] += 1
        timestamp = np.concatenate(timestamp_parts)
        item = np.concatenate(item_parts)
        event_type = np.concatenate(type_parts)
        order = np.lexsort((event_type, item, timestamp))
        timestamp, item, event_type = timestamp[order], item[order], event_type[order]
        real = real_lookup[item]
        count = len(item)
        sequence_all[uid] = count
        sequence_real[uid] = int(real.sum())
        feedback_all += np.bincount(event_type, minlength=4).astype(np.uint64)
        feedback_real += np.bincount(event_type[real], minlength=4).astype(np.uint64)

        group_start_flag = np.empty(count, dtype=bool)
        group_start_flag[0] = True
        group_start_flag[1:] = timestamp[1:] != timestamp[:-1]
        group_starts = np.flatnonzero(group_start_flag)
        group_ends = np.r_[group_starts[1:], count]
        group_sizes = group_ends - group_starts
        if int(group_sizes.max()) > MAX_TIMESTAMP_GROUP:
            raise RuntimeError(f"timestamp group exceeds {MAX_TIMESTAMP_GROUP}")
        real_group_sizes = np.add.reduceat(real.astype(np.uint32), group_starts)
        group_hist_all += np.bincount(group_sizes, minlength=len(group_hist_all)).astype(np.uint64)
        positive_real_group = real_group_sizes > 0
        if positive_real_group.any():
            group_hist_real += np.bincount(
                real_group_sizes[positive_real_group], minlength=len(group_hist_real)
            ).astype(np.uint64)
        group_change["groups_original"] += len(group_sizes)
        group_change["groups_retained"] += int(positive_real_group.sum())
        group_change["groups_removed_entirely"] += int((~positive_real_group).sum())
        group_change["groups_unchanged"] += int((real_group_sizes == group_sizes).sum())
        group_change["groups_shrunk"] += int(((real_group_sizes > 0) & (real_group_sizes < group_sizes)).sum())
        original_multi = group_sizes > 1
        group_change["original_multi_groups"] += int(original_multi.sum())
        group_change["original_multi_to_singleton"] += int((original_multi & (real_group_sizes == 1)).sum())
        group_change["events_in_original_multi_groups"] += int(group_sizes[original_multi].sum())
        group_change["events_in_real_multi_groups"] += int(real_group_sizes[real_group_sizes > 1].sum())

        pair_start_flag = np.empty(count, dtype=bool)
        pair_start_flag[0] = True
        pair_start_flag[1:] = (timestamp[1:] != timestamp[:-1]) | (item[1:] != item[:-1])
        pair_starts = np.flatnonzero(pair_start_flag)
        pair_ends = np.r_[pair_starts[1:], count]
        active_like: dict[int, int] = {}
        active_dislike: dict[int, int] = {}
        for start, end in zip(pair_starts.tolist(), pair_ends.tolist(), strict=True):
            first = int(event_type[start])
            if first != int(event_type[end - 1]):
                strict["ambiguous_same_item_timestamp_groups"] += 1
                continue
            current_item = int(item[start])
            current_time = int(timestamp[start])
            if first == 0:
                active_like.setdefault(current_item, current_time)
            elif first == 1:
                active_dislike.setdefault(current_item, current_time)
            elif first == 2:
                origin = active_like.pop(current_item, None)
                if origin is not None:
                    if origin >= current_time:
                        raise RuntimeError("non-positive like->unlike delay")
                    strict["like_to_unlike"]["all"] += 1
                    strict["like_to_unlike"]["real_item"] += int(real_lookup[current_item])
            else:
                origin = active_dislike.pop(current_item, None)
                if origin is not None:
                    if origin >= current_time:
                        raise RuntimeError("non-positive dislike->undislike delay")
                    strict["dislike_to_undislike"]["all"] += 1
                    strict["dislike_to_undislike"]["real_item"] += int(real_lookup[current_item])
        users += 1
        if users % args.progress_every == 0:
            progress = {"status": "running", "users": users, "last_uid": uid, "elapsed_seconds": time.time() - started}
            atomic_json(args.progress, progress)
            print(progress, flush=True)

    all_users = sequence_all > 0
    real_users = sequence_real > 0
    total_all = int(feedback_all.sum())
    total_real = int(feedback_real.sum())
    for name in ("like_to_unlike", "dislike_to_undislike"):
        row = strict[name]
        row["retention_rate"] = row["real_item"] / row["all"]
    group_change["group_retention_rate"] = group_change["groups_retained"] / group_change["groups_original"]
    group_change["removed_group_rate"] = group_change["groups_removed_entirely"] / group_change["groups_original"]
    group_change["shrunken_group_rate"] = group_change["groups_shrunk"] / group_change["groups_original"]
    report = {
        "status": "complete_real_audio_only_bias_audit" if not args.max_users else "partial_smoke",
        "data_contract": {
            "source": str(args.dall.resolve()),
            "dataset": "deduplicated D_all",
            "filter": "retain event iff item has a real source audio embedding",
            "listen_used": False,
            "same_timestamp_policy": "shared group; no invented order",
            "strict_revision_policy": "same user/item; active prerequisite at strictly earlier timestamp; ambiguous same-item same-time feedback groups do not update state",
            "events_deleted_or_output_dataset_written": False,
        },
        "events": {
            "all": total_all,
            "real_audio_only": total_real,
            "retention_rate": total_real / total_all,
            "by_feedback": {
                mark: {
                    "all": int(feedback_all[index]),
                    "real_audio_only": int(feedback_real[index]),
                    "retention_rate": float(feedback_real[index] / feedback_all[index]),
                }
                for index, mark in enumerate(MARKS)
            },
        },
        "strict_revision_pairs": strict,
        "user_sequences": {
            "all_users": int(all_users.sum()),
            "retained_users": int(real_users.sum()),
            "users_lost_all_events": int((all_users & ~real_users).sum()),
            "user_retention_rate": float(real_users.sum() / all_users.sum()),
            "length_before_all_users": describe(sequence_all[all_users]),
            "length_after_same_original_users_including_zero": describe(sequence_real[all_users]),
            "length_after_retained_users": describe(sequence_real[real_users]),
            "events_removed_per_original_user": describe((sequence_all - sequence_real)[all_users]),
            "users_with_unchanged_length": int(np.count_nonzero(all_users & (sequence_all == sequence_real))),
        },
        "same_timestamp_groups": {
            "before": {
                "groups": int(group_hist_all.sum()),
                "group_size": describe_histogram(group_hist_all),
                "events_in_multi_groups": group_change["events_in_original_multi_groups"],
                "events_in_multi_groups_fraction": group_change["events_in_original_multi_groups"] / total_all,
            },
            "after_real_audio_only": {
                "groups": int(group_hist_real.sum()),
                "group_size": describe_histogram(group_hist_real),
                "events_in_multi_groups": group_change["events_in_real_multi_groups"],
                "events_in_multi_groups_fraction": group_change["events_in_real_multi_groups"] / total_real,
            },
            "changes": group_change,
        },
        "elapsed_seconds": time.time() - started,
        "boundaries": {"gate2_started": False, "snmpp_or_hpn_trained": False},
    }
    if not args.max_users:
        if total_all != manifest["dedup"]["events_after"]:
            raise RuntimeError("D_all event conservation failed")
        expected = [manifest["dedup"]["by_feedback"][mark]["events_after_exact_dedup"] for mark in MARKS]
        if expected != feedback_all.tolist():
            raise RuntimeError("per-feedback D_all conservation failed")
    atomic_json(args.output, report)
    atomic_json(args.progress, {"status": "complete", "users": users, "elapsed_seconds": time.time() - started})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
