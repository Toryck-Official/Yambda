#!/usr/bin/env python3
"""Train-only audit of positive gaps between unique D_SID timestamp groups."""

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
OUT = ROOT / "phase2_timestamp_gate" / "artifacts" / "train_time_scale_audit.json"
MARKS = ("like", "dislike", "unlike", "undislike")
MAX_GAPS = 77_565_283


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dall", type=Path, default=DALL)
    p.add_argument("--support", type=Path, default=SUPPORT)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--progress-users", type=int, default=50_000)
    return p.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = parse_args(); started = time.time()
    manifest = json.loads(args.dall.read_text())
    cutoff = int(manifest["global_split"]["train_cutoff_timestamp_inclusive"])
    with np.load(args.support, allow_pickle=False) as z:
        items = z["item_id"]; real = z["real_embedding"]
    lookup = np.zeros(int(items.max()) + 1, bool); lookup[items] = real
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
    gaps = np.empty(MAX_GAPS, dtype=np.uint32); cursor = users = train_groups = zero = 0
    while True:
        active = [s for s in streams if s["row"] < len(s["uid"])]
        if not active:
            break
        uid = min(int(s["uid"][s["row"]]) for s in active); times = []
        for stream in streams:
            row = stream["row"]
            if row >= len(stream["uid"]) or int(stream["uid"][row]) != uid:
                continue
            a, b = int(stream["offsets"][row]), int(stream["offsets"][row + 1])
            ts = np.asarray(stream["timestamp"][a:b], dtype=np.uint32)
            it = np.asarray(stream["item"][a:b], dtype=np.uint32)
            keep = lookup[it] & (ts <= cutoff)
            if np.any(keep):
                times.append(ts[keep])
            stream["row"] += 1
        if not times:
            continue
        unique = np.unique(np.concatenate(times)); users += 1; train_groups += len(unique)
        delta = np.diff(unique.astype(np.int64))
        zero += int(np.count_nonzero(delta == 0))
        positive = delta[delta > 0].astype(np.uint32)
        gaps[cursor:cursor + len(positive)] = positive; cursor += len(positive)
        if users % args.progress_users == 0:
            print({"users": users, "train_groups": train_groups, "positive_gaps": cursor, "elapsed_seconds": round(time.time() - started, 1)}, flush=True)
    gaps = gaps[:cursor]
    if not len(gaps) or zero:
        raise RuntimeError("group-level gaps missing or contain zero")
    quantiles = {}
    for name, q in (("p50", .5), ("p75", .75), ("p90", .9), ("p95", .95), ("p99", .99), ("p99_9", .999)):
        quantiles[name] = int(np.quantile(gaps, q, method="higher"))
    histogram = np.bincount(gaps.astype(np.int64))
    mode = int(np.argmax(histogram[1:]) + 1)
    gcd = int(np.gcd.reduce(gaps.astype(np.uint64)))
    timestamp_values_quantized = gcd > 1 and int(np.count_nonzero(gaps % gcd)) == 0
    # Freeze an interpretable horizon using train only: p99 rounded upward to a
    # whole day.  P99.9 is retained as a long-tail diagnostic rather than
    # forcing an impractical roughly 99-day first implementation horizon.
    day = 86_400
    horizon = int(math.ceil(quantiles["p99"] / day) * day)
    horizon_p95 = int(math.ceil(quantiles["p95"] / day) * day)
    horizon_p99_9 = int(math.ceil(quantiles["p99_9"] / day) * day)
    model_unit = 3_600
    report = {
        "status": "complete_train_only_time_scale_audit",
        "data_contract": {"dataset": "D_SID", "split": "train only", "cutoff_inclusive": cutoff, "validation_or_test_used_for_decision": False, "unique_timestamp_groups": True},
        "counts": {"users_with_train_groups": users, "train_groups": train_groups, "positive_adjacent_group_gaps": len(gaps), "zero_group_gaps": zero},
        "raw_timestamp_gap": {
            "min": int(gaps.min()), "mode": mode, "mode_count": int(histogram[mode]),
            "gcd": gcd, "all_positive_gaps_divisible_by_gcd": timestamp_values_quantized,
            **quantiles, "max": int(gaps.max()),
        },
        "official_evidence": {
            "dataset_card": "https://huggingface.co/datasets/yandex/yambda/blob/main/README.md",
            "field_description": "timestamp is a delta-time field on a 5-second quantized grid; local values and train-only gap GCD independently agree",
            "raw_value_interpretation": "seconds from anonymized origin, quantized to multiples of 5 seconds",
        },
        "frozen_train_only_decision": {
            "raw_timestamp_unit_seconds": 1,
            "timestamp_quantum_seconds": gcd,
            "model_time_unit_seconds": model_unit,
            "model_time_unit_name": "hour",
            "normalization": "model_delta = raw_timestamp_delta / 3600",
            "prediction_horizon_seconds": horizon,
            "prediction_horizon_hours": horizon / model_unit,
            "selection": "train positive group-gap p99 rounded upward to the next whole 24-hour boundary",
            "train_gap_mass_covered": float(np.mean(gaps <= horizon)),
            "candidate_horizons": {
                "p95_rounded_day": {"seconds": horizon_p95, "mass_covered": float(np.mean(gaps <= horizon_p95))},
                "p99_rounded_day_selected": {"seconds": horizon, "mass_covered": float(np.mean(gaps <= horizon))},
                "p99_9_rounded_day_long_tail_diagnostic": {"seconds": horizon_p99_9, "mass_covered": float(np.mean(gaps <= horizon_p99_9))},
            },
            "selection_reason": "p99 preserves a fixed high train-only event mass while avoiding the approximately 99-day p99.9 tail as the first prediction grid",
            "train_p99_seconds": quantiles["p99"],
            "train_p99_9_seconds": quantiles["p99_9"],
        },
        "elapsed_seconds": time.time() - started,
        "boundaries": {"minimal_snmpp_training_started": False, "validation_test_inspected_for_horizon": False},
    }
    atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
