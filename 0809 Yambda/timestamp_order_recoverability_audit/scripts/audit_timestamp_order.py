#!/usr/bin/env python3
"""Timestamp order recoverability audit for the real-audio D_SID view.

This script is audit-only.  It never writes a canonical dataset, changes a
timestamp group, or trains a model.  The four explicit streams remain separate
and are merged only transiently per user.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DALL = ROOT / "phase1_canonical" / "artifacts" / "dall_manifest.json"
SUPPORT = ROOT / "phase1_gates" / "artifacts" / "metadata_loo_support.npz"
RAW = ROOT / "phase0_5b_audit" / "raw" / "sequential" / "5b"
OUT = ROOT / "timestamp_order_recoverability_audit" / "artifacts"
MARKS = ("like", "dislike", "unlike", "undislike")
RAW_FILES = ("likes.parquet", "dislikes.parquet", "unlikes.parquet", "undislikes.parquet")
BIN_LABELS = ("1", "2", "3-5", "6-10", "11-20", "21-50", "51-100", "101-200", "201-500", "501-1000", "1000+")
BIN_UPPER = np.asarray((1, 2, 5, 10, 20, 50, 100, 200, 500, 1000), dtype=np.int64)
FILTER_THRESHOLDS = (20, 50, 100, 200, 500, 1000)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dall", type=Path, default=DALL)
    p.add_argument("--support", type=Path, default=SUPPORT)
    p.add_argument("--raw", type=Path, default=RAW)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--max-users", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=10_000)
    p.add_argument("--sample-user-modulus", type=int, default=211)
    p.add_argument("--provenance-only", action="store_true")
    return p.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha_arrays(uid: np.ndarray, timestamps: list[np.ndarray], items: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(uid, dtype=np.uint32).tobytes())
    for ts, it in zip(timestamps, items, strict=True):
        h.update(np.asarray([len(ts)], dtype=np.uint64).tobytes())
        h.update(np.asarray(ts, dtype=np.uint32).tobytes())
        h.update(np.asarray(it, dtype=np.uint32).tobytes())
    return h.hexdigest()


def read_prefix_signature(path: Path, batch_size: int, limit_users: int = 10_000) -> tuple[str, int, int]:
    parquet = pq.ParquetFile(path)
    out_uid: list[np.ndarray] = []
    out_ts: list[np.ndarray] = []
    out_item: list[np.ndarray] = []
    rows = events = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["uid", "timestamp", "item_id"], use_threads=False):
        uid = batch.column(0).to_numpy(zero_copy_only=False)
        ts_col, item_col = batch.column(1), batch.column(2)
        take = min(len(uid), limit_users - rows)
        if take <= 0:
            break
        out_uid.append(np.asarray(uid[:take], dtype=np.uint32))
        for j in range(take):
            ts = np.asarray(ts_col[j].as_py(), dtype=np.uint32)
            item = np.asarray(item_col[j].as_py(), dtype=np.uint32)
            out_ts.append(ts); out_item.append(item); events += len(ts)
        rows += take
        if rows >= limit_users:
            break
    return sha_arrays(np.concatenate(out_uid), out_ts, out_item), rows, events


def provenance_audit(raw_dir: Path, manifest: dict) -> dict:
    files = {}
    for name in RAW_FILES:
        path = raw_dir / name
        parquet = pq.ParquetFile(path)
        signatures = {}
        for batch in (7, 128, 1000):
            sig, users, events = read_prefix_signature(path, batch)
            signatures[str(batch)] = {"sha256": sig, "users": users, "events": events}
        files[name] = {
            "schema": str(parquet.schema_arrow),
            "columns": parquet.schema_arrow.names,
            "row_groups": int(parquet.metadata.num_row_groups),
            "user_rows": int(parquet.metadata.num_rows),
            "prefix_signatures_by_batch_size": signatures,
            "same_signature_across_batch_sizes": len({x["sha256"] for x in signatures.values()}) == 1,
            "ordering_fields_present": [x for x in parquet.schema_arrow.names if x in {"event_id", "original_row_id", "ingestion_index", "source_sequence", "sequence_id"}],
        }

    # Verify that exact dedup keeps the relative order of surviving rows inside
    # each homogeneous feedback stream.  Cross-stream order cannot be tested,
    # because it is absent from D_all by construction.
    dedup_checks = {}
    flat_manifest = json.loads((ROOT / "phase0_5b_audit" / "work" / "flat_explicit_5b" / "manifest.json").read_text())
    for mark in MARKS:
        source = flat_manifest["records"][mark]["output_files"]
        target = manifest["dedup"]["by_feedback"][mark]["output_files"]
        suid = np.load(source["uid.npy"], mmap_mode="r")
        soff = np.load(source["offsets.npy"], mmap_mode="r")
        sts = np.load(source["timestamp.npy"], mmap_mode="r")
        sit = np.load(source["item_id.npy"], mmap_mode="r")
        toff = np.load(target["offsets"], mmap_mode="r")
        tts = np.load(target["timestamp"], mmap_mode="r")
        tit = np.load(target["item_id"], mmap_mode="r")
        checked = mismatches = 0
        for row in np.linspace(0, len(suid) - 1, num=min(2000, len(suid)), dtype=np.int64):
            a, b = int(soff[row]), int(soff[row + 1])
            key = (np.asarray(sts[a:b], dtype=np.uint64) << np.uint64(24)) | np.asarray(sit[a:b], dtype=np.uint64)
            _, first = np.unique(key, return_index=True)
            keep = np.zeros(b - a, dtype=bool); keep[first] = True
            c, d = int(toff[row]), int(toff[row + 1])
            ok = np.array_equal(sts[a:b][keep], tts[c:d]) and np.array_equal(sit[a:b][keep], tit[c:d])
            mismatches += int(not ok); checked += 1
        dedup_checks[mark] = {"sampled_users": checked, "relative_survivor_order_mismatches": mismatches, "relative_survivor_order_preserved": mismatches == 0}

    return {
        "status": "complete",
        "evidence_hierarchy": {
            "A_semantic_provenance": "No event_id/original_row_id/ingestion/source-sequence field in the four local explicit sequential files. Official documentation promises chronological timestamp order, but does not define a business order inside equal 5-second timestamps.",
            "B_engineering_stability": "Repeated local reads with different Arrow batch sizes reproduce the same list order. The official multi-event builder concatenates fixed event-type files and stable-sorts only by uid,timestamp, so its equal-timestamp order is an engineering type-block order. The local flattening script copies within-stream list values in source order, and exact dedup preserves surviving within-stream order.",
            "critical_non_equivalence": "Engineering stability is not semantic provenance and cannot establish ground-truth chronology.",
        },
        "official_sources": {
            "dataset_card": "https://huggingface.co/datasets/yandex/yambda/blob/main/README.md",
            "transform_script": "https://huggingface.co/datasets/yandex/yambda/blob/main/benchmarks/scripts/transform2sequential.py",
            "multi_event_builder": "https://huggingface.co/datasets/yandex/yambda/blob/main/benchmarks/scripts/make_multievent.py",
            "documented_fact": "files sorted by (uid,timestamp); sequential lists maintain chronological order",
            "missing_fact": "no documented tie-break rule or sub-5-second event sequence for equal timestamps",
            "decisive_engineering_fact": "make_multievent.py concatenates homogeneous files in fixed type-block order [listen, dislike, like, undislike, unlike], then performs a stable sort only by uid,timestamp; equal-timestamp type order is therefore inherited from file concatenation, not demonstrated user chronology",
        },
        "local_files": files,
        "pipeline": {
            "four_feedbacks_are_separate_files": True,
            "cross_feedback_row_order_available_in_current_D_all_or_D_SID": False,
            "official_multi_event_equal_timestamp_order": "fixed input-file/type-block concatenation retained by stable uid,timestamp sort",
            "flattening": "lossless within each separate feedback stream",
            "D_all_merge_rule": "downstream audits merge per user and sort by timestamp,item,feedback; this is a canonical engineering order, not source chronology",
            "exact_dedup": dedup_checks,
            "dedup_effect": "removes duplicate multiplicity; preserves relative order among surviving events within each homogeneous stream; cannot preserve a cross-feedback order that is not present",
        },
        "classification": "B",
        "classification_text": "row order is engineering-stable but has no defensible cross-feedback business chronology for tied timestamps",
        "may_use_as_ground_truth_chronology": False,
    }


def bin_index(size: np.ndarray) -> np.ndarray:
    return np.searchsorted(BIN_UPPER, size, side="left").astype(np.uint8)


def new_bin_stats() -> dict[str, np.ndarray | list]:
    n = len(BIN_LABELS)
    return {
        "groups": np.zeros(n, np.uint64),
        "events": np.zeros(n, np.uint64),
        "unique_items": np.zeros(n, np.uint64),
        "same_item_multifeedback_groups": np.zeros(n, np.uint64),
        "single_feedback_groups": np.zeros(n, np.uint64),
        "feedback": np.zeros((n, 4), np.uint64),
        "same_time_revision_events": np.zeros(n, np.uint64),
        "users": np.zeros(n, np.uint64),
        "prev_gap_count": np.zeros(n, np.uint64),
        "prev_gap_sum": np.zeros(n, np.float64),
        "prev_gap_min": np.full(n, np.iinfo(np.uint32).max, np.uint32),
        "prev_gap_max": np.zeros(n, np.uint32),
        "next_gap_count": np.zeros(n, np.uint64),
        "next_gap_sum": np.zeros(n, np.float64),
        "next_gap_min": np.full(n, np.iinfo(np.uint32).max, np.uint32),
        "next_gap_max": np.zeros(n, np.uint32),
        "prev_samples": [[] for _ in range(n)],
        "next_samples": [[] for _ in range(n)],
    }


def update_gap(stats: dict, bins: np.ndarray, gaps: np.ndarray, which: str, sample: np.ndarray) -> None:
    if not len(gaps):
        return
    count = np.bincount(bins, minlength=len(BIN_LABELS)).astype(np.uint64)
    sums = np.bincount(bins, weights=gaps.astype(np.float64), minlength=len(BIN_LABELS))
    stats[f"{which}_gap_count"] += count
    stats[f"{which}_gap_sum"] += sums
    for b in np.unique(bins):
        vals = gaps[bins == b]
        stats[f"{which}_gap_min"][b] = min(int(stats[f"{which}_gap_min"][b]), int(vals.min()))
        stats[f"{which}_gap_max"][b] = max(int(stats[f"{which}_gap_max"][b]), int(vals.max()))
        sampled = vals[sample[bins == b]]
        if len(sampled):
            stats[f"{which}_samples"][int(b)].append(np.asarray(sampled, dtype=np.uint32))


def update_bins(stats: dict, uid: int, group_t: np.ndarray, sizes: np.ndarray, unique_items: np.ndarray,
                same_multi: np.ndarray, single_feedback: np.ndarray, group_feedback: np.ndarray,
                event_type: np.ndarray, event_group: np.ndarray, same_revision_events_group: np.ndarray,
                select: np.ndarray) -> None:
    idx = np.flatnonzero(select)
    if not len(idx):
        return
    bins_all = bin_index(sizes)
    bins = bins_all[idx]
    stats["groups"] += np.bincount(bins, minlength=len(BIN_LABELS)).astype(np.uint64)
    stats["events"] += np.bincount(bins, weights=sizes[idx], minlength=len(BIN_LABELS)).astype(np.uint64)
    stats["unique_items"] += np.bincount(bins, weights=unique_items[idx], minlength=len(BIN_LABELS)).astype(np.uint64)
    stats["same_item_multifeedback_groups"] += np.bincount(bins, weights=same_multi[idx], minlength=len(BIN_LABELS)).astype(np.uint64)
    stats["single_feedback_groups"] += np.bincount(bins, weights=single_feedback[idx], minlength=len(BIN_LABELS)).astype(np.uint64)
    stats["same_time_revision_events"] += np.bincount(bins, weights=same_revision_events_group[idx], minlength=len(BIN_LABELS)).astype(np.uint64)
    stats["users"][np.unique(bins)] += 1
    event_select = select[event_group]
    eb = bins_all[event_group[event_select]]
    stats["feedback"] += np.bincount(eb * 4 + event_type[event_select], minlength=len(BIN_LABELS) * 4).reshape(len(BIN_LABELS), 4).astype(np.uint64)

    # Exact moments plus deterministic samples.  Rare large-group bins are all sampled.
    if len(group_t) > 1:
        hashes = (group_t.astype(np.uint64) * np.uint64(11400714819323198485)) ^ np.uint64(uid * 0x9E3779B1)
        sampled = ((hashes & np.uint64(1023)) == 0) | (bins_all >= 7)
        prev_ok = select[1:] & select[:-1]
        update_gap(stats, bins_all[1:][prev_ok], np.diff(group_t.astype(np.int64))[prev_ok].astype(np.uint32), "prev", sampled[1:][prev_ok])
        next_ok = select[:-1] & select[1:]
        update_gap(stats, bins_all[:-1][next_ok], np.diff(group_t.astype(np.int64))[next_ok].astype(np.uint32), "next", sampled[:-1][next_ok])


def apply_event(state: int, event: int, strict: bool = True) -> int | None:
    like = bool(state & 1); dislike = bool(state & 2)
    if event == 0:
        like = True
    elif event == 1:
        dislike = True
    elif event == 2:
        if strict and not like:
            return None
        like = False
    else:
        if strict and not dislike:
            return None
        dislike = False
    return int(like) | (int(dislike) << 1)


def enumerate_orders(events: tuple[int, ...], prestates: set[int]) -> tuple[str, tuple[int, ...] | None, set[int]]:
    all_orders = set(itertools.permutations(events))
    legal: set[tuple[int, ...]] = set()
    posts: set[int] = set()
    for order in all_orders:
        for pre in prestates:
            state: int | None = pre
            for event in order:
                state = apply_event(int(state), event, strict=True) if state is not None else None
                if state is None:
                    break
            if state is not None:
                legal.add(order); posts.add(state)
    if not legal:
        # Preserve the observation without claiming semantic consistency.  The
        # union of unconstrained outcomes is carried forward as uncertainty.
        for order in all_orders:
            for pre in prestates:
                state = pre
                for event in order:
                    state = int(apply_event(state, event, strict=False))
                posts.add(state)
        return "inconsistent", None, posts
    if len(legal) == len(all_orders):
        return "no_order_constraint", None, posts
    if len(legal) == 1:
        return "uniquely_recoverable", next(iter(legal)), posts
    return "ambiguous", None, posts


def classify_user_recovery(timestamp: np.ndarray, item: np.ndarray, event_type: np.ndarray,
                           group_starts: np.ndarray, group_sizes: np.ndarray) -> tuple[dict, np.ndarray, dict]:
    counts = {name: {"subgroups": 0, "events": 0, "parent_groups": 0, "parent_group_events": 0}
              for name in ("uniquely_recoverable", "ambiguous", "inconsistent", "no_order_constraint")}
    parent_flags = {name: np.zeros(len(group_starts), dtype=bool) for name in counts}
    whole_unique = np.zeros(len(group_starts), dtype=bool)
    recovered_orders: dict[int, tuple[int, ...]] = {}

    order = np.lexsort((event_type, timestamp, item))
    it, tt, et = item[order], timestamp[order], event_type[order]
    item_starts = np.r_[0, 1 + np.flatnonzero(it[1:] != it[:-1])]
    item_ends = np.r_[item_starts[1:], len(it)]
    for a, b in zip(item_starts.tolist(), item_ends.tolist(), strict=True):
        if b - a < 2 or not np.any(tt[a + 1:b] == tt[a:b - 1]):
            continue
        prestates: set[int] = {0}
        tstarts = np.r_[a, a + 1 + np.flatnonzero(tt[a + 1:b] != tt[a:b - 1])]
        tends = np.r_[tstarts[1:], b]
        for x, y in zip(tstarts.tolist(), tends.tolist(), strict=True):
            events = tuple(int(v) for v in et[x:y])
            parent = int(np.searchsorted(timestamp[group_starts], int(tt[x])))
            if len(events) == 1:
                posts = {out for pre in prestates if (out := apply_event(pre, events[0], strict=True)) is not None}
                if not posts:
                    posts = {int(apply_event(pre, events[0], strict=False)) for pre in prestates}
                prestates = posts
                continue
            label, unique_order, posts = enumerate_orders(events, prestates)
            counts[label]["subgroups"] += 1
            counts[label]["events"] += len(events)
            parent_flags[label][parent] = True
            prestates = posts
            if unique_order is not None and int(group_sizes[parent]) == len(events):
                whole_unique[parent] = True
                recovered_orders[parent] = unique_order
    for name, flags in parent_flags.items():
        counts[name]["parent_groups"] = int(flags.sum())
        counts[name]["parent_group_events"] = int(group_sizes[flags].sum())
    return counts, whole_unique, recovered_orders


def add_nested_counts(target: dict, source: dict) -> None:
    for label, values in source.items():
        for key, value in values.items():
            target[label][key] += int(value)


def prepare_streams(manifest: dict) -> list[dict]:
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
    return streams


def next_user(streams: list[dict], real_lookup: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    active = [s for s in streams if s["row"] < len(s["uid"])]
    if not active:
        return None
    uid = min(int(s["uid"][s["row"]]) for s in active)
    tsp: list[np.ndarray] = []; ip: list[np.ndarray] = []; ep: list[np.ndarray] = []; pp: list[np.ndarray] = []
    for mark, stream in enumerate(streams):
        row = stream["row"]
        if row >= len(stream["uid"]) or int(stream["uid"][row]) != uid:
            continue
        a, b = int(stream["offsets"][row]), int(stream["offsets"][row + 1])
        ts = np.asarray(stream["timestamp"][a:b], dtype=np.uint32)
        it = np.asarray(stream["item"][a:b], dtype=np.uint32)
        keep = real_lookup[it]
        if np.any(keep):
            tsp.append(ts[keep]); ip.append(it[keep]); ep.append(np.full(int(keep.sum()), mark, np.uint8)); pp.append(np.flatnonzero(keep).astype(np.uint32))
        stream["row"] += 1
    if not tsp:
        return uid, np.empty(0, np.uint32), np.empty(0, np.uint32), np.empty(0, np.uint8), np.empty(0, np.uint32)
    timestamp = np.concatenate(tsp); item = np.concatenate(ip); event_type = np.concatenate(ep); source_pos = np.concatenate(pp)
    # Sort only to identify timestamp/item sets.  event_type is a deterministic
    # engineering tie-break and is never interpreted as chronology.
    order = np.lexsort((event_type, item, timestamp))
    return uid, timestamp[order], item[order], event_type[order], source_pos[order]


def transition_stats_for_user(groups: list[tuple[np.ndarray, np.ndarray]], mode: str, seed: int = 0,
                              recovered: dict[int, tuple[int, ...]] | None = None) -> dict:
    matrix = np.zeros((4, 4), np.float64)
    pair = np.zeros((4, 4, 2), np.float64)
    zero = 0.0; rev_like = 0.0; rev_dislike = 0.0

    def add_sources(sf: np.ndarray, si: np.ndarray, tf: np.ndarray, ti: np.ndarray, weight: float) -> None:
        nonlocal rev_like, rev_dislike
        for a in range(len(sf)):
            for b in range(len(tf)):
                w = weight
                matrix[int(sf[a]), int(tf[b])] += w
                same = int(si[a] == ti[b]); pair[int(sf[a]), int(tf[b]), same] += w
                if same and int(sf[a]) == 0 and int(tf[b]) == 2: rev_like += w
                if same and int(sf[a]) == 1 and int(tf[b]) == 3: rev_dislike += w

    ordered_groups: list[tuple[np.ndarray, np.ndarray] | None] = []
    rng = np.random.default_rng(seed)
    for gi, (f, i) in enumerate(groups):
        if len(f) == 1:
            ordered_groups.append((f, i)); continue
        if mode == "grouped" or (mode == "conservative_mixed" and (recovered is None or gi not in recovered)):
            ordered_groups.append(None); continue
        if mode == "engineering_original":
            order = np.lexsort((i, f))
        elif mode == "reversed":
            order = np.lexsort((i, f))[::-1]
        elif mode.startswith("random"):
            order = rng.permutation(len(f))
        elif mode == "conservative_mixed":
            wanted = recovered[gi]
            order_list = []
            for value in wanted:
                candidates = np.flatnonzero(f == value)
                order_list.append(int(candidates[0]))
            order = np.asarray(order_list, dtype=np.int64)
        else:
            raise ValueError(mode)
        of, oi = f[order], i[order]
        ordered_groups.append((of, oi))
        for j in range(len(of) - 1):
            add_sources(of[j:j+1], oi[j:j+1], of[j+1:j+2], oi[j+1:j+2], 1.0)
            zero += 1

    for gi in range(len(groups) - 1):
        lf, li = groups[gi]; rf, ri = groups[gi + 1]
        left = ordered_groups[gi]; right = ordered_groups[gi + 1]
        if left is not None:
            lf, li = left[0][-1:], left[1][-1:]
        if right is not None:
            rf, ri = right[0][:1], right[1][:1]
        weight = 1.0 / (len(lf) * len(rf))
        add_sources(lf, li, rf, ri, weight)
    return {"matrix": matrix, "pair": pair, "zero": zero, "like_revision": rev_like, "dislike_revision": rev_dislike}


def strict_and_recovery_for_user(timestamp: np.ndarray, item: np.ndarray, event_type: np.ndarray,
                                 group_starts: np.ndarray, group_sizes: np.ndarray) -> tuple[dict, np.ndarray, dict, np.ndarray, np.ndarray]:
    recovery, whole_unique, recovered = classify_user_recovery(timestamp, item, event_type, group_starts, group_sizes)
    strict_matrix_like = np.zeros((len(BIN_LABELS), len(BIN_LABELS)), np.uint64)
    strict_matrix_dislike = np.zeros_like(strict_matrix_like)
    # Strict legacy path: ambiguous same-item same-time groups neither pair nor update state.
    order = np.lexsort((event_type, timestamp, item))
    it, tt, et = item[order], timestamp[order], event_type[order]
    starts = np.r_[0, 1 + np.flatnonzero((it[1:] != it[:-1]) | (tt[1:] != tt[:-1]))]
    ends = np.r_[starts[1:], len(it)]
    active_like: dict[int, tuple[int, int]] = {}; active_dislike: dict[int, tuple[int, int]] = {}
    group_t = timestamp[group_starts]
    bins = bin_index(group_sizes)
    for a, b in zip(starts.tolist(), ends.tolist(), strict=True):
        if b - a != 1:
            continue
        current_item = int(it[a]); typ = int(et[a]); time_value = int(tt[a])
        parent = int(np.searchsorted(group_t, time_value)); current_bin = int(bins[parent])
        if typ == 0:
            active_like.setdefault(current_item, (time_value, current_bin))
        elif typ == 1:
            active_dislike.setdefault(current_item, (time_value, current_bin))
        elif typ == 2:
            origin = active_like.pop(current_item, None)
            if origin is not None and origin[0] < time_value:
                strict_matrix_like[origin[1], current_bin] += 1
        else:
            origin = active_dislike.pop(current_item, None)
            if origin is not None and origin[0] < time_value:
                strict_matrix_dislike[origin[1], current_bin] += 1
    return recovery, whole_unique, recovered, strict_matrix_like, strict_matrix_dislike


def scan_groups(manifest: dict, real_lookup: np.ndarray, args: argparse.Namespace) -> tuple[dict, dict]:
    started = time.time(); streams = prepare_streams(manifest)
    train_cutoff = int(manifest["global_split"]["train_cutoff_timestamp_inclusive"])
    stats_all = new_bin_stats(); stats_train = new_bin_stats()
    composition = defaultdict(int)
    recovery_total = {name: {"subgroups": 0, "events": 0, "parent_groups": 0, "parent_group_events": 0}
                      for name in ("uniquely_recoverable", "ambiguous", "inconsistent", "no_order_constraint")}
    strict_like = np.zeros((len(BIN_LABELS), len(BIN_LABELS)), np.uint64)
    strict_dislike = np.zeros_like(strict_like)
    user_threshold_retained = {str(t): 0 for t in FILTER_THRESHOLDS}
    sample_aggregate: dict[str, dict] = {}
    sample_group_count = sample_multi_count = sample_users = 0
    scanned_users = total_users = total_events = 0
    while True:
        result = next_user(streams, real_lookup)
        if result is None or (args.max_users and scanned_users >= args.max_users):
            break
        uid, timestamp, item, event_type, _ = result
        scanned_users += 1
        if not len(item):
            continue
        total_users += 1
        total_events += len(item)
        group_starts = np.r_[0, 1 + np.flatnonzero(timestamp[1:] != timestamp[:-1])]
        group_ends = np.r_[group_starts[1:], len(item)]
        sizes = group_ends - group_starts; group_t = timestamp[group_starts]
        event_group = np.repeat(np.arange(len(group_starts), dtype=np.int64), sizes)
        item_start = np.empty(len(item), bool); item_start[0] = True
        item_start[1:] = (timestamp[1:] != timestamp[:-1]) | (item[1:] != item[:-1])
        pair_starts = np.flatnonzero(item_start); pair_ends = np.r_[pair_starts[1:], len(item)]
        pair_sizes = pair_ends - pair_starts
        unique_items = np.add.reduceat(item_start.astype(np.uint16), group_starts).astype(np.uint32)
        pair_masks = np.bitwise_or.reduceat((np.uint8(1) << event_type), pair_starts)
        pair_parent = event_group[pair_starts]
        same_multi = np.zeros(len(group_starts), bool)
        np.logical_or.at(same_multi, pair_parent, pair_sizes > 1)
        group_masks = np.bitwise_or.reduceat((np.uint8(1) << event_type), group_starts)
        single_feedback = (group_masks & (group_masks - 1)) == 0
        c_pair = (pair_masks & 0b0101) == 0b0101
        d_pair = (pair_masks & 0b1010) == 0b1010
        multitype = (pair_masks & (pair_masks - 1)) != 0
        other_pair = multitype & ~((pair_masks == 0b0101) | (pair_masks == 0b1010))
        c_group = np.zeros(len(group_starts), bool); d_group = np.zeros_like(c_group); other_group = np.zeros_like(c_group)
        np.logical_or.at(c_group, pair_parent, c_pair); np.logical_or.at(d_group, pair_parent, d_pair); np.logical_or.at(other_group, pair_parent, other_pair)
        same_revision_events_group = np.zeros(len(group_starts), np.uint32)
        np.add.at(same_revision_events_group, pair_parent[c_pair | d_pair], pair_sizes[c_pair | d_pair].astype(np.uint32))

        multi = sizes > 1; all_different = multi & (unique_items == sizes); same_item = multi & (unique_items < sizes)
        composition["groups"] += len(sizes); composition["events"] += len(item)
        composition["singleton_groups"] += int((sizes == 1).sum()); composition["singleton_events"] += int((sizes == 1).sum())
        composition["multi_groups"] += int(multi.sum()); composition["multi_events"] += int(sizes[multi].sum())
        for label, flags in (("all_different_item", all_different), ("same_item_multi", same_item),
                             ("like_unlike", c_group), ("dislike_undislike", d_group), ("other_multifeedback", other_group)):
            composition[f"{label}_groups"] += int(flags.sum())
            composition[f"{label}_parent_group_events"] += int(sizes[flags].sum())
        composition["like_unlike_direct_events"] += int(pair_sizes[c_pair].sum())
        composition["dislike_undislike_direct_events"] += int(pair_sizes[d_pair].sum())
        composition["other_multifeedback_direct_events"] += int(pair_sizes[other_pair].sum())

        train_select = group_t <= train_cutoff
        update_bins(stats_all, uid, group_t, sizes, unique_items, same_multi, single_feedback, group_masks,
                    event_type, event_group, same_revision_events_group, np.ones(len(sizes), bool))
        update_bins(stats_train, uid, group_t, sizes, unique_items, same_multi, single_feedback, group_masks,
                    event_type, event_group, same_revision_events_group, train_select)
        for threshold in FILTER_THRESHOLDS:
            user_threshold_retained[str(threshold)] += int(np.any(sizes <= threshold))

        rec, whole_unique, recovered, sl, sd = strict_and_recovery_for_user(timestamp, item, event_type, group_starts, sizes)
        add_nested_counts(recovery_total, rec); strict_like += sl; strict_dislike += sd
        composition["fully_uniquely_ordered_groups"] += int(whole_unique.sum())
        composition["fully_uniquely_ordered_events"] += int(sizes[whole_unique].sum())

        if uid % args.sample_user_modulus == 0 and np.any(multi):
            groups = [(event_type[a:b].copy(), item[a:b].copy()) for a, b in zip(group_starts, group_ends, strict=True)]
            sample_users += 1; sample_group_count += len(groups); sample_multi_count += int(multi.sum())
            modes = (("engineering_original", 0), ("reversed", 0), ("random_1", 113), ("random_2", 271), ("random_3", 619), ("grouped", 0), ("conservative_mixed", 0))
            for mode, seed in modes:
                row = transition_stats_for_user(groups, mode, seed=seed + uid, recovered=recovered)
                agg = sample_aggregate.setdefault(mode, {"matrix": np.zeros((4,4)), "pair": np.zeros((4,4,2)), "zero": 0., "like_revision": 0., "dislike_revision": 0.})
                for key in agg:
                    agg[key] += row[key]

        if scanned_users % args.progress_every == 0:
            atomic_json(args.output_dir / "progress.json", {"status": "running", "scanned_users": scanned_users, "D_SID_users": total_users, "events": total_events, "elapsed_seconds": time.time() - started})
            print({"scanned_users": scanned_users, "D_SID_users": total_users, "events": total_events, "elapsed_seconds": round(time.time() - started, 1)}, flush=True)

    def gap_desc(stats: dict, b: int, which: str) -> dict:
        chunks = stats[f"{which}_samples"][b]
        sample = np.concatenate(chunks) if chunks else np.empty(0, np.uint32)
        count = int(stats[f"{which}_gap_count"][b])
        if not count:
            return {"count": 0}
        return {
            "count": count,
            "mean_seconds": float(stats[f"{which}_gap_sum"][b] / count),
            "min_seconds": int(stats[f"{which}_gap_min"][b]),
            "median_seconds_sample": int(np.quantile(sample, .5, method="higher")) if len(sample) else None,
            "p90_seconds_sample": int(np.quantile(sample, .9, method="higher")) if len(sample) else None,
            "p99_seconds_sample": int(np.quantile(sample, .99, method="higher")) if len(sample) else None,
            "max_seconds": int(stats[f"{which}_gap_max"][b]),
            "sample_size": int(len(sample)),
        }

    def serialize_bins(stats: dict) -> dict:
        total_g = int(stats["groups"].sum()); total_e = int(stats["events"].sum())
        out = {}
        for b, label in enumerate(BIN_LABELS):
            groups = int(stats["groups"][b]); events = int(stats["events"][b]); fb = stats["feedback"][b]
            out[label] = {
                "groups": groups, "group_fraction": groups / total_g if total_g else 0,
                "events": events, "event_fraction": events / total_e if total_e else 0,
                "unique_items_per_group_mean": float(stats["unique_items"][b] / groups) if groups else None,
                "same_item_multifeedback_group_fraction": float(stats["same_item_multifeedback_groups"][b] / groups) if groups else None,
                "single_feedback_group_fraction": float(stats["single_feedback_groups"][b] / groups) if groups else None,
                "feedback_counts": {MARKS[k]: int(fb[k]) for k in range(4)},
                "feedback_composition": {MARKS[k]: float(fb[k] / events) if events else 0 for k in range(4)},
                "same_time_revision_related_events": int(stats["same_time_revision_events"][b]),
                "same_time_revision_related_event_fraction": float(stats["same_time_revision_events"][b] / events) if events else 0,
                "users": int(stats["users"][b]),
                "previous_timestamp_gap": gap_desc(stats, b, "prev"),
                "next_timestamp_gap": gap_desc(stats, b, "next"),
            }
        return {"totals": {"groups": total_g, "events": total_e}, "bins": out}

    total_groups = composition["groups"]; tied_events = composition["multi_events"]
    comp = dict(composition)
    for label in ("all_different_item", "same_item_multi", "like_unlike", "dislike_undislike", "other_multifeedback"):
        comp[f"{label}_group_fraction_of_multi"] = comp[f"{label}_groups"] / comp["multi_groups"] if comp["multi_groups"] else 0
        comp[f"{label}_parent_event_fraction_of_tied"] = comp[f"{label}_parent_group_events"] / tied_events if tied_events else 0
    for label, row in recovery_total.items():
        row["subgroup_fraction"] = row["subgroups"] / max(1, sum(v["subgroups"] for v in recovery_total.values()))
        row["event_fraction_of_tied"] = row["events"] / tied_events if tied_events else 0
    comp["fully_uniquely_ordered_group_fraction_of_multi"] = comp["fully_uniquely_ordered_groups"] / comp["multi_groups"]
    comp["fully_uniquely_ordered_event_fraction_of_tied"] = comp["fully_uniquely_ordered_events"] / tied_events

    # Hypothetical thresholds are arithmetic summaries only; no data are written or deleted.
    threshold_effects = {}
    all_bins = stats_all
    for threshold in FILTER_THRESHOLDS:
        max_bin = int(np.searchsorted(BIN_UPPER, threshold, side="left"))
        keep = np.arange(len(BIN_LABELS)) <= max_bin
        events = int(all_bins["events"][keep].sum()); groups = int(all_bins["groups"][keep].sum())
        fb = all_bins["feedback"][keep].sum(axis=0)
        multi_events = int(all_bins["events"][keep & (np.arange(len(BIN_LABELS)) > 0)].sum())
        sl = int(strict_like[np.ix_(keep, keep)].sum()); sd = int(strict_dislike[np.ix_(keep, keep)].sum())
        threshold_effects[str(threshold)] = {
            "rule": f"hypothetically retain timestamp groups with size <= {threshold}",
            "event_retention": events / int(all_bins["events"].sum()),
            "group_retention": groups / int(all_bins["groups"].sum()),
            "user_retention": user_threshold_retained[str(threshold)] / total_users,
            "feedback_retention": {MARKS[k]: float(fb[k] / all_bins["feedback"][:, k].sum()) for k in range(4)},
            "tied_event_fraction_after": multi_events / events if events else 0,
            "multi_group_bins_retained": {BIN_LABELS[b]: int(all_bins["groups"][b]) for b in range(1, max_bin + 1)},
            "strict_like_to_unlike_pairs_retained": sl,
            "strict_dislike_to_undislike_pairs_retained": sd,
            "strict_like_to_unlike_pair_retention": sl / int(strict_like.sum()) if strict_like.sum() else 0,
            "strict_dislike_to_undislike_pair_retention": sd / int(strict_dislike.sum()) if strict_dislike.sum() else 0,
        }

    report = {
        "status": "complete" if not args.max_users else "partial",
        "data_contract": {"dataset": "D_SID real-audio view of exact-deduplicated D_all", "listen_used": False, "events_modified": False, "group_protocol_modified": False, "train_cutoff_inclusive": train_cutoff},
        "counts": {"users": total_users, "events": total_events, "groups": total_groups},
        "tied_group_composition": comp,
        "conservative_recovery": {"unit": "same user+item+timestamp subgroup", "rules": "only like->unlike and dislike->undislike; like/dislike are not mutually exclusive; repeated base feedback is not forbidden", "classifications": recovery_total,
                                  "important_scope": "locally unique order inside an item subgroup does not order that subgroup relative to other items in the same timestamp group"},
        "extreme_burst_all": serialize_bins(stats_all),
        "extreme_burst_train": serialize_bins(stats_train),
        "strict_revision_pair_origin_bin_by_revision_bin": {"like_to_unlike": strict_like.tolist(), "dislike_to_undislike": strict_dislike.tolist(), "bin_labels": BIN_LABELS},
        "hypothetical_filter_sensitivity": threshold_effects,
        "elapsed_seconds": time.time() - started,
    }

    sensitivity = {"status": "complete", "sample": {"selection": f"uid % {args.sample_user_modulus} == 0 among users with multi-event groups", "users": sample_users, "groups": sample_group_count, "multi_groups": sample_multi_count}, "protocol_note": "engineering_original is deterministic source-type order, not business chronology; grouped has no within-timestamp transitions; conservative_mixed orders only fully uniquely recoverable groups"}
    grouped_matrix = sample_aggregate.get("grouped", {}).get("matrix", np.zeros((4,4)))
    grouped_dist = grouped_matrix / grouped_matrix.sum() if grouped_matrix.sum() else grouped_matrix
    for mode, row in sample_aggregate.items():
        matrix = row["matrix"]; dist = matrix / matrix.sum() if matrix.sum() else matrix
        sensitivity[mode] = {
            "feedback_transition_matrix": matrix.tolist(),
            "feedback_transition_distribution": dist.tolist(),
            "total_variation_vs_grouped": float(.5 * np.abs(dist - grouped_dist).sum()),
            "zero_delta_ordered_transitions": row["zero"],
            "adjacent_like_to_unlike_same_item": row["like_revision"],
            "adjacent_dislike_to_undislike_same_item": row["dislike_revision"],
            "source_target_same_item_tensor": row["pair"].tolist(),
        }
    return report, sensitivity


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.dall.read_text())
    print("[1/3] provenance", flush=True)
    provenance = provenance_audit(args.raw, manifest)
    atomic_json(args.output_dir / "row_order_provenance.json", provenance)
    if args.provenance_only:
        print(json.dumps(provenance, ensure_ascii=False, indent=2))
        return
    with np.load(args.support, allow_pickle=False) as z:
        explicit = z["item_id"]; real = z["real_embedding"]
    lookup = np.zeros(int(explicit.max()) + 1, bool); lookup[explicit] = real
    print("[2/3] full D_SID group/recovery/burst scan", flush=True)
    group_report, sensitivity = scan_groups(manifest, lookup, args)
    atomic_json(args.output_dir / "group_recoverability_and_burst.json", group_report)
    print("[3/3] order sensitivity", flush=True)
    atomic_json(args.output_dir / "row_order_sensitivity.json", sensitivity)
    atomic_json(args.output_dir / "progress.json", {"status": "complete", "counts": group_report["counts"]})
    print(json.dumps({"provenance": provenance["classification_text"], "counts": group_report["counts"], "composition": group_report["tied_group_composition"], "recovery": group_report["conservative_recovery"], "outputs": [str((args.output_dir / x).resolve()) for x in ("row_order_provenance.json", "group_recoverability_and_burst.json", "row_order_sensitivity.json")]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
