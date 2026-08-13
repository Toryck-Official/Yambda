#!/usr/bin/env python3
"""Materialize exact-deduplicated D_all and global 80/10/10 split views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "phase0_5b_audit" / "work" / "flat_explicit_5b" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "phase1_canonical" / "d_all"
DEFAULT_AUDIT = ROOT / "phase1_canonical" / "artifacts" / "dall_manifest.json"
DEFAULT_FREQ = ROOT / "phase0_5b_audit" / "artifacts" / "item_explicit_frequency.npz"
DEFAULT_ORIG2DENSE = Path("/root/autodl-tmp/0626/0626 Predictor/01_data/processed/raw_rqkmeans/orig2dense_item_id.npy")
EXPECTED_RAW = 136_292_476
EXPECTED_DUPLICATE_SURPLUS = 1_213_728
EXPECTED_DEDUP = 135_078_748


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--item-frequency", type=Path, default=DEFAULT_FREQ)
    p.add_argument("--orig2dense", type=Path, default=DEFAULT_ORIG2DENSE)
    return p.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    tmp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_mark(mark: str, record: dict, output_dir: Path) -> dict:
    files = {key: Path(value) for key, value in record["output_files"].items()}
    uid = np.load(files["uid.npy"], mmap_mode="r")
    offsets = np.load(files["offsets.npy"], mmap_mode="r")
    timestamp = np.load(files["timestamp.npy"], mmap_mode="r")
    item = np.load(files["item_id.npy"], mmap_mode="r")
    organic = np.load(files["is_organic.npy"], mmap_mode="r")
    keep_path = output_dir / f".{mark}_keep.bool.npy"
    keep = np.lib.format.open_memmap(keep_path, mode="w+", dtype=bool, shape=(len(timestamp),))
    counts = np.empty(len(uid), dtype=np.uint64)
    duplicate_surplus = 0
    duplicate_keys_with_mixed_organic = 0
    for row in range(len(uid)):
        start, end = int(offsets[row]), int(offsets[row + 1])
        ts = np.asarray(timestamp[start:end], dtype=np.uint64)
        items = np.asarray(item[start:end], dtype=np.uint64)
        key = (ts << np.uint64(24)) | items
        _, first, inverse, key_counts = np.unique(
            key, return_index=True, return_inverse=True, return_counts=True
        )
        local_keep = np.zeros(end - start, dtype=bool)
        local_keep[first] = True
        keep[start:end] = local_keep
        counts[row] = len(first)
        duplicate_surplus += (end - start) - len(first)
        repeated_groups = np.flatnonzero(key_counts > 1)
        if len(repeated_groups):
            org = np.asarray(organic[start:end], dtype=np.uint8)
            group_min = np.full(len(key_counts), 1, dtype=np.uint8)
            group_max = np.zeros(len(key_counts), dtype=np.uint8)
            np.minimum.at(group_min, inverse, org)
            np.maximum.at(group_max, inverse, org)
            duplicate_keys_with_mixed_organic += int(
                np.count_nonzero((key_counts > 1) & (group_min != group_max))
            )
        if (row + 1) % 100_000 == 0:
            print(f"{mark}: dedup scanned {row + 1:,}/{len(uid):,} users", flush=True)
    keep.flush()
    total = int(counts.sum(dtype=np.uint64))
    mark_dir = output_dir / mark
    mark_dir.mkdir(parents=True, exist_ok=True)
    output_timestamp = np.lib.format.open_memmap(
        mark_dir / "timestamp.npy", mode="w+", dtype=np.uint32, shape=(total,)
    )
    output_item = np.lib.format.open_memmap(
        mark_dir / "item_id.npy", mode="w+", dtype=np.uint32, shape=(total,)
    )
    cursor = 0
    for start in range(0, len(timestamp), 5_000_000):
        end = min(start + 5_000_000, len(timestamp))
        mask = np.asarray(keep[start:end])
        count = int(mask.sum())
        output_timestamp[cursor : cursor + count] = timestamp[start:end][mask]
        output_item[cursor : cursor + count] = item[start:end][mask]
        cursor += count
    if cursor != total:
        raise RuntimeError(f"{mark} output conservation failed")
    output_timestamp.flush(); output_item.flush()
    output_offsets = np.empty(len(uid) + 1, dtype=np.uint64)
    output_offsets[0] = 0
    np.cumsum(counts, out=output_offsets[1:])
    atomic_npy(mark_dir / "uid.npy", np.asarray(uid))
    atomic_npy(mark_dir / "offsets.npy", output_offsets)
    keep_path.unlink()
    if int(record["events"]) - total != duplicate_surplus:
        raise RuntimeError(f"{mark} duplicate conservation failed")
    return {
        "feedback": mark,
        "raw_events": int(record["events"]),
        "events_after_exact_dedup": total,
        "exact_duplicate_surplus_removed": int(duplicate_surplus),
        "duplicate_keys_with_mixed_is_organic": int(duplicate_keys_with_mixed_organic),
        "users": int(len(uid)),
        "output_files": {
            "uid": str((mark_dir / "uid.npy").resolve()),
            "offsets": str((mark_dir / "offsets.npy").resolve()),
            "timestamp": str((mark_dir / "timestamp.npy").resolve()),
            "item_id": str((mark_dir / "item_id.npy").resolve()),
        },
    }


def global_cutoffs(mark_reports: dict[str, dict]) -> tuple[int, int, dict]:
    maximum = 26_000_000
    resolution = 5
    hist = np.zeros(maximum // resolution + 1, dtype=np.uint64)
    for report in mark_reports.values():
        timestamp = np.load(report["output_files"]["timestamp"], mmap_mode="r")
        for start in range(0, len(timestamp), 5_000_000):
            chunk = np.asarray(timestamp[start : start + 5_000_000], dtype=np.uint32)
            hist += np.bincount(chunk // resolution, minlength=len(hist)).astype(np.uint64)
    cumulative = np.cumsum(hist, dtype=np.uint64)
    total = int(cumulative[-1])
    cutoffs = []
    for fraction in (0.8, 0.9):
        target = int(np.ceil(total * fraction))
        index = int(np.searchsorted(cumulative, target, side="left"))
        cutoffs.append(index * resolution)
    train_end, validation_end = cutoffs
    train_events = int(cumulative[train_end // resolution])
    train_validation_events = int(cumulative[validation_end // resolution])
    return train_end, validation_end, {
        "train_cutoff_timestamp_inclusive": train_end,
        "validation_cutoff_timestamp_inclusive": validation_end,
        "target_fractions": [0.8, 0.1, 0.1],
        "actual_event_counts": {
            "train": train_events,
            "validation": train_validation_events - train_events,
            "test": total - train_validation_events,
        },
        "actual_event_fractions": {
            "train": train_events / total,
            "validation": (train_validation_events - train_events) / total,
            "test": (total - train_validation_events) / total,
        },
        "same_timestamp_group_not_split": True,
    }


def materialize_split_views_and_stats(
    mark_reports: dict[str, dict], train_cutoff: int, validation_cutoff: int,
    orig2dense: np.ndarray, catalog_size: int, output_dir: Path,
) -> tuple[dict, dict]:
    split_names = ("train", "validation", "test")
    user_seen = {name: np.zeros(1_000_001, dtype=bool) for name in split_names}
    item_seen = {name: np.zeros(catalog_size, dtype=bool) for name in split_names}
    split_events = {name: 0 for name in split_names}
    missing_events = {name: 0 for name in split_names}
    by_feedback: dict[str, dict] = {}
    split_view_reports = {}
    for mark, report in mark_reports.items():
        uid = np.load(report["output_files"]["uid"], mmap_mode="r")
        offsets = np.load(report["output_files"]["offsets"], mmap_mode="r")
        timestamp = np.load(report["output_files"]["timestamp"], mmap_mode="r")
        item = np.load(report["output_files"]["item_id"], mmap_mode="r")
        train_end = np.empty(len(uid), dtype=np.uint64)
        validation_end = np.empty(len(uid), dtype=np.uint64)
        mark_counts = {name: 0 for name in split_names}
        for row in range(len(uid)):
            start, end = int(offsets[row]), int(offsets[row + 1])
            times = timestamp[start:end]
            first_end = start + int(np.searchsorted(times, train_cutoff, side="right"))
            second_end = start + int(np.searchsorted(times, validation_cutoff, side="right"))
            train_end[row] = first_end
            validation_end[row] = second_end
            bounds = ((start, first_end), (first_end, second_end), (second_end, end))
            for split, (left, right) in zip(split_names, bounds, strict=True):
                if right <= left:
                    continue
                count = right - left
                mark_counts[split] += count
                split_events[split] += count
                user_seen[split][int(uid[row])] = True
                chunk_items = np.asarray(item[left:right], dtype=np.uint32)
                item_seen[split][chunk_items] = True
                missing_events[split] += int(np.count_nonzero(orig2dense[chunk_items] <= 0))
        split_dir = output_dir / "split_views" / mark
        split_dir.mkdir(parents=True, exist_ok=True)
        atomic_npy(split_dir / "train_end.npy", train_end)
        atomic_npy(split_dir / "validation_end.npy", validation_end)
        by_feedback[mark] = mark_counts
        split_view_reports[mark] = {
            "uid": report["output_files"]["uid"],
            "offsets": report["output_files"]["offsets"],
            "shared_timestamp": report["output_files"]["timestamp"],
            "shared_item_id": report["output_files"]["item_id"],
            "train_end": str((split_dir / "train_end.npy").resolve()),
            "validation_end": str((split_dir / "validation_end.npy").resolve()),
            "semantics": "per user: [offset,train_end)=train, [train_end,validation_end)=validation, [validation_end,next_offset)=test",
        }
    stats = {}
    for split in split_names:
        items = item_seen[split]
        present_ids = np.flatnonzero(items)
        missing_items = int(np.count_nonzero(orig2dense[present_ids] <= 0))
        stats[split] = {
            "events": int(split_events[split]),
            "users": int(user_seen[split].sum()),
            "items": int(items.sum()),
            "real_audio_items": int(items.sum() - missing_items),
            "missing_audio_items": missing_items,
            "missing_audio_events": int(missing_events[split]),
            "feedback_counts": {mark: int(by_feedback[mark][split]) for mark in mark_reports},
        }
    return split_view_reports, stats


def main() -> None:
    args = parse_args()
    source = json.loads(args.source_manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for mark, record in source["records"].items():
        print(f"[dedup] {mark}", flush=True)
        reports[mark] = build_mark(mark, record, args.output_dir)
    raw = sum(row["raw_events"] for row in reports.values())
    dedup = sum(row["events_after_exact_dedup"] for row in reports.values())
    removed = sum(row["exact_duplicate_surplus_removed"] for row in reports.values())
    if (raw, removed, dedup) != (EXPECTED_RAW, EXPECTED_DUPLICATE_SURPLUS, EXPECTED_DEDUP):
        raise RuntimeError(f"global dedup totals changed: {(raw, removed, dedup)}")
    print("[split] global event-count cutoffs", flush=True)
    train_cutoff, validation_cutoff, split_contract = global_cutoffs(reports)
    orig2dense = np.load(args.orig2dense, mmap_mode="r")
    split_views, split_stats = materialize_split_views_and_stats(
        reports, train_cutoff, validation_cutoff, orig2dense, len(orig2dense), args.output_dir
    )
    if sum(value["events"] for value in split_stats.values()) != dedup:
        raise RuntimeError("split event conservation failed")
    with np.load(args.item_frequency, allow_pickle=False) as z:
        explicit_items = int(len(z["item_id"]))
    manifest = {
        "status": "complete_phase1_dall_exact_dedup_global_split",
        "data_contract": {
            "events": ["like", "dislike", "unlike", "undislike"],
            "listen_used": False,
            "only_deletion": "surplus rows sharing exact user,item,timestamp,feedback key",
            "state_anomalies_removed": False,
            "timestamp_bursts_removed": False,
            "is_organic_used": False,
            "dataset": "D_all",
        },
        "dedup": {
            "events_before": raw,
            "exact_duplicate_surplus_removed": removed,
            "events_after": dedup,
            "explicit_item_universe_before": explicit_items,
            "by_feedback": reports,
        },
        "global_split": split_contract,
        "split_stats": split_stats,
        "split_views": split_views,
        "checksums": {
            mark: {
                key: sha256(Path(path))
                for key, path in report["output_files"].items()
            }
            for mark, report in reports.items()
        },
        "boundaries": {
            "gate1c_allowed": True,
            "gate2_started": False,
            "final_sid_materialized": False,
        },
    }
    atomic_json(args.audit_output, manifest)
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"dedup": manifest["dedup"], "global_split": split_contract, "split_stats": split_stats, "output": str(args.audit_output)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

