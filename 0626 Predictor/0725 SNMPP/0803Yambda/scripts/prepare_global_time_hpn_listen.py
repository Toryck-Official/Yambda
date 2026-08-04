#!/usr/bin/env python3
"""Prepare a global-time HPN dataset whose history may include listen events.

Targets are always the next observed explicit-feedback group (like/dislike/
unlike/undislike), identical to the no-listen control.  The only difference is
the history contract:

- ``--include-listen``: the user history includes listen events as mark id 5
  (padding 0; like=1, dislike=2, unlike=3, undislike=4, listen=5).
- ``--include-source``: every history event carries ``is_organic`` as source id
  (0 padding, 1 recommendation-driven, 2 organic).

This supports the explicit ablation ``HPN(no listen) vs HPN(+listen history)
vs HPN(+listen history + source flag)`` under the same global-time split, same
targets, same catalog, and same test protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

THIS_ROOT = Path(__file__).resolve().parents[1]
SNMPP_ROOT = THIS_ROOT.parent
sys.path.insert(0, str(SNMPP_ROOT))

from snmpp.recommendation.data import (  # noqa: E402
    RQCodebookStore,
    recommendation_catalog_fingerprints,
)

EVENT_NAMES = ("listen", "like", "dislike", "unlike", "undislike")
EXPLICIT_NAMES = EVENT_NAMES[1:]
SPLITS = ("train", "validation", "test")
GAP_SECONDS = 1_800
HISTORY_MARK_ID = {
    "listen": 5,
    "like": 1,
    "dislike": 2,
    "unlike": 3,
    "undislike": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/root/autodl-tmp/0330Yambda/data/sequential-50m/multi_event.parquet"),
    )
    parser.add_argument(
        "--codebook-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/0626/0626 Predictor/0725 SNMPP/0731dataprocess/"
            "artifacts/metadata_imputation_fixed_codebook_seed2026"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_ROOT / "artifacts" / "global_time_explicit_hpn",
    )
    parser.add_argument("--history-len", type=int, default=50)
    parser.add_argument("--min-history-events", type=int, default=1)
    parser.add_argument("--shard-rows", type=int, default=50_000)
    parser.add_argument("--max-users", type=int, default=0, help="Smoke check only; 0 means all.")
    parser.add_argument("--include-listen", action="store_true")
    parser.add_argument("--include-source", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path, *, include_sha256: bool = False) -> dict[str, Any]:
    stat = path.stat()
    output: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        output["sha256"] = sha256_file(path)
    return output


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def boundaries_from_arrays(timestamp: np.ndarray) -> dict[str, int]:
    minimum = int(timestamp.min())
    maximum = int(timestamp.max())
    span = maximum - minimum + 1
    train_end = minimum + int(span * 0.80)
    validation_end = minimum + int(span * 0.90)
    if validation_end - train_end <= GAP_SECONDS or maximum - validation_end <= GAP_SECONDS:
        raise RuntimeError(
            "The requested 80/10/10 temporal split cannot contain its 30-minute gaps"
        )
    return {
        "min_timestamp_seconds": minimum,
        "max_timestamp_seconds": maximum,
        "train_end_exclusive": train_end,
        "validation_start_inclusive": train_end + GAP_SECONDS,
        "validation_end_exclusive": validation_end,
        "test_start_inclusive": validation_end + GAP_SECONDS,
        "gap_seconds": GAP_SECONDS,
    }


def split_of(timestamp: int, bounds: dict[str, int]) -> str:
    if timestamp < bounds["train_end_exclusive"]:
        return "train"
    if timestamp < bounds["validation_start_inclusive"]:
        return "gap_before_validation"
    if timestamp < bounds["validation_end_exclusive"]:
        return "validation"
    if timestamp < bounds["test_start_inclusive"]:
        return "gap_before_test"
    return "test"


def recommendation_schema(history_len: int, sid_levels: int, include_source: bool) -> pa.Schema:
    fields = [
            ("uid", pa.uint32()),
            ("timestamp_seconds", pa.uint32()),
            ("timestamp_group_id", pa.uint32()),
            ("source_group_size", pa.uint16()),
            ("history_dense_item_ids", pa.list_(pa.int32(), history_len)),
            ("history_event_type_ids", pa.list_(pa.int8(), history_len)),
            ("history_age_seconds", pa.list_(pa.float32(), history_len)),
            ("history_mask", pa.list_(pa.bool_(), history_len)),
            ("positive_orig_item_ids", pa.list_(pa.uint32())),
            ("positive_dense_item_ids", pa.list_(pa.int32())),
            ("positive_sids", pa.list_(pa.list_(pa.int16(), sid_levels))),
            ("positive_event_type_mask", pa.list_(pa.list_(pa.bool_(), len(EXPLICIT_NAMES)))),
        ]
    if include_source:
        fields.insert(4, ("history_source_ids", pa.list_(pa.int8(), history_len)))
    return pa.schema(fields)


class SplitWriter:
    def __init__(self, root: Path, schema: pa.Schema, shard_rows: int) -> None:
        self.root = root
        self.schema = schema
        self.shard_rows = int(shard_rows)
        self.buffers: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
        self.indices: dict[str, int] = {split: 0 for split in SPLITS}
        self.row_counts: Counter[str] = Counter()
        for split in SPLITS:
            (root / split).mkdir(parents=True, exist_ok=True)

    def append(self, split: str, row: dict[str, Any]) -> None:
        self.buffers[split].append(row)
        self.row_counts[split] += 1
        if len(self.buffers[split]) >= self.shard_rows:
            self.flush(split)

    def flush(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return
        path = self.root / split / f"part-{self.indices[split]:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=self.schema), path, compression="zstd")
        self.indices[split] += 1
        self.buffers[split] = []

    def close(self) -> None:
        for split in SPLITS:
            self.flush(split)


def load_arrays(path: Path, max_users: int) -> tuple[np.ndarray, ...]:
    table = pq.read_table(path, columns=["uid", "timestamp", "item_id", "is_organic", "event_type"])
    if max_users:
        table = table.slice(0, min(int(max_users), table.num_rows))
    uid = table["uid"].combine_chunks().to_numpy(zero_copy_only=False)
    timestamp_array = table["timestamp"].combine_chunks()
    item_array = table["item_id"].combine_chunks()
    organic_array = table["is_organic"].combine_chunks()
    event_array = table["event_type"].combine_chunks()
    offsets = timestamp_array.offsets.to_numpy(zero_copy_only=False)
    timestamp = timestamp_array.values.to_numpy(zero_copy_only=False)
    item = item_array.values.to_numpy(zero_copy_only=False)
    organic = organic_array.values.to_numpy(zero_copy_only=False)
    raw_event = event_array.values.indices.to_numpy(zero_copy_only=False)
    dictionary = tuple(str(value) for value in event_array.values.dictionary.to_pylist())
    if set(dictionary) != set(EVENT_NAMES):
        raise ValueError(f"unexpected raw event dictionary: {dictionary}")
    raw_to_canonical = np.empty(len(dictionary), dtype=np.int8)
    for canonical, name in enumerate(EVENT_NAMES):
        raw_to_canonical[dictionary.index(name)] = canonical
    event = raw_to_canonical[raw_event]
    if len(offsets) != len(uid) + 1 or not (
        len(timestamp) == len(item) == len(organic) == len(event)
    ):
        raise RuntimeError("Yambda nested arrays are inconsistent")
    return uid, offsets, timestamp, item, organic, event


def map_one(store: RQCodebookStore, item_id: int) -> int:
    return int(store.map_orig_ids(np.asarray([item_id], dtype=np.int64))[0])


def history_arrays(
    history: deque[tuple[int, int, int, int]], timestamp: int, history_len: int
) -> tuple[list[int], list[int], list[float], list[bool], list[int]]:
    selected = list(history)[-history_len:]
    pad = history_len - len(selected)
    return (
        [0] * pad + [entry[0] for entry in selected],
        [0] * pad + [entry[1] for entry in selected],
        [0.0] * pad + [float(timestamp - entry[2]) for entry in selected],
        [False] * pad + [True] * len(selected),
        [0] * pad + [entry[3] for entry in selected],
    )


def cleaned_group(
    indices: np.ndarray,
    item: np.ndarray,
    organic: np.ndarray,
    event: np.ndarray,
) -> tuple[bool, list[tuple[int, int, int]]]:
    """Keep source-aware unique explicit rows; flag impossible timestamp bursts."""

    if len({int(item[index]) for index in indices}) > 50:
        return True, []
    seen: set[tuple[int, int, int]] = set()
    output: list[tuple[int, int, int]] = []
    for index in indices.tolist():
        key = (int(item[index]), int(event[index]), int(organic[index]))
        if key not in seen:
            seen.add(key)
            output.append(key)
    return False, output


def prepare_output(root: Path, overwrite: bool) -> None:
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output is non-empty: {root}; pass --overwrite after review")
        allowed = {
            "train",
            "validation",
            "test",
            "manifest.json",
            "catalog_dense_ids.npy",
            "train_dense_ids.npy",
            "train_item_counts.npz",
            "unavailable_target_audit.npz",
        }
        unknown = sorted(path.name for path in root.iterdir() if path.name not in allowed)
        if unknown:
            raise RuntimeError(f"refusing to overwrite unknown files: {unknown}")
        for split in SPLITS:
            shutil.rmtree(root / split, ignore_errors=True)
        for name in allowed - set(SPLITS):
            (root / name).unlink(missing_ok=True)
    root.mkdir(parents=True, exist_ok=True)


def as_json_counter(counter: Counter[str]) -> dict[str, int]:
    return {name: int(value) for name, value in sorted(counter.items())}


def main() -> None:
    args = parse_args()
    if (
        args.history_len <= 0
        or args.min_history_events < 0
        or args.min_history_events > args.history_len
    ):
        raise ValueError("history limits are invalid")
    if args.shard_rows <= 0:
        raise ValueError("shard rows must be positive")
    input_path = args.input.expanduser().resolve()
    codebook_dir = args.codebook_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prepare_output(output_dir, args.overwrite)
    store = RQCodebookStore(codebook_dir)
    uid, offsets, timestamp, item, organic, event = load_arrays(input_path, args.max_users)
    bounds = boundaries_from_arrays(timestamp)
    writer = SplitWriter(
        output_dir,
        recommendation_schema(args.history_len, store.sid_levels, args.include_source),
        args.shard_rows,
    )
    quality = Counter()
    cleaning = Counter()
    target_events: dict[str, Counter[str]] = defaultdict(Counter)
    unavailable_targets: dict[str, Counter[str]] = defaultdict(Counter)
    source_target_events: dict[str, Counter[str]] = defaultdict(Counter)
    train_item_counts: Counter[int] = Counter()
    unavailable_audit: dict[str, list[int]] = {
        "uid": [],
        "timestamp_seconds": [],
        "timestamp_group_id": [],
        "item_id": [],
        "event_type_id": [],
    }
    group_id = 0

    for user_pos, raw_uid in enumerate(uid.tolist()):
        start, stop = int(offsets[user_pos]), int(offsets[user_pos + 1])
        user_time = timestamp[start:stop]
        if len(user_time) > 1 and np.any(user_time[1:] < user_time[:-1]):
            quality["users_with_unsorted_timestamps"] += 1
        if np.any(user_time % 5 != 0):
            quality["timestamps_not_divisible_by_5"] += int(np.count_nonzero(user_time % 5 != 0))
        positions = np.arange(start, stop, dtype=np.int64)
        all_positions = positions
        history: deque[tuple[int, int, int, int]] = deque(maxlen=args.history_len)
        explicit_eligibility: deque[tuple[int, int, int, int]] = deque(maxlen=args.history_len)
        local = 0
        while local < len(all_positions):
            group_time = int(timestamp[all_positions[local]])
            group_end = local + 1
            while (
                group_end < len(all_positions)
                and int(timestamp[all_positions[group_end]]) == group_time
            ):
                group_end += 1
            raw_group = all_positions[local:group_end]
            split = split_of(group_time, bounds)
            explicit_indices = raw_group[event[raw_group] != 0]
            if len(explicit_indices):
                group_id += 1
                cleaning["explicit_timestamp_groups"] += 1
                cleaning["explicit_rows_before_cleaning"] += int(len(explicit_indices))
                extreme, rows = cleaned_group(explicit_indices, item, organic, event)
                if extreme:
                    cleaning["extreme_groups_removed"] += 1
                    cleaning["rows_removed_in_extreme_groups"] += int(len(explicit_indices))
                    local = group_end
                    continue
                cleaning["explicit_rows_after_source_aware_dedup"] += len(rows)
                cleaning["exact_duplicates_removed"] += int(len(explicit_indices)) - len(rows)

                mapped_by_dense: dict[int, dict[str, Any]] = {}
                unavailable_in_group: list[tuple[int, int]] = []
                for original, canonical, source in rows:
                    event_name = EVENT_NAMES[canonical]
                    target_events[split][event_name] += 1
                    source_target_events[split][
                        f"{event_name}|{'recommendation_driven' if source == 0 else 'organic'}"
                    ] += 1
                    dense = map_one(store, original)
                    if dense <= 0:
                        unavailable_targets[split][event_name] += 1
                        unavailable_in_group.append((original, canonical))
                        continue
                    position = mapped_by_dense.get(dense)
                    if position is None:
                        mapped_by_dense[dense] = {
                            "original": original,
                            "dense": dense,
                            "marks": [False] * len(EXPLICIT_NAMES),
                        }
                        position = mapped_by_dense[dense]
                    position["marks"][canonical - 1] = True

                eligible_history = len(explicit_eligibility) >= args.min_history_events
                if split in SPLITS and eligible_history and mapped_by_dense:
                    dense_history, mark_history, age_history, mask_history, source_history = (
                        history_arrays(history, group_time, args.history_len)
                    )
                    positives = list(mapped_by_dense.values())
                    dense_positive = [int(value["dense"]) for value in positives]
                    row = {
                        "uid": int(raw_uid),
                        "timestamp_seconds": group_time,
                        "timestamp_group_id": group_id,
                        "source_group_size": min(len(rows), np.iinfo(np.uint16).max),
                        "history_dense_item_ids": dense_history,
                        "history_event_type_ids": mark_history,
                        "history_age_seconds": age_history,
                        "history_mask": mask_history,
                        "positive_orig_item_ids": [int(value["original"]) for value in positives],
                        "positive_dense_item_ids": dense_positive,
                        "positive_sids": store.lookup_sids(
                            np.asarray(dense_positive, dtype=np.int64)
                        )
                        .astype(np.int16)
                        .tolist(),
                        "positive_event_type_mask": [value["marks"] for value in positives],
                    }
                    if args.include_source:
                        row["history_source_ids"] = source_history
                    writer.append(split, row)
                    if split == "train":
                        train_item_counts.update(dense_positive)

                if split in SPLITS and unavailable_in_group:
                    for original, canonical in unavailable_in_group:
                        unavailable_audit["uid"].append(int(raw_uid))
                        unavailable_audit["timestamp_seconds"].append(group_time)
                        unavailable_audit["timestamp_group_id"].append(group_id)
                        unavailable_audit["item_id"].append(original)
                        unavailable_audit["event_type_id"].append(canonical)

            if split not in ("gap_before_validation", "gap_before_test"):
                for idx in raw_group:
                    dense = map_one(store, int(item[idx]))
                    if dense <= 0:
                        continue
                    event_name = EVENT_NAMES[int(event[idx])]
                    if event_name == "listen" and not args.include_listen:
                        continue
                    history.append(
                        (
                            dense,
                            HISTORY_MARK_ID[event_name],
                            group_time,
                            1 if int(organic[idx]) == 0 else 2,
                        )
                    )
                    if event_name != "listen":
                        explicit_eligibility.append(
                            (
                                dense,
                                HISTORY_MARK_ID[event_name],
                                group_time,
                                1 if int(organic[idx]) == 0 else 2,
                            )
                        )
            local = group_end

    writer.close()
    catalog_dense = np.arange(1, store.catalog_size + 1, dtype=np.int64)
    train_dense = np.asarray(sorted(train_item_counts), dtype=np.int64)
    train_counts = np.asarray(
        [train_item_counts[int(value)] for value in train_dense], dtype=np.int64
    )
    np.save(output_dir / "catalog_dense_ids.npy", catalog_dense)
    np.save(output_dir / "train_dense_ids.npy", train_dense)
    np.savez(output_dir / "train_item_counts.npz", dense_item_id=train_dense, count=train_counts)
    np.savez(
        output_dir / "unavailable_target_audit.npz",
        uid=np.asarray(unavailable_audit["uid"], dtype=np.uint32),
        timestamp_seconds=np.asarray(unavailable_audit["timestamp_seconds"], dtype=np.uint32),
        timestamp_group_id=np.asarray(unavailable_audit["timestamp_group_id"], dtype=np.uint32),
        item_id=np.asarray(unavailable_audit["item_id"], dtype=np.int64),
        event_type_id=np.asarray(unavailable_audit["event_type_id"], dtype=np.int8),
    )
    codebook_fingerprints = recommendation_catalog_fingerprints(store, catalog_dense)
    source_sha256 = sha256_file(input_path)
    manifest: dict[str, Any] = {
        "format_version": 2,
        "created_from": {
            "source_events": str(input_path),
            "source_sha256": source_sha256,
            "codebook_dir": str(codebook_dir),
            "codebook_manifest": fingerprint(codebook_dir / "manifest.json", include_sha256=True),
        },
        "prepare_config": {
            "history_len": int(args.history_len),
            "min_history_events": int(args.min_history_events),
            "shard_rows": int(args.shard_rows),
            "max_users": int(args.max_users),
            "include_listen": bool(args.include_listen),
            "include_source": bool(args.include_source),
            "output_dir": str(output_dir),
        },
        "protocol": {
            "name": (
                "global_time_observed_explicit_item_ranking_v2_listen"
                if args.include_listen
                else "global_time_observed_explicit_item_ranking_v1"
            ),
            "intended_use": "supervised HPN training-chain validation only",
            "not_intended_use": [
                "not an exposure-aware recommendation-policy evaluation",
                "not a replay buffer",
                "not a user simulator",
                "not an action-conditioned feedback dataset",
            ],
            "temporal_split": bounds,
            "global_not_per_user_split": True,
            "gap_handling": "30-minute gaps have no targets and do not update history",
            "strict_tie_history": (
                "all targets in one timestamp group share only strictly earlier history"
            ),
            "explicit_deduplication_key": ["item_id", "event_type", "is_organic"],
            "extreme_group_rule": (
                "drop an explicit timestamp group with more than 50 distinct items"
            ),
            "target_definition": "next observed explicit item(s), irrespective of target source",
        },
        "data_contract": {
            "event_type_names": list(EXPLICIT_NAMES),
            "listen_included": bool(args.include_listen),
            "is_organic_used": bool(args.include_source),
            "history_mark_ids": "0 padding; 1 like; 2 dislike; 3 unlike; 4 undislike; 5 listen",
            "history_source_ids": "0 padding; 1 recommendation-driven; 2 organic",
            "sessionization": False,
            "adjacent_item_aggregation": False,
            "tie_policy": "shared_strict_pre_group_history",
            "target_identity": "observed interaction item, not verified recommendation exposure",
        },
        "mapping": {
            "store_format": store.mapping_format,
            "sid_levels": store.sid_levels,
            "semantic_levels": store.semantic_levels,
            "has_disambiguation_level": store.has_disambiguation_level,
            "sid_vocab_size_observed": store.sid_vocab_size_observed,
            "item_feature_dim": store.item_dim,
            "source_catalog_items": store.catalog_size,
            "candidate_catalog": (
                "all mapped static item codes; evaluation must stratify train-seen targets"
            ),
            "codebook_catalog_fingerprints": codebook_fingerprints,
        },
        "quality": as_json_counter(quality),
        "cleaning": as_json_counter(cleaning),
        "target_events_before_mapping": {
            split: as_json_counter(target_events[split]) for split in sorted(target_events)
        },
        "target_events_by_source_before_mapping": {
            split: as_json_counter(source_target_events[split])
            for split in sorted(source_target_events)
        },
        "unavailable_target_events": {
            split: as_json_counter(unavailable_targets[split])
            for split in sorted(unavailable_targets)
        },
        "splits": {split: {"rows": int(writer.row_counts[split])} for split in SPLITS},
        "train_catalog": {
            "unique_observed_target_items": int(train_dense.shape[0]),
            "item_counts_file": "train_item_counts.npz",
        },
    }
    atomic_json(output_dir / "manifest.json", manifest)
    print(output_dir / "manifest.json")


if __name__ == "__main__":
    main()
