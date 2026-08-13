#!/usr/bin/env python3
"""Convert the four exact Yambda-5B explicit streams to memory-safe arrays.

This is a lossless staging step for Phase 0, not cleaning.  It preserves every
uid, timestamp, item_id and is_organic value and keeps the original per-user
row boundaries.  Only one source Parquet reader is open at a time so the audit
fits the 2 GiB cgroup limit.
"""

from __future__ import annotations

import argparse
import json
import mmap
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "phase0_5b_audit" / "raw" / "sequential" / "5b"
DEFAULT_OUTPUT = PROJECT_ROOT / "phase0_5b_audit" / "work" / "flat_explicit_5b"
DEFAULT_MANIFEST = DEFAULT_OUTPUT / "manifest.json"
DEFAULT_PROGRESS = PROJECT_ROOT / "phase0_5b_audit" / "work" / "flat_progress.json"
EVENT_FILES = {
    "like": "likes.parquet",
    "dislike": "dislikes.parquet",
    "unlike": "unlikes.parquet",
    "undislike": "undislikes.parquet",
}
VALUE_COLUMNS = (
    ("timestamp", np.uint32),
    ("item_id", np.uint32),
    ("is_organic", np.uint8),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--batch-users", type=int, default=128)
    parser.add_argument(
        "--events",
        nargs="+",
        choices=tuple(EVENT_FILES),
        default=list(EVENT_FILES),
    )
    parser.add_argument("--progress-every-users", type=int, default=50_000)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def nested_value_count(parquet: pq.ParquetFile, column_name: str) -> int:
    suffix = f"{column_name}.list.element"
    total = 0
    found = False
    for row_group_index in range(parquet.metadata.num_row_groups):
        row_group = parquet.metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            if column.path_in_schema == suffix:
                total += int(column.num_values)
                found = True
    if not found:
        raise ValueError(f"nested column not found: {column_name}")
    return total


def flush_and_release(arrays: tuple[np.memmap, ...]) -> None:
    for array in arrays:
        array.flush()
        try:
            array._mmap.madvise(mmap.MADV_DONTNEED)  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass


def flatten_one(
    event_name: str,
    source_path: Path,
    output_dir: Path,
    batch_users: int,
    progress_every_users: int,
    progress_path: Path,
    started: float,
) -> dict[str, Any]:
    parquet = pq.ParquetFile(source_path)
    expected_columns = ("uid", "timestamp", "item_id", "is_organic")
    if tuple(parquet.schema_arrow.names) != expected_columns:
        raise ValueError(f"unexpected schema in {source_path}: {parquet.schema_arrow}")
    user_rows = int(parquet.metadata.num_rows)
    event_rows = nested_value_count(parquet, "timestamp")
    for name, _ in VALUE_COLUMNS[1:]:
        if nested_value_count(parquet, name) != event_rows:
            raise ValueError(f"nested column count mismatch in {source_path}: {name}")

    event_dir = output_dir / event_name
    event_dir.mkdir(parents=True, exist_ok=True)
    uid_out = np.lib.format.open_memmap(
        event_dir / "uid.npy", mode="w+", dtype=np.uint32, shape=(user_rows,)
    )
    offsets_out = np.lib.format.open_memmap(
        event_dir / "offsets.npy", mode="w+", dtype=np.uint64, shape=(user_rows + 1,)
    )
    value_outputs = {
        name: np.lib.format.open_memmap(
            event_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=(event_rows,)
        )
        for name, dtype in VALUE_COLUMNS
    }
    outputs = (uid_out, offsets_out, *value_outputs.values())
    offsets_out[0] = 0
    row_position = 0
    value_position = 0
    previous_uid = -1
    minimum_item: int | None = None
    maximum_item: int | None = None
    minimum_timestamp: int | None = None
    maximum_timestamp: int | None = None
    organic_rows = 0

    iterator = parquet.iter_batches(
        batch_size=batch_users,
        columns=list(expected_columns),
        use_threads=False,
    )
    for batch in iterator:
        uid = batch.column(0).to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
        if uid.size == 0:
            continue
        if int(uid[0]) <= previous_uid or np.any(np.diff(uid.astype(np.int64)) <= 0):
            raise ValueError(f"uid order violation in {source_path}")
        previous_uid = int(uid[-1])

        list_columns = [batch.column(index) for index in (1, 2, 3)]
        offset_arrays = [
            column.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            for column in list_columns
        ]
        normalized_offsets = [offset - int(offset[0]) for offset in offset_arrays]
        if not all(np.array_equal(normalized_offsets[0], other) for other in normalized_offsets[1:]):
            raise ValueError(f"nested offsets differ in {source_path}")
        batch_offsets = normalized_offsets[0]
        batch_event_rows = int(batch_offsets[-1])
        values = [
            column.values.to_numpy(zero_copy_only=False).astype(dtype, copy=False)
            for column, (_, dtype) in zip(list_columns, VALUE_COLUMNS)
        ]
        if any(value.size != batch_event_rows for value in values):
            raise ValueError(f"nested value length mismatch in {source_path}")
        timestamp, item_id, is_organic = values
        if np.any(is_organic > 1):
            raise ValueError(f"is_organic outside 0/1 in {source_path}")

        if batch_event_rows > 1:
            negative = np.diff(timestamp.astype(np.int64, copy=False)) < 0
            boundaries = batch_offsets[1:-1]
            if boundaries.size:
                negative[boundaries - 1] = False
            if bool(np.any(negative)):
                raise ValueError(f"timestamp order violation in {source_path}")

        next_row = row_position + int(uid.size)
        next_value = value_position + batch_event_rows
        uid_out[row_position:next_row] = uid
        offsets_out[row_position + 1 : next_row + 1] = (
            value_position + batch_offsets[1:].astype(np.uint64, copy=False)
        )
        for (name, _), value in zip(VALUE_COLUMNS, values):
            value_outputs[name][value_position:next_value] = value

        organic_rows += int(is_organic.sum(dtype=np.int64))
        batch_min_item = int(item_id.min())
        batch_max_item = int(item_id.max())
        batch_min_time = int(timestamp.min())
        batch_max_time = int(timestamp.max())
        minimum_item = batch_min_item if minimum_item is None else min(minimum_item, batch_min_item)
        maximum_item = batch_max_item if maximum_item is None else max(maximum_item, batch_max_item)
        minimum_timestamp = (
            batch_min_time if minimum_timestamp is None else min(minimum_timestamp, batch_min_time)
        )
        maximum_timestamp = (
            batch_max_time if maximum_timestamp is None else max(maximum_timestamp, batch_max_time)
        )
        row_position = next_row
        value_position = next_value

        if row_position % progress_every_users < uid.size:
            flush_and_release(outputs)
            progress = {
                "status": "running",
                "event": event_name,
                "users": row_position,
                "events": value_position,
                "elapsed_seconds": time.time() - started,
            }
            atomic_json(progress_path, progress)
            print(progress, flush=True)

    if row_position != user_rows or value_position != event_rows:
        raise RuntimeError(
            f"flatten count mismatch for {event_name}: "
            f"users={row_position}/{user_rows}, events={value_position}/{event_rows}"
        )
    flush_and_release(outputs)
    return {
        "event": event_name,
        "source": str(source_path.resolve()),
        "source_size_bytes": source_path.stat().st_size,
        "users": user_rows,
        "events": event_rows,
        "organic_events": organic_rows,
        "recommendation_driven_events": event_rows - organic_rows,
        "uid_min": int(uid_out[0]),
        "uid_max": int(uid_out[-1]),
        "item_id_min": minimum_item,
        "item_id_max": maximum_item,
        "timestamp_min": minimum_timestamp,
        "timestamp_max": maximum_timestamp,
        "output_files": {
            name: str((event_dir / name).resolve())
            for name in ("uid.npy", "offsets.npy", "timestamp.npy", "item_id.npy", "is_organic.npy")
        },
        "lossless_contract": (
            "all source rows and nested values retained; only Parquet encoding replaced "
            "by NumPy memory-mapped arrays"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.batch_users <= 0 or args.progress_every_users <= 0:
        raise ValueError("batch and progress settings must be positive")
    started = time.time()
    records: dict[str, Any] = {}
    for event_name in args.events:
        source_path = args.source_dir / EVENT_FILES[event_name]
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        records[event_name] = flatten_one(
            event_name,
            source_path,
            args.output_dir,
            args.batch_users,
            args.progress_every_users,
            args.progress,
            started,
        )
        print(json.dumps(records[event_name], ensure_ascii=False), flush=True)

    manifest = {
        "format_version": 1,
        "status": "complete" if set(args.events) == set(EVENT_FILES) else "partial",
        "created_for": "Yambda explicit-feedback Phase 0 memory-safe exact audit",
        "cleaning_or_filtering": False,
        "source_dir": str(args.source_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "batch_users": args.batch_users,
        "records": records,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.manifest, manifest)
    atomic_json(
        args.progress,
        {
            "status": manifest["status"],
            "events_completed": list(records),
            "elapsed_seconds": manifest["elapsed_seconds"],
            "manifest": str(args.manifest.resolve()),
        },
    )
    print(args.manifest.resolve(), flush=True)


if __name__ == "__main__":
    main()
