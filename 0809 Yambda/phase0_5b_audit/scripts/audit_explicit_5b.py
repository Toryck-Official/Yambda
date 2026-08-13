#!/usr/bin/env python3
"""Stream-audit all four Yambda-5B explicit-feedback sequences.

No row is removed or rewritten.  The four per-mark sequential Parquet streams
are merged by user, then events are grouped by timestamp.  Events sharing a
timestamp never receive an invented causal order.  Revision transitions are
counted only when the corresponding active state was established strictly
before the revision timestamp.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_0804_ROOT = Path("/root/autodl-tmp/0804 Yambda")
DEFAULT_EXPLICIT_DIR = PROJECT_ROOT / "phase0_5b_audit" / "raw" / "sequential" / "5b"
DEFAULT_CODEBOOK_DIR = (
    LEGACY_0804_ROOT
    / "dataprocess"
    / "artifacts"
    / "metadata_imputation_fixed_codebook_seed2026"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "phase0_5b_audit" / "work" / "explicit_audit.json"
DEFAULT_ITEM_OUTPUT = (
    PROJECT_ROOT / "phase0_5b_audit" / "artifacts" / "item_explicit_frequency.npz"
)
DEFAULT_PROGRESS = PROJECT_ROOT / "phase0_5b_audit" / "work" / "explicit_progress.json"
REPOSITORY = "yandex/yambda"
REVISION = "dd6f3a19eef5866e346c3270e098baa641a44948"
REMOTE_ROOT = f"hf://datasets/{REPOSITORY}@{REVISION}/sequential/5b"
OFFICIAL_RAW_CATALOG_ITEMS = 9_390_623
EVENT_SPECS = (
    (0, "like", "likes.parquet"),
    (1, "dislike", "dislikes.parquet"),
    (2, "unlike", "unlikes.parquet"),
    (3, "undislike", "undislikes.parquet"),
)
EVENT_NAMES = tuple(spec[1] for spec in EVENT_SPECS)
EVENT_COMBINATION_NAMES = tuple(
    "+".join(EVENT_NAMES[index] for index in range(4) if mask & (1 << index))
    for mask in range(16)
)
STATE_NAMES = (
    "neutral_or_left_censored",
    "liked",
    "disliked",
    "liked+disliked",
)
EXPECTED_SCHEMA_COLUMNS = {
    "likes": ("uid", "timestamp", "item_id", "is_organic"),
    "dislikes": ("uid", "timestamp", "item_id", "is_organic"),
    "unlikes": ("uid", "timestamp", "item_id", "is_organic"),
    "undislikes": ("uid", "timestamp", "item_id", "is_organic"),
    "listens": (
        "uid",
        "timestamp",
        "item_id",
        "is_organic",
        "played_ratio_pct",
        "track_length_seconds",
    ),
    "multi_event": (
        "uid",
        "timestamp",
        "item_id",
        "is_organic",
        "played_ratio_pct",
        "track_length_seconds",
        "event_type",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--explicit-dir", type=Path, default=DEFAULT_EXPLICIT_DIR)
    parser.add_argument(
        "--flat-dir",
        type=Path,
        default=None,
        help="optional lossless memory-mapped staging directory",
    )
    parser.add_argument("--codebook-dir", type=Path, default=DEFAULT_CODEBOOK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--item-output", type=Path, default=DEFAULT_ITEM_OUTPUT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--batch-users", type=int, default=4096)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--progress-every-users", type=int, default=10_000)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_npz_streaming(
    path: Path,
    arrays: Iterator[tuple[str, np.ndarray]],
) -> None:
    """Write an ordinary NPZ without retaining every payload array in RAM."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as archive:
        for name, value in arrays:
            array = np.asarray(value)
            with archive.open(f"{name}.npy", mode="w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, array, allow_pickle=False)
            del array
    temporary.replace(path)


class DiscreteHistogram:
    """Exact histogram for non-negative integer values on a fixed unit grid."""

    def __init__(self, unit: int = 1) -> None:
        self.unit = int(unit)
        # Every audited population is bounded by the 136,292,476 explicit
        # rows, so a bin cannot overflow uint32.  uint64 doubled resident
        # memory and exceeded this container's 2 GiB cgroup limit.
        self.counts = np.zeros(1, dtype=np.uint32)
        self.total = 0
        self.value_sum = 0
        self.minimum: int | None = None
        self.maximum: int | None = None

    def add(self, values: np.ndarray | list[int] | tuple[int, ...]) -> None:
        array = np.asarray(values, dtype=np.int64).reshape(-1)
        if not array.size:
            return
        if np.any(array < 0):
            raise ValueError("histogram values must be non-negative")
        if np.any(array % self.unit != 0):
            raise ValueError(f"values are not divisible by histogram unit {self.unit}")
        indices = array // self.unit
        maximum_index = int(indices.max())
        if maximum_index >= self.counts.size:
            # Grow geometrically.  The previous implementation called
            # np.bincount(indices) for every user.  For a single multi-year
            # delay that allocates an array with tens of millions of entries
            # even though only one bin changes.  Direct indexed accumulation
            # is exact and keeps the streaming audit proportional to the
            # number of observed values rather than the largest timestamp.
            expanded_size = max(
                maximum_index + 1,
                int(self.counts.size * 1.5) + 1,
            )
            expanded = np.zeros(expanded_size, dtype=np.uint32)
            expanded[: self.counts.size] = self.counts
            self.counts = expanded
        np.add.at(self.counts, indices, np.uint32(1))
        self.total += int(array.size)
        self.value_sum += int(array.sum(dtype=np.int64))
        current_min = int(array.min())
        current_max = int(array.max())
        self.minimum = current_min if self.minimum is None else min(self.minimum, current_min)
        self.maximum = current_max if self.maximum is None else max(self.maximum, current_max)

    def add_repeated(self, value: int, count: int) -> None:
        if count <= 0:
            return
        if value < 0 or value % self.unit:
            raise ValueError("invalid repeated histogram value")
        index = value // self.unit
        if index >= self.counts.size:
            self.counts = np.pad(self.counts, (0, index + 1 - self.counts.size))
        self.counts[index] += np.uint32(count)
        self.total += int(count)
        self.value_sum += int(value) * int(count)
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def quantile(self, probability: float) -> int | None:
        if self.total == 0:
            return None
        rank = max(1, int(math.ceil(float(probability) * self.total)))
        index = int(np.searchsorted(np.cumsum(self.counts), rank, side="left"))
        return index * self.unit

    def describe(self, scale: float = 1.0) -> dict[str, Any]:
        if self.total == 0:
            return {"count": 0, "unit": self.unit, "scale": scale}
        cumulative = np.cumsum(self.counts)

        def exact_quantile(probability: float) -> int:
            rank = max(1, int(math.ceil(float(probability) * self.total)))
            return int(np.searchsorted(cumulative, rank, side="left")) * self.unit

        result: dict[str, Any] = {
            "count": self.total,
            "min": float(self.minimum) / scale,
            "mean": float(self.value_sum) / self.total / scale,
            "median": float(exact_quantile(0.5)) / scale,
            "p75": float(exact_quantile(0.75)) / scale,
            "p90": float(exact_quantile(0.9)) / scale,
            "p95": float(exact_quantile(0.95)) / scale,
            "p99": float(exact_quantile(0.99)) / scale,
            "max": float(self.maximum) / scale,
            "unit": self.unit,
            "scale": scale,
            "quantile_rule": "nearest observed grid value at ceil(q*n)",
        }
        return result

    def sparse_counts(self) -> dict[str, int]:
        positions = np.flatnonzero(self.counts)
        return {
            str(int(position) * self.unit): int(self.counts[position])
            for position in positions
        }


@dataclass
class UserRow:
    uid: int
    timestamp: np.ndarray
    item_id: np.ndarray
    is_organic: np.ndarray


class ParquetUserStream:
    """Batch-stream one sequential Parquet row at a time without flattening it."""

    def __init__(self, path: Path, batch_users: int) -> None:
        self.path = path
        self.parquet = pq.ParquetFile(path)
        self.iterator: Iterator[pa.RecordBatch] = self.parquet.iter_batches(
            batch_size=batch_users,
            columns=["uid", "timestamp", "item_id", "is_organic"],
            use_threads=True,
        )
        self.batch: pa.RecordBatch | None = None
        self.batch_index = 0
        self.uid_values: np.ndarray | None = None
        self.list_offsets: list[np.ndarray] = []
        self.list_values: list[np.ndarray] = []
        self.previous_uid = -1
        self.current: UserRow | None = None
        self._advance()

    def _load_batch(self) -> bool:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.batch = None
            return False
        if batch.num_rows == 0:
            return self._load_batch()
        self.batch = batch
        self.batch_index = 0
        uid_column = batch.column(0)
        if uid_column.null_count:
            raise ValueError(f"null uid in {self.path}")
        self.uid_values = uid_column.to_numpy(zero_copy_only=False).astype(
            np.uint32, copy=False
        )
        self.list_offsets = []
        self.list_values = []
        expected_dtypes = (np.uint32, np.uint32, np.uint8)
        for column_index, dtype in zip((1, 2, 3), expected_dtypes):
            column = batch.column(column_index)
            if column.null_count or column.values.null_count:
                raise ValueError(f"null nested values in {self.path}")
            self.list_offsets.append(
                column.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            )
            self.list_values.append(
                column.values.to_numpy(zero_copy_only=False).astype(dtype, copy=False)
            )
        return True

    def _advance(self) -> None:
        if self.batch is None or self.batch_index >= self.batch.num_rows:
            if not self._load_batch():
                self.current = None
                return
        assert self.batch is not None and self.uid_values is not None
        index = self.batch_index
        uid = int(self.uid_values[index])
        if uid <= self.previous_uid:
            raise ValueError(f"uids are not strictly increasing in {self.path}: {uid}")
        self.previous_uid = uid
        arrays: list[np.ndarray] = []
        for offsets, values in zip(self.list_offsets, self.list_values):
            start = int(offsets[index])
            end = int(offsets[index + 1])
            arrays.append(values[start:end])
        if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
            raise ValueError(f"misaligned nested columns for uid={uid} in {self.path}")
        if arrays[0].size == 0:
            raise ValueError(f"empty sequential row for uid={uid} in {self.path}")
        if np.any(np.diff(arrays[0].astype(np.int64, copy=False)) < 0):
            raise ValueError(f"timestamps are not sorted for uid={uid} in {self.path}")
        if np.any(arrays[2] > 1):
            raise ValueError(f"is_organic outside 0/1 for uid={uid} in {self.path}")
        self.current = UserRow(uid, arrays[0], arrays[1], arrays[2])
        self.batch_index += 1

    def pop(self) -> UserRow:
        if self.current is None:
            raise StopIteration
        result = self.current
        self._advance()
        return result


class FlatUserStream:
    """Read a losslessly staged per-user stream with negligible resident RAM."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.uid_values = np.load(path / "uid.npy", mmap_mode="r")
        self.offsets = np.load(path / "offsets.npy", mmap_mode="r")
        self.timestamp_values = np.load(path / "timestamp.npy", mmap_mode="r")
        self.item_values = np.load(path / "item_id.npy", mmap_mode="r")
        self.organic_values = np.load(path / "is_organic.npy", mmap_mode="r")
        if self.uid_values.ndim != 1 or self.offsets.shape != (self.uid_values.size + 1,):
            raise ValueError(f"invalid staged uid/offset shape in {path}")
        event_rows = int(self.timestamp_values.size)
        if not (
            self.item_values.shape == (event_rows,)
            and self.organic_values.shape == (event_rows,)
            and int(self.offsets[0]) == 0
            and int(self.offsets[-1]) == event_rows
        ):
            raise ValueError(f"invalid staged value shapes in {path}")
        if self.uid_values.size and np.any(
            np.diff(self.uid_values.astype(np.int64, copy=False)) <= 0
        ):
            raise ValueError(f"staged uids are not strictly increasing in {path}")
        self.user_rows = int(self.uid_values.size)
        self.event_rows = event_rows
        self.index = 0
        self.current: UserRow | None = None
        self._advance()

    def _advance(self) -> None:
        if self.index >= self.user_rows:
            self.current = None
            return
        start = int(self.offsets[self.index])
        end = int(self.offsets[self.index + 1])
        if end <= start:
            raise ValueError(f"empty staged sequential row at index={self.index}")
        timestamp = self.timestamp_values[start:end]
        item_id = self.item_values[start:end]
        is_organic = self.organic_values[start:end]
        if np.any(np.diff(timestamp.astype(np.int64, copy=False)) < 0):
            raise ValueError(f"staged timestamps are not sorted at index={self.index}")
        if np.any(is_organic > 1):
            raise ValueError(f"staged is_organic outside 0/1 at index={self.index}")
        self.current = UserRow(
            int(self.uid_values[self.index]),
            timestamp,
            item_id,
            is_organic,
        )
        self.index += 1

    def pop(self) -> UserRow:
        if self.current is None:
            raise StopIteration
        result = self.current
        self._advance()
        return result


def remote_metadata() -> dict[str, Any]:
    fs = HfFileSystem()
    records: dict[str, Any] = {}
    for name in ("listens", "likes", "dislikes", "unlikes", "undislikes", "multi_event"):
        path = f"{REMOTE_ROOT}/{name}.parquet"
        info = fs.info(path)
        with fs.open(path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            schema_names = tuple(parquet.schema_arrow.names)
            if schema_names != EXPECTED_SCHEMA_COLUMNS[name]:
                raise ValueError(f"unexpected remote schema for {name}: {schema_names}")
            leaf_counts: Counter[str] = Counter()
            compressed_sizes: Counter[str] = Counter()
            for row_group_index in range(parquet.metadata.num_row_groups):
                row_group = parquet.metadata.row_group(row_group_index)
                for column_index in range(row_group.num_columns):
                    column = row_group.column(column_index)
                    leaf_counts[column.path_in_schema] += int(column.num_values)
                    compressed_sizes[column.path_in_schema] += int(
                        column.total_compressed_size
                    )
            records[name] = {
                "path": path,
                "size_bytes": int(info["size"]),
                "sha256": getattr(info.get("lfs"), "sha256", None),
                "user_rows": int(parquet.metadata.num_rows),
                "row_groups": int(parquet.metadata.num_row_groups),
                "schema": str(parquet.schema_arrow),
                "nested_value_counts": dict(leaf_counts),
                "compressed_column_bytes": dict(compressed_sizes),
            }
    return records


def ensure_item_capacity(counts: np.ndarray, maximum: int) -> np.ndarray:
    if maximum < counts.shape[1]:
        return counts
    expanded_size = max(maximum + 1, int(counts.shape[1] * 1.2))
    expanded = np.zeros((counts.shape[0], expanded_size), dtype=counts.dtype)
    expanded[:, : counts.shape[1]] = counts
    return expanded


def threshold_add(
    destination: dict[str, dict[str, int]],
    values: np.ndarray,
    thresholds: tuple[int, ...],
) -> None:
    for threshold in thresholds:
        mask = values >= threshold
        entry = destination[str(threshold)]
        entry["groups"] += int(np.count_nonzero(mask))
        entry["event_rows"] += int(values[mask].sum(dtype=np.int64))


def main() -> None:
    args = parse_args()
    if args.batch_users <= 0 or args.max_users < 0 or args.progress_every_users <= 0:
        raise ValueError("invalid positive batch/progress settings")
    started = time.time()
    metadata = remote_metadata()

    raw_count = int(metadata["multi_event"]["nested_value_counts"]["timestamp.list.element"])
    source_event_counts = {
        name[:-1] if name.endswith("s") else name: int(
            metadata[name]["nested_value_counts"]["timestamp.list.element"]
        )
        for name in ("likes", "dislikes", "unlikes", "undislikes")
    }
    # Avoid the incorrect singular "dislike" transformation for undislikes.
    source_event_counts = {
        "like": int(metadata["likes"]["nested_value_counts"]["timestamp.list.element"]),
        "dislike": int(metadata["dislikes"]["nested_value_counts"]["timestamp.list.element"]),
        "unlike": int(metadata["unlikes"]["nested_value_counts"]["timestamp.list.element"]),
        "undislike": int(metadata["undislikes"]["nested_value_counts"]["timestamp.list.element"]),
    }
    listen_count = int(
        metadata["listens"]["nested_value_counts"]["timestamp.list.element"]
    )
    explicit_count_from_metadata = sum(source_event_counts.values())
    if listen_count + explicit_count_from_metadata != raw_count:
        raise RuntimeError("remote metadata counts do not reconcile")

    local_paths = {
        event_name: args.explicit_dir / filename
        for _, event_name, filename in EVENT_SPECS
    }
    for path in local_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    mapping_path = args.codebook_dir / "orig2catalog_row.npy"
    catalog_path = args.codebook_dir / "catalog_item_ids.npy"
    if not mapping_path.is_file() or not catalog_path.is_file():
        raise FileNotFoundError("current frozen SID mapping files are missing")
    sid_mapping = np.load(mapping_path, mmap_mode="r")
    sid_catalog_items = np.load(catalog_path, mmap_mode="r")
    if sid_mapping.ndim != 1 or sid_catalog_items.ndim != 1:
        raise ValueError("invalid SID mapping shapes")

    if args.flat_dir is None:
        streams = {
            event_name: ParquetUserStream(local_paths[event_name], args.batch_users)
            for _, event_name, _ in EVENT_SPECS
        }
        staging_contract = "direct Parquet streams"
    else:
        manifest_path = args.flat_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        flat_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if flat_manifest.get("status") != "complete":
            raise ValueError(f"flat staging manifest is not complete: {manifest_path}")
        streams = {
            event_name: FlatUserStream(args.flat_dir / event_name)
            for _, event_name, _ in EVENT_SPECS
        }
        for _, event_name, filename in EVENT_SPECS:
            remote_name = filename.removesuffix(".parquet")
            stream = streams[event_name]
            if (
                stream.user_rows != int(metadata[remote_name]["user_rows"])
                or stream.event_rows != source_event_counts[event_name]
            ):
                raise ValueError(f"flat staging count mismatch for {event_name}")
        staging_contract = (
            f"lossless NumPy memory maps from {manifest_path.resolve()}; "
            "no row filtering or value transformation"
        )
    # Full Yambda-5B item ids are one-based and the observed maximum equals
    # the official catalog cardinality (9,390,623).  Allocate the inclusive
    # id domain; cardinality is not itself a valid zero-based array length.
    item_count_shape = (
        max(OFFICIAL_RAW_CATALOG_ITEMS + 1, int(sid_mapping.shape[0])),
    )
    item_count_work_path = args.item_output.with_suffix(".counts.uint32.mmap")
    item_count_work_path.parent.mkdir(parents=True, exist_ok=True)
    item_counts = np.memmap(
        item_count_work_path,
        mode="w+",
        dtype=np.uint32,
        shape=item_count_shape,
    )
    item_counts[:] = 0

    sequence_length = DiscreteHistogram()
    active_span = DiscreteHistogram(unit=5)
    inter_group_gap = DiscreteHistogram(unit=5)
    raw_adjacent_gap = DiscreteHistogram(unit=5)
    timestamp_group_size = DiscreteHistogram()
    like_unlike_delay = DiscreteHistogram(unit=5)
    dislike_undislike_delay = DiscreteHistogram(unit=5)

    scanned_by_type = Counter()
    organic_by_type = Counter()
    exact_duplicate_by_type = Counter()
    timestamp_combo_counts = Counter()
    user_item_timestamp_combo_counts = Counter()
    user_item_timestamp_combo_rows = Counter()
    state_before_event_groups = Counter()
    state_before_event_rows = Counter()
    transition_counts = Counter()
    anomaly_counts = Counter()
    users_by_type = Counter()
    large_thresholds = {
        str(value): {"groups": 0, "event_rows": 0}
        for value in (2, 5, 10, 20, 50, 100, 500, 1000, 5000)
    }
    top_groups: list[tuple[int, int, int, int, tuple[int, int, int, int]]] = []
    explicit_users = 0
    explicit_events_scanned = 0
    uncovered_sid_users = 0
    timestamp_groups = 0
    events_in_multi_groups = 0
    exact_duplicate_excess = 0
    conflict_user_item_timestamp_groups = 0
    conflict_event_rows = 0
    timestamps_not_multiple_of_five = 0

    while True:
        active_streams = [stream for stream in streams.values() if stream.current is not None]
        if not active_streams:
            break
        if args.max_users and explicit_users >= args.max_users:
            break
        uid = min(int(stream.current.uid) for stream in active_streams if stream.current)
        timestamps_parts: list[np.ndarray] = []
        items_parts: list[np.ndarray] = []
        types_parts: list[np.ndarray] = []
        organic_parts: list[np.ndarray] = []
        for event_id, event_name, _ in EVENT_SPECS:
            stream = streams[event_name]
            if stream.current is None or stream.current.uid != uid:
                continue
            row = stream.pop()
            count = int(row.timestamp.size)
            users_by_type[event_name] += 1
            scanned_by_type[event_name] += count
            organic_by_type[event_name] += int(row.is_organic.sum(dtype=np.int64))
            timestamps_parts.append(row.timestamp)
            items_parts.append(row.item_id)
            organic_parts.append(row.is_organic)
            types_parts.append(np.full(count, event_id, dtype=np.uint8))
            maximum_item = int(row.item_id.max())
            if maximum_item >= item_counts.size:
                raise ValueError(
                    f"item_id={maximum_item} exceeds audited catalog bound "
                    f"{item_counts.size - 1}"
                )
            np.add.at(item_counts, row.item_id, np.uint32(1))

        timestamp = np.concatenate(timestamps_parts).astype(np.uint32, copy=False)
        item_id = np.concatenate(items_parts).astype(np.uint32, copy=False)
        event_type = np.concatenate(types_parts).astype(np.uint8, copy=False)
        is_organic = np.concatenate(organic_parts).astype(np.uint8, copy=False)
        count = int(timestamp.size)
        if not (count == item_id.size == event_type.size == is_organic.size):
            raise RuntimeError(f"merged arrays differ for uid={uid}")
        order = np.lexsort((event_type, item_id, timestamp))
        timestamp = timestamp[order]
        item_id = item_id[order]
        event_type = event_type[order]
        is_organic = is_organic[order]
        del is_organic

        explicit_users += 1
        explicit_events_scanned += count
        sequence_length.add([count])
        timestamps_not_multiple_of_five += int(np.count_nonzero(timestamp % 5))
        if int(item_id.max()) >= sid_mapping.shape[0]:
            covered = np.zeros(count, dtype=bool)
            in_range = item_id < sid_mapping.shape[0]
            covered[in_range] = sid_mapping[item_id[in_range]] >= 0
        else:
            covered = sid_mapping[item_id] >= 0
        if not bool(np.all(covered)):
            uncovered_sid_users += 1

        group_start_mask = np.empty(count, dtype=bool)
        group_start_mask[0] = True
        group_start_mask[1:] = timestamp[1:] != timestamp[:-1]
        group_starts = np.flatnonzero(group_start_mask)
        group_ends = np.r_[group_starts[1:], count]
        group_sizes = (group_ends - group_starts).astype(np.int64, copy=False)
        group_timestamps = timestamp[group_starts]
        timestamp_group_size.add(group_sizes)
        timestamp_groups += int(group_sizes.size)
        multi_mask = group_sizes > 1
        events_in_multi_groups += int(group_sizes[multi_mask].sum(dtype=np.int64))
        threshold_add(large_thresholds, group_sizes, tuple(int(k) for k in large_thresholds))
        active_span.add([int(group_timestamps[-1]) - int(group_timestamps[0])])
        if group_timestamps.size > 1:
            strict_gaps = np.diff(group_timestamps.astype(np.int64, copy=False))
            inter_group_gap.add(strict_gaps)
            raw_adjacent_gap.add(strict_gaps)
        raw_adjacent_gap.add_repeated(0, count - int(group_sizes.size))

        duplicate = (
            (timestamp[1:] == timestamp[:-1])
            & (item_id[1:] == item_id[:-1])
            & (event_type[1:] == event_type[:-1])
        )
        exact_duplicate_excess += int(np.count_nonzero(duplicate))
        for event_id, event_name, _ in EVENT_SPECS:
            exact_duplicate_by_type[event_name] += int(
                np.count_nonzero(duplicate & (event_type[1:] == event_id))
            )

        pair_start_mask = np.empty(count, dtype=bool)
        pair_start_mask[0] = True
        pair_start_mask[1:] = (
            (timestamp[1:] != timestamp[:-1]) | (item_id[1:] != item_id[:-1])
        )
        pair_starts = np.flatnonzero(pair_start_mask)
        pair_ends = np.r_[pair_starts[1:], count]
        type_bits = np.left_shift(np.uint16(1), event_type.astype(np.uint16))
        pair_type_masks = np.bitwise_or.reduceat(type_bits, pair_starts)
        group_pair_positions = np.searchsorted(pair_starts, group_starts)
        distinct_items_per_group = np.diff(np.r_[group_pair_positions, pair_starts.size])
        group_type_bits = np.bitwise_or.reduceat(type_bits, group_starts)
        combo_hist = np.bincount(group_type_bits.astype(np.int64), minlength=16)
        for mask, value in enumerate(combo_hist.tolist()):
            if value:
                names = "+".join(
                    EVENT_NAMES[index]
                    for index in range(4)
                    if mask & (1 << index)
                )
                timestamp_combo_counts[names] += int(value)

        for group_index in np.flatnonzero(
            group_sizes >= (top_groups[0][0] if len(top_groups) >= 100 else 1)
        ):
            start = int(group_starts[group_index])
            end = int(group_ends[group_index])
            type_counts = tuple(
                int(value)
                for value in np.bincount(event_type[start:end], minlength=4)[:4]
            )
            record = (
                int(group_sizes[group_index]),
                uid,
                int(group_timestamps[group_index]),
                int(distinct_items_per_group[group_index]),
                type_counts,
            )
            if len(top_groups) < 100:
                heapq.heappush(top_groups, record)
            elif record > top_groups[0]:
                heapq.heapreplace(top_groups, record)

        active_like: dict[int, int] = {}
        active_dislike: dict[int, int] = {}
        for start, end, pair_mask_value in zip(
            pair_starts.tolist(), pair_ends.tolist(), pair_type_masks.tolist(), strict=True
        ):
            raw_rows = int(end - start)
            item = int(item_id[start])
            current_time = int(timestamp[start])
            pair_mask = int(pair_mask_value)
            pair_names = EVENT_COMBINATION_NAMES[pair_mask]
            user_item_timestamp_combo_counts[pair_names] += 1
            user_item_timestamp_combo_rows[pair_names] += raw_rows
            # Events are sorted by (timestamp, item, type), so the first and
            # last mark differ iff this user-item-timestamp has a conflict.
            first_mark = int(event_type[start])
            if first_mark != int(event_type[end - 1]):
                conflict_user_item_timestamp_groups += 1
                conflict_event_rows += raw_rows
                anomaly_counts["ambiguous_state_groups_not_applied"] += 1
                continue
            mark = first_mark
            state_code = int(item in active_like) + 2 * int(item in active_dislike)
            state_name = STATE_NAMES[state_code]
            transition_key = f"{state_name}->{EVENT_NAMES[mark]}"
            state_before_event_groups[transition_key] += 1
            state_before_event_rows[transition_key] += raw_rows
            if mark == 0:  # like
                if item in active_like:
                    transition_counts["repeated_like_groups"] += 1
                    transition_counts["repeated_like_rows"] += raw_rows
                else:
                    active_like[item] = current_time
                if item in active_dislike:
                    transition_counts["like_while_disliked_groups"] += 1
            elif mark == 1:  # dislike
                if item in active_dislike:
                    transition_counts["repeated_dislike_groups"] += 1
                    transition_counts["repeated_dislike_rows"] += raw_rows
                else:
                    active_dislike[item] = current_time
                if item in active_like:
                    transition_counts["dislike_while_liked_groups"] += 1
            elif mark == 2:  # unlike
                origin = active_like.pop(item, None)
                if origin is None:
                    transition_counts["unlike_without_active_prior_like_groups"] += 1
                    transition_counts["unlike_without_active_prior_like_rows"] += raw_rows
                else:
                    delay = current_time - origin
                    if delay <= 0:
                        raise RuntimeError("non-positive strict like->unlike delay")
                    transition_counts["like_to_unlike_pairs"] += 1
                    transition_counts["like_to_unlike_revision_rows"] += raw_rows
                    like_unlike_delay.add([delay])
            else:  # undislike
                origin = active_dislike.pop(item, None)
                if origin is None:
                    transition_counts["undislike_without_active_prior_dislike_groups"] += 1
                    transition_counts["undislike_without_active_prior_dislike_rows"] += raw_rows
                else:
                    delay = current_time - origin
                    if delay <= 0:
                        raise RuntimeError("non-positive strict dislike->undislike delay")
                    transition_counts["dislike_to_undislike_pairs"] += 1
                    transition_counts["dislike_to_undislike_revision_rows"] += raw_rows
                    dislike_undislike_delay.add([delay])

        if explicit_users % args.progress_every_users == 0:
            item_counts.flush()
            progress = {
                "status": "running",
                "users": explicit_users,
                "events": explicit_events_scanned,
                "last_uid": uid,
                "elapsed_seconds": time.time() - started,
                "max_users": args.max_users,
            }
            atomic_json(args.progress, progress)
            print(progress, flush=True)

    full_scan = args.max_users == 0
    if full_scan:
        if explicit_events_scanned != explicit_count_from_metadata:
            raise RuntimeError(
                f"scanned {explicit_events_scanned} explicit events, metadata says "
                f"{explicit_count_from_metadata}"
            )
        if dict(scanned_by_type) != source_event_counts:
            raise RuntimeError(
                f"per-type scan counts {dict(scanned_by_type)} differ from metadata "
                f"{source_event_counts}"
            )

    item_counts.flush()
    explicit_item_ids = np.flatnonzero(item_counts).astype(np.uint32, copy=False)
    explicit_item_counts = item_counts[explicit_item_ids]
    sid_covered_item = np.zeros(explicit_item_ids.size, dtype=bool)
    in_mapping = explicit_item_ids < sid_mapping.shape[0]
    sid_covered_item[in_mapping] = sid_mapping[explicit_item_ids[in_mapping]] >= 0
    sid_covered_events = int(explicit_item_counts[sid_covered_item].sum(dtype=np.uint64))
    sid_uncovered_events = int(explicit_item_counts[~sid_covered_item].sum(dtype=np.uint64))

    def item_arrays() -> Iterator[tuple[str, np.ndarray]]:
        yield "item_id", explicit_item_ids
        yield "total_count", explicit_item_counts
        yield "sid_covered", sid_covered_item

    write_npz_streaming(args.item_output, item_arrays())

    top_indices = np.argsort(explicit_item_counts)[-100:][::-1]
    top_items = [
        {
            "item_id": int(explicit_item_ids[index]),
            "total": int(explicit_item_counts[index]),
            "sid_covered": bool(sid_covered_item[index]),
        }
        for index in top_indices
    ]
    top_group_records = [
        {
            "size": size,
            "uid": uid,
            "timestamp": timestamp,
            "distinct_items": distinct_items,
            "feedback_counts": dict(zip(EVENT_NAMES, type_counts)),
        }
        for size, uid, timestamp, distinct_items, type_counts in sorted(
            top_groups, reverse=True
        )
    ]

    item_frequency_hist = DiscreteHistogram()
    item_frequency_hist.add(explicit_item_counts.astype(np.int64, copy=False))
    report: dict[str, Any] = {
        "format_version": 1,
        "status": "complete" if full_scan else "partial_smoke",
        "scope": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "explicit_dir": str(args.explicit_dir.resolve()),
            "included_events": list(EVENT_NAMES),
            "listen_removed_from_explicit_analysis": True,
            "is_organic_filtering": False,
            "sessionization": False,
            "cleaning_or_row_deletion": False,
            "same_timestamp_policy": "shared group; no invented within-group order",
            "revision_policy": (
                "state must be active at a strictly earlier timestamp; conflicting "
                "user-item-timestamp groups are counted but not applied"
            ),
            "streaming": True,
            "staging_contract": staging_contract,
            "batch_users": args.batch_users,
            "max_users": args.max_users,
            "item_count_work_path": str(item_count_work_path.resolve()),
        },
        "remote_dataset": {
            "files": metadata,
            "raw_event_count": raw_count,
            "raw_unique_users": int(metadata["multi_event"]["user_rows"]),
            "raw_unique_items_official_reference": OFFICIAL_RAW_CATALOG_ITEMS,
            "raw_unique_items_provenance": (
                "Yambda paper Table 3; independent listen-item scan is a separate Phase 0 "
                "artifact because the item_id column alone is 14,039,281,312 compressed bytes"
            ),
            "listen_count": listen_count,
            "explicit_count_after_removing_listen": explicit_count_from_metadata,
            "event_counts": {"listen": listen_count, **source_event_counts},
            "count_reconciliation": (
                listen_count + explicit_count_from_metadata == raw_count
            ),
        },
        "explicit_scan": {
            "users": explicit_users,
            "events": explicit_events_scanned,
            "items": int(explicit_item_ids.size),
            "event_counts": dict(scanned_by_type),
            "users_by_feedback": dict(users_by_type),
            "organic_counts": dict(organic_by_type),
            "recommendation_driven_counts": {
                name: int(scanned_by_type[name] - organic_by_type[name])
                for name in EVENT_NAMES
            },
            "explicit_item_fraction_of_raw_catalog": (
                float(explicit_item_ids.size) / OFFICIAL_RAW_CATALOG_ITEMS
            ),
            "sequence_length": sequence_length.describe(),
            "active_span_seconds": active_span.describe(),
            "active_span_days": active_span.describe(scale=86400.0),
            "strict_positive_inter_group_gap_seconds": inter_group_gap.describe(),
            "strict_positive_inter_group_gap_hours": inter_group_gap.describe(
                scale=3600.0
            ),
            "raw_adjacent_inter_event_gap_seconds_including_ties": (
                raw_adjacent_gap.describe()
            ),
            "raw_adjacent_inter_event_gap_hours_including_ties": (
                raw_adjacent_gap.describe(scale=3600.0)
            ),
            "timestamp_groups": timestamp_groups,
            "timestamp_group_size": timestamp_group_size.describe(),
            "timestamp_group_size_histogram": timestamp_group_size.sparse_counts(),
            "events_in_multi_event_groups": events_in_multi_groups,
            "events_in_multi_event_groups_fraction": (
                float(events_in_multi_groups) / explicit_events_scanned
                if explicit_events_scanned
                else 0.0
            ),
            "same_timestamp_feedback_combinations": dict(timestamp_combo_counts),
            "same_user_item_timestamp_feedback_combinations": {
                "groups": dict(user_item_timestamp_combo_counts),
                "event_rows": dict(user_item_timestamp_combo_rows),
            },
            "large_timestamp_groups": large_thresholds,
            "largest_timestamp_groups": top_group_records,
            "timestamps_not_multiple_of_five": timestamps_not_multiple_of_five,
            "exact_duplicate_excess_rows": exact_duplicate_excess,
            "exact_duplicate_excess_by_feedback": dict(exact_duplicate_by_type),
            "conflicting_user_item_timestamp_groups": (
                conflict_user_item_timestamp_groups
            ),
            "conflicting_event_rows": conflict_event_rows,
            "transition_counts": dict(transition_counts),
            "state_before_event_counts": {
                "groups": dict(state_before_event_groups),
                "event_rows": dict(state_before_event_rows),
            },
            "transition_anomalies": dict(anomaly_counts),
            "like_to_unlike_delay_seconds": like_unlike_delay.describe(),
            "like_to_unlike_delay_days": like_unlike_delay.describe(scale=86400.0),
            "dislike_to_undislike_delay_seconds": dislike_undislike_delay.describe(),
            "dislike_to_undislike_delay_days": dislike_undislike_delay.describe(
                scale=86400.0
            ),
            "item_explicit_frequency": item_frequency_hist.describe(),
            "top_items": top_items,
        },
        "sid_coverage": {
            "mapping_path": str(mapping_path.resolve()),
            "mapping_sha256": sha256_file(mapping_path),
            "current_sid_catalog_items": int(sid_catalog_items.size),
            "covered_explicit_items": int(np.count_nonzero(sid_covered_item)),
            "uncovered_explicit_items": int(np.count_nonzero(~sid_covered_item)),
            "item_coverage_fraction": (
                float(np.count_nonzero(sid_covered_item)) / explicit_item_ids.size
                if explicit_item_ids.size
                else 0.0
            ),
            "covered_explicit_events": sid_covered_events,
            "uncovered_explicit_events": sid_uncovered_events,
            "event_coverage_fraction": (
                float(sid_covered_events) / explicit_events_scanned
                if explicit_events_scanned
                else 0.0
            ),
            "users_with_at_least_one_uncovered_event": uncovered_sid_users,
            "user_coverage_fraction": (
                float(explicit_users - uncovered_sid_users) / explicit_users
                if explicit_users
                else 0.0
            ),
            "important_scope_note": (
                "This is coverage of the existing 50M-derived frozen SID mapping over "
                "the full 5B explicit-feedback item universe."
            ),
        },
        "supporting_artifacts": {
            "item_frequency_npz": str(args.item_output.resolve()),
            "item_frequency_npz_sha256": sha256_file(args.item_output),
        },
        "runtime": {
            "elapsed_seconds": time.time() - started,
        },
    }
    atomic_json(args.output, report)
    atomic_json(
        args.progress,
        {
            "status": "complete" if full_scan else "partial_smoke",
            "users": explicit_users,
            "events": explicit_events_scanned,
            "elapsed_seconds": time.time() - started,
            "output": str(args.output.resolve()),
        },
    )
    print(json.dumps(args.progress.read_text(encoding="utf-8")), flush=True)
    print(str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
