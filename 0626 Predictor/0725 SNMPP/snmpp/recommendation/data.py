"""Prepared group-level data loading for no-listen recommendation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from snmpp.constants import EVENT_TYPE_NAMES

SPLIT_NAMES = ("train", "validation", "test")
LEGACY_MAPPING_FILES = (
    "orig2dense_item_id.npy",
    "dense2orig_item_id.npy",
    "dense_item2sid.npy",
    "dense_item_features.npy",
)
CATALOG_MAPPING_FILES = (
    "orig2catalog_row.npy",
    "catalog_item_ids.npy",
    "semantic_codes.npy",
    "catalog_features.npy",
)


class RQCodebookStore:
    """Memory-mapped item features and residual-quantization assignments.

    Two on-disk contracts are supported:

    * ``legacy_padded`` stores a physical zero row and an original-to-dense map.
    * ``catalog_row_v1`` stores real items only.  Its public dense ids are the
      zero-based catalog rows plus one, so zero remains reserved for padding or
      an unavailable item throughout the recommendation stack.

    A catalog artifact may contain ``full_codes.npy``.  In that case the first
    four tokens remain semantic RQ codes and the final token is a deterministic
    disambiguation suffix.  Recommendation uses the full path so two different
    items never silently become the same target.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        legacy_complete = all((self.root / name).is_file() for name in LEGACY_MAPPING_FILES)
        catalog_complete = all((self.root / name).is_file() for name in CATALOG_MAPPING_FILES)
        if legacy_complete and catalog_complete:
            raise ValueError(
                "RQ codebook store ambiguously contains both legacy and catalog-row contracts"
            )
        if legacy_complete:
            self.mapping_format = "legacy_padded"
            self._orig_to_dense = np.load(self.root / "orig2dense_item_id.npy", mmap_mode="r")
            self._catalog_item_ids = np.load(self.root / "dense2orig_item_id.npy", mmap_mode="r")
            self._semantic_codes = np.load(self.root / "dense_item2sid.npy", mmap_mode="r")
            self._full_codes = self._semantic_codes
            self._features = np.load(self.root / "dense_item_features.npy", mmap_mode="r")
            self._physical_padding_row = True
            self._validate_legacy()
        elif catalog_complete:
            self.mapping_format = "catalog_row_v1"
            self._orig_to_dense = np.load(self.root / "orig2catalog_row.npy", mmap_mode="r")
            self._catalog_item_ids = np.load(self.root / "catalog_item_ids.npy", mmap_mode="r")
            self._semantic_codes = np.load(self.root / "semantic_codes.npy", mmap_mode="r")
            full_codes_path = self.root / "full_codes.npy"
            self._full_codes = (
                np.load(full_codes_path, mmap_mode="r")
                if full_codes_path.is_file()
                else self._semantic_codes
            )
            self._features = np.load(self.root / "catalog_features.npy", mmap_mode="r")
            self._physical_padding_row = False
            self._validate_catalog()
        else:
            legacy_missing = [
                name for name in LEGACY_MAPPING_FILES if not (self.root / name).is_file()
            ]
            catalog_missing = [
                name for name in CATALOG_MAPPING_FILES if not (self.root / name).is_file()
            ]
            raise FileNotFoundError(
                "RQ codebook store matches neither supported contract; "
                f"legacy missing={legacy_missing}, catalog missing={catalog_missing}"
            )

    def _validate_legacy(self) -> None:
        if self._orig_to_dense.ndim != 1 or self._catalog_item_ids.ndim != 1:
            raise ValueError("legacy item-id mapping arrays must be one-dimensional")
        if self._semantic_codes.ndim != 2:
            raise ValueError("dense_item2sid.npy must be two-dimensional")
        if self._features.ndim != 2:
            raise ValueError("dense_item_features.npy must be two-dimensional")
        expected = int(self._catalog_item_ids.shape[0])
        if expected == 0:
            raise ValueError("legacy store must contain the reserved zero row")
        if self._semantic_codes.shape[0] != expected:
            raise ValueError("dense_item2sid and dense2orig row counts differ")
        if self._features.shape[0] != expected:
            raise ValueError("dense_item_features and dense2orig row counts differ")

    def _validate_catalog(self) -> None:
        if self._orig_to_dense.ndim != 1 or self._catalog_item_ids.ndim != 1:
            raise ValueError("catalog item-id mapping arrays must be one-dimensional")
        if self._semantic_codes.ndim != 2 or self._full_codes.ndim != 2:
            raise ValueError("semantic and full code arrays must be two-dimensional")
        if self._features.ndim != 2:
            raise ValueError("catalog_features.npy must be two-dimensional")
        expected = int(self._catalog_item_ids.shape[0])
        if expected == 0:
            raise ValueError("catalog-row store must contain at least one item")
        for name, array in (
            ("semantic_codes.npy", self._semantic_codes),
            ("full_codes.npy", self._full_codes),
            ("catalog_features.npy", self._features),
        ):
            if int(array.shape[0]) != expected:
                raise ValueError(f"{name} and catalog_item_ids.npy row counts differ")
        if self._full_codes.shape[1] < self._semantic_codes.shape[1]:
            raise ValueError("full codes cannot have fewer levels than semantic codes")
        item_ids = np.asarray(self._catalog_item_ids, dtype=np.int64)
        if np.any(item_ids < 0) or np.any(item_ids >= self._orig_to_dense.shape[0]):
            raise ValueError("catalog item id is outside orig2catalog_row.npy")
        expected_rows = np.arange(expected, dtype=np.int64)
        observed_rows = np.asarray(self._orig_to_dense[item_ids], dtype=np.int64)
        if not np.array_equal(observed_rows, expected_rows):
            raise ValueError("orig2catalog_row.npy is not inverse-consistent with catalog_item_ids")

    @property
    def item_dim(self) -> int:
        return int(self._features.shape[1])

    @property
    def sid_levels(self) -> int:
        """Number of tokens used to identify a recommendation target."""

        return int(self._full_codes.shape[1])

    @property
    def semantic_levels(self) -> int:
        """Number of genuinely semantic RQ levels before disambiguation."""

        return int(self._semantic_codes.shape[1])

    @property
    def has_disambiguation_level(self) -> bool:
        return self.sid_levels > self.semantic_levels

    @property
    def catalog_size(self) -> int:
        if self._physical_padding_row:
            return int(self._catalog_item_ids.shape[0]) - 1
        return int(self._catalog_item_ids.shape[0])

    @property
    def sid_vocab_size_observed(self) -> int:
        return int(np.max(self._full_codes)) + 1

    def map_orig_ids(self, item_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        """Map original ids to dense ids; unmapped or out-of-range ids become zero."""

        values = np.asarray(item_ids, dtype=np.int64)
        result = np.zeros(values.shape, dtype=np.int64)
        valid = (values >= 0) & (values < self._orig_to_dense.shape[0])
        if np.any(valid):
            mapped = np.asarray(self._orig_to_dense[values[valid]], dtype=np.int64)
            if self._physical_padding_row:
                result[valid] = mapped
            else:
                result[valid] = np.where(mapped >= 0, mapped + 1, 0)
        return result

    def _catalog_rows(self, dense_ids: np.ndarray | Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(dense_ids, dtype=np.int64)
        if np.any(values < 0) or np.any(values > self.catalog_size):
            raise IndexError("dense item id is outside the feature store")
        if self._physical_padding_row:
            return values, values == 0
        return np.maximum(values - 1, 0), values == 0

    def lookup_features(self, dense_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        """Look up fixed item features, preserving zero as padding/unknown."""

        rows, padding = self._catalog_rows(dense_ids)
        output = np.asarray(self._features[rows], dtype=np.float32).copy()
        output[padding] = 0.0
        return output

    def lookup_sids(self, dense_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        rows, padding = self._catalog_rows(dense_ids)
        output = np.asarray(self._full_codes[rows], dtype=np.int64).copy()
        output[padding] = 0
        return output

    def lookup_semantic_sids(self, dense_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        """Look up only the semantic RQ prefix, excluding an identity suffix."""

        rows, padding = self._catalog_rows(dense_ids)
        output = np.asarray(self._semantic_codes[rows], dtype=np.int64).copy()
        output[padding] = 0
        return output

    def lookup_orig_ids(self, dense_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        """Invert valid dense ids; zero is returned unchanged as unknown/padding."""

        rows, padding = self._catalog_rows(dense_ids)
        output = np.asarray(self._catalog_item_ids[rows], dtype=np.int64).copy()
        output[padding] = 0
        return output


def _sha256_array(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _sha256_catalog_lookup(
    lookup: Callable[[np.ndarray], np.ndarray],
    dense_ids: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    probe = np.ascontiguousarray(lookup(dense_ids[:1]))
    digest.update(str(probe.dtype).encode("ascii"))
    shape = (int(dense_ids.shape[0]), *[int(value) for value in probe.shape[1:]])
    digest.update(np.asarray(shape, dtype="<i8").tobytes())
    for start in range(0, int(dense_ids.shape[0]), 16_384):
        selected = np.ascontiguousarray(lookup(dense_ids[start : start + 16_384]))
        digest.update(selected.tobytes())
    return digest.hexdigest()


def recommendation_catalog_fingerprints(
    store: RQCodebookStore,
    dense_ids: np.ndarray,
) -> dict[str, str]:
    """Content fingerprints that bind prepared rows to a concrete codebook."""

    catalog_dense = np.asarray(dense_ids, dtype=np.int64)
    return {
        "catalog_dense_ids_sha256": _sha256_array(catalog_dense),
        "catalog_orig_item_ids_sha256": _sha256_array(store.lookup_orig_ids(catalog_dense)),
        "catalog_semantic_sids_sha256": _sha256_catalog_lookup(
            store.lookup_semantic_sids, catalog_dense
        ),
        "catalog_full_sids_sha256": _sha256_catalog_lookup(store.lookup_sids, catalog_dense),
        "catalog_item_features_sha256": _sha256_catalog_lookup(
            store.lookup_features, catalog_dense
        ),
    }


def validate_codebook_store_against_manifest(
    store: RQCodebookStore,
    manifest: dict[str, Any],
    dataset_dir: str | Path,
) -> None:
    """Reject a training/evaluation store that differs from prepared data."""

    mapping = manifest.get("mapping", {})
    metadata_checks = {
        "store_format": store.mapping_format,
        "sid_levels": store.sid_levels,
        "semantic_levels": store.semantic_levels,
        "item_feature_dim": store.item_dim,
        "source_catalog_items": store.catalog_size,
    }
    for key, observed in metadata_checks.items():
        expected = mapping.get(key)
        if expected is not None and expected != observed:
            raise ValueError(
                f"recommendation codebook {key} mismatch: prepared={expected!r}, "
                f"loaded={observed!r}"
            )

    catalog_path = Path(dataset_dir).expanduser().resolve() / "catalog_dense_ids.npy"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Missing recommendation catalog: {catalog_path}")
    catalog_dense = np.asarray(np.load(catalog_path), dtype=np.int64)
    observed_fingerprints = recommendation_catalog_fingerprints(store, catalog_dense)
    expected_fingerprints = mapping.get("codebook_catalog_fingerprints", {})
    for key, observed in observed_fingerprints.items():
        expected = expected_fingerprints.get(key)
        # Old format-version-2 manifests predate the normalized fingerprint keys.
        if expected is not None and expected != observed:
            raise ValueError(f"recommendation codebook content fingerprint differs: {key}")


def load_recommendation_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    """Load and enforce the scientific data contract."""

    root = Path(dataset_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing recommendation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 2:
        raise ValueError("recommendation manifest must use format_version=2")
    contract = manifest.get("data_contract", {})
    if not isinstance(contract.get("listen_included"), bool):
        raise ValueError("recommendation dataset must declare listen_included explicitly")
    if not isinstance(contract.get("is_organic_used"), bool):
        raise ValueError("recommendation dataset must declare is_organic_used explicitly")
    if tuple(contract.get("event_type_names", ())) != EVENT_TYPE_NAMES:
        raise ValueError("recommendation event type order does not match the four-mark contract")
    if contract.get("tie_policy") != "shared_strict_pre_group_history":
        raise ValueError("recommendation dataset does not enforce shared pre-group history")
    if contract.get("sessionization") is not False:
        raise ValueError("recommendation dataset must not introduce sessionization")
    if contract.get("adjacent_item_aggregation") is not False:
        raise ValueError("recommendation dataset must not aggregate adjacent item events")
    fingerprints = manifest.get("mapping", {}).get("codebook_catalog_fingerprints")
    if not isinstance(fingerprints, dict) or not fingerprints:
        raise ValueError("recommendation manifest has no catalog codebook fingerprints")
    return manifest


def _parquet_files(root: Path, split: str) -> list[Path]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {split}")
    files = sorted((root / split).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards under {root / split}")
    return files


def _buffered_shuffle(
    rows: Iterable[dict[str, Any]],
    rng: np.random.Generator,
    buffer_size: int,
) -> Iterator[dict[str, Any]]:
    if buffer_size <= 1:
        yield from rows
        return
    buffer: list[dict[str, Any]] = []
    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue
        position = int(rng.integers(0, len(buffer)))
        yield buffer[position]
        buffer[position] = row
    if buffer:
        order = rng.permutation(len(buffer))
        for position in order.tolist():
            yield buffer[int(position)]


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Return a platform-stable uint64 mix without Python's salted hash."""

    with np.errstate(over="ignore"):
        mixed = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return mixed ^ (mixed >> np.uint64(31))


def _selection_identity_hashes(
    files: tuple[str, ...],
    split: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    split_salt = {
        "train": np.uint64(0x243F6A8885A308D3),
        "validation": np.uint64(0x13198A2E03707344),
        "test": np.uint64(0xA4093822299F31D0),
    }[split]
    uid_chunks: list[np.ndarray] = []
    hash_chunks: list[np.ndarray] = []
    for file_name in files:
        parquet = pq.ParquetFile(file_name)
        for batch in parquet.iter_batches(
            columns=["uid", "timestamp_seconds", "timestamp_group_id"],
            batch_size=65_536,
        ):
            uid = batch.column(0).to_numpy(zero_copy_only=False).astype(np.uint64)
            timestamp = batch.column(1).to_numpy(zero_copy_only=False).astype(np.uint64)
            group = batch.column(2).to_numpy(zero_copy_only=False).astype(np.uint64)
            with np.errstate(over="ignore"):
                identity = (
                    uid * np.uint64(0xD6E8FEB86659FD93)
                    ^ timestamp * np.uint64(0xA5A3564E27F8862B)
                    ^ group * np.uint64(0x9E3779B185EBCA87)
                    ^ np.uint64(seed)
                    ^ split_salt
                )
            uid_chunks.append(uid)
            hash_chunks.append(_splitmix64(identity))
    if not uid_chunks:
        raise RuntimeError(f"Cannot select rows from an empty {split} split")
    return np.concatenate(uid_chunks), np.concatenate(hash_chunks)


@lru_cache(maxsize=32)
def _user_stratified_selected_rows(
    files: tuple[str, ...],
    split: str,
    seed: int,
    max_rows: int,
) -> tuple[np.ndarray, int, int]:
    """Choose exact rows deterministically while spreading coverage across users."""

    uids, hashes = _selection_identity_hashes(files, split, seed)
    total_rows = int(uids.shape[0])
    target_rows = min(int(max_rows), total_rows)
    positions = np.arange(total_rows, dtype=np.int64)
    if target_rows == total_rows:
        return positions, int(np.unique(uids).shape[0]), int(np.unique(uids).shape[0])

    # The lowest-hash row of each user is mandatory when the row budget permits.
    by_user = np.lexsort((positions, hashes, uids))
    sorted_users = uids[by_user]
    first_for_user = np.concatenate([np.asarray([True]), sorted_users[1:] != sorted_users[:-1]])
    representatives = by_user[first_for_user]
    total_users = int(representatives.shape[0])
    representative_order = np.lexsort((representatives, hashes[representatives]))
    if target_rows <= total_users:
        selected = representatives[representative_order[:target_rows]]
    else:
        mandatory = representatives
        available = np.ones(total_rows, dtype=bool)
        available[mandatory] = False
        remaining = positions[available]
        remaining_order = np.lexsort((remaining, hashes[remaining]))
        fill = remaining[remaining_order[: target_rows - total_users]]
        selected = np.concatenate([mandatory, fill])
    selected = np.sort(selected.astype(np.int64, copy=False))
    selected_users = int(np.unique(uids[selected]).shape[0])
    return selected, selected_users, total_users


def _selection_fingerprint(indices: np.ndarray) -> str:
    values = np.ascontiguousarray(indices, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


class ExplicitRecommendationDataset(IterableDataset):
    """Stream one row per timestamp group with a set of mapped positive items."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        *,
        seed: int = 2026,
        epoch: int = 0,
        shuffle: bool = False,
        shuffle_buffer_size: int = 0,
        max_rows: int | None = None,
        limited_row_selection: str = "head",
        row_selection_seed: int = 2026,
        batch_read_size: int = 2048,
    ) -> None:
        super().__init__()
        self.root = Path(dataset_dir).expanduser().resolve()
        self.split = split
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.shuffle = bool(shuffle)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.max_rows = int(max_rows) if max_rows is not None else None
        self.limited_row_selection = str(limited_row_selection)
        self.row_selection_seed = int(row_selection_seed)
        self.batch_read_size = int(batch_read_size)
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows must be positive or None")
        if self.batch_read_size <= 0:
            raise ValueError("batch_read_size must be positive")
        if self.limited_row_selection not in {"head", "user_stratified_hash"}:
            raise ValueError("limited_row_selection must be head or user_stratified_hash")
        if self.row_selection_seed < 0:
            raise ValueError("row_selection_seed cannot be negative")
        self.manifest = load_recommendation_manifest(self.root)
        self.files = _parquet_files(self.root, split)

    def _selected_global_rows(self) -> np.ndarray | None:
        total = int(self.manifest.get("splits", {}).get(self.split, {}).get("rows", 0))
        if self.max_rows is None or self.max_rows >= total or self.limited_row_selection == "head":
            return None
        selected, _, _ = _user_stratified_selected_rows(
            tuple(str(path) for path in self.files),
            self.split,
            self.row_selection_seed,
            self.max_rows,
        )
        return selected

    @property
    def selection_metadata(self) -> dict[str, Any]:
        total_rows = int(self.manifest.get("splits", {}).get(self.split, {}).get("rows", 0))
        if self.max_rows is None or self.max_rows >= total_rows:
            return {
                "mode": "all",
                "seed": self.row_selection_seed,
                "selected_rows": total_rows,
                "total_rows": total_rows,
                "selected_indices_sha256": None,
            }
        if self.limited_row_selection == "head":
            indices = np.arange(self.max_rows, dtype=np.int64)
            return {
                "mode": "head",
                "seed": self.row_selection_seed,
                "selected_rows": int(indices.shape[0]),
                "total_rows": total_rows,
                "selected_indices_sha256": _selection_fingerprint(indices),
            }
        selected, selected_users, total_users = _user_stratified_selected_rows(
            tuple(str(path) for path in self.files),
            self.split,
            self.row_selection_seed,
            self.max_rows,
        )
        return {
            "mode": "user_stratified_hash",
            "seed": self.row_selection_seed,
            "selected_rows": int(selected.shape[0]),
            "total_rows": total_rows,
            "selected_users": selected_users,
            "total_users": total_users,
            "selected_indices_sha256": _selection_fingerprint(selected),
        }

    @property
    def expected_rows(self) -> int | None:
        value = self.manifest.get("splits", {}).get(self.split, {}).get("rows")
        if value is None:
            return None
        rows = int(value)
        return min(rows, self.max_rows) if self.max_rows is not None else rows

    def _iter_files(self, files: list[Path]) -> Iterator[dict[str, Any]]:
        for file_path in files:
            parquet = pq.ParquetFile(file_path)
            for batch in parquet.iter_batches(batch_size=self.batch_read_size):
                yield from batch.to_pylist()

    def _iter_selected_files(
        self,
        files: list[Path],
        selected_rows: np.ndarray,
    ) -> Iterator[dict[str, Any]]:
        """Read all row groups but materialize only selected rows as Python objects."""

        selected_cursor = 0
        global_position = 0
        for file_path in files:
            parquet = pq.ParquetFile(file_path)
            for batch in parquet.iter_batches(batch_size=self.batch_read_size):
                batch_end = global_position + batch.num_rows
                next_cursor = int(
                    np.searchsorted(selected_rows, batch_end, side="left")
                )
                if next_cursor > selected_cursor:
                    local_positions = (
                        selected_rows[selected_cursor:next_cursor] - global_position
                    )
                    selected_batch = batch.take(pa.array(local_positions))
                    yield from selected_batch.to_pylist()
                    selected_cursor = next_cursor
                global_position = batch_end
                if selected_cursor >= selected_rows.shape[0]:
                    return

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        files = list(self.files)
        rng_seed = self.seed + 1_000_003 * self.epoch
        if worker is not None:
            files = files[worker.id :: worker.num_workers]
            rng_seed += 97_409 * worker.id
        rng = np.random.default_rng(rng_seed)
        selected_rows = self._selected_global_rows()
        if self.shuffle and selected_rows is None:
            rng.shuffle(files)
        rows: Iterable[dict[str, Any]]
        if selected_rows is None:
            rows = self._iter_files(files)
        else:
            rows = self._iter_selected_files(files, selected_rows)
        if self.shuffle:
            rows = _buffered_shuffle(rows, rng, self.shuffle_buffer_size)
        emitted = 0
        for row in rows:
            yield row
            emitted += 1
            if self.max_rows is not None and emitted >= self.max_rows:
                return


def _pad_positive_arrays(
    rows: list[dict[str, Any]],
    sid_levels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_positives = max(len(row["positive_dense_item_ids"]) for row in rows)
    dense = np.zeros((len(rows), max_positives), dtype=np.int64)
    sids = np.zeros((len(rows), max_positives, sid_levels), dtype=np.int64)
    mask = np.zeros((len(rows), max_positives), dtype=bool)
    feedback = np.zeros(
        (len(rows), max_positives, len(EVENT_TYPE_NAMES)),
        dtype=bool,
    )
    for row_index, row in enumerate(rows):
        values = np.asarray(row["positive_dense_item_ids"], dtype=np.int64)
        row_sids = np.asarray(row["positive_sids"], dtype=np.int64)
        if row_sids.ndim != 2 or row_sids.shape[1] != sid_levels:
            raise ValueError("positive_sids has an invalid shape")
        if values.shape[0] != row_sids.shape[0]:
            raise ValueError("positive item and SID counts differ")
        row_feedback = np.asarray(row["positive_event_type_mask"], dtype=bool)
        if row_feedback.shape != (values.shape[0], len(EVENT_TYPE_NAMES)):
            raise ValueError("positive feedback mask has an invalid shape")
        if not np.all(row_feedback.any(axis=1)):
            raise ValueError("every target item must retain at least one explicit mark")
        count = int(values.shape[0])
        dense[row_index, :count] = values
        sids[row_index, :count] = row_sids
        mask[row_index, :count] = True
        feedback[row_index, :count] = row_feedback
    return dense, sids, mask, feedback


def collate_recommendation_groups(
    rows: list[dict[str, Any]],
    store: RQCodebookStore,
) -> dict[str, torch.Tensor]:
    """Collate grouped positives without inventing an order within timestamp ties."""

    if not rows:
        raise ValueError("cannot collate an empty recommendation batch")
    history_dense = np.asarray(
        [row["history_dense_item_ids"] for row in rows],
        dtype=np.int64,
    )
    history_marks = np.asarray(
        [row["history_event_type_ids"] for row in rows],
        dtype=np.int64,
    )
    history_ages = np.asarray(
        [row["history_age_seconds"] for row in rows],
        dtype=np.float32,
    )
    history_mask = np.asarray([row["history_mask"] for row in rows], dtype=bool)
    if not (history_dense.shape == history_marks.shape == history_ages.shape == history_mask.shape):
        raise ValueError("history tensors must share [batch, history] shape")
    if np.any((history_marks < 0) | (history_marks > len(EVENT_TYPE_NAMES) + 1)):
        raise ValueError("history mark id is outside padding, four explicit marks, and listen")
    if any("history_source_ids" in row for row in rows):
        history_sources = np.asarray(
            [row.get("history_source_ids", [0] * len(row["history_mask"])) for row in rows],
            dtype=np.int64,
        )
        if history_sources.shape != history_marks.shape:
            raise ValueError("history source ids must share the mark shape")
        if np.any((history_sources < 0) | (history_sources > 2)):
            raise ValueError("history source id must be 0 (pad), 1 (rec-driven), or 2 (organic)")
    else:
        history_sources = np.zeros_like(history_marks)

    positive_dense, positive_sids, positive_mask, positive_feedback = _pad_positive_arrays(
        rows,
        store.sid_levels,
    )
    features = store.lookup_features(history_dense)
    return {
        "uid": torch.tensor([int(row["uid"]) for row in rows], dtype=torch.long),
        "timestamp_seconds": torch.tensor(
            [int(row["timestamp_seconds"]) for row in rows],
            dtype=torch.long,
        ),
        "timestamp_group_id": torch.tensor(
            [int(row["timestamp_group_id"]) for row in rows],
            dtype=torch.long,
        ),
        "history_dense_item_ids": torch.from_numpy(history_dense),
        "history_item_features": torch.from_numpy(features),
        "history_event_type_ids": torch.from_numpy(history_marks),
        "history_source_ids": torch.from_numpy(history_sources),
        "history_age_seconds": torch.from_numpy(history_ages),
        "history_mask": torch.from_numpy(history_mask),
        "positive_dense_item_ids": torch.from_numpy(positive_dense),
        "positive_sids": torch.from_numpy(positive_sids),
        "positive_mask": torch.from_numpy(positive_mask),
        "positive_event_type_mask": torch.from_numpy(positive_feedback),
    }


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}
