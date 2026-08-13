#!/usr/bin/env python3
"""Audit whether missing explicit-item embeddings have artist/album support.

This script is read-only.  It does not create proxy vectors and does not alter the
explicit event universe.  A missing item is called metadata-imputable only when it
belongs to an official artist/album group containing at least one different item
with a source audio embedding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


CATALOG_MAX_ITEM_ID = 9_390_623
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_0804_ROOT = Path("/root/autodl-tmp/0804 Yambda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--item-frequency",
        type=Path,
        default=PROJECT_ROOT
        / "phase0_5b_audit"
        / "artifacts"
        / "item_explicit_frequency.npz",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=LEGACY_0804_ROOT / "data" / "embeddings.parquet",
    )
    parser.add_argument(
        "--artist-mapping",
        type=Path,
        default=LEGACY_0804_ROOT
        / "dataprocess"
        / "data"
        / "official_metadata"
        / "artist_item_mapping.parquet",
    )
    parser.add_argument(
        "--album-mapping",
        type=Path,
        default=LEGACY_0804_ROOT
        / "dataprocess"
        / "data"
        / "official_metadata"
        / "album_item_mapping.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "phase0_5b_audit"
        / "artifacts"
        / "missing_embedding_metadata_audit.json",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def embedding_presence(path: Path) -> np.ndarray:
    present = np.zeros(CATALOG_MAX_ITEM_ID + 1, dtype=bool)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        columns=["item_id"], batch_size=524_288, use_threads=False
    ):
        item_ids = batch.column(0).to_numpy(zero_copy_only=False).astype(
            np.int64, copy=False
        )
        valid = (item_ids >= 0) & (item_ids <= CATALOG_MAX_ITEM_ID)
        present[item_ids[valid]] = True
    return present


def mapping_support(
    path: Path,
    group_column: str,
    source_present: np.ndarray,
    missing_explicit_lookup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    table = pq.read_table(path, columns=[group_column, "item_id"], use_threads=False)
    groups = table[group_column].to_numpy(zero_copy_only=False).astype(
        np.uint32, copy=False
    )
    items = table["item_id"].to_numpy(zero_copy_only=False).astype(
        np.uint32, copy=False
    )
    if len(groups) and np.any(groups[1:] < groups[:-1]):
        raise ValueError(f"{path} must be sorted by {group_column}")
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    ends = np.r_[starts[1:], len(groups)]
    lengths = ends - starts
    group_has_source = np.logical_or.reduceat(source_present[items], starts)
    row_group_has_source = np.repeat(group_has_source, lengths)
    target_rows = missing_explicit_lookup[items]

    mapped = np.zeros(CATALOG_MAX_ITEM_ID + 1, dtype=bool)
    supported = np.zeros(CATALOG_MAX_ITEM_ID + 1, dtype=bool)
    mapped[items[target_rows]] = True
    supported[items[target_rows & row_group_has_source]] = True
    report = {
        "path": str(path.resolve()),
        "group_column": group_column,
        "mapping_rows": int(len(items)),
        "groups": int(len(starts)),
        "groups_with_source_embedding": int(group_has_source.sum()),
        "missing_explicit_items_with_mapping": int(mapped.sum()),
        "missing_explicit_items_with_source_supported_group": int(supported.sum()),
        "definition": (
            "target belongs to an official group containing at least one item "
            "with a source audio embedding"
        ),
    }
    return mapped, supported, report


def slice_summary(mask: np.ndarray, item_ids: np.ndarray, event_counts: np.ndarray) -> dict:
    selected = mask[item_ids]
    return {
        "items": int(selected.sum()),
        "events": int(event_counts[selected].sum(dtype=np.uint64)),
    }


def main() -> None:
    args = parse_args()
    with np.load(args.item_frequency, allow_pickle=False) as payload:
        item_ids = payload["item_id"].astype(np.uint32, copy=False)
        event_counts = payload["total_count"].astype(np.uint32, copy=False)
    source_present = embedding_presence(args.embeddings)
    missing_item_mask = ~source_present[item_ids]
    missing_ids = item_ids[missing_item_mask]
    missing_counts = event_counts[missing_item_mask]
    missing_lookup = np.zeros(CATALOG_MAX_ITEM_ID + 1, dtype=bool)
    missing_lookup[missing_ids] = True

    artist_mapped, artist_supported, artist_report = mapping_support(
        args.artist_mapping, "artist_id", source_present, missing_lookup
    )
    album_mapped, album_supported, album_report = mapping_support(
        args.album_mapping, "album_id", source_present, missing_lookup
    )
    either_mapped = artist_mapped | album_mapped
    either_supported = artist_supported | album_supported
    both_supported = artist_supported & album_supported

    output = {
        "status": "complete_read_only_audit",
        "data_contract": {
            "explicit_universe": str(args.item_frequency.resolve()),
            "source_embeddings": str(args.embeddings.resolve()),
            "artist_mapping": str(args.artist_mapping.resolve()),
            "album_mapping": str(args.album_mapping.resolve()),
            "no_proxy_vectors_created": True,
            "no_rows_removed": True,
            "collaborative_fallback_not_audited_here": True,
        },
        "explicit_items": int(len(item_ids)),
        "explicit_events": int(event_counts.sum(dtype=np.uint64)),
        "source_embedding": {
            "covered": {
                "items": int((~missing_item_mask).sum()),
                "events": int(event_counts[~missing_item_mask].sum(dtype=np.uint64)),
            },
            "missing": {
                "items": int(missing_item_mask.sum()),
                "events": int(missing_counts.sum(dtype=np.uint64)),
            },
        },
        "artist": artist_report,
        "album": album_report,
        "missing_explicit_slices": {
            "artist_mapping": slice_summary(artist_mapped, missing_ids, missing_counts),
            "album_mapping": slice_summary(album_mapped, missing_ids, missing_counts),
            "either_mapping": slice_summary(either_mapped, missing_ids, missing_counts),
            "artist_source_supported": slice_summary(
                artist_supported, missing_ids, missing_counts
            ),
            "album_source_supported": slice_summary(
                album_supported, missing_ids, missing_counts
            ),
            "either_source_supported": slice_summary(
                either_supported, missing_ids, missing_counts
            ),
            "both_source_supported": slice_summary(
                both_supported, missing_ids, missing_counts
            ),
            "no_source_supported_metadata": slice_summary(
                missing_lookup & ~either_supported, missing_ids, missing_counts
            ),
        },
        "interpretation_boundary": (
            "metadata support is only potential proxy-vector coverage; quality must "
            "still be established by leave-one-out reconstruction before RQKMeans"
        ),
    }
    atomic_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
