#!/usr/bin/env python3
"""Construct strict LOO artist/album proxy vectors for the frozen Gate-1 sample."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "phase1_gates" / "configs" / "gates.json"
DEFAULT_SAMPLE = ROOT / "phase1_gates" / "artifacts" / "proxy_validation_sample.npz"
DEFAULT_OUTPUT = ROOT / "phase1_gates" / "work" / "proxy_vectors"
DEFAULT_REPORT = ROOT / "phase1_gates" / "artifacts" / "proxy_vector_build_report.json"


@dataclass
class ProxyResult:
    vectors: np.ndarray
    available: np.ndarray
    source_count: np.ndarray
    relation_count: np.ndarray
    report: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def norm_audit(vectors: np.ndarray, available: np.ndarray) -> dict:
    norms = np.linalg.norm(vectors[available], axis=1)
    return {
        "rows": int(len(vectors)),
        "available": int(available.sum()),
        "finite": bool(np.isfinite(vectors).all()),
        "available_norm_min": float(norms.min(initial=np.inf)),
        "available_norm_mean": float(norms.mean()) if len(norms) else None,
        "available_norm_max": float(norms.max(initial=-np.inf)),
    }


def metadata_group_predictions_streaming(
    mapping_path: Path,
    group_column: str,
    target_item_ids: np.ndarray,
    orig2dense: np.ndarray,
    dense_features: np.ndarray,
) -> ProxyResult:
    """Compute per-relation strict LOO centroids with row-group bounded memory."""
    target_item_ids = np.asarray(target_item_ids, dtype=np.int64)
    if np.any(target_item_ids[1:] <= target_item_ids[:-1]):
        raise ValueError("target_item_ids must be strictly increasing")
    target_lookup = np.zeros(len(orig2dense), dtype=np.int32)
    target_lookup[target_item_ids] = np.arange(
        1, len(target_item_ids) + 1, dtype=np.int32
    )
    output_sums = np.zeros((len(target_item_ids), 128), dtype=np.float32)
    output_sources = np.zeros(len(target_item_ids), dtype=np.uint64)
    output_relations = np.zeros(len(target_item_ids), dtype=np.uint32)
    parquet = pq.ParquetFile(mapping_path)
    carry_groups = np.empty(0, dtype=np.uint32)
    carry_items = np.empty(0, dtype=np.uint32)
    mapping_rows = 0
    groups_processed = 0
    relevant_groups = 0
    previous_group: int | None = None

    def process_complete_groups(groups: np.ndarray, items: np.ndarray) -> tuple[int, int]:
        if not len(groups):
            return 0, 0
        starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
        ends = np.r_[starts[1:], len(groups)]
        group_target = np.logical_or.reduceat(target_lookup[items] > 0, starts)
        selected_groups = np.flatnonzero(group_target)
        for group_index in selected_groups.tolist():
            start = int(starts[group_index])
            end = int(ends[group_index])
            group_items = items[start:end]
            group_sum = np.zeros(128, dtype=np.float32)
            source_total = 0
            # Some official metadata groups are extremely large. Bound feature
            # gathering inside a group as well as at the Parquet row-group level.
            for chunk_start in range(0, len(group_items), 65_536):
                chunk_items = group_items[chunk_start : chunk_start + 65_536]
                chunk_dense = orig2dense[chunk_items].astype(np.int64, copy=False)
                known_dense = chunk_dense[chunk_dense > 0]
                if not len(known_dense):
                    continue
                group_sum += np.asarray(
                    dense_features[known_dense], dtype=np.float32
                ).sum(axis=0, dtype=np.float32)
                source_total += int(len(known_dense))
            if source_total == 0:
                continue
            targets = group_items[target_lookup[group_items] > 0]
            for target_item in targets.tolist():
                target_position = int(target_lookup[target_item]) - 1
                target_dense = int(orig2dense[target_item])
                remaining = source_total - int(target_dense > 0)
                if remaining <= 0:
                    continue
                centroid = group_sum.copy()
                if target_dense > 0:
                    centroid -= np.asarray(
                        dense_features[target_dense], dtype=np.float32
                    )
                centroid /= remaining
                norm = float(np.linalg.norm(centroid))
                if not np.isfinite(norm) or norm <= 1e-12:
                    continue
                output_sums[target_position] += centroid / norm
                output_sources[target_position] += remaining
                output_relations[target_position] += 1
        return int(len(starts)), int(len(selected_groups))

    for row_group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(
            row_group, columns=[group_column, "item_id"], use_threads=False
        )
        groups = table[group_column].to_numpy(zero_copy_only=False).astype(
            np.uint32, copy=False
        )
        items = table["item_id"].to_numpy(zero_copy_only=False).astype(
            np.uint32, copy=False
        )
        mapping_rows += len(groups)
        if len(groups) and previous_group is not None and int(groups[0]) < previous_group:
            raise ValueError(f"mapping is not sorted by {group_column}")
        if len(groups):
            previous_group = int(groups[-1])
        if len(carry_groups):
            groups = np.concatenate([carry_groups, groups])
            items = np.concatenate([carry_items, items])
        if not len(groups):
            continue
        last_start = int(np.searchsorted(groups, groups[-1], side="left"))
        complete, relevant = process_complete_groups(
            groups[:last_start], items[:last_start]
        )
        groups_processed += complete
        relevant_groups += relevant
        carry_groups = groups[last_start:].copy()
        carry_items = items[last_start:].copy()
    complete, relevant = process_complete_groups(carry_groups, carry_items)
    groups_processed += complete
    relevant_groups += relevant

    raw_available = output_relations > 0
    output_sums[raw_available] /= output_relations[raw_available, None]
    norms = np.linalg.norm(output_sums, axis=1)
    available = raw_available & np.isfinite(norms) & (norms > 1e-12)
    output_sums[available] /= norms[available, None]
    output_sums[~available] = 0.0
    return ProxyResult(
        vectors=output_sums,
        available=available,
        source_count=output_sources,
        relation_count=output_relations,
        report={
            "mapping_path": str(mapping_path.resolve()),
            "group_column": group_column,
            "mapping_rows": int(mapping_rows),
            "row_groups_streamed": int(parquet.metadata.num_row_groups),
            "groups_processed": int(groups_processed),
            "relevant_groups": int(relevant_groups),
            "target_items": int(len(target_item_ids)),
            "predicted_items": int(available.sum()),
            "prediction_coverage": float(available.mean()),
            "leave_one_out_for_source_covered_targets": True,
            "centroid_per_relation_then_relation_average": True,
        },
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {key: Path(value) for key, value in config["paths"].items()}
    with np.load(args.sample, allow_pickle=False) as sample:
        item_ids = sample["item_id"].astype(np.int64, copy=False)
        dense_ids = sample["dense_id"].astype(np.int64, copy=False)
        expected_artist = sample["artist_loo_source_count"] > 0
        expected_album = sample["album_loo_source_count"] > 0
    if np.any(item_ids[1:] <= item_ids[:-1]):
        raise ValueError("sample item IDs must be strictly increasing")
    if np.any(dense_ids <= 0):
        raise ValueError("every Gate-1 target must have a real embedding")

    orig2dense = np.load(paths["orig2dense"], mmap_mode="r")
    dense_features = np.load(paths["dense_features"], mmap_mode="r")
    truth = np.asarray(dense_features[dense_ids], dtype=np.float32)
    truth_norms = np.linalg.norm(truth, axis=1)
    if not np.isfinite(truth).all() or not np.allclose(
        truth_norms, 1.0, atol=2e-5, rtol=2e-5
    ):
        raise ValueError("official normalized embedding cache is not unit-L2")

    print("[1/3] Strict LOO artist proxy", flush=True)
    artist = metadata_group_predictions_streaming(
        paths["artist_mapping"],
        "artist_id",
        item_ids,
        orig2dense,
        dense_features,
    )
    if not np.array_equal(artist.available, expected_artist):
        differing = int(np.count_nonzero(artist.available != expected_artist))
        raise RuntimeError(f"artist availability differs from support audit: {differing}")

    print("[2/3] Strict LOO album proxy", flush=True)
    album = metadata_group_predictions_streaming(
        paths["album_mapping"],
        "album_id",
        item_ids,
        orig2dense,
        dense_features,
    )
    if not np.array_equal(album.available, expected_album):
        differing = int(np.count_nonzero(album.available != expected_album))
        raise RuntimeError(f"album availability differs from support audit: {differing}")

    print("[3/3] Saving proxy vectors and provenance", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "artist_proxy.npy": artist.vectors,
        "artist_available.npy": artist.available,
        "artist_source_count.npy": artist.source_count,
        "artist_relation_count.npy": artist.relation_count,
        "album_proxy.npy": album.vectors,
        "album_available.npy": album.available,
        "album_source_count.npy": album.source_count,
        "album_relation_count.npy": album.relation_count,
    }
    for name, value in outputs.items():
        atomic_npy(args.output_dir / name, value)

    report = {
        "status": "complete_strict_leave_one_out_proxy_vectors",
        "data_contract": {
            "targets": str(args.sample.resolve()),
            "targets_have_real_embedding": True,
            "target_embedding_removed_from_each_metadata_group": True,
            "target_embedding_used_only_as_evaluation_truth": True,
            "vectors_normalized": True,
            "collaborative_information_used": False,
            "event_rows_modified": False,
        },
        "truth": norm_audit(truth, np.ones(len(truth), dtype=bool)),
        "artist": {
            "norm_audit": norm_audit(artist.vectors, artist.available),
            "builder_report": artist.report,
        },
        "album": {
            "norm_audit": norm_audit(album.vectors, album.available),
            "builder_report": album.report,
        },
        "outputs": {
            name: {
                "path": str((args.output_dir / name).resolve()),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in outputs.items()
        },
    }
    atomic_json(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
