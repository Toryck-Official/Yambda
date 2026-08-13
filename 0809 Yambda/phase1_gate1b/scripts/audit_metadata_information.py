#!/usr/bin/env python3
"""Audit actual metadata fields and context identifiability for Gate 1B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "phase1_gate1b" / "configs" / "gate1b.json"
DEFAULT_OUTPUT = ROOT / "phase1_gate1b" / "artifacts" / "metadata_information_audit.json"
MASK64 = (1 << 64) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def splitmix64(value: np.ndarray) -> np.ndarray:
    x = value.astype(np.uint64, copy=False)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def relation_signature(
    path: Path,
    group_column: str,
    explicit_lookup: np.ndarray,
    rows: int,
    salt: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    xor_hash = np.zeros(rows, dtype=np.uint64)
    sum_hash = np.zeros(rows, dtype=np.uint64)
    count = np.zeros(rows, dtype=np.uint16)
    mapping_rows = 0
    relevant_rows = 0
    duplicate_explicit_relations = 0
    parquet = pq.ParquetFile(path)
    previous_pair: tuple[int, int] | None = None
    for rg in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(rg, columns=[group_column, "item_id"], use_threads=False)
        group = table[group_column].to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
        item = table["item_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        mapping_rows += len(item)
        pos = explicit_lookup[item].astype(np.int64) - 1
        keep = pos >= 0
        if np.any(keep):
            kept_group = group[keep]
            kept_item = item[keep]
            kept_pos = pos[keep]
            relevant_rows += len(kept_pos)
            hashed = splitmix64(kept_group ^ np.uint64(salt))
            np.bitwise_xor.at(xor_hash, kept_pos, hashed)
            np.add.at(sum_hash, kept_pos, hashed)
            np.add.at(count, kept_pos, np.uint16(1))
            # Official mapping is group-sorted. Count adjacent duplicate pairs,
            # including a pair split over a Parquet row-group boundary.
            if previous_pair is not None and len(kept_item):
                duplicate_explicit_relations += int(
                    previous_pair == (int(kept_group[0]), int(kept_item[0]))
                )
            if len(kept_item) > 1:
                duplicate_explicit_relations += int(
                    np.count_nonzero(
                        (kept_group[1:] == kept_group[:-1])
                        & (kept_item[1:] == kept_item[:-1])
                    )
                )
            previous_pair = (int(kept_group[-1]), int(kept_item[-1]))
    return xor_hash, sum_hash, count, {
        "path": str(path),
        "schema": str(parquet.schema_arrow),
        "rows": int(mapping_rows),
        "explicit_relation_rows": int(relevant_rows),
        "explicit_items_with_relation": int(np.count_nonzero(count)),
        "maximum_relations_per_item": int(count.max()),
        "adjacent_duplicate_explicit_relation_rows": int(duplicate_explicit_relations),
    }


def bucket_summary(signature: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    _, inverse, counts = np.unique(signature, return_inverse=True, return_counts=True, axis=0)
    item_bucket_size = counts[inverse]
    colliding = counts > 1
    return {
        "items": int(len(signature)),
        "unique_contexts": int(len(counts)),
        "singleton_contexts": int(np.count_nonzero(counts == 1)),
        "singleton_items": int(np.count_nonzero(item_bucket_size == 1)),
        "items_in_nonunique_context": int(np.count_nonzero(item_bucket_size > 1)),
        "nonunique_context_buckets": int(np.count_nonzero(colliding)),
        "bucket_size_p50": float(np.quantile(counts, 0.50)),
        "bucket_size_p90": float(np.quantile(counts, 0.90)),
        "bucket_size_p99": float(np.quantile(counts, 0.99)),
        "maximum_context_collision_bucket": int(counts.max()),
    }, inverse, counts


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    paths = {key: Path(value) for key, value in config["paths"].items()}
    with np.load(paths["item_frequency"], allow_pickle=False) as payload:
        item_ids = payload["item_id"].astype(np.uint32, copy=False)
        event_count = payload["total_count"].astype(np.uint32, copy=False)
    orig2dense = np.load(paths["orig2dense"], mmap_mode="r")
    real = np.asarray(orig2dense[item_ids] > 0)
    missing = ~real
    explicit_lookup = np.zeros(len(orig2dense), dtype=np.int32)
    explicit_lookup[item_ids] = np.arange(1, len(item_ids) + 1, dtype=np.int32)
    print("[1/3] artist relation signatures", flush=True)
    ax, asum, acount, artist_report = relation_signature(
        paths["artist_mapping"], "artist_id", explicit_lookup, len(item_ids), 0xA77157
    )
    print("[2/3] album relation signatures", flush=True)
    bx, bsum, bcount, album_report = relation_signature(
        paths["album_mapping"], "album_id", explicit_lookup, len(item_ids), 0xA1B00D
    )
    # A 128-bit content digest per relation type plus relation counts. This is an
    # audit key, not a learned feature. The collision probability is negligible;
    # the maximum bucket is separately described and can be relation-replayed.
    context = np.column_stack([ax, asum, acount, bx, bsum, bcount]).astype(np.uint64)
    missing_summary, inverse, counts = bucket_summary(context[missing])
    maximum_bucket = int(counts.max())
    maximum_bucket_id = int(np.argmax(counts))
    missing_positions = np.flatnonzero(missing)
    max_members = missing_positions[inverse == maximum_bucket_id]
    max_member_items = item_ids[max_members]
    support_report = json.loads(paths["support_report"].read_text())
    reconstructable_real = sum(support_report["real_reconstructable"].values())
    local_manifest = json.loads(paths["metadata_manifest"].read_text())
    actual_fields = {
        "artist_item_mapping": ["artist_id", "item_id"],
        "album_item_mapping": ["album_id", "item_id"],
        "audio_embedding": ["item_id", "embed", "normalized_embed"],
        "historical_item_tokens": ["item_id_as_json_key", "four_historical_sid_tokens"],
    }
    absent = ["track_title", "genre", "release_date", "release_year", "track_text"]
    no_relation = (acount == 0) & (bcount == 0)
    metadata_available = (acount > 0) | (bcount > 0)
    report = {
        "status": "complete_gate1b_metadata_information_audit",
        "data_contract": {
            "explicit_items": int(len(item_ids)),
            "real_embedding_items": int(real.sum()),
            "missing_embedding_items": int(missing.sum()),
            "interaction_frequency_used_as_metadata_input": False,
            "historical_sid_used_as_metadata_input": False,
            "context_signature": "artist and album relation sets represented by independent xor+sum 64-bit digests plus relation counts; audit-only, not model input",
        },
        "actual_local_static_fields": actual_fields,
        "requested_but_not_present_in_pinned_local_yambda_assets": absent,
        "pinned_official_metadata_manifest": local_manifest,
        "relation_tables": {"artist": artist_report, "album": album_report},
        "real_items_strict_leave_one_out_context": {
            "count": int(reconstructable_real),
            "fraction_of_real_items": float(reconstructable_real / real.sum()),
            "artist_only": int(support_report["real_reconstructable"]["artist_only"]),
            "album_only": int(support_report["real_reconstructable"]["album_only"]),
            "both": int(support_report["real_reconstructable"]["both"]),
            "none": int(real.sum() - reconstructable_real),
        },
        "missing_items": {
            "with_any_item_associated_artist_or_album_relation": int(np.count_nonzero(missing & metadata_available)),
            "without_artist_or_album_relation": int(np.count_nonzero(missing & no_relation)),
            "with_artist_relation": int(np.count_nonzero(missing & (acount > 0))),
            "with_album_relation": int(np.count_nonzero(missing & (bcount > 0))),
            "item_specific_title_genre_release_fields": 0,
            "same_context_collision": missing_summary,
            "maximum_bucket": {
                "size": maximum_bucket,
                "member_item_id_sample": [int(x) for x in max_member_items[:20]],
                "all_members_have_same_digest_and_relation_counts": True,
                "artist_relation_count": int(acount[max_members[0]]),
                "album_relation_count": int(bcount[max_members[0]]),
                "events_in_bucket": int(event_count[max_members].sum(dtype=np.uint64)),
            },
        },
        "interpretation": {
            "metadata_can_distinguish_every_missing_item": bool(missing_summary["items_in_nonunique_context"] == 0),
            "identifiability_limit": "items with identical available artist/album relation sets receive identical permitted context inputs; no deterministic model can distinguish their track-specific audio embedding or SID from these inputs alone",
            "old_item_tokens_are_targets_or_history_not_new_static_metadata": True,
        },
    }
    atomic_json(args.output, report)
    print(json.dumps({
        "real_loo_context": report["real_items_strict_leave_one_out_context"],
        "missing": report["missing_items"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

