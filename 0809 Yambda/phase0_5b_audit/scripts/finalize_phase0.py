#!/usr/bin/env python3
"""Validate and consolidate the read-only Yambda-5B Phase 0 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_0804_ROOT = Path("/root/autodl-tmp/0804 Yambda")
AUDIT_ROOT = PROJECT_ROOT / "phase0_5b_audit"
DEFAULT_EXPLICIT_AUDIT = AUDIT_ROOT / "work" / "explicit_audit.json"
DEFAULT_ITEM_FREQUENCY = AUDIT_ROOT / "artifacts" / "item_explicit_frequency.npz"
DEFAULT_DOWNLOAD_MANIFEST = AUDIT_ROOT / "artifacts" / "download_manifest.json"
DEFAULT_FLAT_MANIFEST = AUDIT_ROOT / "work" / "flat_explicit_5b" / "manifest.json"
DEFAULT_EMBEDDING_AUDIT = AUDIT_ROOT / "artifacts" / "embedding_catalog_audit.json"
DEFAULT_MISSING_METADATA_AUDIT = (
    AUDIT_ROOT / "artifacts" / "missing_embedding_metadata_audit.json"
)
DEFAULT_EMBEDDINGS = LEGACY_0804_ROOT / "data" / "embeddings.parquet"
DEFAULT_PROTOCOL = PROJECT_ROOT / "docs" / "2026-08-09_Yambda显式反馈主线协议.md"
DEFAULT_OUTPUT = AUDIT_ROOT / "artifacts" / "data_audit.json"
OFFICIAL_CATALOG_ITEMS = 9_390_623
EVENT_NAMES = ("like", "dislike", "unlike", "undislike")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--explicit-audit", type=Path, default=DEFAULT_EXPLICIT_AUDIT)
    parser.add_argument("--item-frequency", type=Path, default=DEFAULT_ITEM_FREQUENCY)
    parser.add_argument("--download-manifest", type=Path, default=DEFAULT_DOWNLOAD_MANIFEST)
    parser.add_argument("--flat-manifest", type=Path, default=DEFAULT_FLAT_MANIFEST)
    parser.add_argument("--embedding-audit", type=Path, default=DEFAULT_EMBEDDING_AUDIT)
    parser.add_argument(
        "--missing-metadata-audit",
        type=Path,
        default=DEFAULT_MISSING_METADATA_AUDIT,
    )
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def embedding_presence(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    if "item_id" not in parquet.schema_arrow.names:
        raise ValueError(f"embedding file has no item_id: {path}")
    seen = np.zeros(OFFICIAL_CATALOG_ITEMS + 1, dtype=bool)
    rows = 0
    duplicate_rows = 0
    out_of_range_rows = 0
    previous = -1
    order_violations = 0
    for batch in parquet.iter_batches(
        batch_size=262_144,
        columns=["item_id"],
        use_threads=False,
    ):
        values = batch.column(0).to_numpy(zero_copy_only=False).astype(
            np.uint32, copy=False
        )
        rows += int(values.size)
        if values.size:
            order_violations += int(int(values[0]) <= previous)
            order_violations += int(np.count_nonzero(values[1:] <= values[:-1]))
            previous = int(values[-1])
        valid = values <= OFFICIAL_CATALOG_ITEMS
        out_of_range_rows += int(np.count_nonzero(~valid))
        valid_values = values[valid]
        duplicate_rows += int(np.count_nonzero(seen[valid_values]))
        seen[valid_values] = True
    return seen, {
        "path": str(path.resolve()),
        "rows": rows,
        "unique_items": int(np.count_nonzero(seen)),
        "duplicate_rows": duplicate_rows,
        "strict_order_violations": order_violations,
        "out_of_range_rows": out_of_range_rows,
    }


def main() -> None:
    args = parse_args()
    detailed = read_json(args.explicit_audit)
    download_manifest = read_json(args.download_manifest)
    flat_manifest = read_json(args.flat_manifest)
    embedding_audit = read_json(args.embedding_audit)
    missing_metadata_audit = read_json(args.missing_metadata_audit)
    if not args.protocol.is_file():
        raise FileNotFoundError(args.protocol)
    if detailed.get("status") != "complete":
        raise ValueError("explicit audit is not a complete full scan")
    if flat_manifest.get("status") != "complete":
        raise ValueError("lossless staging manifest is not complete")

    remote = detailed["remote_dataset"]
    explicit = detailed["explicit_scan"]
    sid = detailed["sid_coverage"]
    raw_events = int(remote["raw_event_count"])
    listens = int(remote["listen_count"])
    explicit_events = int(remote["explicit_count_after_removing_listen"])
    event_counts = {name: int(remote["event_counts"][name]) for name in EVENT_NAMES}
    if listens + explicit_events != raw_events:
        raise ValueError("raw/listen/explicit counts do not reconcile")
    if sum(event_counts.values()) != explicit_events:
        raise ValueError("four explicit source counts do not reconcile")
    if int(explicit["events"]) != explicit_events:
        raise ValueError("full explicit scan event count differs from metadata")
    if {name: int(explicit["event_counts"][name]) for name in EVENT_NAMES} != event_counts:
        raise ValueError("full explicit scan mark counts differ from source metadata")
    flat_counts = {
        name: int(flat_manifest["records"][name]["events"]) for name in EVENT_NAMES
    }
    if flat_counts != event_counts:
        raise ValueError("lossless staging event counts differ from source metadata")

    with np.load(args.item_frequency) as item_data:
        explicit_item_ids = np.asarray(item_data["item_id"], dtype=np.int64)
        explicit_item_counts = np.asarray(item_data["total_count"], dtype=np.uint64)
        sid_covered_flags = np.asarray(item_data["sid_covered"], dtype=bool)
    if explicit_item_ids.size != int(explicit["items"]):
        raise ValueError("item-frequency artifact item count differs from audit")
    if int(explicit_item_counts.sum(dtype=np.uint64)) != explicit_events:
        raise ValueError("item-frequency artifact event sum differs from audit")
    if int(np.count_nonzero(sid_covered_flags)) != int(sid["covered_explicit_items"]):
        raise ValueError("item-frequency artifact SID flags differ from audit")

    embedding_seen, embedding_live = embedding_presence(args.embeddings)
    if embedding_live["unique_items"] != int(embedding_audit["unique_items"]):
        raise ValueError("live embedding item scan differs from embedding audit artifact")
    valid_explicit = (
        (explicit_item_ids >= 0) & (explicit_item_ids < embedding_seen.size)
    )
    explicit_embedding_covered = np.zeros(explicit_item_ids.size, dtype=bool)
    explicit_embedding_covered[valid_explicit] = embedding_seen[
        explicit_item_ids[valid_explicit]
    ]
    embedding_covered_events = int(
        explicit_item_counts[explicit_embedding_covered].sum(dtype=np.uint64)
    )
    if int(missing_metadata_audit["explicit_items"]) != int(explicit["items"]):
        raise ValueError("metadata-support audit explicit item count differs")
    if int(missing_metadata_audit["explicit_events"]) != explicit_events:
        raise ValueError("metadata-support audit explicit event count differs")
    if int(missing_metadata_audit["source_embedding"]["missing"]["items"]) != int(
        np.count_nonzero(~explicit_embedding_covered)
    ):
        raise ValueError("metadata-support audit missing item count differs")

    sequence = explicit["sequence_length"]
    same_time = explicit["timestamp_group_size"]
    transitions = explicit["transition_counts"]
    anomalies = {
        "exact_duplicate_excess_rows": int(explicit["exact_duplicate_excess_rows"]),
        "conflicting_user_item_timestamp_groups": int(
            explicit["conflicting_user_item_timestamp_groups"]
        ),
        "conflicting_event_rows": int(explicit["conflicting_event_rows"]),
        "largest_same_timestamp_group": int(same_time["max"]),
        "events_in_multi_event_groups": int(explicit["events_in_multi_event_groups"]),
        "events_in_multi_event_groups_fraction": float(
            explicit["events_in_multi_event_groups_fraction"]
        ),
        "repeated_like_groups": int(transitions.get("repeated_like_groups", 0)),
        "repeated_dislike_groups": int(transitions.get("repeated_dislike_groups", 0)),
        "unlike_without_active_prior_like_groups": int(
            transitions.get("unlike_without_active_prior_like_groups", 0)
        ),
        "undislike_without_active_prior_dislike_groups": int(
            transitions.get("undislike_without_active_prior_dislike_groups", 0)
        ),
    }
    has_obvious_anomaly = any(
        (
            anomalies["exact_duplicate_excess_rows"] > 0,
            anomalies["conflicting_user_item_timestamp_groups"] > 0,
            anomalies["largest_same_timestamp_group"] >= 100,
        )
    )

    report: dict[str, Any] = {
        "format_version": 2,
        "phase": 0,
        "status": "complete_awaiting_user_confirmation",
        "protocol": str(args.protocol.resolve()),
        "hypothesis": (
            "The pinned full Yambda-5B sequential data can be audited without "
            "cleaning, and its explicit-only event universe and current SID support "
            "can be measured exactly."
        ),
        "data_contract": {
            "repository": remote["files"]["multi_event"]["path"].split(
                "/sequential/5b/"
            )[0],
            "revision": detailed["scope"]["revision"],
            "raw_source": "sequential/5b/multi_event.parquet metadata",
            "explicit_sources": [
                "sequential/5b/likes.parquet",
                "sequential/5b/dislikes.parquet",
                "sequential/5b/unlikes.parquet",
                "sequential/5b/undislikes.parquet",
            ],
            "included_main_events": list(EVENT_NAMES),
            "listen_is_not_a_main_event": True,
            "listen_rows_physically_deleted": False,
            "is_organic_filtering": False,
            "cleaning_or_row_deletion": False,
            "same_timestamp_policy": "shared event group; no invented order",
            "revision_pairing_policy": (
                "same user and same item; the prerequisite state must exist at a "
                "strictly earlier timestamp; ambiguous same-item same-time groups "
                "are not applied to state"
            ),
            "lossless_staging": str(args.flat_manifest.resolve()),
        },
        "data_counts": {
            "raw_events": raw_events,
            "raw_unique_users": int(remote["raw_unique_users"]),
            "raw_unique_items": {
                "value": OFFICIAL_CATALOG_ITEMS,
                "provenance": (
                    "Yambda paper Table 3 / official dataset cardinality; local full "
                    "item-id and remote Parquet statistics confirm the inclusive id "
                    "range 1..9,390,623, but the 14.45 GB compressed multi_event item "
                    "column was not independently distinct-scanned"
                ),
                "independently_distinct_scanned": False,
            },
            "listen_events": listens,
            "explicit_events_after_removing_listen": explicit_events,
            "explicit_users": int(explicit["users"]),
            "explicit_items": int(explicit["items"]),
            "explicit_item_fraction_of_raw_catalog": float(
                explicit["explicit_item_fraction_of_raw_catalog"]
            ),
            "feedback_events": event_counts,
            "users_by_feedback": explicit["users_by_feedback"],
            "organic_counts_retained_for_audit_only": explicit["organic_counts"],
            "recommendation_driven_counts_retained_for_audit_only": explicit[
                "recommendation_driven_counts"
            ],
            "explicit_sequence_length": sequence,
            "user_active_span_seconds": explicit["active_span_seconds"],
            "inter_event_time_seconds": explicit[
                "raw_adjacent_inter_event_gap_seconds_including_ties"
            ],
        },
        "schema": {
            "separate_explicit_files": [
                "uid:uint32",
                "timestamp:list<uint32>",
                "item_id:list<uint32>",
                "is_organic:list<uint8>",
            ],
            "listens_add": [
                "played_ratio_pct:list<uint16>",
                "track_length_seconds:list<uint32>",
            ],
            "multi_event_add": "event_type:list<dictionary<string>>",
            "important_note": (
                "one Parquet row is one user with list-valued events; Parquet row "
                "count is not interaction count"
            ),
        },
        "model_input": "not applicable; Phase 0 prohibits model training",
        "model_output": "not applicable; Phase 0 produces audit artifacts only",
        "training_objective": "not applicable; no model was trained",
        "metrics": {
            "counts": "exact stream counts reconciled against pinned Parquet metadata",
            "quantiles": "exact 5-second-grid empirical quantiles",
            "sid_coverage": "current frozen 50M-derived SID mapping over full-5B explicit items",
            "raw_item_count_limitation": (
                "official exact catalog cardinality, not an independent full distinct scan"
            ),
        },
        "results": {
            "same_timestamp": {
                "groups": int(explicit["timestamp_groups"]),
                "group_size": same_time,
                "events_in_multi_groups": anomalies["events_in_multi_event_groups"],
                "events_in_multi_groups_fraction": anomalies[
                    "events_in_multi_event_groups_fraction"
                ],
                "largest_groups": explicit["largest_timestamp_groups"],
            },
            "duplicates_and_conflicts": {
                "exact_duplicate_excess_rows": anomalies[
                    "exact_duplicate_excess_rows"
                ],
                "exact_duplicate_excess_by_feedback": explicit[
                    "exact_duplicate_excess_by_feedback"
                ],
                "conflicting_user_item_timestamp_groups": anomalies[
                    "conflicting_user_item_timestamp_groups"
                ],
                "conflicting_event_rows": anomalies["conflicting_event_rows"],
                "same_user_item_timestamp_feedback_combinations": explicit[
                    "same_user_item_timestamp_feedback_combinations"
                ],
            },
            "revision_paths": {
                "counts": transitions,
                "state_before_event_counts": explicit["state_before_event_counts"],
                "like_to_unlike_delay_seconds": explicit[
                    "like_to_unlike_delay_seconds"
                ],
                "dislike_to_undislike_delay_seconds": explicit[
                    "dislike_to_undislike_delay_seconds"
                ],
            },
            "item_frequency": explicit["item_explicit_frequency"],
            "top_items": explicit["top_items"],
            "source_embedding_coverage": {
                "raw_catalog_items": OFFICIAL_CATALOG_ITEMS,
                "items_with_embedding": embedding_live["unique_items"],
                "catalog_item_coverage_fraction": (
                    embedding_live["unique_items"] / OFFICIAL_CATALOG_ITEMS
                ),
                "items_without_embedding": (
                    OFFICIAL_CATALOG_ITEMS - embedding_live["unique_items"]
                ),
                "covered_explicit_items": int(
                    np.count_nonzero(explicit_embedding_covered)
                ),
                "uncovered_explicit_items": int(
                    np.count_nonzero(~explicit_embedding_covered)
                ),
                "explicit_item_coverage_fraction": float(
                    np.mean(explicit_embedding_covered)
                ),
                "covered_explicit_events": embedding_covered_events,
                "uncovered_explicit_events": explicit_events - embedding_covered_events,
                "explicit_event_coverage_fraction": (
                    embedding_covered_events / explicit_events
                ),
                "live_scan": embedding_live,
            },
            "missing_embedding_metadata_support": missing_metadata_audit,
            "current_sid_coverage": sid,
        },
        "failure_or_anomaly": {
            "obvious_anomaly_present": has_obvious_anomaly,
            "headline": anomalies,
            "no_rows_removed": True,
            "engineering_constraint": (
                "The full 87.2 GB raw sequential files do not fit the available "
                "data disk and direct four-stream Parquet decoding exceeds the 2 GiB "
                "cgroup limit. Four explicit files were therefore downloaded and "
                "losslessly staged one at a time."
            ),
        },
        "conclusion": (
            "Phase 0 audit evidence is complete for the explicit-only universe. "
            "No cleaning rule has been applied. Existing source embedding and SID "
            "coverage must be reviewed before constructing Phase 1."
        ),
        "sufficient_to_enter_next_phase": False,
        "next_phase_recommendation": (
            "STOP for user confirmation. Decide Phase 1 cleaning rules from the "
            "reported anomalies and resolve how explicit items lacking source "
            "embeddings/current SID are handled without changing the confirmed protocol."
        ),
        "provenance": {
            "download_manifest": download_manifest,
            "flat_manifest": flat_manifest,
            "embedding_catalog_audit": embedding_audit,
            "missing_embedding_metadata_audit": missing_metadata_audit,
            "detailed_explicit_audit": detailed,
            "official_dataset_url": (
                "https://huggingface.co/datasets/yandex/yambda/tree/main/sequential/5b"
            ),
            "paper_url": "https://arxiv.org/abs/2505.22238",
        },
    }
    atomic_json(args.output, report)
    summary = {
        "status": report["status"],
        "raw_events": raw_events,
        "explicit_events": explicit_events,
        "explicit_users": int(explicit["users"]),
        "explicit_items": int(explicit["items"]),
        "feedback_events": event_counts,
        "obvious_anomaly_present": has_obvious_anomaly,
        "sid_item_coverage": float(sid["item_coverage_fraction"]),
        "sid_event_coverage": float(sid["event_coverage_fraction"]),
        "output": str(args.output.resolve()),
        "next": "STOP and await user confirmation",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
