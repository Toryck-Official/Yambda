#!/usr/bin/env python3
"""Validate the full timestamp-burst audit against the frozen Phase-0 counts."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = (
    PROJECT_ROOT
    / "phase1_pre_audit"
    / "artifacts"
    / "timestamp_burst_sensitivity.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "phase1_pre_audit"
    / "artifacts"
    / "timestamp_burst_validation.json"
)

EXPECTED = {
    "users": 857_499,
    "events": 136_292_476,
    "timestamp_groups": 80_259_999,
    "feedback_events": {
        "like": 89_334_605,
        "dislike": 11_579_143,
        "unlike": 32_944_520,
        "undislike": 2_434_208,
    },
    "strict_revision_pairs": {
        "like_to_unlike": 6_364_678,
        "dislike_to_undislike": 308_204,
    },
    "ambiguous_same_item_time_groups": 3_612_417,
}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    counts = audit["counts"]
    histograms = audit["histograms"]

    checks = {
        "audit_status_complete": audit["status"] == "complete_read_only_audit",
        "no_filtering_performed": not audit["data_contract"][
            "cleaning_or_filtering_performed"
        ],
        "no_same_time_order_invented": not audit["data_contract"][
            "same_timestamp_order_invented"
        ],
        "users_match_phase0": counts["users"] == EXPECTED["users"],
        "events_match_phase0": counts["events"] == EXPECTED["events"],
        "groups_match_phase0": (
            counts["timestamp_groups"] == EXPECTED["timestamp_groups"]
        ),
        "feedback_match_phase0": (
            counts["feedback_events"] == EXPECTED["feedback_events"]
        ),
        "strict_pairs_match_phase0": (
            counts["strict_revision_pairs"] == EXPECTED["strict_revision_pairs"]
        ),
        "ambiguity_groups_match_phase0": (
            counts["ambiguous_same_item_time_groups"]
            == EXPECTED["ambiguous_same_item_time_groups"]
        ),
        "group_histogram_conserves_groups": (
            sum(histograms["group_count_by_size"])
            == counts["timestamp_groups"]
        ),
        "event_histogram_conserves_events": (
            sum(histograms["event_count_by_group_size"]) == counts["events"]
        ),
        "feedback_histograms_conserve_events": (
            sum(
                sum(values)
                for values in histograms["feedback_count_by_group_size"].values()
            )
            == counts["events"]
        ),
        "user_max_histogram_conserves_users": (
            sum(histograms["user_count_by_max_group_size"]) == counts["users"]
        ),
        "like_pair_histogram_conserves_pairs": (
            sum(histograms["strict_like_to_unlike_count_by_max_endpoint_group_size"])
            == counts["strict_revision_pairs"]["like_to_unlike"]
        ),
        "dislike_pair_histogram_conserves_pairs": (
            sum(
                histograms[
                    "strict_dislike_to_undislike_count_by_max_endpoint_group_size"
                ]
            )
            == counts["strict_revision_pairs"]["dislike_to_undislike"]
        ),
    }
    passed = all(checks.values())

    concise_effects = {}
    for label, effect in audit["threshold_effects"].items():
        concise_effects[label] = {
            "threshold_keep_group_size_leq": effect[
                "threshold_inclusive_keep_max"
            ],
            "excluded": effect["excluded"],
            "strict_revision_pairs_touched_at_either_endpoint": effect[
                "strict_revision_pairs_touched_at_either_endpoint"
            ],
        }

    output = {
        "status": "passed" if passed else "failed",
        "source": str(AUDIT_PATH),
        "checks": checks,
        "counts": counts,
        "group_size_quantiles": audit["group_size_quantiles"],
        "group_size_max": audit["group_size_max"],
        "threshold_effects": concise_effects,
        "important_boundary": (
            "The audit is over raw explicit rows before the deterministic exact-"
            "duplicate removal. Phase 1 must recompute group sizes and selected-"
            "threshold effects after deduplication before materializing "
            "batch_event_flag."
        ),
    }
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

