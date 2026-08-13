#!/usr/bin/env python3
"""Combine the three read-only SID Feasibility Gate audits."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "sid_feasibility_gate" / "artifacts"
OUTPUT_JSON = ARTIFACTS / "sid_feasibility_gate_report.json"
OUTPUT_TEXT = ROOT / "sid_feasibility_gate" / "sid_feasibility_gate_report.txt"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    collision = json.loads((ARTIFACTS / "real_sid_collision.json").read_text())
    bias = json.loads((ARTIFACTS / "real_audio_subset_bias.json").read_text())
    source = json.loads((ARTIFACTS / "missing_embedding_sources.json").read_text())
    c = collision["collision"]
    e = bias["events"]
    p = bias["strict_revision_pairs"]
    u = bias["user_sequences"]
    g = bias["same_timestamp_groups"]
    retention_values = [e["by_feedback"][mark]["retention_rate"] for mark in ("like", "dislike", "unlike", "undislike")]
    feedback_retention_gap = max(retention_values) - min(retention_values)
    report = {
        "status": "sid_feasibility_gate_complete_stop_before_gate2",
        "audit_1_real_embedding_sid_collision": collision,
        "audit_2_real_audio_only_bias": bias,
        "audit_3_missing_embedding_sources": source,
        "decisions": {
            "four_layer_sid_can_serve_exact_item_identity": False,
            "identity_reason": (
                f"{c['collision_items']:,}/{collision['data_contract']['items']:,} real items "
                f"({100*c['collision_item_ratio']:.2f}%) collide; maximum bucket size is "
                f"{c['all_bucket_size']['max']:,}."
            ),
            "real_audio_only_has_serious_feedback_type_bias": True,
            "feedback_bias_reason": (
                f"like/dislike retain {100*e['by_feedback']['like']['retention_rate']:.2f}%/"
                f"{100*e['by_feedback']['dislike']['retention_rate']:.2f}%, but unlike/undislike "
                f"retain only {100*e['by_feedback']['unlike']['retention_rate']:.2f}%/"
                f"{100*e['by_feedback']['undislike']['retention_rate']:.2f}%; max-min gap is "
                f"{100*feedback_retention_gap:.2f} percentage points."
            ),
            "real_audio_only_strict_revision_pair_bias": "limited but nonzero",
            "strict_revision_reason": (
                f"strict like->unlike retains {100*p['like_to_unlike']['retention_rate']:.2f}% and "
                f"strict dislike->undislike retains {100*p['dislike_to_undislike']['retention_rate']:.2f}%."
            ),
            "real_audio_only_recommended_role": "sensitivity subset only; not a distribution-preserving replacement for D_all",
            "missing_is_mapping_bug": source["conclusion"]["missing_is_current_mapping_bug"],
            "alternate_item_specific_embedding_found": source["conclusion"]["alternate_item_specific_embedding_covering_missing_items_found"],
            "proceed_to_gate2": False,
        },
        "headline": {
            "semantic_sid": {
                "real_items": collision["data_contract"]["items"],
                "unique_paths": c["unique_semantic_sid_paths"],
                "singleton_item_ratio": c["singleton_item_ratio"],
                "collision_item_ratio": c["collision_item_ratio"],
                "collision_buckets": c["collision_buckets"],
                "bucket_p90": c["all_bucket_size"]["p90"],
                "bucket_p99": c["all_bucket_size"]["p99"],
                "bucket_p99_9": c["all_bucket_size"]["p99_9"],
                "bucket_max": c["all_bucket_size"]["max"],
            },
            "real_audio_subset": {
                "event_retention": e["retention_rate"],
                "feedback_retention": {mark: e["by_feedback"][mark]["retention_rate"] for mark in e["by_feedback"]},
                "strict_pair_retention": {
                    "like_to_unlike": p["like_to_unlike"]["retention_rate"],
                    "dislike_to_undislike": p["dislike_to_undislike"]["retention_rate"],
                },
                "user_retention": u["user_retention_rate"],
                "median_length_before": u["length_before_all_users"]["median"],
                "median_length_after": u["length_after_same_original_users_including_zero"]["median"],
                "timestamp_group_retention": g["changes"]["group_retention_rate"],
                "multi_event_fraction_before": g["before"]["events_in_multi_groups_fraction"],
                "multi_event_fraction_after": g["after_real_audio_only"]["events_in_multi_groups_fraction"],
            },
            "missing_source": source["conclusion"],
        },
        "boundaries": {
            "unk_sid_generated": False,
            "suffix_designed": False,
            "gate2_started": False,
            "snmpp_or_hpn_trained": False,
        },
    }
    atomic_json(OUTPUT_JSON, report)
    lines = [
        "SID Feasibility Gate（STOP before Gate 2）",
        "",
        "1. Real-embedding SID collision",
        f"real items: {collision['data_contract']['items']:,}",
        f"unique 4-level paths: {c['unique_semantic_sid_paths']:,}",
        f"singleton item ratio: {100*c['singleton_item_ratio']:.4f}%",
        f"collision item ratio: {100*c['collision_item_ratio']:.4f}% ({c['collision_items']:,} items)",
        f"collision buckets: {c['collision_buckets']:,}",
        f"all-bucket P90/P99/P99.9/max: {c['all_bucket_size']['p90']}/{c['all_bucket_size']['p99']}/{c['all_bucket_size']['p99_9']}/{c['all_bucket_size']['max']}",
        "Decision: 4-layer semantic SID cannot serve exact-item identity.",
        "Note: audited codebook is the requested frozen 4x256 candidate, whose own report marks it provisional and fit on 100,000 real items.",
        "",
        "2. Real-audio-only D_all bias",
        f"overall event retention: {100*e['retention_rate']:.4f}% ({e['real_audio_only']:,}/{e['all']:,})",
        f"like retention: {100*e['by_feedback']['like']['retention_rate']:.4f}%",
        f"dislike retention: {100*e['by_feedback']['dislike']['retention_rate']:.4f}%",
        f"unlike retention: {100*e['by_feedback']['unlike']['retention_rate']:.4f}%",
        f"undislike retention: {100*e['by_feedback']['undislike']['retention_rate']:.4f}%",
        f"strict like->unlike pair retention: {100*p['like_to_unlike']['retention_rate']:.4f}% ({p['like_to_unlike']['real_item']:,}/{p['like_to_unlike']['all']:,})",
        f"strict dislike->undislike pair retention: {100*p['dislike_to_undislike']['retention_rate']:.4f}% ({p['dislike_to_undislike']['real_item']:,}/{p['dislike_to_undislike']['all']:,})",
        f"user retention: {100*u['user_retention_rate']:.4f}% ({u['retained_users']:,}/{u['all_users']:,})",
        f"sequence median: {u['length_before_all_users']['median']} -> {u['length_after_same_original_users_including_zero']['median']}",
        f"sequence mean: {u['length_before_all_users']['mean']:.3f} -> {u['length_after_same_original_users_including_zero']['mean']:.3f}",
        f"timestamp groups retained: {100*g['changes']['group_retention_rate']:.4f}%",
        f"multi-group event fraction: {100*g['before']['events_in_multi_groups_fraction']:.4f}% -> {100*g['after_real_audio_only']['events_in_multi_groups_fraction']:.4f}%",
        "Decision: serious feedback-type composition bias, especially loss of unmatched/left-censored revision events; strict matched revision paths are much less affected. Use only as a sensitivity subset.",
        "",
        "3. Missing embedding source",
        f"source parquet rows: {source['source_embedding']['rows']:,}",
        f"explicit real/missing items: {source['explicit_membership_cross_check']['covered_by_source_parquet']:,}/{source['explicit_membership_cross_check']['missing_from_source_parquet']:,}",
        f"support-vs-parquet membership mismatches: {source['explicit_membership_cross_check']['mismatch_vs_existing_support_real_flag']}",
        f"real mapping round-trip: {source['dense_mapping_cross_check']['real_item_roundtrip_passed']:,}/{source['dense_mapping_cross_check']['real_item_roundtrip_total']:,}",
        "Decision: missing is genuine absence from the local official embedding parquet, not the current item mapping. No alternate item-specific embedding covering these items was found.",
        "",
        "STOP: no UNK SID, no suffix design, no Gate 2, no SNMPP/HPN training.",
    ]
    OUTPUT_TEXT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"output_json": str(OUTPUT_JSON), "output_text": str(OUTPUT_TEXT), "decisions": report["decisions"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
