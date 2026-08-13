#!/usr/bin/env python3
"""Create concise human and machine summaries from completed audit JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "timestamp_order_recoverability_audit" / "artifacts"


def dump(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    provenance = json.loads((ART / "row_order_provenance.json").read_text())
    group = json.loads((ART / "group_recoverability_and_burst.json").read_text())
    sensitivity = json.loads((ART / "row_order_sensitivity.json").read_text())
    comp = group["tied_group_composition"]
    rec = group["conservative_recovery"]["classifications"]
    filt = group["hypothetical_filter_sensitivity"]["500"]
    bins = group["extreme_burst_train"]["bins"]

    def dist(mode: str) -> np.ndarray:
        return np.asarray(sensitivity[mode]["feedback_transition_distribution"], dtype=np.float64)

    strict_like = np.asarray(group["strict_revision_pair_origin_bin_by_revision_bin"]["like_to_unlike"], dtype=np.uint64)
    strict_dislike = np.asarray(group["strict_revision_pair_origin_bin_by_revision_bin"]["dislike_to_undislike"], dtype=np.uint64)
    tied = comp["multi_events"]
    local_unique = rec["uniquely_recoverable"]["events"]
    whole_unique = comp["fully_uniquely_ordered_events"]
    summary = {
        "status": "complete_timestamp_order_recoverability_audit",
        "guardrails": {
            "SNMPP_or_HPN_trained": False,
            "group_protocol_modified": False,
            "Gate2_executed": False,
            "events_deleted": False,
            "SNMPP_Phase2_v1_1_modified": False,
        },
        "conservation": {
            **group["counts"],
            "expected_D_SID_events": 121_819_651,
            "expected_D_SID_users": 854_649,
            "event_conservation_pass": group["counts"]["events"] == 121_819_651,
            "user_conservation_pass": group["counts"]["users"] == 854_649,
            "strict_like_to_unlike": int(strict_like.sum()),
            "strict_dislike_to_undislike": int(strict_dislike.sum()),
        },
        "final_answers": {
            "row_order": {
                "answer": "No defensible ground-truth chronology; evidence class B only.",
                "reason": "All reads are engineering-stable, but there is no sub-5-second/event-sequence field and current explicit streams are separate by feedback type.",
            },
            "recoverability": {
                "tied_events": tied,
                "locally_unique_same_item_events": local_unique,
                "locally_unique_fraction_of_tied_events": local_unique / tied,
                "fully_uniquely_orderable_group_events": whole_unique,
                "fully_unique_fraction_of_tied_events": whole_unique / tied,
                "fully_uniquely_orderable_groups": comp["fully_uniquely_ordered_groups"],
                "fully_unique_fraction_of_multi_groups": comp["fully_uniquely_ordered_group_fraction_of_multi"],
            },
            "mixed_protocol": {
                "recommended_as_main": False,
                "reason": "Only 1.96% of tied events belong to a fully uniquely orderable timestamp group; local item order does not order events relative to other items.",
                "allowed_role": "diagnostic/sensitivity analysis only",
            },
            "main_protocol": {
                "keep_grouped_shared_history": True,
                "reason": "It is the only protocol that covers all ties without inventing chronology.",
            },
            "extreme_burst": {
                "separate_sensitivity_subset_recommended": True,
                "candidate_exclusion_rule_for_sensitivity_only": "group_size > 500",
                "not_approved_as_cleaning": True,
                "evidence": {
                    "train_201_500": bins["201-500"],
                    "train_501_1000": bins["501-1000"],
                    "train_1000_plus": bins["1000+"],
                    "hypothetical_retain_le_500": filt,
                },
            },
        },
        "order_sensitivity": {
            "sample": sensitivity["sample"],
            "TV_engineering_original_vs_reversed": float(.5 * np.abs(dist("engineering_original") - dist("reversed")).sum()),
            "TV_engineering_original_vs_random_1": float(.5 * np.abs(dist("engineering_original") - dist("random_1")).sum()),
            "TV_random_1_vs_random_2": float(.5 * np.abs(dist("random_1") - dist("random_2")).sum()),
            "TV_conservative_mixed_vs_grouped": float(.5 * np.abs(dist("conservative_mixed") - dist("grouped")).sum()),
            "like_to_unlike_adjacent": {mode: sensitivity[mode]["adjacent_like_to_unlike_same_item"] for mode in ("engineering_original", "reversed", "random_1", "random_2", "random_3", "grouped", "conservative_mixed")},
            "dislike_to_undislike_adjacent": {mode: sensitivity[mode]["adjacent_dislike_to_undislike_same_item"] for mode in ("engineering_original", "reversed", "random_1", "random_2", "random_3", "grouped", "conservative_mixed")},
            "zero_delta_ordered_transitions": {mode: sensitivity[mode]["zero_delta_ordered_transitions"] for mode in ("engineering_original", "reversed", "random_1", "random_2", "random_3", "grouped", "conservative_mixed")},
            "conclusion": "Arbitrary ordering materially changes revision adjacency and introduces many zero-delta transitions; random permutations are mutually stable but differ from deterministic type ordering, demonstrating assumption-driven structure.",
        },
        "artifacts": {
            "provenance": str((ART / "row_order_provenance.json").resolve()),
            "group_and_burst": str((ART / "group_recoverability_and_burst.json").resolve()),
            "sensitivity": str((ART / "row_order_sensitivity.json").resolve()),
        },
    }
    dump(ART / "audit_summary.json", summary)

    r = summary["final_answers"]["recoverability"]
    lines = [
        "Timestamp Order Recoverability Audit",
        "====================================",
        "",
        "Scope: audit only; no model training, event deletion, Gate 2, or protocol modification.",
        "",
        f"D_SID: {group['counts']['users']:,} users, {group['counts']['events']:,} events, {group['counts']['groups']:,} timestamp groups.",
        f"Row order: evidence class {provenance['classification']} (engineering-stable, not business chronology).",
        f"Tied events: {tied:,} ({tied / group['counts']['events']:.4%} of D_SID).",
        f"Locally unique same-item order: {r['locally_unique_same_item_events']:,} events ({r['locally_unique_fraction_of_tied_events']:.4%} of tied events).",
        f"Fully unique whole timestamp group: {r['fully_uniquely_orderable_group_events']:,} events ({r['fully_unique_fraction_of_tied_events']:.4%} of tied events).",
        "Decision: do not adopt a mixed protocol as the main protocol; retain grouped shared pre-history.",
        "",
        "Extreme burst:",
        "A natural batch-like composition shift appears above group size 500. Treat >500 only as an independent sensitivity/exclusion candidate, not as approved cleaning.",
        f"Hypothetical <=500 retention: events {filt['event_retention']:.4%}, groups {filt['group_retention']:.4%}, users {filt['user_retention']:.4%}.",
        f"Strict pair retention: like->unlike {filt['strict_like_to_unlike_pair_retention']:.4%}; dislike->undislike {filt['strict_dislike_to_undislike_pair_retention']:.4%}.",
        "",
        "Order sensitivity:",
        f"Sample: {sensitivity['sample']['users']:,} users, {sensitivity['sample']['multi_groups']:,} multi-event groups.",
        f"Like->unlike adjacent counts: engineering order {sensitivity['engineering_original']['adjacent_like_to_unlike_same_item']:.0f}, reversed {sensitivity['reversed']['adjacent_like_to_unlike_same_item']:.0f}, random seeds {sensitivity['random_1']['adjacent_like_to_unlike_same_item']:.0f}/{sensitivity['random_2']['adjacent_like_to_unlike_same_item']:.0f}/{sensitivity['random_3']['adjacent_like_to_unlike_same_item']:.0f}.",
        f"Ordered variants create {sensitivity['engineering_original']['zero_delta_ordered_transitions']:.0f} zero-delta transitions; grouped creates 0.",
    ]
    (ART / "audit_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
