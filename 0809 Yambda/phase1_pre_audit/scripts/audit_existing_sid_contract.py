#!/usr/bin/env python3
"""Read-only audit of the historical RQ/SID contract used by 0804 Yambda."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = Path("/root/autodl-tmp/0804 Yambda")
DEFAULT_STORE = (
    LEGACY_ROOT
    / "dataprocess"
    / "artifacts"
    / "metadata_imputation_fixed_codebook_seed2026"
)
DEFAULT_SOURCE_STORE = (
    LEGACY_ROOT / "dataprocess" / "artifacts" / "rq4_collision_aware_seed2026"
)
DEFAULT_HPN_CONFIG = LEGACY_ROOT / "configs" / "nolisten_hpn_high_coverage.yaml"
DEFAULT_HPN_DATA = LEGACY_ROOT / "snmpp" / "recommendation" / "data.py"
DEFAULT_HPN_EVALUATION = LEGACY_ROOT / "snmpp" / "recommendation" / "evaluation.py"
DEFAULT_OLD_SPLIT = (
    LEGACY_ROOT / "processed" / "snmpp_yambda_nolisten_max50" / "manifest.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "phase1_pre_audit" / "artifacts" / "existing_sid_contract.json"
)
OFFICIAL_MAX_ITEM_ID = 9_390_623
NEW_EXPLICIT_ITEMS = 2_875_071
NEW_REAL_EMBEDDING_ITEMS = 2_367_341
NEW_METADATA_SUPPORTED_PROXY_ITEMS = 450_594
NEW_COLD_ITEMS = 57_136


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--source-store", type=Path, default=DEFAULT_SOURCE_STORE)
    parser.add_argument("--hpn-config", type=Path, default=DEFAULT_HPN_CONFIG)
    parser.add_argument("--old-split", type=Path, default=DEFAULT_OLD_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    store = args.store.resolve()
    codebooks = np.load(store / "codebooks.npy", mmap_mode="r")
    semantic = np.load(store / "semantic_codes.npy", mmap_mode="r")
    full = np.load(store / "full_codes.npy", mmap_mode="r")
    suffix = np.load(store / "collision_suffix.npy", mmap_mode="r")
    item_ids = np.load(store / "catalog_item_ids.npy", mmap_mode="r")
    inverse = np.load(store / "orig2catalog_row.npy", mmap_mode="r")
    features = np.load(store / "catalog_features.npy", mmap_mode="r")
    feature_source = np.load(store / "feature_source.npy", mmap_mode="r")
    manifest = read_json(store / "manifest.json")
    metrics = read_json(store / "metrics.json")
    source_manifest = read_json(args.source_store / "manifest.json")
    split_manifest = read_json(args.old_split)
    hpn_config = yaml.safe_load(args.hpn_config.read_text(encoding="utf-8"))
    data_source = DEFAULT_HPN_DATA.read_text(encoding="utf-8")
    evaluation_source = DEFAULT_HPN_EVALUATION.read_text(encoding="utf-8")

    expected_rows = np.arange(len(item_ids), dtype=np.int64)
    inverse_rows = np.asarray(inverse[np.asarray(item_ids, dtype=np.int64)], dtype=np.int64)
    semantic_unique = int(np.unique(semantic, axis=0).shape[0])
    full_unique = int(np.unique(full, axis=0).shape[0])
    semantic_levels = int(semantic.shape[1])
    full_levels = int(full.shape[1])
    codebook_levels, codebook_size, code_dim = map(int, codebooks.shape)
    suffix_vocab = int(suffix.max()) + 1
    minimum_base256_cold_disambiguation_levels = math.ceil(
        math.log(max(NEW_COLD_ITEMS, 1), 256)
    )

    checks = {
        "four_semantic_levels": semantic_levels == 4,
        "four_codebooks": codebook_levels == semantic_levels,
        "codebook_size_256_each": codebook_size == 256,
        "codebook_vector_dimension_128": code_dim == 128,
        "fifth_level_exists": full_levels == semantic_levels + 1,
        "fifth_level_equals_collision_suffix": bool(
            np.array_equal(full[:, -1], suffix)
        ),
        "first_four_full_levels_equal_semantic_codes": bool(
            np.array_equal(full[:, :semantic_levels], semantic)
        ),
        "fifth_level_has_no_codebook_vectors": codebook_levels < full_levels,
        "full_sid_unique_per_catalog_item": full_unique == len(item_ids),
        "semantic_sid_not_unique_per_catalog_item": semantic_unique < len(item_ids),
        "catalog_item_ids_unique": int(np.unique(item_ids).size) == len(item_ids),
        "orig2catalog_inverse_consistent": bool(
            np.array_equal(inverse_rows, expected_rows)
        ),
        "feature_rows_align": features.shape[0] == len(item_ids),
        "feature_source_rows_align": feature_source.shape == (len(item_ids),),
        "codebooks_finite": bool(np.isfinite(codebooks).all()),
    }
    if not all(checks.values()):
        raise ValueError(f"historical SID contract check failed: {checks}")

    output = {
        "status": "complete_read_only_audit",
        "historical_store": str(store),
        "structure": {
            "catalog_items": int(len(item_ids)),
            "semantic_levels": semantic_levels,
            "semantic_vocabulary_per_level": [
                int(np.unique(semantic[:, level]).size)
                for level in range(semantic_levels)
            ],
            "codebooks_shape": list(map(int, codebooks.shape)),
            "codebook_vector_dimension": code_dim,
            "full_identity_levels": full_levels,
            "fifth_level": {
                "kind": "deterministic item-id-ordered collision suffix",
                "semantic": False,
                "has_codebook_vector": False,
                "observed_min": int(suffix.min()),
                "observed_max": int(suffix.max()),
                "observed_vocabulary_size": suffix_vocab,
            },
            "semantic_unique_paths": semantic_unique,
            "semantic_excess_colliding_items": int(len(item_ids) - semantic_unique),
            "full_unique_paths": full_unique,
            "item_id_min": int(item_ids.min()),
            "item_id_max": int(item_ids.max()),
            "orig2catalog_length": int(len(inverse)),
            "exact_item_recovery": (
                "full 5-token path is unique inside this historical catalog; "
                "catalog_item_ids plus inverse map recover the item"
            ),
        },
        "checks": checks,
        "historical_training_contract": {
            "source_codebook_items": int(
                source_manifest["outputs"]["catalog_item_ids.npy"]["shape"][0]
            ),
            "source_feature_type": "normalized 128-d audio embedding",
            "fit_scope": source_manifest["data_contract"]["fit_scope"],
            "fit_uses_interaction_labels": source_manifest["data_contract"][
                "fit_uses_interaction_labels"
            ],
            "codebook_refitted_after_proxy_imputation": bool(
                metrics["encoding"]["codebooks_refitted"]
            ),
            "old_proxy_collaborative_split_strategy": split_manifest["split"][
                "strategy"
            ],
            "old_proxy_collaborative_train_users": split_manifest["splits"]["train"][
                "user_count"
            ],
            "current_protocol_compatible": False,
            "incompatibility": (
                "old item universe and collaborative fallback came from a 50M-derived "
                "pipeline with user-disjoint splitting, not the current full-5B global "
                "chronological protocol"
            ),
        },
        "interaction_leakage_audit": {
            "kmeans_objective_uses_feedback_labels": False,
            "item_universe_selected_from_full_old_explicit_log": True,
            "global_time_inductive_codebook_claim_supported": False,
            "classification": "transductive item-universe selection",
            "note": (
                "Static audio vectors are not future labels, but selecting the fit catalog "
                "from all explicit periods reveals which items occur later. The new fixed "
                "protocol may deliberately retain the full explicit universe, but this must "
                "be disclosed rather than called inductive or leakage-free."
            ),
        },
        "historical_hpn_contract": {
            "configured_predicted_levels": int(hpn_config["model"]["sid_levels"]),
            "stored_semantic_levels": semantic_levels,
            "stored_identity_levels": full_levels,
            "predicts_fifth_suffix": int(hpn_config["model"]["sid_levels"]) == full_levels,
            "separate_collision_resolution": (
                "matched-training popularity then dense item id inside equal semantic prefix"
            ),
            "code_evidence": {
                "store_exposes_full_codes": "self._full_codes" in data_source,
                "evaluation_has_separate_collision_resolver": (
                    "resolve_semantic_prefix_ties" in evaluation_source
                ),
            },
        },
        "new_full5b_requirements": {
            "explicit_items": NEW_EXPLICIT_ITEMS,
            "real_embedding_items_for_codebook_fit": NEW_REAL_EMBEDDING_ITEMS,
            "metadata_supported_proxy_items_quantized_after_freeze": (
                NEW_METADATA_SUPPORTED_PROXY_ITEMS
            ),
            "cold_unknown_items": NEW_COLD_ITEMS,
            "orig2catalog_required_length": OFFICIAL_MAX_ITEM_ID + 1,
            "single_256_way_suffix_can_distinguish_all_cold_items": (
                NEW_COLD_ITEMS <= 256
            ),
            "minimum_base256_disambiguation_levels_for_cold_bucket": (
                minimum_base256_cold_disambiguation_levels
            ),
            "hard_issue": (
                "If all cold items share one unknown semantic vector, they share one "
                "four-level semantic path. One historical 256-way suffix is insufficient; "
                "identity needs a larger vocabulary, at least two base-256 suffix levels, "
                "or a separate exact-item resolver. Proxy centroids may create additional "
                "large identical-vector buckets, so the final maximum bucket must be audited."
            ),
        },
        "decision_boundary": (
            "Do not train a new codebook or assign final SIDs until the disambiguation "
            "contract is selected and proxy leave-one-out quality is measured."
        ),
        "provenance": {
            "store_manifest": manifest,
            "source_store_manifest": source_manifest,
            "old_split_manifest": split_manifest,
            "hpn_config": str(args.hpn_config.resolve()),
            "hpn_data_source": str(DEFAULT_HPN_DATA.resolve()),
            "hpn_evaluation_source": str(DEFAULT_HPN_EVALUATION.resolve()),
        },
    }
    atomic_json(args.output, output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "structure": output["structure"],
                "historical_training_contract": output[
                    "historical_training_contract"
                ],
                "interaction_leakage_audit": output["interaction_leakage_audit"],
                "historical_hpn_contract": output["historical_hpn_contract"],
                "new_full5b_requirements": output["new_full5b_requirements"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
