#!/usr/bin/env python3
"""Read-only provenance audit for explicit items missing source audio embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS = Path("/root/autodl-tmp/0804 Yambda/data/embeddings.parquet")
SUPPORT = ROOT / "phase1_gates" / "artifacts" / "metadata_loo_support.npz"
FEATURE_ROOT = Path("/root/autodl-tmp/0626/0626 Predictor/01_data/processed/raw_rqkmeans")
TOKENS = Path("/root/autodl-tmp/0804 Yambda/data/item_tokens.json")
OUTPUT = ROOT / "sid_feasibility_gate" / "artifacts" / "missing_embedding_sources.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    with np.load(SUPPORT, allow_pickle=False) as support:
        explicit_item = support["item_id"]
        audited_real = support["real_embedding"]
        audited_dense = support["dense_id"]
    parquet = pq.ParquetFile(EMBEDDINGS)
    source_present = np.zeros(int(explicit_item.max()) + 1, dtype=bool)
    source_rows = 0
    source_min = None
    source_max = None
    duplicate_rows = 0
    for row_group in range(parquet.num_row_groups):
        item = parquet.read_row_group(row_group, columns=["item_id"])["item_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        source_rows += len(item)
        source_min = int(item.min()) if source_min is None else min(source_min, int(item.min()))
        source_max = int(item.max()) if source_max is None else max(source_max, int(item.max()))
        in_explicit_domain = item <= int(explicit_item.max())
        selected = item[in_explicit_domain]
        duplicate_rows += int(source_present[selected].sum())
        source_present[selected] = True
    independently_real = source_present[explicit_item]
    mismatch = independently_real != audited_real

    orig2dense = np.load(FEATURE_ROOT / "orig2dense_item_id.npy", mmap_mode="r")
    dense2orig = np.load(FEATURE_ROOT / "dense2orig_item_id.npy", mmap_mode="r")
    dense_features = np.load(FEATURE_ROOT / "dense_item_features.npy", mmap_mode="r")
    mapped_dense = np.asarray(orig2dense[explicit_item], dtype=np.int64)
    # Dense row 0 is the explicit missing/sentinel row; source-backed items
    # start at dense row 1.
    mapping_real = mapped_dense > 0
    real_roundtrip = dense2orig[mapped_dense[audited_real]] == explicit_item[audited_real]
    missing_dense_values, missing_dense_counts = np.unique(mapped_dense[~audited_real], return_counts=True)
    support_dense_match = audited_dense == mapped_dense

    mapping_meta = json.loads((FEATURE_ROOT / "mapping.meta.json").read_text())
    old_mapping_meta = json.loads(Path("/root/autodl-tmp/0408Yambda/artifacts/mappings/yambda_item_id_mapping.meta.json").read_text())
    metadata_proxy_manifest = json.loads(Path("/root/autodl-tmp/0804 Yambda/dataprocess/artifacts/metadata_imputation_fixed_codebook_seed2026/manifest.json").read_text())
    rq_manifest = json.loads(Path("/root/autodl-tmp/0804 Yambda/dataprocess/artifacts/rq4_collision_aware_seed2026/manifest.json").read_text())
    with np.load("/root/autodl-tmp/0408Yambda/artifacts/env/yambda_user_env.feature_cache.npz", allow_pickle=False) as cache:
        cache_shapes = {key: list(cache[key].shape) for key in cache.files}
    token_last_key = str(source_rows - 1)
    with TOKENS.open("rb") as handle:
        handle.seek(max(0, TOKENS.stat().st_size - 4096))
        token_tail = handle.read().decode("utf-8", errors="ignore")

    result = {
        "status": "complete_missing_embedding_source_audit",
        "source_embedding": {
            "path": str(EMBEDDINGS.resolve()),
            "sha256": sha256(EMBEDDINGS),
            "schema": str(parquet.schema_arrow),
            "rows": source_rows,
            "row_groups": parquet.num_row_groups,
            "item_id_min": source_min,
            "item_id_max": source_max,
            "duplicate_item_rows_during_explicit_domain_membership_scan": duplicate_rows,
        },
        "explicit_membership_cross_check": {
            "explicit_items": int(len(explicit_item)),
            "covered_by_source_parquet": int(independently_real.sum()),
            "missing_from_source_parquet": int((~independently_real).sum()),
            "mismatch_vs_existing_support_real_flag": int(mismatch.sum()),
            "conclusion": "existing 507,730 missing flags exactly match absence from the source parquet" if not mismatch.any() else "mapping mismatch detected",
        },
        "dense_mapping_cross_check": {
            "mapping_meta": mapping_meta,
            "orig2dense_shape": list(orig2dense.shape),
            "dense2orig_shape": list(dense2orig.shape),
            "dense_features_shape": list(dense_features.shape),
            "explicit_real_items_with_valid_dense_mapping": int(mapping_real[audited_real].sum()),
            "explicit_missing_items_with_valid_dense_mapping": int(mapping_real[~audited_real].sum()),
            "real_item_roundtrip_passed": int(real_roundtrip.sum()),
            "real_item_roundtrip_total": int(len(real_roundtrip)),
            "support_dense_id_mismatch": int((~support_dense_match).sum()),
            "missing_dense_id_values": {str(int(value)): int(count) for value, count in zip(missing_dense_values, missing_dense_counts)},
        },
        "other_local_artifacts": {
            "item_tokens": {
                "path": str(TOKENS.resolve()),
                "size_bytes": TOKENS.stat().st_size,
                "last_expected_dense_key_seen_in_tail": f'"{token_last_key}"' in token_tail,
                "role": "four discrete tokens for the same 7,721,749 source-embedding rows; no additional continuous item embedding coverage",
            },
            "historical_dense_store": {
                "role": "float32 conversion of normalized_embed from the same embeddings parquet",
                "source": mapping_meta,
            },
            "old_0408_mapping": {
                "role": "mapping generated from the same-size 7,721,749-row embeddings parquet, not an alternate source",
                "metadata": old_mapping_meta,
            },
            "0408_feature_cache": {
                "shapes": cache_shapes,
                "role": "144,482-row task subset indexed by existing dense IDs; not additional catalog coverage",
            },
            "metadata_imputation_artifact": {
                "catalog_items": metadata_proxy_manifest["outputs"]["catalog_item_ids.npy"]["shape"][0],
                "feature_source_labels": metadata_proxy_manifest["feature_source_labels"],
                "role": "derived real/metadata/collaborative proxy features for an older 244,553-item protocol; not a new source embedding",
            },
            "old_rq_artifact": {
                "catalog_items": rq_manifest["outputs"]["catalog_item_ids.npy"]["shape"][0],
                "fit_scope": rq_manifest["data_contract"]["fit_scope"],
                "missing_feature_policy": rq_manifest["data_contract"]["missing_feature_policy"],
                "role": "RQ codes/features derived from the same embeddings parquet for an older filtered item universe",
            },
            "0330_data_directory": {
                "exists": Path("/root/autodl-tmp/0330Yambda/data").exists(),
                "note": "historical manifests point there, but the current directory is absent; current embeddings.parquet has the identical recorded byte size",
            },
            "llama_embedding_generator": {
                "script": "/root/autodl-tmp/0626/0626 Predictor/0525DIGER/scripts/generate_llama_item_embeddings.py",
                "item_embedding_output_found_in_workspace": False,
                "role": "generator script only; no produced item-specific embedding artifact found",
            },
        },
        "conclusion": {
            "missing_is_current_mapping_bug": False,
            "missing_is_absent_from_local_source_embedding_parquet": True,
            "alternate_item_specific_embedding_covering_missing_items_found": False,
            "missing_items": int((~audited_real).sum()),
        },
        "boundaries": {
            "proxy_training_performed": False,
            "unk_sid_generated": False,
            "suffix_designed": False,
            "gate2_started": False,
            "snmpp_or_hpn_trained": False,
        },
    }
    if source_rows != 7_721_749 or int((~independently_real).sum()) != 507_730:
        raise RuntimeError("source count changed")
    if mismatch.any() or not real_roundtrip.all() or not support_dense_match.all():
        raise RuntimeError("embedding mapping cross-check failed")
    atomic_json(OUTPUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
