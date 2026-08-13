#!/usr/bin/env python3
"""Evaluate strict metadata proxies under the frozen Gate-1 protocol."""

from __future__ import annotations

import argparse
import gc
import json
import mmap
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LEGACY_DATAPROCESS = Path("/root/autodl-tmp/0804 Yambda/dataprocess")
sys.path.insert(0, str(LEGACY_DATAPROCESS))

from rqdataprocess.rq import encode_with_codebooks  # noqa: E402


DEFAULT_CONFIG = ROOT / "phase1_gates" / "configs" / "gates.json"
DEFAULT_SAMPLE = ROOT / "phase1_gates" / "artifacts" / "proxy_validation_sample.npz"
DEFAULT_SUPPORT = ROOT / "phase1_gates" / "artifacts" / "metadata_loo_support.npz"
DEFAULT_PROXIES = ROOT / "phase1_gates" / "work" / "proxy_vectors"
DEFAULT_CODEBOOK = ROOT / "phase1_gates" / "artifacts" / "gate1_candidate_rq" / "codebooks.npy"
DEFAULT_SUPPORT_REPORT = ROOT / "phase1_gates" / "artifacts" / "metadata_support_sample_report.json"
DEFAULT_OUTPUT = ROOT / "phase1_gates" / "artifacts" / "gate1_proxy_metrics.json"

AVAILABILITY_NAMES = {1: "artist_only", 2: "album_only", 3: "both"}
SIZE_TIER_NAMES = {1: "small", 2: "medium", 3: "large"}
FREQUENCY_TIER_NAMES = {0: "tail", 1: "mid", 2: "head"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--proxy-dir", type=Path, default=DEFAULT_PROXIES)
    parser.add_argument("--codebooks", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--support-report", type=Path, default=DEFAULT_SUPPORT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def normalized_blend(
    artist: np.ndarray,
    album: np.ndarray,
    artist_available: np.ndarray,
    album_available: np.ndarray,
    album_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    available = artist_available | album_available
    both = artist_available & album_available
    output = np.zeros_like(artist)
    output[artist_available & ~album_available] = artist[
        artist_available & ~album_available
    ]
    output[album_available & ~artist_available] = album[
        album_available & ~artist_available
    ]
    output[both] = (
        (1.0 - album_weight) * artist[both] + album_weight * album[both]
    )
    norms = np.linalg.norm(output, axis=1)
    valid_norm = np.isfinite(norms) & (norms > 1e-12)
    available &= valid_norm
    output[available] /= norms[available, None]
    return output, available


def rowwise_diagnostics(
    truth: np.ndarray,
    proxy: np.ndarray,
    true_codes: np.ndarray,
    codebooks: np.ndarray,
) -> dict[str, np.ndarray]:
    proxy_codes = encode_with_codebooks(proxy, codebooks)
    equal = proxy_codes == true_codes
    prefix = np.cumprod(equal, axis=1, dtype=np.uint8).astype(bool)
    return {
        "cosine": np.sum(truth * proxy, axis=1),
        "squared_error": np.mean(np.square(truth - proxy), axis=1),
        "prefix": prefix,
        "proxy_codes": proxy_codes,
    }


def aggregate(diagnostics: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {"count": 0}
    cosine = diagnostics["cosine"][mask]
    mse = diagnostics["squared_error"][mask]
    prefix = diagnostics["prefix"][mask]
    return {
        "count": count,
        "cosine": {
            "mean": float(cosine.mean()),
            "median": float(np.median(cosine)),
            "p10": float(np.quantile(cosine, 0.10)),
            "p25": float(np.quantile(cosine, 0.25)),
            "p75": float(np.quantile(cosine, 0.75)),
            "p90": float(np.quantile(cosine, 0.90)),
        },
        "mse": {
            "mean": float(mse.mean()),
            "median_item_mse": float(np.median(mse)),
        },
        "prefix_accuracy": {
            f"prefix_acc_at_{level + 1}": float(prefix[:, level].mean())
            for level in range(prefix.shape[1])
        },
        "mean_common_prefix_length": float(prefix.sum(axis=1).mean()),
    }


def selection_score(metrics: dict[str, Any]) -> tuple[float, ...]:
    prefix = metrics["prefix_accuracy"]
    return (
        prefix["prefix_acc_at_4"],
        prefix["prefix_acc_at_3"],
        prefix["prefix_acc_at_2"],
        prefix["prefix_acc_at_1"],
        metrics["cosine"]["mean"],
        -metrics["mse"]["mean"],
    )


def stratified_metrics(
    diagnostics: dict[str, np.ndarray],
    base_mask: np.ndarray,
    sample: dict[str, np.ndarray],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    output["availability"] = {
        name: aggregate(diagnostics, base_mask & (sample["availability"] == code))
        for code, name in AVAILABILITY_NAMES.items()
    }
    output["artist_group_size"] = {
        name: aggregate(
            diagnostics, base_mask & (sample["artist_size_tier"] == code)
        )
        for code, name in SIZE_TIER_NAMES.items()
    }
    output["album_group_size"] = {
        name: aggregate(
            diagnostics, base_mask & (sample["album_size_tier"] == code)
        )
        for code, name in SIZE_TIER_NAMES.items()
    }
    output["explicit_frequency"] = {
        name: aggregate(diagnostics, base_mask & (sample["frequency_tier"] == code))
        for code, name in FREQUENCY_TIER_NAMES.items()
    }
    return output


def neighbor_overlap(reference: np.ndarray, truth: np.ndarray, proxy: np.ndarray) -> dict:
    index = faiss.IndexFlatIP(reference.shape[1])
    index.add(np.ascontiguousarray(reference, dtype=np.float32))
    _, truth_neighbors = index.search(np.ascontiguousarray(truth, dtype=np.float32), 50)
    _, proxy_neighbors = index.search(np.ascontiguousarray(proxy, dtype=np.float32), 50)
    overlap10 = np.empty(len(truth), dtype=np.float32)
    overlap50 = np.empty(len(truth), dtype=np.float32)
    for row in range(len(truth)):
        overlap10[row] = len(
            set(truth_neighbors[row, :10]).intersection(proxy_neighbors[row, :10])
        ) / 10.0
        overlap50[row] = len(
            set(truth_neighbors[row]).intersection(proxy_neighbors[row])
        ) / 50.0
    return {
        "queries": int(len(truth)),
        "reference_items": int(len(reference)),
        "exact_index": "FAISS IndexFlatIP over normalized vectors",
        "neighbor_overlap_at_10_mean": float(overlap10.mean()),
        "neighbor_overlap_at_10_median": float(np.median(overlap10)),
        "neighbor_overlap_at_50_mean": float(overlap50.mean()),
        "neighbor_overlap_at_50_median": float(np.median(overlap50)),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    gate = config["gate1"]
    paths = {key: Path(value) for key, value in config["paths"].items()}
    seed = int(config["seed"])
    sample_payload = np.load(args.sample, allow_pickle=False)
    sample = {key: sample_payload[key] for key in sample_payload.files}
    sample_payload.close()

    dense_features = np.load(paths["dense_features"], mmap_mode="r")
    truth = np.ascontiguousarray(
        dense_features[sample["dense_id"].astype(np.int64)], dtype=np.float32
    )
    try:
        dense_features._mmap.madvise(mmap.MADV_DONTNEED)
    except (AttributeError, OSError):
        pass
    codebooks = np.load(args.codebooks)
    true_codes = encode_with_codebooks(truth, codebooks)
    artist = np.load(args.proxy_dir / "artist_proxy.npy", mmap_mode="r")
    album = np.load(args.proxy_dir / "album_proxy.npy", mmap_mode="r")
    artist_available = np.load(args.proxy_dir / "artist_available.npy")
    album_available = np.load(args.proxy_dir / "album_available.npy")
    if not np.array_equal(artist_available, sample["artist_loo_source_count"] > 0):
        raise RuntimeError("artist availability contract changed")
    if not np.array_equal(album_available, sample["album_loo_source_count"] > 0):
        raise RuntimeError("album availability contract changed")

    validation_both = (
        (sample["split"] == 0) & artist_available & album_available
    )
    alpha_trials: dict[str, Any] = {}
    best_weight: float | None = None
    best_score: tuple[float, ...] | None = None
    for raw_weight in gate["album_weight_grid"]:
        weight = float(raw_weight)
        blended, _ = normalized_blend(
            artist,
            album,
            artist_available,
            album_available,
            album_weight=weight,
        )
        indices = np.flatnonzero(validation_both)
        diagnostics = rowwise_diagnostics(
            truth[indices], blended[indices], true_codes[indices], codebooks
        )
        metrics = aggregate(diagnostics, np.ones(len(indices), dtype=bool))
        alpha_trials[str(weight)] = metrics
        score = selection_score(metrics)
        if best_score is None or score > best_score:
            best_score = score
            best_weight = weight
    if best_weight is None:
        raise RuntimeError("no alpha candidate was evaluated")

    combined, combined_available = normalized_blend(
        artist,
        album,
        artist_available,
        album_available,
        album_weight=best_weight,
    )
    methods = {
        "artist_centroid": (np.asarray(artist), artist_available),
        "album_centroid": (np.asarray(album), album_available),
        "conditional_artist_album": (combined, combined_available),
    }
    method_reports: dict[str, Any] = {}
    diagnostics_by_method: dict[str, dict[str, np.ndarray]] = {}
    test = sample["split"] == 1
    for name, (vectors, available) in methods.items():
        diagnostics = rowwise_diagnostics(truth, vectors, true_codes, codebooks)
        diagnostics_by_method[name] = diagnostics
        test_mask = test & available
        method_reports[name] = {
            "test_overall": aggregate(diagnostics, test_mask),
            "test_strata": stratified_metrics(diagnostics, test_mask, sample),
            "test_available": int(test_mask.sum()),
            "test_requested": int(test.sum()),
            "test_coverage": float(test_mask.sum() / test.sum()),
        }

    # Fixed real-embedding reference pool, disjoint from all proxy targets.
    with np.load(args.support, allow_pickle=False) as support:
        support_item_id = support["item_id"]
        support_dense_id = support["dense_id"]
        support_real = support["real_embedding"]
    target_lookup = np.zeros(len(np.load(paths["orig2dense"], mmap_mode="r")), dtype=bool)
    target_lookup[sample["item_id"]] = True
    reference_candidates = np.flatnonzero(
        support_real & ~target_lookup[support_item_id]
    )
    rng = np.random.default_rng(seed + 101)
    reference_positions = rng.choice(
        reference_candidates,
        size=int(gate["neighbor_reference_items"]),
        replace=False,
    )
    reference = np.ascontiguousarray(
        dense_features[support_dense_id[reference_positions].astype(np.int64)],
        dtype=np.float32,
    )
    try:
        dense_features._mmap.madvise(mmap.MADV_DONTNEED)
    except (AttributeError, OSError):
        pass
    del dense_features
    gc.collect()
    faiss.omp_set_num_threads(2)
    for name, (vectors, available) in methods.items():
        candidates = np.flatnonzero(test & available)
        query_count = min(int(gate["neighbor_query_items"]), len(candidates))
        query_positions = np.sort(rng.choice(candidates, size=query_count, replace=False))
        method_reports[name]["nearest_neighbor_consistency"] = neighbor_overlap(
            reference,
            truth[query_positions],
            np.asarray(vectors[query_positions], dtype=np.float32),
        )

    support_report = json.loads(args.support_report.read_text(encoding="utf-8"))
    report = {
        "status": "complete_gate1_metrics_no_pass_threshold_applied",
        "priority_order": [
            "SID PrefixAcc",
            "nearest-neighbor consistency",
            "cosine similarity",
            "MSE",
        ],
        "data_contract": {
            "sample": str(args.sample.resolve()),
            "validation_split_used_for_alpha_only": True,
            "test_used_once_after_alpha_selection": True,
            "candidate_codebook": str(args.codebooks.resolve()),
            "candidate_codebook_real_only": True,
            "candidate_codebook_is_not_gate2_final": True,
            "normalized_embeddings": True,
        },
        "sample_counts": {
            "total": int(len(truth)),
            "validation": int(np.count_nonzero(sample["split"] == 0)),
            "test": int(test.sum()),
            "validation_both_for_alpha": int(validation_both.sum()),
        },
        "alpha_selection": {
            "formula": "alpha * album + (1-alpha) * artist, then L2 normalize",
            "selected_album_weight": best_weight,
            "lexicographic_selection": (
                "PrefixAcc@4, @3, @2, @1, mean cosine, negative MSE"
            ),
            "validation_trials_on_both_available": alpha_trials,
        },
        "methods": method_reports,
        "missing_item_coverage": support_report["missing_item_coverage"],
        "interpretation_boundary": (
            "This file reports evidence without inventing a universal pass threshold. "
            "The technical A/B/C judgment is made in the Gate-1 report after checking "
            "task-priority metrics and stratum stability."
        ),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
