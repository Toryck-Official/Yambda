#!/usr/bin/env python3
"""Evaluate Gate 1B learned embedding and direct SID predictors fairly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LEGACY = Path("/root/autodl-tmp/0804 Yambda/dataprocess")
sys.path.insert(0, str(LEGACY))
from rqdataprocess.rq import encode_with_codebooks  # noqa: E402

DEFAULT_CONFIG = ROOT / "phase1_gate1b" / "configs" / "gate1b.json"
DEFAULT_ARRAYS = ROOT / "phase1_gate1b" / "work" / "learning_arrays"
DEFAULT_SAMPLE = ROOT / "phase1_gate1b" / "artifacts" / "learning_sample.npz"
DEFAULT_PROXIES = ROOT / "phase1_gate1b" / "work" / "learning_proxy_vectors"
DEFAULT_RUNS = ROOT / "phase1_gate1b" / "artifacts" / "predictor_runs"
DEFAULT_OUTPUT = ROOT / "phase1_gate1b" / "artifacts" / "gate1b_metrics.json"

AVAILABILITY = {1: "artist_only", 2: "album_only", 3: "both"}
SIZE = {1: "small", 2: "medium", 3: "large"}
FREQUENCY = {0: "tail", 1: "mid", 2: "head"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    p.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    p.add_argument("--proxy-dir", type=Path, default=DEFAULT_PROXIES)
    p.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def embedding_rows(truth: np.ndarray, prediction: np.ndarray, true_codes: np.ndarray, codebooks: np.ndarray) -> dict:
    prediction = np.ascontiguousarray(prediction, dtype=np.float32)
    prediction /= np.maximum(np.linalg.norm(prediction, axis=1, keepdims=True), 1e-12)
    pred_codes = encode_with_codebooks(prediction, codebooks)
    equal = pred_codes == true_codes
    return {
        "cosine": np.sum(truth * prediction, axis=1),
        "mse": np.mean(np.square(truth - prediction), axis=1),
        "token": equal,
        "prefix": np.logical_and.accumulate(equal, axis=1),
        "codes": pred_codes,
        "vectors": prediction,
    }


def sid_rows(true_codes: np.ndarray, prediction: np.ndarray) -> dict:
    equal = prediction == true_codes
    return {"token": equal, "prefix": np.logical_and.accumulate(equal, axis=1), "codes": prediction}


def aggregate(rows: dict, mask: np.ndarray | None = None) -> dict:
    if mask is None:
        mask = np.ones(len(rows["prefix"]), dtype=bool)
    count = int(mask.sum())
    result = {
        "count": count,
        "token_accuracy": {f"level_{i + 1}": float(rows["token"][mask, i].mean()) for i in range(4)},
        "prefix_accuracy": {f"prefix_at_{i + 1}": float(rows["prefix"][mask, i].mean()) for i in range(4)},
        "exact_semantic_sid_at_4": float(rows["prefix"][mask, 3].mean()),
    }
    if "cosine" in rows:
        cosine = rows["cosine"][mask]
        mse = rows["mse"][mask]
        result.update({
            "cosine": {"mean": float(cosine.mean()), "median": float(np.median(cosine)), "p10": float(np.quantile(cosine, .1)), "p90": float(np.quantile(cosine, .9))},
            "mse": {"mean": float(mse.mean()), "median": float(np.median(mse))},
        })
    return result


def strata(rows: dict, sample: dict) -> dict:
    return {
        "availability": {name: aggregate(rows, sample["availability"] == code) for code, name in AVAILABILITY.items()},
        "artist_group_size": {name: aggregate(rows, sample["artist_size_tier"] == code) for code, name in SIZE.items()},
        "album_group_size": {name: aggregate(rows, sample["album_size_tier"] == code) for code, name in SIZE.items()},
        "explicit_frequency": {name: aggregate(rows, sample["frequency_tier"] == code) for code, name in FREQUENCY.items()},
    }


def bootstrap_difference(delta: np.ndarray, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 50):
        end = min(start + 50, repeats)
        indices = rng.integers(0, len(delta), size=(end - start, len(delta)), dtype=np.int32)
        means[start:end] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean_difference": float(delta.mean()), "bootstrap_95_ci": [float(low), float(high)], "ci_excludes_zero": bool(low > 0 or high < 0), "paired_items": int(len(delta)), "repeats": repeats}


def neighbor_indices(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    index = faiss.IndexFlatIP(reference.shape[1])
    index.add(np.ascontiguousarray(reference, dtype=np.float32))
    return index.search(np.ascontiguousarray(query, dtype=np.float32), 50)[1]


def neighbor_overlap(truth_neighbors: np.ndarray, prediction_neighbors: np.ndarray) -> dict:
    overlap10 = np.empty(len(truth_neighbors), dtype=np.float32)
    overlap50 = np.empty(len(truth_neighbors), dtype=np.float32)
    for i in range(len(truth_neighbors)):
        overlap10[i] = len(set(truth_neighbors[i, :10]).intersection(prediction_neighbors[i, :10])) / 10
        overlap50[i] = len(set(truth_neighbors[i]).intersection(prediction_neighbors[i])) / 50
    return {"queries": int(len(overlap10)), "nn_overlap_at_10_mean": float(overlap10.mean()), "nn_overlap_at_10_median": float(np.median(overlap10)), "nn_overlap_at_50_mean": float(overlap50.mean()), "nn_overlap_at_50_median": float(np.median(overlap50))}


def mean_std(reports: list[dict], path: tuple[str, ...]) -> dict:
    values = []
    for report in reports:
        value = report
        for key in path:
            value = value[key]
        values.append(float(value))
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "values": values}


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    paths = {key: Path(value) for key, value in config["paths"].items()}
    learning = config["learning"]
    with np.load(args.sample, allow_pickle=False) as z:
        whole_sample = {key: z[key] for key in z.files}
    test_mask = whole_sample["split"] == 2
    test_sample = {key: value[test_mask] for key, value in whole_sample.items()}
    truth_all = np.load(args.arrays / "truth.float32.npy", mmap_mode="r")
    true_codes_all = np.load(args.arrays / "true_codes.uint16.npy", mmap_mode="r")
    truth = np.ascontiguousarray(truth_all[test_mask], dtype=np.float32)
    true_codes = np.asarray(true_codes_all[test_mask])
    codebooks = np.load(paths["codebooks"])
    artist_all = np.load(args.proxy_dir / "artist_proxy.npy", mmap_mode="r")
    album_all = np.load(args.proxy_dir / "album_proxy.npy", mmap_mode="r")
    artist = np.asarray(artist_all[test_mask], dtype=np.float32)
    album = np.asarray(album_all[test_mask], dtype=np.float32)
    aa = np.load(args.proxy_dir / "artist_available.npy")[test_mask]
    ba = np.load(args.proxy_dir / "album_available.npy")[test_mask]
    conditional = np.zeros_like(artist)
    conditional[aa & ~ba] = artist[aa & ~ba]
    conditional[ba] = album[ba]
    baseline_rows = {
        "artist_centroid": embedding_rows(truth, artist, true_codes, codebooks),
        "album_centroid": embedding_rows(truth, album, true_codes, codebooks),
        "conditional_centroid": embedding_rows(truth, conditional, true_codes, codebooks),
    }
    baseline_reports = {
        "artist_centroid": {"overall_available": aggregate(baseline_rows["artist_centroid"], aa)},
        "album_centroid": {"overall_available": aggregate(baseline_rows["album_centroid"], ba)},
        "conditional_centroid": {"overall": aggregate(baseline_rows["conditional_centroid"]), "strata": strata(baseline_rows["conditional_centroid"], test_sample)},
    }
    learned_rows = []
    learned_reports = []
    direct_rows = []
    direct_reports = []
    for raw_seed in learning["seeds"]:
        seed = int(raw_seed)
        run = args.runs / f"seed_{seed}"
        learned_vector = np.load(run / "embedding_test_prediction.float32.npy")
        direct_code = np.load(run / "direct_sid_test_prediction.uint16.npy")
        erows = embedding_rows(truth, learned_vector, true_codes, codebooks)
        drows = sid_rows(true_codes, direct_code)
        learned_rows.append(erows)
        direct_rows.append(drows)
        learned_reports.append({"seed": seed, "overall": aggregate(erows), "strata": strata(erows, test_sample)})
        direct_reports.append({"seed": seed, "overall": aggregate(drows), "strata": strata(drows, test_sample), "test_inference": "strict autoregressive predicted prefix; no truth prefix accepted by inference API"})

    # Fixed reference/query pool for every embedding-producing method.
    with np.load(paths["support"], allow_pickle=False) as support:
        support_item = support["item_id"]
        support_dense = support["dense_id"]
        support_real = support["real_embedding"]
    maximum_item = int(max(support_item.max(), whole_sample["item_id"].max()))
    target = np.zeros(maximum_item + 1, dtype=bool)
    target[whole_sample["item_id"]] = True
    candidates = np.flatnonzero(support_real & ~target[support_item])
    rng = np.random.default_rng(int(config["seed"]) + 101)
    ref_pos = rng.choice(candidates, int(learning["neighbor_reference_items"]), replace=False)
    dense = np.load(paths["dense_features"], mmap_mode="r")
    reference = np.ascontiguousarray(dense[support_dense[ref_pos].astype(np.int64)], dtype=np.float32)
    query_rng = np.random.default_rng(int(config["seed"]) + 303)
    query = np.sort(query_rng.choice(len(truth), int(learning["neighbor_query_items"]), replace=False))
    faiss.omp_set_num_threads(2)
    truth_neighbors = neighbor_indices(reference, truth[query])
    baseline_reports["conditional_centroid"]["nearest_neighbor_consistency"] = neighbor_overlap(truth_neighbors, neighbor_indices(reference, conditional[query]))
    for offset, (name, vectors, available) in enumerate([
        ("artist_centroid", artist, aa),
        ("album_centroid", album, ba),
    ]):
        candidates_for_method = np.flatnonzero(available)
        method_rng = np.random.default_rng(int(config["seed"]) + 304 + offset)
        method_query = np.sort(
            method_rng.choice(
                candidates_for_method,
                min(int(learning["neighbor_query_items"]), len(candidates_for_method)),
                replace=False,
            )
        )
        baseline_reports[name]["nearest_neighbor_consistency"] = neighbor_overlap(
            neighbor_indices(reference, truth[method_query]),
            neighbor_indices(reference, vectors[method_query]),
        )
    for report, rows in zip(learned_reports, learned_rows):
        report["nearest_neighbor_consistency"] = neighbor_overlap(truth_neighbors, neighbor_indices(reference, rows["vectors"][query]))

    repeats = int(learning["bootstrap_repeats"])
    conditional_rows = baseline_rows["conditional_centroid"]
    comparison = {
        "learned_embedding_vs_conditional_centroid": {},
        "direct_sid_vs_learned_embedding": {},
        "direct_sid_vs_conditional_centroid": {},
    }
    seed_mean_learned_cos = np.mean(np.stack([rows["cosine"] for rows in learned_rows]), axis=0)
    comparison["learned_embedding_vs_conditional_centroid"]["cosine"] = bootstrap_difference(seed_mean_learned_cos - conditional_rows["cosine"], repeats, int(config["seed"]) + 501)
    for level in range(4):
        learned_prefix_mean = np.mean(np.stack([rows["prefix"][:, level] for rows in learned_rows]), axis=0)
        direct_prefix_mean = np.mean(np.stack([rows["prefix"][:, level] for rows in direct_rows]), axis=0)
        comparison["learned_embedding_vs_conditional_centroid"][f"prefix_at_{level + 1}"] = bootstrap_difference(learned_prefix_mean - conditional_rows["prefix"][:, level], repeats, int(config["seed"]) + 510 + level)
        comparison["direct_sid_vs_learned_embedding"][f"prefix_at_{level + 1}"] = bootstrap_difference(direct_prefix_mean - learned_prefix_mean, repeats, int(config["seed"]) + 520 + level)
        comparison["direct_sid_vs_conditional_centroid"][f"prefix_at_{level + 1}"] = bootstrap_difference(direct_prefix_mean - conditional_rows["prefix"][:, level], repeats, int(config["seed"]) + 530 + level)

    summary = {
        "learned_embedding": {
            "cosine": mean_std(learned_reports, ("overall", "cosine", "mean")),
            "mse": mean_std(learned_reports, ("overall", "mse", "mean")),
            **{f"prefix_at_{level}": mean_std(learned_reports, ("overall", "prefix_accuracy", f"prefix_at_{level}")) for level in range(1, 5)},
            "nn_at_10": mean_std(learned_reports, ("nearest_neighbor_consistency", "nn_overlap_at_10_mean")),
            "nn_at_50": mean_std(learned_reports, ("nearest_neighbor_consistency", "nn_overlap_at_50_mean")),
        },
        "direct_sid": {**{f"token_level_{level}": mean_std(direct_reports, ("overall", "token_accuracy", f"level_{level}")) for level in range(1, 5)},
                       **{f"prefix_at_{level}": mean_std(direct_reports, ("overall", "prefix_accuracy", f"prefix_at_{level}")) for level in range(1, 5)}},
    }
    report = {
        "status": "complete_gate1b_predictor_metrics_no_gate2_started",
        "data_contract": {
            "train_validation_test": [300000, 50000, 50000],
            "test_is_fixed_gate1a_test": True,
            "all_context_is_strict_leave_one_out": True,
            "model_input": "artist/album LOO centroids, availability, train-standardized source/relation counts",
            "item_id_or_interaction_frequency_as_model_input": False,
            "codebook": str(paths["codebooks"]),
            "codebook_frozen": True,
            "candidate_codebook_not_gate2_final": True,
            "seeds": [int(x) for x in learning["seeds"]],
        },
        "baselines": baseline_reports,
        "learned_embedding_runs": learned_reports,
        "direct_sid_runs": direct_reports,
        "three_seed_summary": summary,
        "paired_bootstrap_comparisons": comparison,
        "neighbor_protocol": {"reference_items": int(len(reference)), "queries": int(len(query)), "exact_index": "FAISS IndexFlatIP", "same_reference_and_queries_for_all_methods": True},
    }
    atomic_json(args.output, report)
    print(json.dumps({"summary": summary, "comparisons": comparison, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
