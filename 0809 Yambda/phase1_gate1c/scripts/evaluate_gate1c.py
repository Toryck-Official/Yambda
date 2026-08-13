#!/usr/bin/env python3
"""Evaluate collaborative-to-audio against frozen metadata-only baselines."""

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

DEFAULT_CONFIG = ROOT / "phase1_gate1c" / "configs" / "experiment.json"
DEFAULT_OUTPUT = ROOT / "phase1_gate1c" / "artifacts" / "gate1c_metrics.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def row_metrics(truth, prediction, true_codes, codebooks):
    prediction = np.ascontiguousarray(prediction, dtype=np.float32)
    prediction /= np.maximum(np.linalg.norm(prediction, axis=1, keepdims=True), 1e-12)
    codes = encode_with_codebooks(prediction, codebooks)
    token = codes == true_codes
    return {
        "cosine": np.sum(truth * prediction, axis=1),
        "mse": np.mean(np.square(truth - prediction), axis=1),
        "token": token,
        "prefix": np.logical_and.accumulate(token, axis=1),
        "vectors": prediction,
    }


def aggregate(rows, mask, nn10=None, nn50=None, query_mask=None):
    result = {
        "count": int(mask.sum()),
        "cosine": float(rows["cosine"][mask].mean()),
        "mse": float(rows["mse"][mask].mean()),
        "token_accuracy": {f"level_{i + 1}": float(rows["token"][mask, i].mean()) for i in range(4)},
        "prefix_accuracy": {f"prefix_at_{i + 1}": float(rows["prefix"][mask, i].mean()) for i in range(4)},
        "exact_semantic_sid_at_4": float(rows["prefix"][mask, 3].mean()),
    }
    if nn10 is not None and query_mask is not None:
        qmask = mask[query_mask]
        result["nearest_neighbor"] = {
            "query_count": int(qmask.sum()),
            "nn_at_10": float(nn10[qmask].mean()) if qmask.any() else None,
            "nn_at_50": float(nn50[qmask].mean()) if qmask.any() else None,
        }
    return result


def bootstrap(delta, repeats, seed):
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 50):
        end = min(start + 50, repeats)
        index = rng.integers(0, len(delta), size=(end - start, len(delta)), dtype=np.int32)
        means[start:end] = delta[index].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean_improvement": float(delta.mean()), "bootstrap_95_ci": [float(low), float(high)], "ci_strictly_positive": bool(low > 0), "count": int(len(delta)), "repeats": repeats}


def stratified_query(labels, requested, seed):
    rng = np.random.default_rng(seed)
    selected = []
    unique, counts = np.unique(labels, return_counts=True)
    base = {int(label): min(500, int(count)) for label, count in zip(unique, counts)}
    remaining = requested - sum(base.values())
    capacity = {int(label): int(count) - base[int(label)] for label, count in zip(unique, counts)}
    while remaining > 0 and sum(capacity.values()) > 0:
        total = sum(capacity.values())
        for label in sorted(capacity):
            if remaining == 0: break
            take = min(capacity[label], max(1, round(remaining * capacity[label] / total)))
            take = min(take, remaining)
            base[label] += take; capacity[label] -= take; remaining -= take
    for label in unique:
        members = np.flatnonzero(labels == label)
        selected.append(rng.choice(members, base[int(label)], replace=False))
    result = np.sort(np.concatenate(selected))
    if len(result) != requested: raise RuntimeError("query allocation failed")
    return result


def neighbor_rows(reference, truth, prediction):
    index = faiss.IndexFlatIP(reference.shape[1]); index.add(np.ascontiguousarray(reference, dtype=np.float32))
    truth_neighbors = index.search(np.ascontiguousarray(truth, dtype=np.float32), 50)[1]
    prediction_neighbors = index.search(np.ascontiguousarray(prediction, dtype=np.float32), 50)[1]
    nn10 = np.empty(len(truth), dtype=np.float32); nn50 = np.empty(len(truth), dtype=np.float32)
    for row in range(len(truth)):
        nn10[row] = len(set(truth_neighbors[row, :10]).intersection(prediction_neighbors[row, :10])) / 10
        nn50[row] = len(set(truth_neighbors[row]).intersection(prediction_neighbors[row])) / 50
    return nn10, nn50


def mean_seed_reports(reports):
    paths = {
        "cosine": ("overall", "cosine"), "mse": ("overall", "mse"),
        "prefix_at_1": ("overall", "prefix_accuracy", "prefix_at_1"),
        "prefix_at_2": ("overall", "prefix_accuracy", "prefix_at_2"),
        "prefix_at_3": ("overall", "prefix_accuracy", "prefix_at_3"),
        "prefix_at_4": ("overall", "prefix_accuracy", "prefix_at_4"),
        "nn_at_10": ("overall", "nearest_neighbor", "nn_at_10"),
        "nn_at_50": ("overall", "nearest_neighbor", "nn_at_50"),
    }
    output = {}
    for name, path in paths.items():
        values = []
        for report in reports:
            value = report
            for key in path: value = value[key]
            values.append(float(value))
        output[name] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "values": values}
    return output


def main() -> None:
    args = parse_args(); config = json.loads(args.config.read_text())
    paths = {key: Path(value) for key, value in config["paths"].items()}
    graph = paths["graph"]
    explicit_item = np.load(graph / "explicit_item_id.uint32.npy", mmap_mode="r")
    lookup = np.full(int(explicit_item.max()) + 1, -1, dtype=np.int32)
    lookup[explicit_item] = np.arange(len(explicit_item), dtype=np.int32)
    unique_users_all = np.load(graph / "unique_train_user_count.uint32.npy", mmap_mode="r")
    event_relation_all = np.load(graph / "event_count_by_relation.uint32.npy", mmap_mode="r")
    revision_share_all = np.load(graph / "revision_event_share.float32.npy", mmap_mode="r")
    with np.load(paths["gate1b_sample"], allow_pickle=False) as z:
        sample = {key: z[key] for key in z.files}
    sample_position = lookup[sample["item_id"]]
    full_test = np.flatnonzero(sample["split"] == 2)
    evidence_test = full_test[unique_users_all[sample_position[full_test]] > 0]
    test_position = sample_position[evidence_test]
    test_slot = np.searchsorted(full_test, evidence_test)
    truth_all = np.load(paths["gate1b_arrays"] / "truth.float32.npy", mmap_mode="r")
    codes_all = np.load(paths["gate1b_arrays"] / "true_codes.uint16.npy", mmap_mode="r")
    truth = np.ascontiguousarray(truth_all[evidence_test], dtype=np.float32)
    true_codes = np.asarray(codes_all[evidence_test])
    codebooks = np.load(paths["codebooks"])
    unique_users = np.asarray(unique_users_all[test_position])
    event_relation = np.asarray(event_relation_all[:, test_position])
    revision_share = np.asarray(revision_share_all[test_position])
    dominant_relation = event_relation.argmax(axis=0)
    user_label = np.empty(len(unique_users), dtype=np.uint8)
    strata_masks = {}
    for code, row in enumerate(config["evaluation"]["unique_train_user_strata"]):
        maximum = row["max"]
        mask = unique_users >= int(row["min"])
        if maximum is not None: mask &= unique_users <= int(maximum)
        user_label[mask] = code
        strata_masks[row["name"]] = mask

    artist = np.load(paths["gate1b_proxy_vectors"] / "artist_proxy.npy", mmap_mode="r")
    album = np.load(paths["gate1b_proxy_vectors"] / "album_proxy.npy", mmap_mode="r")
    aa = np.load(paths["gate1b_proxy_vectors"] / "artist_available.npy")
    ba = np.load(paths["gate1b_proxy_vectors"] / "album_available.npy")
    conditional = np.zeros((len(evidence_test), 128), dtype=np.float32)
    selected_aa, selected_ba = aa[evidence_test], ba[evidence_test]
    conditional[selected_aa & ~selected_ba] = artist[evidence_test[selected_aa & ~selected_ba]]
    conditional[selected_ba] = album[evidence_test[selected_ba]]
    centroid_rows = row_metrics(truth, conditional, true_codes, codebooks)

    # Fixed reference pool disjoint from the 400k mapper sample.
    with np.load(paths["support"], allow_pickle=False) as z:
        support_item, support_dense, support_real = z["item_id"], z["dense_id"], z["real_embedding"]
    target = np.zeros(int(max(support_item.max(), sample["item_id"].max())) + 1, dtype=bool)
    target[sample["item_id"]] = True
    candidates = np.flatnonzero(support_real & ~target[support_item])
    rng = np.random.default_rng(2127)
    reference_position = rng.choice(candidates, int(config["evaluation"]["neighbor_reference_items"]), replace=False)
    dense = np.load(paths["dense_features"], mmap_mode="r")
    reference = np.ascontiguousarray(dense[support_dense[reference_position].astype(np.int64)], dtype=np.float32)
    query = stratified_query(user_label, min(int(config["evaluation"]["neighbor_query_items"]), len(truth)), 2329)
    faiss.omp_set_num_threads(2)
    centroid_nn10, centroid_nn50 = neighbor_rows(reference, truth[query], centroid_rows["vectors"][query])
    all_mask = np.ones(len(truth), dtype=bool)
    centroid_report = {
        "overall": aggregate(centroid_rows, all_mask, centroid_nn10, centroid_nn50, query),
        "unique_user_strata": {name: aggregate(centroid_rows, mask, centroid_nn10, centroid_nn50, query) for name, mask in strata_masks.items()},
    }
    metadata_reports = []; collaborative_reports = []
    metadata_rows_all = []; collaborative_rows_all = []
    metadata_nn_all = []; collaborative_nn_all = []
    for seed in config["seeds"]:
        seed = int(seed)
        meta_prediction_full_test = np.load(paths["gate1b_predictor_runs"] / f"seed_{seed}" / "embedding_test_prediction.float32.npy")
        meta_rows = row_metrics(truth, meta_prediction_full_test[test_slot], true_codes, codebooks)
        run = paths["graph"].parent / "collaborative_runs" / f"seed_{seed}"
        mapper_index = np.load(run / "mapper_test_sample_index.int64.npy")
        if not np.array_equal(mapper_index, evidence_test):
            raise RuntimeError(f"seed {seed} mapper test index changed")
        collab_prediction = np.load(run / "collaborative_audio_test_prediction.float32.npy")
        collab_rows = row_metrics(truth, collab_prediction, true_codes, codebooks)
        meta_nn10, meta_nn50 = neighbor_rows(reference, truth[query], meta_rows["vectors"][query])
        collab_nn10, collab_nn50 = neighbor_rows(reference, truth[query], collab_rows["vectors"][query])
        metadata_rows_all.append(meta_rows); collaborative_rows_all.append(collab_rows)
        metadata_nn_all.append((meta_nn10, meta_nn50)); collaborative_nn_all.append((collab_nn10, collab_nn50))
        def method_report(rows, nn10, nn50):
            revision_masks = {
                "revision_share_0": revision_share == 0,
                "revision_share_(0,0.25)": (revision_share > 0) & (revision_share < .25),
                "revision_share_[0.25,0.5)": (revision_share >= .25) & (revision_share < .5),
                "revision_share_[0.5,0.75)": (revision_share >= .5) & (revision_share < .75),
                "revision_share_[0.75,1)": (revision_share >= .75) & (revision_share < 1),
                "revision_share_1": revision_share == 1,
            }
            return {
                "seed": seed,
                "overall": aggregate(rows, all_mask, nn10, nn50, query),
                "unique_user_strata": {name: aggregate(rows, mask, nn10, nn50, query) for name, mask in strata_masks.items()},
                "revision_share_strata": {name: aggregate(rows, mask, nn10, nn50, query) for name, mask in revision_masks.items() if mask.any()},
                "dominant_feedback": {name: aggregate(rows, dominant_relation == code, nn10, nn50, query) for code, name in enumerate(["like", "dislike", "unlike", "undislike"]) if np.any(dominant_relation == code)},
            }
        metadata_reports.append(method_report(meta_rows, meta_nn10, meta_nn50))
        collaborative_reports.append(method_report(collab_rows, collab_nn10, collab_nn50))

    # Seed-averaged paired item outcomes for fair uncertainty intervals.
    repeats = int(config["evaluation"]["bootstrap_repeats"])
    comparisons = {"collaborative_vs_centroid": {}, "collaborative_vs_metadata_mlp": {}}
    collab_cos = np.mean(np.stack([x["cosine"] for x in collaborative_rows_all]), axis=0)
    meta_cos = np.mean(np.stack([x["cosine"] for x in metadata_rows_all]), axis=0)
    comparisons["collaborative_vs_centroid"]["cosine"] = bootstrap(collab_cos - centroid_rows["cosine"], repeats, 3001)
    comparisons["collaborative_vs_metadata_mlp"]["cosine"] = bootstrap(collab_cos - meta_cos, repeats, 3002)
    comparisons["collaborative_vs_centroid"]["mse"] = bootstrap(centroid_rows["mse"] - np.mean(np.stack([x["mse"] for x in collaborative_rows_all]), axis=0), repeats, 3003)
    comparisons["collaborative_vs_metadata_mlp"]["mse"] = bootstrap(np.mean(np.stack([x["mse"] for x in metadata_rows_all]), axis=0) - np.mean(np.stack([x["mse"] for x in collaborative_rows_all]), axis=0), repeats, 3004)
    for level in range(4):
        collab = np.mean(np.stack([x["prefix"][:, level] for x in collaborative_rows_all]), axis=0)
        meta = np.mean(np.stack([x["prefix"][:, level] for x in metadata_rows_all]), axis=0)
        comparisons["collaborative_vs_centroid"][f"prefix_at_{level + 1}"] = bootstrap(collab - centroid_rows["prefix"][:, level], repeats, 3010 + level)
        comparisons["collaborative_vs_metadata_mlp"][f"prefix_at_{level + 1}"] = bootstrap(collab - meta, repeats, 3020 + level)
    collab_nn10 = np.mean(np.stack([x[0] for x in collaborative_nn_all]), axis=0); collab_nn50 = np.mean(np.stack([x[1] for x in collaborative_nn_all]), axis=0)
    meta_nn10 = np.mean(np.stack([x[0] for x in metadata_nn_all]), axis=0); meta_nn50 = np.mean(np.stack([x[1] for x in metadata_nn_all]), axis=0)
    for name, cvalue, mvalue, bvalue, seed in [
        ("nn_at_10", collab_nn10, meta_nn10, centroid_nn10, 3030),
        ("nn_at_50", collab_nn50, meta_nn50, centroid_nn50, 3031),
    ]:
        comparisons["collaborative_vs_centroid"][name] = bootstrap(cvalue - bvalue, repeats, seed)
        comparisons["collaborative_vs_metadata_mlp"][name] = bootstrap(cvalue - mvalue, repeats, seed + 10)

    # Missing-item coverage in exactly the requested unique-user strata.
    with np.load(paths["support"], allow_pickle=False) as z:
        if not np.array_equal(z["item_id"], explicit_item): raise RuntimeError("support item order differs")
        missing = ~z["real_embedding"]
    missing_users = np.asarray(unique_users_all[missing])
    missing_revision = np.asarray(revision_share_all[missing])
    missing_coverage = {"cold_0": int(np.count_nonzero(missing_users == 0))}
    for row in config["evaluation"]["unique_train_user_strata"]:
        mask = missing_users >= int(row["min"])
        if row["max"] is not None: mask &= missing_users <= int(row["max"])
        missing_coverage[row["name"]] = int(mask.sum())
    missing_coverage["revision_heavy_share_ge_0.5_with_evidence"] = int(np.count_nonzero((missing_users > 0) & (missing_revision >= .5)))
    report = {
        "status": "complete_gate1c_three_seed_semantic_recovery_metrics",
        "data_contract": {
            "test_items_with_train_collaborative_evidence": int(len(truth)),
            "fixed_gate1b_test_items_total": int(len(full_test)),
            "test_item_train_edges_allowed": True,
            "test_audio_embedding_used_in_graph_or_mapper_input": False,
            "validation_or_test_period_edges_used": False,
            "relations": ["like", "dislike", "unlike", "undislike"],
            "relations_collapsed": False,
            "codebook_frozen": True,
            "gate2_started": False,
        },
        "query_protocol": {"reference_items": int(len(reference)), "stratified_queries": int(len(query)), "same_reference_and_queries_for_all_methods": True},
        "metadata_centroid": centroid_report,
        "metadata_mlp_runs": metadata_reports,
        "collaborative_audio_runs": collaborative_reports,
        "three_seed_summary": {"metadata_mlp": mean_seed_reports(metadata_reports), "collaborative_audio": mean_seed_reports(collaborative_reports)},
        "paired_bootstrap": comparisons,
        "missing_item_unique_train_user_coverage": missing_coverage,
    }
    atomic_json(args.output, report)
    print(json.dumps({"data_contract": report["data_contract"], "summary": report["three_seed_summary"], "comparisons": comparisons, "missing_coverage": missing_coverage, "output": str(args.output)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
