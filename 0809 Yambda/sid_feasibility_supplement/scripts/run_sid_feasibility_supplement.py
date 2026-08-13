#!/usr/bin/env python3
"""SID feasibility supplement over explicit real-audio items only.

This is an audit, not SID materialization.  It fits four comparable 4x256
residual K-Means variants, measures identity collisions and preservation of the
original audio space, and audits continuous audio/residual uniqueness.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "phase1_gates" / "artifacts" / "metadata_loo_support.npz"
DENSE = Path(
    "/root/autodl-tmp/0626/0626 Predictor/01_data/processed/raw_rqkmeans/"
    "dense_item_features.npy"
)
ORIG2DENSE = Path(
    "/root/autodl-tmp/0626/0626 Predictor/01_data/processed/raw_rqkmeans/"
    "orig2dense_item_id.npy"
)
EMBEDDINGS = Path("/root/autodl-tmp/0804 Yambda/data/embeddings.parquet")
ARTIST = Path(
    "/root/autodl-tmp/0804 Yambda/dataprocess/data/official_metadata/"
    "artist_item_mapping.parquet"
)
ALBUM = Path(
    "/root/autodl-tmp/0804 Yambda/dataprocess/data/official_metadata/"
    "album_item_mapping.parquet"
)
OUT = ROOT / "sid_feasibility_supplement" / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support", type=Path, default=SUPPORT)
    parser.add_argument("--dense", type=Path, default=DENSE)
    parser.add_argument("--orig2dense", type=Path, default=ORIG2DENSE)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS)
    parser.add_argument("--artist", type=Path, default=ARTIST)
    parser.add_argument("--album", type=Path, default=ALBUM)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--context-weight", type=float, default=0.25)
    parser.add_argument("--neighbor-pool", type=int, default=100_000)
    parser.add_argument("--neighbor-queries", type=int, default=3_000)
    parser.add_argument("--near-duplicate-sample", type=int, default=100_000)
    parser.add_argument("--max-items", type=int, default=0)
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: np.ndarray, extra: bool = False) -> dict:
    points = [("p50", .5), ("p90", .9), ("p99", .99), ("p99_9", .999)]
    if extra:
        points = [("p01", .01), ("p10", .1), ("p25", .25)] + points
    return {
        name: float(np.quantile(values, q, method="linear"))
        for name, q in points
    } | {"max": float(values.max())}


def normalize_rows(value: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    np.divide(value, np.maximum(norms, 1e-12), out=value)
    return value


def packed_codes(codes: np.ndarray) -> np.ndarray:
    packed = np.zeros(len(codes), dtype=np.uint32)
    for level in range(4):
        packed |= codes[:, level].astype(np.uint32) << np.uint32(8 * (3 - level))
    return packed


def collision_metrics(codes: np.ndarray) -> dict:
    _, counts = np.unique(packed_codes(codes), return_counts=True)
    singleton = counts == 1
    collided = counts > 1
    singleton_items = int(counts[singleton].sum(dtype=np.uint64))
    collision_items = int(counts[collided].sum(dtype=np.uint64))
    return {
        "unique_semantic_sid_paths": int(len(counts)),
        "singleton_buckets": int(singleton.sum()),
        "singleton_items": singleton_items,
        "singleton_item_ratio": singleton_items / len(codes),
        "collision_buckets": int(collided.sum()),
        "collision_items": collision_items,
        "collision_item_ratio": collision_items / len(codes),
        "all_bucket_size": {
            key: int(value)
            for key, value in quantiles(counts).items()
        },
        "collision_bucket_size": {
            key: int(value)
            for key, value in quantiles(counts[collided]).items()
        },
    }


def reconstruct(codebooks: np.ndarray, codes: np.ndarray) -> np.ndarray:
    output = np.zeros((len(codes), codebooks.shape[2]), dtype=np.float32)
    for level in range(codebooks.shape[0]):
        output += codebooks[level, codes[:, level].astype(np.int64)]
    return output


def fit_rq(
    features: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict], float]:
    started = time.monotonic()
    residual = np.array(features, dtype=np.float32, order="C", copy=True)
    codes = np.empty((len(features), 4), dtype=np.uint8)
    codebooks = np.empty((4, 256, features.shape[1]), dtype=np.float32)
    trace: list[dict] = []
    for level in range(4):
        level_started = time.monotonic()
        kmeans = faiss.Kmeans(
            d=features.shape[1],
            k=256,
            niter=iterations,
            nredo=1,
            seed=seed + level,
            verbose=False,
            gpu=False,
            spherical=False,
            min_points_per_centroid=1,
            max_points_per_centroid=1_000_000,
        )
        kmeans.train(residual)
        centers = np.ascontiguousarray(
            np.asarray(kmeans.centroids, dtype=np.float32).reshape(256, -1)
        )
        distances, assignments = kmeans.index.search(residual, 1)
        code = assignments[:, 0].astype(np.uint8)
        residual -= centers[code]
        codes[:, level] = code
        codebooks[level] = centers
        reconstruction = features - residual
        cosine = np.sum(features * reconstruction, axis=1) / np.maximum(
            np.linalg.norm(features, axis=1)
            * np.linalg.norm(reconstruction, axis=1),
            1e-12,
        )
        trace.append(
            {
                "level": level + 1,
                "mse": float(np.mean(np.square(residual))),
                "mean_residual_l2": float(np.linalg.norm(residual, axis=1).mean()),
                "mean_cosine_similarity": float(cosine.mean()),
                "used_codes": int(np.unique(code).size),
                "assignment_mean_squared_l2": float(distances[:, 0].mean()),
                "training_final_objective": float(kmeans.obj[-1]),
                "elapsed_seconds": time.monotonic() - level_started,
            }
        )
        print(
            f"  level {level + 1}/4: mse={trace[-1]['mse']:.8f}, "
            f"elapsed={trace[-1]['elapsed_seconds']:.1f}s",
            flush=True,
        )
    return codebooks, codes, trace, time.monotonic() - started


def prefix_consistency(reference: np.ndarray, candidate: np.ndarray) -> dict:
    token = (reference == candidate)
    prefix = np.logical_and.accumulate(token, axis=1)
    return {
        "token_accuracy": [float(token[:, level].mean()) for level in range(4)],
        "prefix_accuracy": [float(prefix[:, level].mean()) for level in range(4)],
        "exact_semantic_sid": float(prefix[:, 3].mean()),
    }


def aligned_prefix_consistency(
    reference_codebooks: np.ndarray,
    reference_codes: np.ndarray,
    candidate_codebooks: np.ndarray,
    candidate_codes: np.ndarray,
) -> dict:
    """Align arbitrary K-Means labels one-to-one before token comparison."""
    aligned = np.empty_like(candidate_codes)
    alignment: list[dict] = []
    for level in range(4):
        ref = reference_codebooks[level]
        cand = candidate_codebooks[level]
        distance = (
            np.sum(cand * cand, axis=1, keepdims=True)
            + np.sum(ref * ref, axis=1)[None, :]
            - 2 * cand @ ref.T
        )
        candidate_index, reference_index = linear_sum_assignment(distance)
        remap = np.empty(256, dtype=np.uint8)
        remap[candidate_index] = reference_index.astype(np.uint8)
        aligned[:, level] = remap[candidate_codes[:, level]]
        alignment.append(
            {
                "level": level + 1,
                "mean_matched_codeword_squared_l2": float(
                    distance[candidate_index, reference_index].mean()
                ),
                "maximum_matched_codeword_squared_l2": float(
                    distance[candidate_index, reference_index].max()
                ),
            }
        )
    return {
        **prefix_consistency(reference_codes, aligned),
        "label_alignment": "Hungarian one-to-one minimum squared-L2 codeword matching at each RQ level",
        "alignment_diagnostics": alignment,
        "raw_unaligned_token_ids_not_compared": True,
    }


def remove_self_neighbors(
    neighbors: np.ndarray,
    query_positions: np.ndarray,
    k: int,
) -> np.ndarray:
    output = np.empty((len(neighbors), k), dtype=np.int64)
    for row, own in enumerate(query_positions.tolist()):
        selected = neighbors[row][neighbors[row] != own]
        if len(selected) < k:
            raise RuntimeError("neighbor search did not return enough non-self items")
        output[row] = selected[:k]
    return output


def neighbor_sets(
    features: np.ndarray,
    pool_positions: np.ndarray,
    query_pool_positions: np.ndarray,
    topk: int,
) -> np.ndarray:
    pool = np.ascontiguousarray(features[pool_positions], dtype=np.float32)
    normalize_rows(pool)
    query = np.ascontiguousarray(pool[query_pool_positions], dtype=np.float32)
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(pool)
    _, neighbors = index.search(query, topk + 1)
    return remove_self_neighbors(neighbors, query_pool_positions, topk)


def overlap(reference: np.ndarray, candidate: np.ndarray, k: int) -> float:
    values = np.empty(len(reference), dtype=np.float32)
    for row in range(len(reference)):
        values[row] = len(
            np.intersect1d(reference[row, :k], candidate[row, :k], assume_unique=True)
        ) / k
    return float(values.mean())


def vector_quality(truth: np.ndarray, candidate: np.ndarray) -> dict:
    dot = np.sum(truth * candidate, axis=1)
    denominator = np.maximum(
        np.linalg.norm(truth, axis=1) * np.linalg.norm(candidate, axis=1), 1e-12
    )
    return {
        "cosine_mean": float(np.mean(dot / denominator)),
        "cosine_median": float(np.median(dot / denominator)),
        "mse": float(np.mean(np.square(truth - candidate))),
    }


def build_context_prototype(
    mapping_path: Path,
    group_column: str,
    target_lookup: np.ndarray,
    target_features: np.ndarray,
    orig2dense: np.ndarray,
    dense_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build normalized per-relation LOO prototypes using every real audio source."""
    started = time.monotonic()
    table = pq.read_table(mapping_path, columns=[group_column, "item_id"])
    groups = table[group_column].to_numpy(zero_copy_only=False).astype(np.uint32)
    items = table["item_id"].to_numpy(zero_copy_only=False).astype(np.uint32)
    if np.any(groups[1:] < groups[:-1]):
        raise RuntimeError(f"{mapping_path} is not sorted by {group_column}")
    dense_ids = orig2dense[items].astype(np.int64, copy=False)
    source = dense_ids > 0
    source_groups = groups[source]
    source_dense = dense_ids[source]
    starts = np.r_[0, np.flatnonzero(source_groups[1:] != source_groups[:-1]) + 1]
    ends = np.r_[starts[1:], len(source_groups)]
    group_counts = (ends - starts).astype(np.int64)
    group_sums = np.add.reduceat(
        np.asarray(dense_features[source_dense], dtype=np.float32), starts, axis=0
    )
    source_group_index = np.repeat(
        np.arange(len(starts), dtype=np.int64), group_counts
    )
    source_items = items[source]
    target_pos_plus = target_lookup[source_items]
    target_rows = target_pos_plus > 0
    positions = target_pos_plus[target_rows].astype(np.int64) - 1
    group_index = source_group_index[target_rows]
    remaining = group_counts[group_index] - 1
    usable = remaining > 0
    positions = positions[usable]
    group_index = group_index[usable]
    remaining = remaining[usable]
    relation_proto = group_sums[group_index] - target_features[positions]
    relation_proto /= remaining[:, None]
    normalize_rows(relation_proto)

    order = np.argsort(positions, kind="stable")
    sorted_pos = positions[order]
    item_starts = np.r_[0, np.flatnonzero(sorted_pos[1:] != sorted_pos[:-1]) + 1]
    unique_pos = sorted_pos[item_starts]
    output = np.zeros_like(target_features)
    output[unique_pos] = np.add.reduceat(relation_proto[order], item_starts, axis=0)
    output[unique_pos] = normalize_rows(output[unique_pos])
    available = np.zeros(len(target_features), dtype=bool)
    available[unique_pos] = True
    relation_count = np.zeros(len(target_features), dtype=np.uint16)
    relation_count[unique_pos] = np.diff(np.r_[item_starts, len(sorted_pos)]).astype(
        np.uint16
    )
    report = {
        "mapping": str(mapping_path.resolve()),
        "mapping_sha256": sha256(mapping_path),
        "group_column": group_column,
        "mapping_rows": int(len(items)),
        "real_audio_source_rows": int(source.sum()),
        "real_audio_source_groups": int(len(starts)),
        "target_relation_rows": int(target_rows.sum()),
        "usable_strict_loo_target_relation_rows": int(usable.sum()),
        "target_items_available": int(available.sum()),
        "target_coverage": float(available.mean()),
        "singleton_or_unavailable_targets": int((~available).sum()),
        "maximum_usable_relations_per_target": int(relation_count.max()),
        "prototype": "normalize(mean(other real-audio embeddings in relation)); then normalize(mean across usable relations)",
        "target_self_excluded": True,
        "source_scope": "all 7,721,749 locally available real audio embeddings, not missing-audio items",
        "elapsed_seconds": time.monotonic() - started,
    }
    del table, groups, items, dense_ids, source_groups, source_dense
    del group_sums, source_group_index, relation_proto
    gc.collect()
    return output, available, report


def fuse(
    audio: np.ndarray,
    contexts: list[tuple[np.ndarray, np.ndarray]],
    weight: float,
) -> tuple[np.ndarray, dict]:
    context_sum = np.zeros_like(audio)
    context_count = np.zeros(len(audio), dtype=np.uint8)
    for prototype, available in contexts:
        context_sum[available] += prototype[available]
        context_count += available.astype(np.uint8)
    available = context_count > 0
    context_sum[available] /= context_count[available, None]
    context_sum[available] = normalize_rows(context_sum[available])
    output = np.array(audio, copy=True)
    output[available] += weight * context_sum[available]
    normalize_rows(output)
    return output, {
        "context_weight_relative_to_audio": weight,
        "available_items": int(available.sum()),
        "fallback_audio_only_items": int((~available).sum()),
        "contexts_per_item": {
            "one": int((context_count == 1).sum()),
            "two": int((context_count == 2).sum()),
        },
        "fusion": "L2_normalize(audio + weight * L2_normalize(mean(available LOO prototypes)))",
        "projection": "shared identity projection in the original 128-d audio coordinate system",
        "availability_mask_used_for_gating": True,
    }


def row_hashes(vectors: np.ndarray, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coeff1 = rng.integers(1, np.iinfo(np.uint64).max, 128, dtype=np.uint64) | 1
    coeff2 = rng.integers(1, np.iinfo(np.uint64).max, 128, dtype=np.uint64) | 1
    first = np.empty(len(vectors), dtype=np.uint64)
    second = np.empty(len(vectors), dtype=np.uint64)
    for start in range(0, len(vectors), 65_536):
        end = min(start + 65_536, len(vectors))
        bits = np.ascontiguousarray(vectors[start:end]).view(np.uint32).reshape(-1, 128)
        first[start:end] = np.sum(bits.astype(np.uint64) * coeff1, axis=1, dtype=np.uint64)
        second[start:end] = np.sum(bits.astype(np.uint64) * coeff2, axis=1, dtype=np.uint64)
    return first, second


def exact_duplicate_metrics(
    vectors: np.ndarray,
    extra_key: np.ndarray | None = None,
) -> dict:
    h1, h2 = row_hashes(vectors)
    if extra_key is None:
        order = np.lexsort((h2, h1))
    else:
        order = np.lexsort((h2, h1, extra_key))
    ordered_h1 = h1[order]
    ordered_h2 = h2[order]
    same = (ordered_h1[1:] == ordered_h1[:-1]) & (ordered_h2[1:] == ordered_h2[:-1])
    if extra_key is not None:
        ordered_key = extra_key[order]
        same &= ordered_key[1:] == ordered_key[:-1]
    starts = np.r_[0, np.flatnonzero(~same) + 1]
    ends = np.r_[starts[1:], len(order)]
    candidate = np.flatnonzero((ends - starts) > 1)
    exact_counts: list[int] = []
    hash_collision_groups = 0
    for group in candidate.tolist():
        idx = order[starts[group] : ends[group]]
        _, counts = np.unique(vectors[idx], axis=0, return_counts=True)
        duplicates = counts[counts > 1]
        if len(duplicates):
            exact_counts.extend(duplicates.tolist())
        if len(counts) > 1:
            hash_collision_groups += 1
    counts = np.asarray(exact_counts, dtype=np.int64)
    duplicate_items = int(counts.sum()) if len(counts) else 0
    return {
        "unique_vectors": int(len(vectors) - duplicate_items + len(counts)),
        "duplicate_buckets": int(len(counts)),
        "duplicate_items": duplicate_items,
        "duplicate_item_ratio": duplicate_items / len(vectors),
        "duplicate_surplus": int((counts - 1).sum()) if len(counts) else 0,
        "maximum_duplicate_bucket": int(counts.max()) if len(counts) else 1,
        "candidate_hash_groups_verified_by_exact_float32_comparison": int(len(candidate)),
        "hash_collision_groups_without_exact_equality": int(hash_collision_groups),
    }


def load_raw_explicit_vectors(
    parquet_path: Path,
    target_lookup: np.ndarray,
    rows: int,
) -> tuple[np.ndarray, dict]:
    raw = np.empty((rows, 128), dtype=np.float32)
    filled = np.zeros(rows, dtype=bool)
    parquet = pq.ParquetFile(parquet_path)
    for rg in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(rg, columns=["item_id", "embed"], use_threads=False)
        items = table["item_id"].to_numpy(zero_copy_only=False).astype(np.uint32)
        pos_plus = target_lookup[items]
        selected = pos_plus > 0
        if selected.any():
            array = table["embed"].combine_chunks()
            values = array.values.to_numpy(zero_copy_only=False).reshape(len(items), 128)
            pos = pos_plus[selected].astype(np.int64) - 1
            raw[pos] = values[selected].astype(np.float32)
            filled[pos] = True
    if not filled.all():
        raise RuntimeError(f"raw embedding extraction missed {(~filled).sum()} targets")
    norms = np.linalg.norm(raw, axis=1)
    return raw, {
        "items": rows,
        "finite_items": int(np.isfinite(raw).all(axis=1).sum()),
        "nan_items": int(np.isnan(raw).any(axis=1).sum()),
        "inf_items": int(np.isinf(raw).any(axis=1).sum()),
        "zero_norm_items": int((norms == 0).sum()),
        "norm": quantiles(norms, extra=True),
        "source": str(parquet_path.resolve()),
        "source_sha256": sha256(parquet_path),
    }


def approximate_near_duplicate_audit(
    features: np.ndarray,
    sample_size: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed + 991)
    queries = np.sort(rng.choice(len(features), size=min(sample_size, len(features)), replace=False))
    train = rng.choice(len(features), size=min(200_000, len(features)), replace=False)
    nlist = min(4096, max(64, len(features) // 50))
    quantizer = faiss.IndexFlatIP(features.shape[1])
    index = faiss.IndexIVFFlat(quantizer, features.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(np.ascontiguousarray(features[train], dtype=np.float32))
    for start in range(0, len(features), 131_072):
        index.add(np.ascontiguousarray(features[start : start + 131_072], dtype=np.float32))
    index.nprobe = min(64, nlist)
    distances, neighbors = index.search(
        np.ascontiguousarray(features[queries], dtype=np.float32), 8
    )
    nearest_other = np.empty(len(queries), dtype=np.float32)
    for row, own in enumerate(queries.tolist()):
        keep = neighbors[row] != own
        nearest_other[row] = distances[row][keep][0]
    result = {
        "method": "sampled IVF-Flat approximate nearest-other audit",
        "sample_items": int(len(queries)),
        "reference_items": int(len(features)),
        "nlist": int(nlist),
        "nprobe": int(index.nprobe),
        "search_k": 8,
        "not_an_exhaustive_pair_count": True,
        "nearest_other_cosine": quantiles(nearest_other, extra=True),
        "thresholds": {},
    }
    for threshold in (.99, .999, .9999):
        count = int((nearest_other >= threshold).sum())
        result["thresholds"][str(threshold)] = {
            "sample_items_with_neighbor": count,
            "sample_item_ratio": count / len(queries),
            "estimated_population_items": float(count / len(queries) * len(features)),
        }
    del index, quantizer
    gc.collect()
    return result


def residual_audit(
    audio: np.ndarray,
    codebooks: np.ndarray,
    codes: np.ndarray,
    seed: int,
) -> dict:
    reconstruction = reconstruct(codebooks, codes)
    residual = audio - reconstruction
    sid = packed_codes(codes)
    exact = exact_duplicate_metrics(residual, extra_key=sid)
    order = np.argsort(sid, kind="stable")
    ordered_sid = sid[order]
    starts = np.r_[0, np.flatnonzero(ordered_sid[1:] != ordered_sid[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    collision = np.flatnonzero((ends - starts) > 1)
    rng = np.random.default_rng(seed + 733)
    wanted = 250_000
    selected_bucket = rng.choice(collision, size=wanted, replace=True)
    first = np.empty(wanted, dtype=np.int64)
    second = np.empty(wanted, dtype=np.int64)
    for row, bucket in enumerate(selected_bucket.tolist()):
        members = order[starts[bucket] : ends[bucket]]
        pair = rng.choice(members, size=2, replace=False)
        first[row], second[row] = int(pair[0]), int(pair[1])
    l2 = np.linalg.norm(residual[first] - residual[second], axis=1)
    audio_cosine = np.sum(audio[first] * audio[second], axis=1)
    return {
        "residual_exact_duplicates_within_sid": exact,
        "sid_plus_float32_residual_unique_rate": exact["unique_vectors"] / len(audio),
        "sampled_within_collision_bucket_pairs": int(wanted),
        "sampling": "collision buckets sampled uniformly with replacement, then two distinct items uniformly",
        "residual_pair_l2": quantiles(l2, extra=True),
        "original_audio_pair_cosine": quantiles(audio_cosine, extra=True),
        "identity_note": "within a fixed SID reconstruction is constant, so residual pair differences equal original audio pair differences",
    }


def main() -> None:
    args = parse_args()
    if args.max_items and args.max_items < 512:
        raise ValueError("max-items smoke must be at least 512")
    if not 0 < args.context_weight <= 1:
        raise ValueError("context weight must be in (0,1]")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    faiss.omp_set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    print("[1/8] Loading explicit real-audio universe", flush=True)
    with np.load(args.support, allow_pickle=False) as support:
        mask = support["real_embedding"]
        item_ids = support["item_id"][mask].astype(np.uint32, copy=False)
        dense_ids = support["dense_id"][mask].astype(np.int64, copy=False)
    if args.max_items:
        item_ids = item_ids[: args.max_items]
        dense_ids = dense_ids[: args.max_items]
    if np.any(dense_ids <= 0) or np.any(item_ids[1:] <= item_ids[:-1]):
        raise RuntimeError("invalid real item universe")
    dense_features = np.load(args.dense, mmap_mode="r")
    audio = np.ascontiguousarray(dense_features[dense_ids], dtype=np.float32)
    if not np.isfinite(audio).all() or not np.allclose(
        np.linalg.norm(audio, axis=1), 1.0, atol=2e-5, rtol=2e-5
    ):
        raise RuntimeError("normalized audio feature contract failed")
    target_lookup = np.zeros(len(np.load(args.orig2dense, mmap_mode="r")), dtype=np.int32)
    target_lookup[item_ids.astype(np.int64)] = np.arange(1, len(item_ids) + 1, dtype=np.int32)
    orig2dense = np.load(args.orig2dense, mmap_mode="r")

    rng = np.random.default_rng(args.seed + 101)
    pool_positions = np.sort(
        rng.choice(len(audio), size=min(args.neighbor_pool, len(audio)), replace=False)
    )
    query_pool_positions = np.sort(
        rng.choice(len(pool_positions), size=min(args.neighbor_queries, len(pool_positions)), replace=False)
    )
    truth_neighbors = neighbor_sets(audio, pool_positions, query_pool_positions, 50)

    print("[2/8] Building strict-LOO artist and album prototypes", flush=True)
    artist, artist_mask, artist_report = build_context_prototype(
        args.artist, "artist_id", target_lookup, audio, orig2dense, dense_features
    )
    album, album_mask, album_report = build_context_prototype(
        args.album, "album_id", target_lookup, audio, orig2dense, dense_features
    )
    contexts = {
        "B0_audio_only": (audio, {"fusion": "audio only", "available_items": len(audio)}),
    }
    b1, b1_report = fuse(audio, [(artist, artist_mask)], args.context_weight)
    b2, b2_report = fuse(audio, [(album, album_mask)], args.context_weight)
    b3, b3_report = fuse(
        audio, [(artist, artist_mask), (album, album_mask)], args.context_weight
    )
    contexts["B1_audio_artist"] = (b1, b1_report)
    contexts["B2_audio_album"] = (b2, b2_report)
    contexts["B3_audio_artist_album"] = (b3, b3_report)

    print("[3/8] Fitting full-scale comparable RQKMeans variants", flush=True)
    method_reports: dict[str, dict] = {}
    base_codes: np.ndarray | None = None
    base_codebooks: np.ndarray | None = None
    for method, (features, fusion_report) in contexts.items():
        print(f"Fitting {method}: {len(features):,} items", flush=True)
        method_dir = args.output_dir / method
        codebooks, codes, trace, elapsed = fit_rq(
            features, args.iterations, args.seed
        )
        reconstruction = reconstruct(codebooks, codes)
        normalized_reconstruction = np.array(reconstruction, copy=True)
        normalize_rows(normalized_reconstruction)
        feature_neighbors = neighbor_sets(
            features, pool_positions, query_pool_positions, 50
        )
        reconstruction_neighbors = neighbor_sets(
            normalized_reconstruction, pool_positions, query_pool_positions, 50
        )
        report = {
            "method": method,
            "fusion_contract": fusion_report,
            "fit": {
                "items": int(len(features)),
                "levels": 4,
                "codes_per_level": 256,
                "feature_dimension": 128,
                "iterations": args.iterations,
                "restarts": 1,
                "seed": args.seed,
                "threads": args.threads,
                "all_items_used_by_faiss": True,
                "elapsed_seconds": elapsed,
                "level_trace": trace,
            },
            "identity": collision_metrics(codes),
            "semantic_preservation": {
                "fused_input_vs_original_audio": vector_quality(audio, features),
                "rq_reconstruction_vs_fused_input": vector_quality(features, reconstruction),
                "rq_reconstruction_vs_original_audio": vector_quality(audio, reconstruction),
                "input_neighbor_overlap_at_10": overlap(truth_neighbors, feature_neighbors, 10),
                "input_neighbor_overlap_at_50": overlap(truth_neighbors, feature_neighbors, 50),
                "rq_reconstruction_neighbor_overlap_at_10": overlap(
                    truth_neighbors, reconstruction_neighbors, 10
                ),
                "rq_reconstruction_neighbor_overlap_at_50": overlap(
                    truth_neighbors, reconstruction_neighbors, 50
                ),
                "neighbor_protocol": {
                    "reference_pool": int(len(pool_positions)),
                    "queries": int(len(query_pool_positions)),
                    "exact_flat_inner_product_within_fixed_pool": True,
                    "self_excluded": True,
                    "seed": args.seed + 101,
                },
            },
        }
        if base_codes is not None:
            report["semantic_preservation"]["audio_only_sid_consistency"] = (
                aligned_prefix_consistency(
                    base_codebooks, base_codes, codebooks, codes
                )
            )
        else:
            base_codes = np.array(codes, copy=True)
            base_codebooks = np.array(codebooks, copy=True)
            report["semantic_preservation"]["audio_only_sid_consistency"] = (
                prefix_consistency(codes, codes)
            )
        atomic_npy(method_dir / "codebooks.npy", codebooks)
        atomic_npy(method_dir / "codes.uint8.npy", codes)
        atomic_json(method_dir / "report.json", report)
        method_reports[method] = report
        print(
            f"  unique={report['identity']['unique_semantic_sid_paths']:,}; "
            f"collision={report['identity']['collision_item_ratio']:.4%}; "
            f"audio NN@10={report['semantic_preservation']['input_neighbor_overlap_at_10']:.4f}",
            flush=True,
        )
        del codebooks, codes, reconstruction, normalized_reconstruction
        gc.collect()

    assert base_codes is not None and base_codebooks is not None
    print("[4/8] Auditing raw and normalized embedding uniqueness", flush=True)
    raw, raw_validity = load_raw_explicit_vectors(args.embeddings, target_lookup, len(audio))
    raw_duplicates = exact_duplicate_metrics(raw)
    del raw
    gc.collect()
    normalized_validity = {
        "items": int(len(audio)),
        "finite_items": int(np.isfinite(audio).all(axis=1).sum()),
        "nan_items": int(np.isnan(audio).any(axis=1).sum()),
        "inf_items": int(np.isinf(audio).any(axis=1).sum()),
        "zero_norm_items": int((np.linalg.norm(audio, axis=1) == 0).sum()),
        "norm": quantiles(np.linalg.norm(audio, axis=1), extra=True),
    }
    normalized_duplicates = exact_duplicate_metrics(audio)

    print("[5/8] Auditing near duplicates (sampled IVF-Flat)", flush=True)
    near_duplicates = approximate_near_duplicate_audit(
        audio, args.near_duplicate_sample, args.seed
    )
    print("[6/8] Auditing SID + continuous residual uniqueness", flush=True)
    residual = residual_audit(audio, base_codebooks, base_codes, args.seed)

    candidate_report_path = (
        ROOT / "phase1_gates" / "artifacts" / "gate1_candidate_rq" / "report.json"
    )
    old_collision_path = (
        ROOT / "sid_feasibility_gate" / "artifacts" / "real_sid_collision.json"
    )
    old_report = json.loads(old_collision_path.read_text())
    old_fit = json.loads(candidate_report_path.read_text())
    output = {
        "status": "complete_sid_feasibility_supplement" if not args.max_items else "smoke",
        "data_contract": {
            "real_audio_items": int(len(audio)),
            "explicit_real_audio_only": True,
            "missing_audio_items_processed": False,
            "snmpp_or_hpn_trained": False,
            "suffix_generated": False,
            "gate2_started": False,
        },
        "comparison_contract": {
            "rq_objective": "four sequential global Euclidean residual KMeans fits",
            "levels": 4,
            "codes_per_level": 256,
            "iterations": args.iterations,
            "restarts": 1,
            "seed": args.seed,
            "normalized_input": True,
            "context_weight": args.context_weight,
            "old_candidate_fit_items": old_fit["data_contract"]["fit_items"],
            "old_candidate_collision": old_report["collision"],
        },
        "metadata_prototypes": {
            "artist": artist_report,
            "album": album_report,
        },
        "methods": method_reports,
        "embedding_uniqueness": {
            "raw_audio": {
                "validity": raw_validity,
                "exact_duplicates": raw_duplicates,
            },
            "normalized_audio": {
                "validity": normalized_validity,
                "exact_duplicates": normalized_duplicates,
            },
            "near_duplicates": near_duplicates,
        },
        "residual_uniqueness": residual,
        "artifacts": {
            "support": str(args.support.resolve()),
            "dense_features": str(args.dense.resolve()),
            "orig2dense": str(args.orig2dense.resolve()),
            "embeddings": str(args.embeddings.resolve()),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    print("[7/8] Writing consolidated machine-readable report", flush=True)
    atomic_json(args.output_dir / "sid_feasibility_supplement_metrics.json", output)
    print("[8/8] Complete; STOP before Gate 2", flush=True)
    print(json.dumps({
        "output": str((args.output_dir / "sid_feasibility_supplement_metrics.json").resolve()),
        "elapsed_seconds": output["elapsed_seconds"],
        "collisions": {
            key: value["identity"]["collision_item_ratio"]
            for key, value in method_reports.items()
        },
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
