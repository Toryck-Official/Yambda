#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose SID/codebook quality before policy or predictor training.")
    parser.add_argument("--mapping_root", default=str(ROOT / "01_data" / "processed" / "raw_rqkmeans"))
    parser.add_argument("--transition_root", default=str(ROOT / "01_data" / "processed" / "regret_current_data"))
    parser.add_argument("--transition_split", default="train")
    parser.add_argument("--transition_max_rows", type=int, default=100000)
    parser.add_argument("--target_bucket_splits", default="val,test,replay_val")
    parser.add_argument("--target_bucket_max_rows", type=int, default=0, help="Per split limit for target SID bucket stats. 0 means full split.")
    parser.add_argument("--out_dir", default=str(ROOT / "artifacts" / "sid_diagnostics"))
    parser.add_argument("--path_sample_size", type=int, default=1000000)
    parser.add_argument("--locality_pool_size", type=int, default=50000)
    parser.add_argument("--locality_anchor_count", type=int, default=2000)
    parser.add_argument("--locality_batch_size", type=int, default=256)
    parser.add_argument("--example_count", type=int, default=30)
    parser.add_argument("--recall_k", type=int, nargs="+", default=[1, 4, 16, 32, 64, 128])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_locality", action="store_true")
    parser.add_argument("--skip_transition", action="store_true")
    return parser.parse_args()


def entropy_bits(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log2(p)).sum())


def quantiles(values: np.ndarray, qs=(0.5, 0.9, 0.99)) -> dict[str, float]:
    if values.size == 0:
        return {f"p{int(q * 100)}": 0.0 for q in qs}
    return {f"p{int(q * 100)}": float(np.quantile(values, q)) for q in qs}


def sid_keys(sid: np.ndarray, base: int, prefix_len: int | None = None) -> np.ndarray:
    levels = sid.shape[1] if prefix_len is None else int(prefix_len)
    keys = np.zeros(sid.shape[0], dtype=np.uint64)
    for level in range(levels):
        keys = keys * np.uint64(base) + sid[:, level].astype(np.uint64)
    return keys


def common_prefix_len(a: np.ndarray, b: np.ndarray) -> int:
    n = 0
    for x, y in zip(a.tolist(), b.tolist()):
        if int(x) != int(y):
            break
        n += 1
    return n


def parquet_files(split_root: Path) -> list[Path]:
    if split_root.is_file():
        return [split_root]
    return sorted(split_root.glob("*.parquet"))



def target_bucket_stats(
    transition_root: Path,
    split: str,
    pos_by_dense: np.ndarray,
    count_by_valid_pos: np.ndarray,
    max_rows: int,
) -> dict:
    files = parquet_files(transition_root / split)
    stats = {"split": split, "rows_scanned": 0, "valid_targets": 0}
    if not files:
        stats["missing_split"] = True
        return stats
    bucket_sizes = []
    stop = int(max_rows or 0)
    for file_path in tqdm(files, desc=f"[target bucket {split}]", unit="file", dynamic_ncols=True):
        pf = pq.ParquetFile(file_path)
        if "target_dense_item_id" not in pf.schema_arrow.names:
            continue
        for batch in pf.iter_batches(columns=["target_dense_item_id"], batch_size=65536):
            arr = np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.int64)
            stats["rows_scanned"] += int(arr.size)
            mask = (arr > 0) & (arr < len(pos_by_dense)) & (pos_by_dense[arr] >= 0)
            stats["valid_targets"] += int(mask.sum())
            if mask.any():
                bucket_sizes.append(count_by_valid_pos[pos_by_dense[arr[mask]]])
            if stop and stats["rows_scanned"] >= stop:
                break
        if stop and stats["rows_scanned"] >= stop:
            break
    values = np.concatenate(bucket_sizes).astype(np.int64) if bucket_sizes else np.asarray([], dtype=np.int64)
    def q(p: float) -> float:
        return float(np.quantile(values, p)) if values.size else 0.0
    stats.update(
        {
            "valid_target_share": stats["valid_targets"] / max(stats["rows_scanned"], 1),
            "target_bucket_size_mean": float(values.mean()) if values.size else 0.0,
            "target_bucket_size_p50": q(0.5),
            "target_bucket_size_p90": q(0.9),
            "target_bucket_size_p99": q(0.99),
            "target_bucket_size_max": int(values.max()) if values.size else 0,
            "target_unique_sid_share": float((values == 1).mean()) if values.size else 0.0,
            "target_collision_sid_share": float((values > 1).mean()) if values.size else 0.0,
            "target_bucket_le_4_share": float((values <= 4).mean()) if values.size else 0.0,
            "target_bucket_le_32_share": float((values <= 32).mean()) if values.size else 0.0,
        }
    )
    return stats


def transition_target_stats(transition_root: Path, split: str, valid_dense: np.ndarray, max_rows: int) -> dict:
    files = parquet_files(transition_root / split)
    stats = {
        "split": split,
        "rows_scanned": 0,
        "target_dense_in_range": 0,
        "target_sid_valid": 0,
        "history_items_seen": 0,
        "history_items_valid": 0,
    }
    if not files:
        stats["missing_split"] = True
        return stats
    columns = ["target_dense_item_id", "history_item_ids"]
    stop = int(max_rows or 0)
    for file_path in tqdm(files, desc=f"[transition {split}]", unit="file", dynamic_ncols=True):
        pf = pq.ParquetFile(file_path)
        present = [col for col in columns if col in pf.schema_arrow.names]
        for batch in pf.iter_batches(columns=present, batch_size=4096):
            for row in batch.to_pylist():
                dense = int(row.get("target_dense_item_id") or 0)
                stats["rows_scanned"] += 1
                if 0 < dense < len(valid_dense):
                    stats["target_dense_in_range"] += 1
                    if bool(valid_dense[dense]):
                        stats["target_sid_valid"] += 1
                for item in row.get("history_item_ids") or []:
                    item = int(item)
                    if item <= 0:
                        continue
                    stats["history_items_seen"] += 1
                    if item < len(valid_dense) and bool(valid_dense[item]):
                        stats["history_items_valid"] += 1
                if stop and stats["rows_scanned"] >= stop:
                    break
            if stop and stats["rows_scanned"] >= stop:
                break
        if stop and stats["rows_scanned"] >= stop:
            break
    rows = max(stats["rows_scanned"], 1)
    hist = max(stats["history_items_seen"], 1)
    stats["target_dense_in_range_share"] = stats["target_dense_in_range"] / rows
    stats["target_sid_valid_share"] = stats["target_sid_valid"] / rows
    stats["history_sid_valid_share"] = stats["history_items_valid"] / hist
    return stats


def run_locality(
    features: np.ndarray,
    sid: np.ndarray,
    valid_dense_ids: np.ndarray,
    dense2orig: np.ndarray,
    rng: np.random.Generator,
    pool_size: int,
    anchor_count: int,
    batch_size: int,
    example_count: int,
    out_examples: Path,
) -> dict:
    pool_size = min(int(pool_size), int(valid_dense_ids.shape[0]))
    anchor_count = min(int(anchor_count), pool_size)
    pool_dense = rng.choice(valid_dense_ids, size=pool_size, replace=False)
    anchor_dense = rng.choice(pool_dense, size=anchor_count, replace=False)

    pool_feat = np.asarray(features[pool_dense], dtype=np.float32)
    pool_norm = np.linalg.norm(pool_feat, axis=1, keepdims=True).clip(min=1e-8)
    pool_feat = pool_feat / pool_norm
    pool_index = {int(dense): idx for idx, dense in enumerate(pool_dense.tolist())}

    hamming = []
    common_prefix = []
    same_l1 = 0
    same_l2 = 0
    same_l3 = 0
    same_full = 0
    cosines = []
    random_hamming = []
    random_same_l1 = 0
    examples = []

    for start in tqdm(range(0, anchor_count, batch_size), desc="[locality nn]", unit="batch", dynamic_ncols=True):
        batch_dense = anchor_dense[start : start + batch_size]
        batch_feat = np.asarray(features[batch_dense], dtype=np.float32)
        batch_feat = batch_feat / np.linalg.norm(batch_feat, axis=1, keepdims=True).clip(min=1e-8)
        sims = batch_feat @ pool_feat.T
        for row_idx, dense_id in enumerate(batch_dense.tolist()):
            pool_pos = pool_index.get(int(dense_id))
            if pool_pos is not None:
                sims[row_idx, pool_pos] = -np.inf
        nn_pos = sims.argmax(axis=1)
        nn_sim = sims[np.arange(len(batch_dense)), nn_pos]
        nn_dense = pool_dense[nn_pos]
        for dense_id, neighbor_id, sim in zip(batch_dense.tolist(), nn_dense.tolist(), nn_sim.tolist()):
            a = sid[int(dense_id)]
            b = sid[int(neighbor_id)]
            dist = int(np.sum(a != b))
            pref = common_prefix_len(a, b)
            hamming.append(dist)
            common_prefix.append(pref)
            same_l1 += int(pref >= 1)
            same_l2 += int(pref >= 2)
            same_l3 += int(pref >= 3)
            same_full += int(pref == sid.shape[1])
            cosines.append(float(sim))
            rand_id = int(pool_dense[int(rng.integers(0, pool_size))])
            random_hamming.append(int(np.sum(a != sid[rand_id])))
            random_same_l1 += int(a[0] == sid[rand_id][0])
            if len(examples) < example_count:
                examples.append(
                    {
                        "anchor_dense": int(dense_id),
                        "anchor_orig": int(dense2orig[int(dense_id)]) if int(dense_id) < len(dense2orig) else 0,
                        "anchor_sid": [int(x) for x in a.tolist()],
                        "neighbor_dense": int(neighbor_id),
                        "neighbor_orig": int(dense2orig[int(neighbor_id)]) if int(neighbor_id) < len(dense2orig) else 0,
                        "neighbor_sid": [int(x) for x in b.tolist()],
                        "cosine": float(sim),
                        "sid_hamming": dist,
                        "common_prefix_len": pref,
                    }
                )

    out_examples.parent.mkdir(parents=True, exist_ok=True)
    with out_examples.open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = max(len(hamming), 1)
    result = {
        "pool_size": int(pool_size),
        "anchor_count": int(anchor_count),
        "mean_embedding_nn_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "mean_sid_hamming_to_embedding_nn": float(np.mean(hamming)) if hamming else 0.0,
        "median_sid_hamming_to_embedding_nn": float(np.median(hamming)) if hamming else 0.0,
        "mean_common_prefix_len": float(np.mean(common_prefix)) if common_prefix else 0.0,
        "same_l1_share": same_l1 / n,
        "same_l1_l2_prefix_share": same_l2 / n,
        "same_l1_l2_l3_prefix_share": same_l3 / n,
        "same_full_sid_share": same_full / n,
        "random_pair_mean_sid_hamming": float(np.mean(random_hamming)) if random_hamming else 0.0,
        "random_pair_same_l1_share": random_same_l1 / n,
        "examples_jsonl": str(out_examples),
    }
    return result


def main() -> None:
    args = parse_args()
    t0 = time.time()
    mapping_root = Path(args.mapping_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sid = np.load(mapping_root / "dense_item2sid.npy", mmap_mode="r")
    features = np.load(mapping_root / "dense_item_features.npy", mmap_mode="r")
    dense2orig = np.load(mapping_root / "dense2orig_item_id.npy", mmap_mode="r")
    orig2dense = np.load(mapping_root / "orig2dense_item_id.npy", mmap_mode="r")

    if sid.ndim != 2:
        raise ValueError(f"dense_item2sid must be 2-D, got {sid.shape}")
    levels = int(sid.shape[1])
    valid_dense = np.zeros(sid.shape[0], dtype=bool)
    valid_dense[1:] = (dense2orig[1:] > 0) & (sid[1:] >= 0).all(axis=1)
    valid_dense_ids = np.flatnonzero(valid_dense).astype(np.int64)
    valid_sid = np.asarray(sid[valid_dense_ids], dtype=np.int64)
    vocab_sizes = [int(valid_sid[:, i].max()) + 1 for i in range(levels)] if valid_sid.size else []
    base = int(max(vocab_sizes) if vocab_sizes else 0)

    basic = {
        "mapping_root": str(mapping_root),
        "dense_item2sid_shape": list(sid.shape),
        "dense_item_features_shape": list(features.shape),
        "dense2orig_shape": list(dense2orig.shape),
        "orig2dense_shape": list(orig2dense.shape),
        "levels": levels,
        "vocab_sizes": vocab_sizes,
        "valid_dense_items": int(valid_dense_ids.shape[0]),
        "invalid_or_padding_rows": int(sid.shape[0] - valid_dense_ids.shape[0]),
        "valid_dense_share_excluding_zero": float(valid_dense[1:].mean()) if sid.shape[0] > 1 else 0.0,
    }

    # Dense/original id round-trip check.
    valid_orig_mask = (dense2orig[valid_dense_ids] > 0) & (dense2orig[valid_dense_ids] < len(orig2dense))
    check_dense = valid_dense_ids[valid_orig_mask]
    check_orig = dense2orig[check_dense]
    roundtrip = orig2dense[check_orig] == check_dense
    id_roundtrip = {
        "checked_items": int(check_dense.shape[0]),
        "roundtrip_ok": int(roundtrip.sum()),
        "roundtrip_ok_share": float(roundtrip.mean()) if roundtrip.size else 0.0,
        "roundtrip_bad": int(roundtrip.size - roundtrip.sum()),
    }

    per_level = []
    for level in range(levels):
        tokens = valid_sid[:, level]
        counts = np.bincount(tokens, minlength=base)
        nonzero = counts[counts > 0]
        ent = entropy_bits(counts)
        per_level.append(
            {
                "level": level + 1,
                "vocab_size": base,
                "used_codes": int((counts > 0).sum()),
                "dead_codes": int((counts == 0).sum()),
                "utilization": float((counts > 0).sum() / max(base, 1)),
                "entropy_bits": ent,
                "entropy_ratio": float(ent / math.log2(max(base, 2))),
                "max_bucket_count": int(counts.max()) if counts.size else 0,
                "max_bucket_share": float(counts.max() / max(tokens.shape[0], 1)) if counts.size else 0.0,
                "min_nonzero_bucket_count": int(nonzero.min()) if nonzero.size else 0,
                **quantiles(nonzero.astype(np.float64), qs=(0.5, 0.9, 0.99)),
            }
        )

    prefix_stats = []
    for prefix_len in range(1, levels + 1):
        keys = sid_keys(valid_sid, base=base, prefix_len=prefix_len)
        _, counts = np.unique(keys, return_counts=True)
        prefix_stats.append(
            {
                "prefix_len": prefix_len,
                "unique_prefixes": int(counts.shape[0]),
                "collision_groups": int((counts > 1).sum()),
                "collision_excess_items": int((counts - 1).clip(min=0).sum()),
                "unique_item_share": float((counts == 1).sum() / max(valid_dense_ids.shape[0], 1)) if prefix_len == levels else None,
                "mean_items_per_prefix": float(counts.mean()) if counts.size else 0.0,
                "max_items_per_prefix": int(counts.max()) if counts.size else 0,
                **quantiles(counts.astype(np.float64), qs=(0.5, 0.9, 0.99)),
            }
        )

    # Exact SID reconstruction/candidate upper bound under index-first-K retrieval.
    full_keys = sid_keys(valid_sid, base=base)
    order = np.argsort(full_keys, kind="stable")
    sorted_keys = full_keys[order]
    unique_keys, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
    rank_by_valid_pos = np.empty(valid_dense_ids.shape[0], dtype=np.int32)
    count_by_valid_pos = np.empty(valid_dense_ids.shape[0], dtype=np.int32)
    rank_by_valid_pos[order] = np.arange(valid_dense_ids.shape[0], dtype=np.int32) - np.repeat(starts, counts).astype(np.int32)
    count_by_valid_pos[order] = np.repeat(counts, counts).astype(np.int32)
    exact_recall_all = {f"recall_at_{int(k)}": float((rank_by_valid_pos < int(k)).mean()) for k in args.recall_k}
    sample_size = valid_dense_ids.shape[0] if args.path_sample_size <= 0 else min(args.path_sample_size, valid_dense_ids.shape[0])
    sample_pos = rng.choice(np.arange(valid_dense_ids.shape[0]), size=sample_size, replace=False)
    sample_counts = count_by_valid_pos[sample_pos]
    reconstruction = {
        "full_catalog_item_count": int(valid_dense_ids.shape[0]),
        "sample_size_for_bucket_summary": int(sample_size),
        "unique_full_sid_paths": int(unique_keys.shape[0]),
        "unique_path_item_share_all": float((count_by_valid_pos == 1).mean()),
        "collision_item_share_all": float((count_by_valid_pos > 1).mean()),
        "mean_exact_sid_bucket_size_all_item_weighted": float(count_by_valid_pos.mean()),
        "median_exact_sid_bucket_size_all_item_weighted": float(np.median(count_by_valid_pos)),
        "mean_exact_sid_bucket_size_sample": float(sample_counts.mean()),
        "median_exact_sid_bucket_size_sample": float(np.median(sample_counts)),
        "max_exact_sid_bucket_size_global": int(counts.max()) if counts.size else 0,
        "candidate_recall_if_take_first_k_items_in_exact_sid_bucket_all_items": exact_recall_all,
    }

    transition = None
    target_bucket_by_split = None
    if not args.skip_transition:
        transition_root = Path(args.transition_root)
        transition = transition_target_stats(transition_root, args.transition_split, valid_dense, args.transition_max_rows)
        pos_by_dense = np.full(sid.shape[0], -1, dtype=np.int32)
        pos_by_dense[valid_dense_ids] = np.arange(valid_dense_ids.shape[0], dtype=np.int32)
        target_bucket_by_split = {
            split.strip(): target_bucket_stats(transition_root, split.strip(), pos_by_dense, count_by_valid_pos, args.target_bucket_max_rows)
            for split in args.target_bucket_splits.split(",")
            if split.strip()
        }

    locality = None
    if not args.skip_locality:
        locality = run_locality(
            features=features,
            sid=sid,
            valid_dense_ids=valid_dense_ids,
            dense2orig=dense2orig,
            rng=rng,
            pool_size=args.locality_pool_size,
            anchor_count=args.locality_anchor_count,
            batch_size=args.locality_batch_size,
            example_count=args.example_count,
            out_examples=out_dir / "sid_locality_examples.jsonl",
        )

    result = {
        "summary": {
            "purpose": "SID/codebook diagnostics before HPN, simulator, or policy training.",
            "interpretation": {
                "utilization": "Whether each SID level actually uses the codebook tokens instead of collapsing.",
                "reconstruction": "Whether a full SID path can identify or recall the original item; collisions reduce exact item reconstruction.",
                "locality": "Whether embedding-nearest items also share early SID tokens/prefixes.",
                "transition_coverage": "Whether train/val/test targets and histories point to valid SID rows.",
            },
        },
        "basic": basic,
        "id_roundtrip": id_roundtrip,
        "per_level_utilization": per_level,
        "prefix_and_collision_stats": prefix_stats,
        "reconstruction_and_exact_sid_recall": reconstruction,
        "transition_coverage": transition,
        "target_full_sid_bucket_by_split": target_bucket_by_split,
        "embedding_locality": locality,
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    out_path = out_dir / "sid_codebook_diagnostics.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
