#!/usr/bin/env python3
"""Audit four-level SID collisions for all explicit real-audio items."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "phase1_gates" / "artifacts" / "metadata_loo_support.npz"
DENSE = Path("/root/autodl-tmp/0626/0626 Predictor/01_data/processed/raw_rqkmeans/dense_item_features.npy")
CODEBOOKS = ROOT / "phase1_gates" / "artifacts" / "gate1_candidate_rq" / "codebooks.npy"
CODEBOOK_REPORT = ROOT / "phase1_gates" / "artifacts" / "gate1_candidate_rq" / "report.json"
OUTPUT = ROOT / "sid_feasibility_gate" / "artifacts" / "real_sid_collision.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support", type=Path, default=SUPPORT)
    parser.add_argument("--dense", type=Path, default=DENSE)
    parser.add_argument("--codebooks", type=Path, default=CODEBOOKS)
    parser.add_argument("--codebook-report", type=Path, default=CODEBOOK_REPORT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32_768)
    parser.add_argument("--max-items", type=int, default=0)
    return parser.parse_args()


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


def quantiles(values: np.ndarray) -> dict:
    return {
        name: int(np.quantile(values, q, method="higher"))
        for name, q in (("p50", .5), ("p90", .9), ("p99", .99), ("p99_9", .999))
    } | {"max": int(values.max())}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    report = json.loads(args.codebook_report.read_text())
    codebooks_np = np.load(args.codebooks)
    if codebooks_np.shape != (4, 256, 128):
        raise RuntimeError(f"unexpected codebook shape {codebooks_np.shape}")
    with np.load(args.support, allow_pickle=False) as support:
        real_mask = support["real_embedding"]
        dense_ids = support["dense_id"][real_mask].astype(np.int64, copy=False)
    if args.max_items:
        dense_ids = dense_ids[: args.max_items]
    dense = np.load(args.dense, mmap_mode="r")
    if dense.shape[1] != 128 or dense_ids.min() < 1 or dense_ids.max() >= len(dense):
        raise RuntimeError("dense feature mapping outside expected real-audio store")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.max_items:
        raise RuntimeError("full collision audit requires enabled GPU")
    codebooks = torch.from_numpy(codebooks_np).to(device)
    code_norm = codebooks.square().sum(dim=2)
    packed = np.empty(len(dense_ids), dtype=np.uint32)
    layer_usage = np.zeros((4, 256), dtype=np.uint64)
    residual_mse_sum = np.zeros(4, dtype=np.float64)
    for start in range(0, len(dense_ids), args.batch_size):
        end = min(start + args.batch_size, len(dense_ids))
        vector = torch.from_numpy(np.asarray(dense[dense_ids[start:end]], dtype=np.float32)).to(device)
        residual = vector.clone()
        batch_key = np.zeros(end - start, dtype=np.uint32)
        for level in range(4):
            distance = (
                residual.square().sum(dim=1, keepdim=True)
                + code_norm[level][None, :]
                - 2 * residual @ codebooks[level].T
            )
            code = distance.argmin(dim=1)
            code_np = code.cpu().numpy().astype(np.uint32, copy=False)
            batch_key |= code_np << np.uint32(8 * (3 - level))
            layer_usage[level] += np.bincount(code_np, minlength=256).astype(np.uint64)
            residual -= codebooks[level][code]
            residual_mse_sum[level] += float(residual.square().mean(dim=1).sum())
        packed[start:end] = batch_key
        if end % (10 * args.batch_size) == 0 or end == len(dense_ids):
            print(f"encoded {end:,}/{len(dense_ids):,}", flush=True)
    unique_path, bucket_size = np.unique(packed, return_counts=True)
    singleton = bucket_size == 1
    collision = bucket_size > 1
    singleton_items = int(bucket_size[singleton].sum(dtype=np.uint64))
    collision_items = int(bucket_size[collision].sum(dtype=np.uint64))
    result = {
        "status": "complete_real_embedding_sid_collision_audit" if not args.max_items else "smoke",
        "data_contract": {
            "items": int(len(dense_ids)),
            "scope": "all explicit items with a real source audio embedding" if not args.max_items else "prefix smoke subset",
            "normalized_embedding_column": True,
            "codebook_frozen": True,
            "codebook_shape": list(codebooks_np.shape),
            "codebook_sha256": sha256(args.codebooks),
            "candidate_codebook_report": str(args.codebook_report.resolve()),
            "candidate_codebook_is_provisional_not_gate2": bool(report.get("provisional_not_gate2")),
            "candidate_codebook_fit_items": report["data_contract"]["fit_items"],
            "sid_or_suffix_materialized": False,
        },
        "collision": {
            "unique_semantic_sid_paths": int(len(unique_path)),
            "singleton_buckets": int(singleton.sum()),
            "singleton_items": singleton_items,
            "singleton_item_ratio": singleton_items / len(dense_ids),
            "collision_buckets": int(collision.sum()),
            "collision_items": collision_items,
            "collision_item_ratio": collision_items / len(dense_ids),
            "all_bucket_size": quantiles(bucket_size),
            "collision_bucket_size": quantiles(bucket_size[collision]) if collision.any() else None,
            "quantile_rule": "numpy higher empirical quantile over unique SID buckets",
        },
        "diagnostics": {
            "used_codes_per_level": [int(np.count_nonzero(row)) for row in layer_usage],
            "residual_mse_after_level": [float(value / len(dense_ids)) for value in residual_mse_sum],
        },
        "boundaries": {
            "gate2_started": False,
            "missing_item_sid_generated": False,
            "snmpp_or_hpn_trained": False,
        },
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
