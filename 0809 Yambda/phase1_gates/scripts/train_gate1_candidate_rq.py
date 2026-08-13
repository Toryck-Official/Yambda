#!/usr/bin/env python3
"""Train a real-only candidate RQ codebook for Gate-1 PrefixAcc evaluation.

The candidate fit sample is disjoint from proxy validation targets. This artifact is
explicitly provisional and must never be presented as the Gate-2 full-universe fit.
"""

from __future__ import annotations

import argparse
import gc
import json
import mmap
import sys
import time
from pathlib import Path

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LEGACY_DATAPROCESS = Path("/root/autodl-tmp/0804 Yambda/dataprocess")
sys.path.insert(0, str(LEGACY_DATAPROCESS))

from rqdataprocess.config import RQConfig  # noqa: E402
from rqdataprocess.rq import ResidualKMeans  # noqa: E402


DEFAULT_CONFIG = ROOT / "phase1_gates" / "configs" / "gates.json"
DEFAULT_SAMPLE = ROOT / "phase1_gates" / "artifacts" / "proxy_validation_sample.npz"
DEFAULT_OUTPUT = ROOT / "phase1_gates" / "artifacts" / "gate1_candidate_rq"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fit-items", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    gate = config["gate1"]
    paths = {key: Path(value) for key, value in config["paths"].items()}
    seed = int(config["seed"])
    fit_items = int(args.fit_items or gate["candidate_codebook_fit_items"])
    iterations = int(args.iterations or gate["candidate_codebook_iterations"])

    with np.load(paths["item_frequency"], allow_pickle=False) as payload:
        item_ids = payload["item_id"].astype(np.uint32, copy=False)
    orig2dense = np.load(paths["orig2dense"], mmap_mode="r")
    dense_ids = orig2dense[item_ids].astype(np.int32, copy=False)
    with np.load(args.sample, allow_pickle=False) as sample:
        target_ids = sample["item_id"].astype(np.uint32, copy=False)
    real_candidates = item_ids[dense_ids > 0]
    real_candidates = np.setdiff1d(real_candidates, target_ids, assume_unique=True)
    if fit_items > len(real_candidates):
        raise ValueError("fit sample exceeds remaining real items")
    rng = np.random.default_rng(seed + 17)
    fit_item_ids = np.sort(rng.choice(real_candidates, size=fit_items, replace=False))
    fit_dense_ids = orig2dense[fit_item_ids].astype(np.int64, copy=False)
    if np.any(fit_dense_ids <= 0):
        raise RuntimeError("candidate fit contains a missing embedding")
    dense_features = np.load(paths["dense_features"], mmap_mode="r")
    features = np.ascontiguousarray(dense_features[fit_dense_ids], dtype=np.float32)
    if not np.isfinite(features).all() or not np.allclose(
        np.linalg.norm(features, axis=1), 1.0, atol=2e-5, rtol=2e-5
    ):
        raise ValueError("candidate fit vectors are not finite unit-L2")
    # Random gathers can leave most of the 3.7-GiB memmap resident. The selected
    # matrix is now an owned ~98-MiB array, so release source file pages before FAISS.
    try:
        dense_features._mmap.madvise(mmap.MADV_DONTNEED)
    except (AttributeError, OSError):
        pass
    del dense_features
    gc.collect()

    faiss.omp_set_num_threads(2)
    rq_config = RQConfig(
        levels=int(gate["candidate_codebook_levels"]),
        codebook_size=int(gate["candidate_codebook_size"]),
        iterations=iterations,
        restarts=1,
        use_gpu=False,
        min_points_per_centroid=1,
        max_points_per_centroid=1_000_000,
    )
    started = time.monotonic()
    fit = ResidualKMeans(rq_config, seed=seed).fit_encode(features)
    elapsed = time.monotonic() - started
    if fit.codebooks.shape != (4, 256, 128):
        raise RuntimeError(f"unexpected codebook shape: {fit.codebooks.shape}")
    if not np.isfinite(fit.codebooks).all():
        raise RuntimeError("candidate codebook contains non-finite values")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_npy(args.output_dir / "codebooks.npy", fit.codebooks)
    atomic_npy(args.output_dir / "fit_item_ids.npy", fit_item_ids)
    report = {
        "status": "complete_gate1_candidate_only",
        "provisional_not_gate2": True,
        "data_contract": {
            "fit_items": int(fit_items),
            "real_audio_embedding_only": True,
            "proxy_validation_targets_excluded": True,
            "normalized": True,
            "feature_dimension": 128,
        },
        "algorithm": {
            "name": "FAISS global residual KMeans candidate",
            "levels": 4,
            "codebook_size": 256,
            "iterations": iterations,
            "restarts": 1,
            "seed": seed,
            "device": "cpu",
            "faiss_threads": 2,
        },
        "level_trace": fit.level_trace,
        "elapsed_seconds": float(elapsed),
        "outputs": {
            "codebooks": str((args.output_dir / "codebooks.npy").resolve()),
            "fit_item_ids": str((args.output_dir / "fit_item_ids.npy").resolve()),
        },
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
