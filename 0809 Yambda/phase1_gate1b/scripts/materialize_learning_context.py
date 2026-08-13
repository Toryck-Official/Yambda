#!/usr/bin/env python3
"""Materialize fixed Gate 1B context features and frozen SID targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LEGACY = Path("/root/autodl-tmp/0804 Yambda/dataprocess")
sys.path.insert(0, str(LEGACY))
from rqdataprocess.rq import encode_with_codebooks  # noqa: E402

DEFAULT_CONFIG = ROOT / "phase1_gate1b" / "configs" / "gate1b.json"
DEFAULT_SAMPLE = ROOT / "phase1_gate1b" / "artifacts" / "learning_sample.npz"
DEFAULT_PROXIES = ROOT / "phase1_gate1b" / "work" / "learning_proxy_vectors"
DEFAULT_OUTPUT = ROOT / "phase1_gate1b" / "work" / "learning_arrays"
DEFAULT_REPORT = ROOT / "phase1_gate1b" / "artifacts" / "learning_context_report.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    p.add_argument("--proxy-dir", type=Path, default=DEFAULT_PROXIES)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return p.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    paths = {key: Path(value) for key, value in config["paths"].items()}
    with np.load(args.sample, allow_pickle=False) as z:
        sample = {key: z[key] for key in z.files}
    artist = np.load(args.proxy_dir / "artist_proxy.npy", mmap_mode="r")
    album = np.load(args.proxy_dir / "album_proxy.npy", mmap_mode="r")
    aa = np.load(args.proxy_dir / "artist_available.npy")
    ba = np.load(args.proxy_dir / "album_available.npy")
    if not np.array_equal(aa, sample["artist_loo_source_count"] > 0):
        raise ValueError("artist availability mismatch")
    if not np.array_equal(ba, sample["album_loo_source_count"] > 0):
        raise ValueError("album availability mismatch")
    scalar_raw = np.column_stack([
        aa.astype(np.float32),
        ba.astype(np.float32),
        np.log1p(sample["artist_loo_source_count"]).astype(np.float32),
        np.log1p(sample["album_loo_source_count"]).astype(np.float32),
        np.log1p(sample["artist_usable_relation_count"]).astype(np.float32),
        np.log1p(sample["album_usable_relation_count"]).astype(np.float32),
    ])
    train = sample["split"] == 0
    mean = scalar_raw[train, 2:].mean(axis=0)
    std = scalar_raw[train, 2:].std(axis=0)
    std = np.maximum(std, 1e-6)
    scalar_raw[:, 2:] = (scalar_raw[:, 2:] - mean) / std
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context_path = args.output_dir / "context.float16.npy"
    context = np.lib.format.open_memmap(
        context_path, mode="w+", dtype=np.float16, shape=(len(sample["item_id"]), 262)
    )
    chunk = 16_384
    for start in range(0, len(context), chunk):
        end = min(start + chunk, len(context))
        context[start:end, :128] = artist[start:end]
        context[start:end, 128:256] = album[start:end]
        context[start:end, 256:] = scalar_raw[start:end]
    context.flush()
    dense = np.load(paths["dense_features"], mmap_mode="r")
    truth_path = args.output_dir / "truth.float32.npy"
    truth = np.lib.format.open_memmap(
        truth_path, mode="w+", dtype=np.float32, shape=(len(context), 128)
    )
    for start in range(0, len(truth), chunk):
        end = min(start + chunk, len(truth))
        truth[start:end] = dense[sample["dense_id"][start:end].astype(np.int64)]
    truth.flush()
    codebooks = np.load(paths["codebooks"])
    codes = encode_with_codebooks(np.asarray(truth), codebooks)
    np.save(args.output_dir / "true_codes.uint16.npy", codes, allow_pickle=False)
    for key in ["item_id", "dense_id", "split", "availability", "event_count",
                "artist_size_tier", "album_size_tier", "frequency_tier"]:
        np.save(args.output_dir / f"{key}.npy", sample[key], allow_pickle=False)
    report = {
        "status": "complete_gate1b_learning_arrays",
        "context": {
            "dimension": 262,
            "components": ["artist_loo_centroid_128", "album_loo_centroid_128",
                           "artist_available", "album_available",
                           "standardized_log1p_artist_source_count",
                           "standardized_log1p_album_source_count",
                           "standardized_log1p_artist_relation_count",
                           "standardized_log1p_album_relation_count"],
            "scalar_standardization_fit_on_train_only": True,
            "scalar_train_mean": mean.tolist(),
            "scalar_train_std": std.tolist(),
            "interaction_frequency_used_as_model_input": False,
            "item_id_used_as_model_input": False,
        },
        "targets": {
            "embedding": "true normalized 128-d audio embedding",
            "sid": "same frozen real-only candidate RQKMeans codebook",
        },
        "counts": {name: int(np.count_nonzero(sample["split"] == code)) for name, code in {"train":0,"validation":1,"test":2}.items()},
    }
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
