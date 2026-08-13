#!/usr/bin/env python3
"""Calibrate frozen RQKMeans SID stability against controlled cosine noise."""

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
from rqdataprocess.rq import encode_with_codebooks, nearest_codes  # noqa: E402

DEFAULT_CONFIG = ROOT / "phase1_gate1b" / "configs" / "gate1b.json"
DEFAULT_OUTPUT = ROOT / "phase1_gate1b" / "artifacts" / "rq_perturbation_stability.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def quantile_summary(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def controlled_perturbation(
    truth: np.ndarray, cosine: float, rng: np.random.Generator
) -> np.ndarray:
    noise = rng.standard_normal(truth.shape, dtype=np.float32)
    noise -= np.sum(noise * truth, axis=1, keepdims=True) * truth
    norms = np.linalg.norm(noise, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise RuntimeError("degenerate orthogonal noise")
    noise /= norms
    result = cosine * truth + np.sqrt(1.0 - cosine * cosine) * noise
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return np.ascontiguousarray(result, dtype=np.float32)


def original_margin_diagnostics(
    truth: np.ndarray, codebooks: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = truth.copy()
    levels = codebooks.shape[0]
    codes = np.empty((len(truth), levels), dtype=np.uint16)
    absolute = np.empty((len(truth), levels), dtype=np.float32)
    relative = np.empty((len(truth), levels), dtype=np.float32)
    for level, centers in enumerate(codebooks):
        distances, assignments = nearest_codes(residual, centers, topk=2)
        codes[:, level] = assignments[:, 0].astype(np.uint16)
        absolute[:, level] = distances[:, 1] - distances[:, 0]
        relative[:, level] = absolute[:, level] / np.maximum(distances[:, 1], 1e-12)
        residual -= centers[codes[:, level]]
    return codes, absolute, relative


def margin_flip_report(
    margins: np.ndarray, flipped: np.ndarray, bins: int
) -> dict:
    edges = np.quantile(margins, np.linspace(0.0, 1.0, bins + 1))
    rows = []
    for index in range(bins):
        left, right = float(edges[index]), float(edges[index + 1])
        if index == bins - 1:
            mask = (margins >= left) & (margins <= right)
        else:
            mask = (margins >= left) & (margins < right)
        rows.append(
            {
                "bin": index + 1,
                "margin_min": left,
                "margin_max": right,
                "count": int(mask.sum()),
                "token_flip_rate": float(flipped[mask].mean()) if mask.any() else None,
            }
        )
    return {
        "margin_stable_median": float(np.median(margins[~flipped]))
        if np.any(~flipped)
        else None,
        "margin_flipped_median": float(np.median(margins[flipped]))
        if np.any(flipped)
        else None,
        "decile_flip_rate": rows,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    paths = {key: Path(value) for key, value in config["paths"].items()}
    stability = config["stability"]
    sample = np.load(paths["sample"], allow_pickle=False)
    dense_ids = sample["dense_id"][: int(stability["sample_size"])].astype(np.int64)
    item_ids = sample["item_id"][: len(dense_ids)]
    sample.close()
    dense = np.load(paths["dense_features"], mmap_mode="r")
    truth = np.ascontiguousarray(dense[dense_ids], dtype=np.float32)
    del dense
    norms = np.linalg.norm(truth, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-5, rtol=2e-5):
        raise ValueError("truth embeddings are not unit normalized")
    codebooks = np.ascontiguousarray(np.load(paths["codebooks"]), dtype=np.float32)
    faiss.omp_set_num_threads(2)
    true_codes, absolute_margin, relative_margin = original_margin_diagnostics(
        truth, codebooks
    )
    levels: dict[str, dict] = {}
    rng = np.random.default_rng(int(config["seed"]) + 1301)
    for target in stability["target_cosines"]:
        target = float(target)
        perturbed = controlled_perturbation(truth, target, rng)
        actual_cosine = np.sum(truth * perturbed, axis=1)
        row_mse = np.mean(np.square(truth - perturbed), axis=1)
        noisy_codes = encode_with_codebooks(perturbed, codebooks)
        equal = noisy_codes == true_codes
        prefix = np.logical_and.accumulate(equal, axis=1)
        layer_margins = {}
        for level in range(codebooks.shape[0]):
            layer_margins[f"level_{level + 1}"] = margin_flip_report(
                relative_margin[:, level], ~equal[:, level], int(stability["margin_bins"])
            )
        levels[f"target_{target:.2f}"] = {
            "count": int(len(truth)),
            "target_cosine": target,
            "actual_cosine": quantile_summary(actual_cosine),
            "mse": quantile_summary(row_mse),
            "token_accuracy": {
                f"level_{level + 1}": float(equal[:, level].mean())
                for level in range(equal.shape[1])
            },
            "prefix_accuracy": {
                f"prefix_at_{level + 1}": float(prefix[:, level].mean())
                for level in range(prefix.shape[1])
            },
            "exact_semantic_sid_at_4": float(prefix[:, -1].mean()),
            "margin_relation": layer_margins,
        }
        print(
            f"cos={target:.2f} prefix="
            + "/".join(f"{prefix[:, i].mean():.4f}" for i in range(4)),
            flush=True,
        )
    report = {
        "status": "complete_gate1b_rq_stability_calibration",
        "data_contract": {
            "sample_items": int(len(truth)),
            "sample_item_id_min": int(item_ids.min()),
            "sample_item_id_max": int(item_ids.max()),
            "truth_normalized": True,
            "perturbation": "random Gaussian direction projected orthogonal to each truth vector; target cosine then constructed analytically and renormalized",
            "same_frozen_codebook_for_truth_and_perturbation": True,
            "candidate_codebook_not_gate2_final": True,
            "codebook_path": str(paths["codebooks"]),
        },
        "original_codeword_margin": {
            "definition_absolute": "second_nearest_squared_l2 - nearest_squared_l2 on the original greedy residual",
            "definition_relative": "absolute_margin / max(second_nearest_squared_l2, 1e-12)",
            "absolute_by_level": {
                f"level_{i + 1}": quantile_summary(absolute_margin[:, i])
                for i in range(absolute_margin.shape[1])
            },
            "relative_by_level": {
                f"level_{i + 1}": quantile_summary(relative_margin[:, i])
                for i in range(relative_margin.shape[1])
            },
        },
        "curve": levels,
    }
    atomic_json(args.output, report)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

