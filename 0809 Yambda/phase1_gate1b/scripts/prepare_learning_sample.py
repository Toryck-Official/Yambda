#!/usr/bin/env python3
"""Create fixed train/validation/test targets for Gate 1B C/D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "phase1_gate1b" / "configs" / "gate1b.json"
DEFAULT_OUTPUT = ROOT / "phase1_gate1b" / "artifacts" / "learning_sample.npz"
DEFAULT_REPORT = ROOT / "phase1_gate1b" / "artifacts" / "learning_sample_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def stratified_train_choice(
    candidates: np.ndarray, labels: np.ndarray, requested: int, rng: np.random.Generator
) -> np.ndarray:
    unique, inverse, sizes = np.unique(labels[candidates], return_inverse=True, return_counts=True)
    raw = sizes.astype(np.float64) * requested / len(candidates)
    take = np.minimum(np.floor(raw).astype(np.int64), sizes)
    left = requested - int(take.sum())
    for idx in np.argsort(-(raw - take), kind="stable"):
        if left == 0:
            break
        if take[idx] < sizes[idx]:
            take[idx] += 1
            left -= 1
    selected = []
    for stratum, count in enumerate(take):
        if count:
            members = candidates[inverse == stratum]
            selected.append(rng.choice(members, int(count), replace=False))
    result = np.concatenate(selected)
    rng.shuffle(result)
    if len(result) != requested:
        raise RuntimeError("training sampler did not conserve requested count")
    return result


def size_tier(count: np.ndarray) -> np.ndarray:
    result = np.zeros(len(count), dtype=np.uint8)
    result[(count >= 1) & (count <= 2)] = 1
    result[(count >= 3) & (count <= 9)] = 2
    result[count >= 10] = 3
    return result


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    paths = {key: Path(value) for key, value in config["paths"].items()}
    requested = int(config["learning"]["train_items"])
    with np.load(paths["support"], allow_pickle=False) as support:
        full = {key: support[key] for key in support.files}
    with np.load(paths["sample"], allow_pickle=False) as prior:
        old = {key: prior[key] for key in prior.files}
    if len(old["item_id"]) != 100_000:
        raise ValueError("Gate 1A fixed validation/test sample changed")
    old_lookup = np.zeros(int(max(full["item_id"].max(), old["item_id"].max())) + 1, dtype=bool)
    old_lookup[old["item_id"]] = True
    eligible = full["real_embedding"] & (full["availability"] > 0) & ~old_lookup[full["item_id"]]
    event = full["event_count"]
    eligible_counts = event[eligible]
    p50 = int(np.quantile(eligible_counts, 0.5, method="higher"))
    p90 = int(np.quantile(eligible_counts, 0.9, method="higher"))
    freq = np.ones(len(event), dtype=np.uint8)
    freq[event <= p50] = 0
    freq[event > p90] = 2
    artist_tier = size_tier(full["artist_loo_source_count"])
    album_tier = size_tier(full["album_loo_source_count"])
    label = (
        full["availability"].astype(np.uint32) * 64
        + artist_tier.astype(np.uint32) * 16
        + album_tier.astype(np.uint32) * 4
        + freq.astype(np.uint32)
    )
    rng = np.random.default_rng(int(config["seed"]) + 2101)
    train_pos = stratified_train_choice(np.flatnonzero(eligible), label, requested, rng)
    index_by_item = np.full(len(old_lookup), -1, dtype=np.int32)
    index_by_item[full["item_id"]] = np.arange(len(full["item_id"]), dtype=np.int32)
    old_pos = index_by_item[old["item_id"]]
    if np.any(old_pos < 0):
        raise RuntimeError("fixed Gate 1A sample not in explicit support universe")
    val_pos = old_pos[old["split"] == 0]
    test_pos = old_pos[old["split"] == 1]
    positions = np.concatenate([train_pos, val_pos, test_pos])
    split = np.concatenate([
        np.zeros(len(train_pos), dtype=np.uint8),
        np.ones(len(val_pos), dtype=np.uint8),
        np.full(len(test_pos), 2, dtype=np.uint8),
    ])
    order = np.argsort(full["item_id"][positions], kind="stable")
    positions, split = positions[order], split[order]
    payload = {
        "item_id": full["item_id"][positions],
        "dense_id": full["dense_id"][positions],
        "event_count": full["event_count"][positions],
        "availability": full["availability"][positions],
        "artist_loo_source_count": full["artist_loo_source_count"][positions],
        "album_loo_source_count": full["album_loo_source_count"][positions],
        "artist_usable_relation_count": full["artist_usable_relation_count"][positions],
        "album_usable_relation_count": full["album_usable_relation_count"][positions],
        "artist_size_tier": artist_tier[positions],
        "album_size_tier": album_tier[positions],
        "frequency_tier": freq[positions],
        "split": split,
    }
    atomic_npz(args.output, **payload)
    report = {
        "status": "complete_gate1b_fixed_learning_sample",
        "counts": {
            "train": int(np.count_nonzero(split == 0)),
            "validation": int(np.count_nonzero(split == 1)),
            "test": int(np.count_nonzero(split == 2)),
        },
        "contract": {
            "train_disjoint_from_gate1a_sample": True,
            "validation_and_test_exactly_reuse_gate1a_targets": True,
            "targets_all_have_real_audio_embedding": True,
            "targets_all_have_strict_loo_artist_or_album_context": True,
            "frequency_is_stratification_only_not_model_input": True,
            "frequency_boundaries_from_train_eligible_universe": {"p50": p50, "p90": p90},
        },
        "output": str(args.output),
    }
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

