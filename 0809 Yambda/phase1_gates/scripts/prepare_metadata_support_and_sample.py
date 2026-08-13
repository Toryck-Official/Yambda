#!/usr/bin/env python3
"""Build strict leave-one-out metadata support and a stratified Gate-1 sample.

This script never constructs a proxy vector. It only determines whether an explicit
item has at least one *other* real-embedding item in an official artist/album group,
then samples real-embedding targets across availability, group-size, and frequency
strata. No event row or source artifact is modified.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "phase1_gates" / "configs" / "gates.json"
DEFAULT_SUPPORT = ROOT / "phase1_gates" / "artifacts" / "metadata_loo_support.npz"
DEFAULT_SAMPLE = ROOT / "phase1_gates" / "artifacts" / "proxy_validation_sample.npz"
DEFAULT_REPORT = ROOT / "phase1_gates" / "artifacts" / "metadata_support_sample_report.json"

AVAILABILITY_NAMES = {0: "none", 1: "artist_only", 2: "album_only", 3: "both"}
SIZE_TIER_NAMES = {0: "unavailable", 1: "small", 2: "medium", 3: "large"}
FREQUENCY_TIER_NAMES = {0: "tail", 1: "mid", 2: "head"}
SPLIT_NAMES = {0: "validation", 1: "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--support-output", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--sample-output", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def metadata_loo_support(
    mapping_path: Path,
    group_column: str,
    explicit_lookup: np.ndarray,
    source_present: np.ndarray,
    explicit_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    maximum_other_sources = np.zeros(explicit_rows, dtype=np.uint32)
    usable_relation_count = np.zeros(explicit_rows, dtype=np.uint32)
    mapped = np.zeros(explicit_rows, dtype=bool)
    parquet = pq.ParquetFile(mapping_path)
    carry_groups = np.empty(0, dtype=np.uint32)
    carry_items = np.empty(0, dtype=np.uint32)
    mapping_rows = 0
    groups_processed = 0
    previous_group: int | None = None

    def process_complete_groups(groups: np.ndarray, items: np.ndarray) -> int:
        if not len(groups):
            return 0
        starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
        source_per_group = np.add.reduceat(
            source_present[items].astype(np.uint32), starts
        )
        target_mapping_rows = np.flatnonzero(explicit_lookup[items] > 0)
        if not len(target_mapping_rows):
            return int(len(starts))
        group_index = np.searchsorted(starts, target_mapping_rows, side="right") - 1
        target_items = items[target_mapping_rows]
        target_positions = explicit_lookup[target_items].astype(np.int64) - 1
        mapped[target_positions] = True
        other_source_count = source_per_group[group_index].astype(np.int64)
        other_source_count -= source_present[target_items].astype(np.int64)
        usable = other_source_count > 0
        np.maximum.at(
            maximum_other_sources,
            target_positions[usable],
            other_source_count[usable].astype(np.uint32),
        )
        np.add.at(
            usable_relation_count,
            target_positions[usable],
            np.uint32(1),
        )
        return int(len(starts))

    for row_group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(
            row_group, columns=[group_column, "item_id"], use_threads=False
        )
        groups = table[group_column].to_numpy(zero_copy_only=False).astype(
            np.uint32, copy=False
        )
        items = table["item_id"].to_numpy(zero_copy_only=False).astype(
            np.uint32, copy=False
        )
        mapping_rows += len(groups)
        if len(groups) and previous_group is not None and int(groups[0]) < previous_group:
            raise ValueError(f"mapping is not sorted by {group_column}: {mapping_path}")
        if len(groups):
            previous_group = int(groups[-1])
        if len(carry_groups):
            groups = np.concatenate([carry_groups, groups])
            items = np.concatenate([carry_items, items])
        if not len(groups):
            continue
        last_group_start = int(np.searchsorted(groups, groups[-1], side="left"))
        groups_processed += process_complete_groups(
            groups[:last_group_start], items[:last_group_start]
        )
        carry_groups = groups[last_group_start:].copy()
        carry_items = items[last_group_start:].copy()
        del table, groups, items
    groups_processed += process_complete_groups(carry_groups, carry_items)

    report = {
        "path": str(mapping_path.resolve()),
        "group_column": group_column,
        "mapping_rows": int(mapping_rows),
        "row_groups_streamed": int(parquet.metadata.num_row_groups),
        "groups": int(groups_processed),
        "explicit_items_with_mapping": int(mapped.sum()),
        "explicit_items_with_strict_loo_support": int(
            np.count_nonzero(maximum_other_sources)
        ),
        "strict_loo_definition": (
            "at least one other source-audio item remains after removing the target "
            "from each official metadata group"
        ),
        "items_with_multiple_usable_relations": int(
            np.count_nonzero(usable_relation_count > 1)
        ),
        "maximum_usable_relations": int(usable_relation_count.max(initial=0)),
    }
    del carry_groups, carry_items, mapped
    gc.collect()
    return maximum_other_sources, usable_relation_count, report


def size_tier(count: np.ndarray, small_max: int, medium_max: int) -> np.ndarray:
    result = np.zeros(len(count), dtype=np.uint8)
    result[(count > 0) & (count <= small_max)] = 1
    result[(count > small_max) & (count <= medium_max)] = 2
    result[count > medium_max] = 3
    return result


def frequency_tier(
    counts: np.ndarray, eligible: np.ndarray, tail_q: float, head_q: float
) -> tuple[np.ndarray, int, int]:
    tail_boundary = int(np.quantile(counts[eligible], tail_q, method="higher"))
    head_boundary = int(np.quantile(counts[eligible], head_q, method="higher"))
    result = np.ones(len(counts), dtype=np.uint8)
    result[counts <= tail_boundary] = 0
    result[counts > head_boundary] = 2
    return result, tail_boundary, head_boundary


def availability_code(artist: np.ndarray, album: np.ndarray) -> np.ndarray:
    return (artist > 0).astype(np.uint8) + 2 * (album > 0).astype(np.uint8)


def allocate_category_samples(
    category_sizes: dict[int, int], total: int, minimum: int
) -> dict[int, int]:
    categories = sorted(category_sizes)
    allocation = {
        category: min(minimum, category_sizes[category]) for category in categories
    }
    remaining = total - sum(allocation.values())
    if remaining < 0:
        raise ValueError("minimum category allocation exceeds requested sample")
    capacity = {
        category: category_sizes[category] - allocation[category]
        for category in categories
    }
    while remaining > 0 and sum(capacity.values()) > 0:
        capacity_total = sum(capacity.values())
        progressed = 0
        for category in categories:
            if capacity[category] <= 0:
                continue
            proposed = max(1, int(round(remaining * capacity[category] / capacity_total)))
            take = min(proposed, capacity[category], remaining)
            allocation[category] += take
            capacity[category] -= take
            remaining -= take
            progressed += take
            if remaining == 0:
                break
        if progressed == 0:
            break
    if remaining:
        raise ValueError("not enough eligible items for requested sample")
    return allocation


def stratified_choice(
    positions: np.ndarray,
    labels: np.ndarray,
    requested: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if requested >= len(positions):
        return positions.copy()
    unique, inverse, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    raw = sizes.astype(np.float64) * (requested / len(positions))
    take = np.floor(raw).astype(np.int64)
    take = np.minimum(take, sizes)
    missing = requested - int(take.sum())
    fractional_order = np.argsort(-(raw - take), kind="stable")
    for index in fractional_order:
        if missing == 0:
            break
        if take[index] < sizes[index]:
            take[index] += 1
            missing -= 1
    if missing:
        for index in np.argsort(-sizes, kind="stable"):
            capacity = int(sizes[index] - take[index])
            extra = min(capacity, missing)
            take[index] += extra
            missing -= extra
            if missing == 0:
                break
    selected: list[np.ndarray] = []
    for stratum_index, count in enumerate(take):
        if count == 0:
            continue
        members = positions[inverse == stratum_index]
        selected.append(rng.choice(members, size=int(count), replace=False))
    result = np.concatenate(selected)
    rng.shuffle(result)
    if len(result) != requested:
        raise RuntimeError("stratified sampler did not conserve requested size")
    return result


def count_labels(values: np.ndarray, names: dict[int, str]) -> dict[str, int]:
    observed = Counter(int(value) for value in values.tolist())
    return {names[key]: observed.get(key, 0) for key in names}


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {key: Path(value) for key, value in config["paths"].items()}
    seed = int(config["seed"])
    gate = config["gate1"]

    with np.load(paths["item_frequency"], allow_pickle=False) as payload:
        item_ids = payload["item_id"].astype(np.uint32, copy=False)
        event_counts = payload["total_count"].astype(np.uint32, copy=False)
    orig2dense = np.load(paths["orig2dense"], mmap_mode="r")
    if int(item_ids.max()) >= len(orig2dense):
        raise ValueError("explicit item id lies outside orig2dense")
    dense_ids = orig2dense[item_ids].astype(np.int32, copy=False)
    real_mask = dense_ids > 0
    if len(item_ids) != 2_875_071 or int(real_mask.sum()) != 2_367_341:
        raise ValueError("frozen explicit/real-item counts changed")

    explicit_lookup = np.zeros(len(orig2dense), dtype=np.int32)
    explicit_lookup[item_ids] = np.arange(1, len(item_ids) + 1, dtype=np.int32)
    source_present = np.asarray(orig2dense > 0, dtype=bool)

    print("[1/4] Artist strict leave-one-out support", flush=True)
    artist_count, artist_relations, artist_report = metadata_loo_support(
        paths["artist_mapping"],
        "artist_id",
        explicit_lookup,
        source_present,
        len(item_ids),
    )
    print("[2/4] Album strict leave-one-out support", flush=True)
    album_count, album_relations, album_report = metadata_loo_support(
        paths["album_mapping"],
        "album_id",
        explicit_lookup,
        source_present,
        len(item_ids),
    )

    availability = availability_code(artist_count, album_count)
    eligible = real_mask & (availability > 0)
    category_sizes = {
        category: int(np.count_nonzero(real_mask & (availability == category)))
        for category in (1, 2, 3)
    }
    sample_size = int(gate["sample_size"])
    allocation = allocate_category_samples(
        category_sizes,
        sample_size,
        int(gate["minimum_availability_category_sample"]),
    )
    size_config = gate["group_size_tiers"]
    artist_tier = size_tier(
        artist_count,
        int(size_config["small_max"]),
        int(size_config["medium_max"]),
    )
    album_tier = size_tier(
        album_count,
        int(size_config["small_max"]),
        int(size_config["medium_max"]),
    )
    frequency_config = gate["frequency_tiers"]
    freq_tier, tail_boundary, head_boundary = frequency_tier(
        event_counts,
        eligible,
        float(frequency_config["tail_quantile"]),
        float(frequency_config["head_quantile"]),
    )
    joint_label = (
        freq_tier.astype(np.uint16) * 16
        + artist_tier.astype(np.uint16) * 4
        + album_tier.astype(np.uint16)
    )
    rng = np.random.default_rng(seed)
    selected_parts: list[np.ndarray] = []
    for category in (1, 2, 3):
        candidates = np.flatnonzero(real_mask & (availability == category))
        selected_parts.append(
            stratified_choice(candidates, joint_label[candidates], allocation[category], rng)
        )
    selected = np.concatenate(selected_parts)
    rng.shuffle(selected)

    # Split within each availability/group/frequency stratum. Rare one-item strata
    # are assigned by the seeded shuffle; aggregate split balance is then audited.
    split = np.zeros(len(selected), dtype=np.uint8)
    selected_labels = availability[selected].astype(np.uint32) * 256 + joint_label[selected]
    for label in np.unique(selected_labels):
        members = np.flatnonzero(selected_labels == label)
        rng.shuffle(members)
        test_count = int(round(len(members) * (1.0 - float(gate["validation_fraction"]))))
        if len(members) > 1:
            test_count = min(max(test_count, 1), len(members) - 1)
        split[members[:test_count]] = 1

    order = np.argsort(item_ids[selected], kind="stable")
    selected = selected[order]
    split = split[order]
    sample_arrays = {
        "item_id": item_ids[selected],
        "dense_id": dense_ids[selected],
        "event_count": event_counts[selected],
        "availability": availability[selected],
        "artist_loo_source_count": artist_count[selected],
        "album_loo_source_count": album_count[selected],
        "artist_usable_relation_count": artist_relations[selected],
        "album_usable_relation_count": album_relations[selected],
        "artist_size_tier": artist_tier[selected],
        "album_size_tier": album_tier[selected],
        "frequency_tier": freq_tier[selected],
        "split": split,
    }

    print("[3/4] Saving full support arrays", flush=True)
    atomic_npz(
        args.support_output,
        item_id=item_ids,
        event_count=event_counts,
        dense_id=dense_ids,
        real_embedding=real_mask,
        artist_loo_source_count=artist_count,
        album_loo_source_count=album_count,
        artist_usable_relation_count=artist_relations,
        album_usable_relation_count=album_relations,
        availability=availability,
    )
    print("[4/4] Saving stratified sample and report", flush=True)
    atomic_npz(args.sample_output, **sample_arrays)

    missing = ~real_mask
    missing_artist = missing & (artist_count > 0)
    missing_album = missing & (album_count > 0)
    missing_either = missing_artist | missing_album
    report = {
        "status": "complete_read_only_support_and_sample",
        "data_contract": {
            "target_universe": "full explicit-feedback item universe",
            "leave_one_out": True,
            "target_embedding_used_in_proxy": False,
            "event_rows_modified": False,
            "proxy_vectors_constructed": False,
            "sample_targets_have_real_audio_embedding": True,
        },
        "counts": {
            "explicit_items": int(len(item_ids)),
            "explicit_events": int(event_counts.sum(dtype=np.uint64)),
            "real_embedding_items": int(real_mask.sum()),
            "missing_embedding_items": int(missing.sum()),
        },
        "artist": artist_report,
        "album": album_report,
        "real_reconstructable": {
            AVAILABILITY_NAMES[category]: category_sizes[category]
            for category in (1, 2, 3)
        },
        "missing_item_coverage": {
            "artist_proxy": {
                "items": int(missing_artist.sum()),
                "events": int(event_counts[missing_artist].sum(dtype=np.uint64)),
            },
            "album_proxy": {
                "items": int(missing_album.sum()),
                "events": int(event_counts[missing_album].sum(dtype=np.uint64)),
            },
            "combined_rule": {
                "items": int(missing_either.sum()),
                "events": int(event_counts[missing_either].sum(dtype=np.uint64)),
            },
            "cold_unknown": {
                "items": int((missing & ~missing_either).sum()),
                "events": int(
                    event_counts[missing & ~missing_either].sum(dtype=np.uint64)
                ),
            },
        },
        "stratification": {
            "sample_requested": sample_size,
            "sample_actual": int(len(selected)),
            "availability_allocation": {
                AVAILABILITY_NAMES[key]: allocation[key] for key in allocation
            },
            "availability_observed": count_labels(
                availability[selected], AVAILABILITY_NAMES
            ),
            "artist_size_tier": count_labels(
                artist_tier[selected], SIZE_TIER_NAMES
            ),
            "album_size_tier": count_labels(album_tier[selected], SIZE_TIER_NAMES),
            "frequency_tier": count_labels(freq_tier[selected], FREQUENCY_TIER_NAMES),
            "split": count_labels(split, SPLIT_NAMES),
            "group_size_tier_definition": {
                "small": f"1..{int(size_config['small_max'])} other real items",
                "medium": (
                    f"{int(size_config['small_max']) + 1}.."
                    f"{int(size_config['medium_max'])} other real items"
                ),
                "large": f">={int(size_config['medium_max']) + 1} other real items",
            },
            "frequency_tier_definition": {
                "tail": f"event_count <= empirical P50 = {tail_boundary}",
                "mid": (
                    f"empirical P50 < event_count <= empirical P90 = {head_boundary}"
                ),
                "head": f"event_count > empirical P90 = {head_boundary}",
            },
            "split_selection": (
                "seeded 50/50 split inside availability x artist-size x album-size "
                "x frequency strata"
            ),
        },
        "outputs": {
            "support": str(args.support_output.resolve()),
            "sample": str(args.sample_output.resolve()),
        },
    }
    atomic_json(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
