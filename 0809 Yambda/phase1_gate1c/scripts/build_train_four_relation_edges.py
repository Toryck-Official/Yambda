#!/usr/bin/env python3
"""Build unique train-period user-item edges for four explicit relations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DALL = ROOT / "phase1_canonical" / "artifacts" / "dall_manifest.json"
DEFAULT_FREQ = ROOT / "phase0_5b_audit" / "artifacts" / "item_explicit_frequency.npz"
DEFAULT_OUTPUT = ROOT / "phase1_gate1c" / "artifacts" / "train_four_relation_graph"
MARKS = ("like", "dislike", "unlike", "undislike")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dall-manifest", type=Path, default=DEFAULT_DALL)
    p.add_argument("--item-frequency", type=Path, default=DEFAULT_FREQ)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    dall = json.loads(args.dall_manifest.read_text())
    with np.load(args.item_frequency, allow_pickle=False) as z:
        explicit_item = z["item_id"].astype(np.uint32, copy=False)
    max_item = int(explicit_item.max())
    lookup = np.full(max_item + 1, -1, dtype=np.int32)
    lookup[explicit_item] = np.arange(len(explicit_item), dtype=np.int32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    unique_relation_counts = np.zeros((4, len(explicit_item)), dtype=np.uint32)
    event_relation_counts = np.zeros((4, len(explicit_item)), dtype=np.uint32)
    edge_reports = {}
    edge_paths = []
    for relation, mark in enumerate(MARKS):
        print(f"[{relation + 1}/4] {mark} train edges", flush=True)
        view = dall["split_views"][mark]
        uid = np.load(view["uid"], mmap_mode="r")
        offsets = np.load(view["offsets"], mmap_mode="r")
        train_end = np.load(view["train_end"], mmap_mode="r")
        items = np.load(view["shared_item_id"], mmap_mode="r")
        mark_dir = args.output_dir / mark
        mark_dir.mkdir(parents=True, exist_ok=True)
        uid_path = mark_dir / "uid.uint32.bin"
        item_path = mark_dir / "item_position.uint32.bin"
        edge_count = 0
        train_event_count = 0
        users_with_edge = 0
        with uid_path.open("wb") as uid_handle, item_path.open("wb") as item_handle:
            for row in range(len(uid)):
                start, end = int(offsets[row]), int(train_end[row])
                if end <= start:
                    continue
                raw_items = np.asarray(items[start:end], dtype=np.uint32)
                positions = lookup[raw_items]
                if np.any(positions < 0):
                    raise RuntimeError(f"{mark} item outside explicit universe")
                np.add.at(event_relation_counts[relation], positions, np.uint32(1))
                unique_positions = np.unique(positions).astype(np.uint32)
                edge_uids = np.full(len(unique_positions), int(uid[row]), dtype=np.uint32)
                edge_uids.tofile(uid_handle)
                unique_positions.tofile(item_handle)
                np.add.at(unique_relation_counts[relation], unique_positions, np.uint32(1))
                edge_count += len(unique_positions)
                train_event_count += end - start
                users_with_edge += 1
                if (row + 1) % 100_000 == 0:
                    print(f"{mark}: {row + 1:,}/{len(uid):,} users", flush=True)
        if train_event_count != int(dall["split_stats"]["train"]["feedback_counts"][mark]):
            raise RuntimeError(f"{mark} train event count changed")
        edge_paths.append((uid_path, item_path, edge_count))
        edge_reports[mark] = {
            "relation_id": relation,
            "train_events_after_exact_dedup": int(train_event_count),
            "unique_user_item_relation_edges": int(edge_count),
            "users_with_relation_edge": int(users_with_edge),
            "uid_file": str(uid_path.resolve()),
            "item_position_file": str(item_path.resolve()),
            "edge_semantics": "one edge per observed train-period user,item,feedback triple; repeated times are aggregated",
        }
    np.save(args.output_dir / "unique_user_count_by_relation.uint32.npy", unique_relation_counts, allow_pickle=False)
    np.save(args.output_dir / "event_count_by_relation.uint32.npy", event_relation_counts, allow_pickle=False)
    np.save(args.output_dir / "explicit_item_id.uint32.npy", explicit_item, allow_pickle=False)

    # Exact union over the four relation edge sets to get per-item distinct train
    # users. Relation edges remain separate for model training.
    total_edges = sum(count for _, _, count in edge_paths)
    print(f"[union] sorting {total_edges:,} relation edges by user-item key", flush=True)
    union_keys = np.empty(total_edges, dtype=np.uint64)
    cursor = 0
    for uid_path, item_path, count in edge_paths:
        uids = np.memmap(uid_path, mode="r", dtype=np.uint32, shape=(count,))
        positions = np.memmap(item_path, mode="r", dtype=np.uint32, shape=(count,))
        union_keys[cursor : cursor + count] = (
            uids.astype(np.uint64) << np.uint64(32)
        ) | positions.astype(np.uint64)
        cursor += count
    union_keys.sort()
    first = np.ones(len(union_keys), dtype=bool)
    first[1:] = union_keys[1:] != union_keys[:-1]
    unique_keys = union_keys[first]
    unique_positions = (unique_keys & np.uint64(0xFFFFFFFF)).astype(np.int64)
    unique_user_count = np.bincount(unique_positions, minlength=len(explicit_item)).astype(np.uint32)
    np.save(args.output_dir / "unique_train_user_count.uint32.npy", unique_user_count, allow_pickle=False)
    train_evidence = unique_user_count > 0
    event_total = event_relation_counts.sum(axis=0, dtype=np.uint64)
    revision_event = event_relation_counts[2].astype(np.uint64) + event_relation_counts[3].astype(np.uint64)
    revision_share = np.zeros(len(explicit_item), dtype=np.float32)
    revision_share[train_evidence] = revision_event[train_evidence] / event_total[train_evidence]
    np.save(args.output_dir / "revision_event_share.float32.npy", revision_share, allow_pickle=False)
    report = {
        "status": "complete_train_only_four_relation_graph",
        "data_contract": {
            "source": str(args.dall_manifest),
            "train_cutoff_timestamp_inclusive": dall["global_split"]["train_cutoff_timestamp_inclusive"],
            "validation_or_test_edges_used": False,
            "listen_used": False,
            "relations_preserved": list(MARKS),
            "relations_collapsed_to_positive": False,
            "audio_embedding_used": False,
        },
        "counts": {
            "explicit_items": int(len(explicit_item)),
            "relation_edges": int(total_edges),
            "unique_user_item_pairs_across_relations": int(len(unique_keys)),
            "items_with_train_evidence": int(train_evidence.sum()),
            "items_without_train_evidence": int((~train_evidence).sum()),
        },
        "relations": edge_reports,
        "outputs": {
            "explicit_item_id": str((args.output_dir / "explicit_item_id.uint32.npy").resolve()),
            "unique_user_count_by_relation": str((args.output_dir / "unique_user_count_by_relation.uint32.npy").resolve()),
            "event_count_by_relation": str((args.output_dir / "event_count_by_relation.uint32.npy").resolve()),
            "unique_train_user_count": str((args.output_dir / "unique_train_user_count.uint32.npy").resolve()),
            "revision_event_share": str((args.output_dir / "revision_event_share.float32.npy").resolve()),
        },
    }
    atomic_json(args.output_dir / "manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
