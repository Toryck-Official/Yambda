#!/usr/bin/env python3
"""Materialize the frozen real-audio SID catalog and D_SID event dataset.

The event dataset is partitioned by feedback type.  There is deliberately no
unified within-group row order; `group_id` is the only group membership key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]
DALL = ROOT / "phase1_canonical" / "artifacts" / "dall_manifest.json"
SUPPORT = ROOT / "phase1_gates" / "artifacts" / "metadata_loo_support.npz"
CODEBOOK = ROOT / "sid_feasibility_supplement" / "artifacts" / "B0_audio_only" / "codebooks.npy"
CODES = ROOT / "sid_feasibility_supplement" / "artifacts" / "B0_audio_only" / "codes.uint8.npy"
DENSE = Path("/root/autodl-tmp/0626/0626 Predictor/01_data/processed/raw_rqkmeans/dense_item_features.npy")
OUT_ROOT = ROOT / "phase2_gate2_sid"
FINAL = OUT_ROOT / "materialized_v1_1"
STAGING = OUT_ROOT / ".materialized_v1_1.staging"
MANIFEST = OUT_ROOT / "gate2_manifest.json"
MARKS = ("like", "dislike", "unlike", "undislike")
EXPECTED = {
    "events": 121_819_651,
    "users": 854_649,
    "items": 2_367_341,
    "groups": 77_565_283,
    "feedback": (85_325_332, 11_061_046, 23_667_107, 1_766_166),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dall", type=Path, default=DALL)
    p.add_argument("--support", type=Path, default=SUPPORT)
    p.add_argument("--codebook", type=Path, default=CODEBOOK)
    p.add_argument("--codes", type=Path, default=CODES)
    p.add_argument("--dense", type=Path, default=DENSE)
    p.add_argument("--final", type=Path, default=FINAL)
    p.add_argument("--staging", type=Path, default=STAGING)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--buffer-events", type=int, default=500_000)
    p.add_argument("--catalog-chunk", type=int, default=100_000)
    p.add_argument("--progress-users", type=int, default=25_000)
    return p.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def sha_array_bytes(array: np.ndarray, chunk: int = 1_000_000) -> str:
    h = hashlib.sha256()
    for start in range(0, len(array), chunk):
        h.update(np.ascontiguousarray(array[start:start + chunk]).tobytes())
    return h.hexdigest()


EVENT_SCHEMA = pa.schema([
    ("uid", pa.uint32()),
    ("timestamp", pa.uint32()),
    ("group_id", pa.uint64()),
    ("feedback_type", pa.uint8()),
    ("item_id", pa.uint32()),
    ("sid_1", pa.uint8()),
    ("sid_2", pa.uint8()),
    ("sid_3", pa.uint8()),
    ("sid_4", pa.uint8()),
    ("split", pa.uint8()),
])

CATALOG_SCHEMA = pa.schema([
    ("item_id", pa.uint32()),
    ("sid_1", pa.uint8()),
    ("sid_2", pa.uint8()),
    ("sid_3", pa.uint8()),
    ("sid_4", pa.uint8()),
    ("audio_embedding", pa.list_(pa.float32(), 128)),
])


class EventBuffer:
    def __init__(self, path: Path, threshold: int) -> None:
        self.path = path
        self.threshold = threshold
        self.parts: dict[str, list[np.ndarray]] = {name: [] for name in EVENT_SCHEMA.names}
        self.rows = 0
        self.total = 0
        self.writer = pq.ParquetWriter(
            path,
            EVENT_SCHEMA,
            compression="zstd",
            compression_level=3,
            use_dictionary=["feedback_type", "split"],
            write_statistics=True,
        )

    def add(self, **values: np.ndarray) -> None:
        n = len(values["uid"])
        if not n:
            return
        for name in EVENT_SCHEMA.names:
            self.parts[name].append(values[name])
        self.rows += n
        if self.rows >= self.threshold:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        arrays = [pa.array(np.concatenate(self.parts[name]), type=EVENT_SCHEMA.field(name).type) for name in EVENT_SCHEMA.names]
        self.writer.write_table(pa.Table.from_arrays(arrays, schema=EVENT_SCHEMA), row_group_size=self.threshold)
        self.total += self.rows
        self.parts = {name: [] for name in EVENT_SCHEMA.names}
        self.rows = 0

    def close(self) -> None:
        self.flush(); self.writer.close()


def prepare_streams(manifest: dict) -> list[dict]:
    streams = []
    for mark in MARKS:
        view = manifest["split_views"][mark]
        streams.append({
            "uid": np.load(view["uid"], mmap_mode="r"),
            "offsets": np.load(view["offsets"], mmap_mode="r"),
            "timestamp": np.load(view["shared_timestamp"], mmap_mode="r"),
            "item": np.load(view["shared_item_id"], mmap_mode="r"),
            "train_end": np.load(view["train_end"], mmap_mode="r"),
            "validation_end": np.load(view["validation_end"], mmap_mode="r"),
            "row": 0,
        })
    return streams


def write_catalog(path: Path, item_ids: np.ndarray, dense_ids: np.ndarray, codes: np.ndarray,
                  dense: np.ndarray, chunk_size: int) -> tuple[str, dict]:
    writer = pq.ParquetWriter(path, CATALOG_SCHEMA, compression="zstd", compression_level=3, write_statistics=True)
    selected_embedding_hash = hashlib.sha256()
    norm_min = float("inf"); norm_max = 0.0; invalid = 0
    for start in range(0, len(item_ids), chunk_size):
        end = min(start + chunk_size, len(item_ids))
        embedding = np.ascontiguousarray(dense[dense_ids[start:end]], dtype=np.float32)
        selected_embedding_hash.update(embedding.tobytes())
        norms = np.linalg.norm(embedding, axis=1)
        norm_min = min(norm_min, float(norms.min())); norm_max = max(norm_max, float(norms.max()))
        invalid += int((~np.isfinite(embedding).all(axis=1)).sum())
        fixed = pa.FixedSizeListArray.from_arrays(pa.array(embedding.reshape(-1), type=pa.float32()), 128)
        table = pa.Table.from_arrays([
            pa.array(item_ids[start:end], type=pa.uint32()),
            *[pa.array(codes[start:end, level], type=pa.uint8()) for level in range(4)],
            fixed,
        ], schema=CATALOG_SCHEMA)
        writer.write_table(table, row_group_size=chunk_size)
        if end % 500_000 < chunk_size or end == len(item_ids):
            print(f"catalog {end:,}/{len(item_ids):,}", flush=True)
    writer.close()
    return selected_embedding_hash.hexdigest(), {
        "rows": len(item_ids), "invalid_embeddings": invalid,
        "norm_min": norm_min, "norm_max": norm_max,
    }


def validate_outputs(event_paths: list[Path], expected_uid: np.ndarray, expected_ts: np.ndarray,
                     expected_size: np.ndarray, item_to_pos: np.ndarray, codes: np.ndarray,
                     train_cutoff: int, validation_cutoff: int) -> dict:
    group_count = np.zeros(len(expected_size), dtype=np.uint32)
    user_seen = np.zeros(1_000_001, dtype=bool)
    item_seen = np.zeros(len(item_to_pos), dtype=bool)
    feedback = np.zeros(4, dtype=np.uint64)
    total = mapping_mismatch = membership_mismatch = split_mismatch = 0
    file_reports = {}
    for mark, path in enumerate(event_paths):
        parquet = pq.ParquetFile(path)
        rows = 0
        for batch in parquet.iter_batches(batch_size=1_000_000, use_threads=True):
            columns = {name: batch.column(i).to_numpy(zero_copy_only=False) for i, name in enumerate(EVENT_SCHEMA.names)}
            uid = columns["uid"].astype(np.uint32, copy=False)
            ts = columns["timestamp"].astype(np.uint32, copy=False)
            gid = columns["group_id"].astype(np.uint64, copy=False)
            item = columns["item_id"].astype(np.uint32, copy=False)
            fb = columns["feedback_type"].astype(np.uint8, copy=False)
            split = columns["split"].astype(np.uint8, copy=False)
            if len(gid) and int(gid.max()) >= len(expected_size):
                raise RuntimeError("group_id outside frozen range")
            membership_mismatch += int(np.count_nonzero((expected_uid[gid] != uid) | (expected_ts[gid] != ts)))
            computed_split = np.where(ts <= train_cutoff, 0, np.where(ts <= validation_cutoff, 1, 2)).astype(np.uint8)
            split_mismatch += int(np.count_nonzero(split != computed_split))
            pos = item_to_pos[item]
            if np.any(pos < 0):
                mapping_mismatch += int(np.count_nonzero(pos < 0))
            valid = pos >= 0
            observed_codes = np.stack([columns[f"sid_{level}"] for level in range(1, 5)], axis=1).astype(np.uint8, copy=False)
            mapping_mismatch += int(np.count_nonzero(np.any(observed_codes[valid] != codes[pos[valid]], axis=1)))
            if np.any(fb != mark):
                raise RuntimeError(f"feedback partition mismatch in {path}")
            unique_gid, counts = np.unique(gid, return_counts=True)
            group_count[unique_gid] += counts.astype(np.uint32)
            user_seen[uid] = True; item_seen[item] = True
            feedback += np.bincount(fb, minlength=4).astype(np.uint64)
            rows += len(uid); total += len(uid)
        file_reports[MARKS[mark]] = {"rows": rows, "sha256": sha256(path), "bytes": path.stat().st_size}
    group_size_mismatch = int(np.count_nonzero(group_count != expected_size))
    return {
        "events": total,
        "users": int(user_seen.sum()),
        "items": int(item_seen.sum()),
        "feedback_counts": feedback.tolist(),
        "mapping_mismatch_events": mapping_mismatch,
        "group_membership_mismatch_events": membership_mismatch,
        "split_mismatch_events": split_mismatch,
        "group_size_mismatch_groups": group_size_mismatch,
        "group_count": len(expected_size),
        "files": file_reports,
    }


def main() -> None:
    args = parse_args(); started = time.time()
    if args.final.exists() or args.staging.exists():
        raise FileExistsError(f"refusing to overwrite {args.final} or {args.staging}")
    free = shutil.disk_usage(args.final.parent).free
    if free < 6 * 1024**3:
        raise RuntimeError(f"Gate 2 requires at least 6 GiB free, found {free / 1024**3:.2f} GiB")
    args.staging.mkdir(parents=True)
    (args.staging / "events").mkdir(); (args.staging / "frozen_sid").mkdir()
    manifest = json.loads(args.dall.read_text())
    train_cutoff = int(manifest["global_split"]["train_cutoff_timestamp_inclusive"])
    validation_cutoff = int(manifest["global_split"]["validation_cutoff_timestamp_inclusive"])

    with np.load(args.support, allow_pickle=False) as z:
        real = z["real_embedding"]
        item_ids = np.asarray(z["item_id"][real], dtype=np.uint32)
        dense_ids = np.asarray(z["dense_id"][real], dtype=np.int64)
    codebooks = np.load(args.codebook, mmap_mode="r")
    codes = np.load(args.codes, mmap_mode="r")
    dense = np.load(args.dense, mmap_mode="r")
    if (len(item_ids), codes.shape, codebooks.shape, dense.shape[1]) != (EXPECTED["items"], (EXPECTED["items"], 4), (4, 256, 128), 128):
        raise RuntimeError("frozen input shape mismatch")
    if not np.all(item_ids[1:] > item_ids[:-1]) or int(codes.min()) < 0 or int(codes.max()) > 255:
        raise RuntimeError("item universe or SID range invalid")
    frozen_tensor = torch.from_numpy(np.array(codebooks, copy=True))
    if frozen_tensor.requires_grad:
        raise RuntimeError("codebook unexpectedly trainable")

    frozen_codebook = args.staging / "frozen_sid" / "codebooks.npy"
    frozen_codes = args.staging / "frozen_sid" / "codes.uint8.npy"
    frozen_items = args.staging / "frozen_sid" / "item_ids.uint32.npy"
    frozen_dense_ids = args.staging / "frozen_sid" / "dense_ids.int64.npy"
    shutil.copy2(args.codebook, frozen_codebook); shutil.copy2(args.codes, frozen_codes)
    np.save(frozen_items, item_ids, allow_pickle=False); np.save(frozen_dense_ids, dense_ids, allow_pickle=False)

    print("[1/4] item catalog", flush=True)
    catalog_path = args.staging / "item_catalog.parquet"
    selected_embedding_sha, catalog_stats = write_catalog(
        catalog_path, item_ids, dense_ids, codes, dense, args.catalog_chunk
    )
    item_to_pos = np.full(int(item_ids.max()) + 1, -1, dtype=np.int32)
    item_to_pos[item_ids] = np.arange(len(item_ids), dtype=np.int32)

    print("[2/4] event dataset", flush=True)
    event_paths = [args.staging / "events" / f"feedback_type={mark}.parquet" for mark in MARKS]
    writers = [EventBuffer(path, args.buffer_events) for path in event_paths]
    streams = prepare_streams(manifest)
    expected_uid = np.empty(EXPECTED["groups"], dtype=np.uint32)
    expected_ts = np.empty(EXPECTED["groups"], dtype=np.uint32)
    expected_size = np.empty(EXPECTED["groups"], dtype=np.uint32)
    group_cursor = users = events = split_position_mismatch = 0
    feedback_counts = np.zeros(4, np.uint64)
    user_seen = np.zeros(1_000_001, bool); item_seen = np.zeros(len(item_to_pos), bool)
    while True:
        active = [s for s in streams if s["row"] < len(s["uid"])]
        if not active:
            break
        uid = min(int(s["uid"][s["row"]]) for s in active)
        parts: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        all_times = []
        for mark, stream in enumerate(streams):
            row = stream["row"]
            if row >= len(stream["uid"]) or int(stream["uid"][row]) != uid:
                continue
            a, b = int(stream["offsets"][row]), int(stream["offsets"][row + 1])
            ts_all = np.asarray(stream["timestamp"][a:b], dtype=np.uint32)
            it_all = np.asarray(stream["item"][a:b], dtype=np.uint32)
            keep = item_to_pos[it_all] >= 0
            if np.any(keep):
                ts = ts_all[keep]; it = it_all[keep]
                global_pos = np.arange(a, b, dtype=np.uint64)[keep]
                expected_split = np.where(global_pos < int(stream["train_end"][row]), 0,
                                  np.where(global_pos < int(stream["validation_end"][row]), 1, 2)).astype(np.uint8)
                rule_split = np.where(ts <= train_cutoff, 0, np.where(ts <= validation_cutoff, 1, 2)).astype(np.uint8)
                split_position_mismatch += int(np.count_nonzero(expected_split != rule_split))
                parts.append((mark, ts, it, rule_split)); all_times.append(ts)
            stream["row"] += 1
        if not parts:
            continue
        unique_time, size = np.unique(np.concatenate(all_times), return_counts=True)
        n_group = len(unique_time); end_group = group_cursor + n_group
        if end_group > EXPECTED["groups"]:
            raise RuntimeError("group count exceeds frozen target")
        expected_uid[group_cursor:end_group] = uid
        expected_ts[group_cursor:end_group] = unique_time
        expected_size[group_cursor:end_group] = size.astype(np.uint32)
        for mark, ts, it, split in parts:
            pos = item_to_pos[it]; sid = codes[pos]
            gid = group_cursor + np.searchsorted(unique_time, ts)
            n = len(it)
            writers[mark].add(
                uid=np.full(n, uid, np.uint32), timestamp=ts,
                group_id=gid.astype(np.uint64), feedback_type=np.full(n, mark, np.uint8),
                item_id=it, sid_1=sid[:, 0], sid_2=sid[:, 1], sid_3=sid[:, 2], sid_4=sid[:, 3], split=split,
            )
            feedback_counts[mark] += n; events += n; item_seen[it] = True
        user_seen[uid] = True; users += 1; group_cursor = end_group
        if users % args.progress_users == 0:
            progress = {"status": "materializing_events", "users": users, "events": events, "groups": group_cursor, "elapsed_seconds": time.time() - started}
            atomic_json(OUT_ROOT / "gate2_progress.json", progress); print(progress, flush=True)
    for writer in writers:
        writer.close()

    before = {
        "events": events, "users": int(user_seen.sum()), "items": int(item_seen.sum()),
        "groups": group_cursor, "feedback_counts": feedback_counts.tolist(),
        "split_position_mismatch_events": split_position_mismatch,
    }
    print("[3/4] independent output validation", flush=True)
    after = validate_outputs(event_paths, expected_uid, expected_ts, expected_size, item_to_pos, codes, train_cutoff, validation_cutoff)
    catalog_meta = pq.ParquetFile(catalog_path).metadata
    checks = {
        "item_catalog_rows": int(catalog_meta.num_rows) == EXPECTED["items"],
        "one_sid_per_D_SID_item": len(item_ids) == len(np.unique(item_ids)) == EXPECTED["items"],
        "sid_range_0_255": int(codes.min()) >= 0 and int(codes.max()) <= 255,
        "event_count_conserved": before["events"] == after["events"] == EXPECTED["events"],
        "feedback_counts_conserved": tuple(before["feedback_counts"]) == tuple(after["feedback_counts"]) == EXPECTED["feedback"],
        "users_conserved": before["users"] == after["users"] == EXPECTED["users"],
        "items_conserved": before["items"] == after["items"] == EXPECTED["items"],
        "global_cutoffs_unchanged": (train_cutoff, validation_cutoff) == (22_172_245, 24_154_245),
        "split_assignment_unchanged": split_position_mismatch == after["split_mismatch_events"] == 0,
        "group_membership_size_count_unchanged": group_cursor == after["group_count"] == EXPECTED["groups"] and after["group_membership_mismatch_events"] == 0 and after["group_size_mismatch_groups"] == 0,
        "no_within_group_order_created": True,
        "codebook_requires_grad_false": not frozen_tensor.requires_grad,
        "full_item_to_sid_mapping_matches_frozen_codes": after["mapping_mismatch_events"] == 0,
    }
    if not all(checks.values()):
        failure = {"status": "FAILED_STOP_BEFORE_TIMESTAMP_GATE", "checks": checks, "before": before, "after": after}
        atomic_json(args.manifest, failure); raise RuntimeError(f"Gate 2 conservation failed: {checks}")

    print("[4/4] hashes and freeze", flush=True)
    hashes = {
        "source_codebook_sha256": sha256(args.codebook),
        "frozen_codebook_sha256": sha256(frozen_codebook),
        "source_codes_sha256": sha256(args.codes),
        "frozen_codes_sha256": sha256(frozen_codes),
        "source_dense_embedding_file_sha256": sha256(args.dense),
        "selected_explicit_real_embedding_rows_sha256": selected_embedding_sha,
        "item_universe_values_sha256": sha_array_bytes(item_ids),
        "item_universe_npy_sha256": sha256(frozen_items),
        "dense_ids_npy_sha256": sha256(frozen_dense_ids),
        "item_catalog_sha256": sha256(catalog_path),
        "item_sid_pairs_sha256": sha_array_bytes(np.column_stack((item_ids, codes.astype(np.uint32)))),
    }
    if hashes["source_codebook_sha256"] != hashes["frozen_codebook_sha256"] or hashes["source_codes_sha256"] != hashes["frozen_codes_sha256"]:
        raise RuntimeError("frozen copy hash mismatch")
    report = {
        "status": "complete_gate2_materialized_all_checks_passed",
        "protocol_version": "v1.1",
        "method": "full_fit_audio_only_RQKMeans",
        "fit_contract": {"items": EXPECTED["items"], "levels": 4, "codes_per_level": 256, "input_dimension": 128, "normalized_input": True, "seed": 2026, "iterations_per_level": 25},
        "data_contract": {"dataset": "D_SID real-audio events from exact-deduplicated D_all", "listen_used": False, "missing_audio_used": False, "event_table_partitioning": "four homogeneous feedback parquet files; group_id defines membership; row order has no chronology semantics", "embedding_repeated_in_event_table": False},
        "cutoffs": {"train_inclusive": train_cutoff, "validation_inclusive": validation_cutoff},
        "before_join": before, "after_materialized_validation": after,
        "catalog": {**catalog_stats, "schema": str(CATALOG_SCHEMA), "path": str((args.final / "item_catalog.parquet").resolve())},
        "checks": checks, "all_checks_passed": True,
        "freeze": {"requires_grad": False, "file_mode": "0444", "codebook_path": str((args.final / "frozen_sid" / "codebooks.npy").resolve()), "codes_path": str((args.final / "frozen_sid" / "codes.uint8.npy").resolve())},
        "hashes": hashes,
        "elapsed_seconds": time.time() - started,
        "boundaries": {"minimal_snmpp_training_started": False, "hierarchical_sid_head_added": False, "timestamp_gate_started": False},
    }
    # Final path references are fixed before the atomic directory rename.
    atomic_json(args.staging / "manifest.json", report)
    for path in (frozen_codebook, frozen_codes, frozen_items, frozen_dense_ids):
        os.chmod(path, 0o444)
    args.staging.replace(args.final)
    atomic_json(args.manifest, report)
    atomic_json(OUT_ROOT / "gate2_progress.json", {"status": "complete", "events": events, "groups": group_cursor, "elapsed_seconds": time.time() - started})
    print(json.dumps({"status": report["status"], "checks": checks, "final": str(args.final), "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
