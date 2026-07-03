#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build predictor-specific chronological train/val/test splits from existing "
            "Regret step-level transitions. This does not rebuild sessions or steps."
        )
    )
    parser.add_argument("--transition_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--source_splits", default="train,val,test")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--shard_rows", type=int, default=50000)
    parser.add_argument("--max_users", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4096)
    return parser.parse_args()


def parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.parquet"))


def iter_rows(files: list[Path], batch_size: int) -> Iterable[dict]:
    for file_path in files:
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield row


def sort_user_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("user_step_idx", row.get("episode_start_time", 0)) or 0),
            int(row.get("step_time", row.get("episode_end_time", 0)) or 0),
            int(row.get("transition_id", 0) or 0),
        ),
    )


def assign_seq_splits(n_rows: int, train_frac: float, val_frac: float) -> list[str]:
    if n_rows <= 0:
        return []
    if n_rows == 1:
        return ["train"]
    if n_rows == 2:
        return ["train", "test"]
    train_end = int(n_rows * float(train_frac))
    val_end = int(n_rows * (float(train_frac) + float(val_frac)))
    train_end = min(max(train_end, 1), n_rows - 2)
    val_end = min(max(val_end, train_end + 1), n_rows - 1)
    return ["train"] * train_end + ["val"] * (val_end - train_end) + ["test"] * (n_rows - val_end)


class ShardedWriter:
    def __init__(self, out_root: Path, shard_rows: int) -> None:
        self.out_root = out_root
        self.shard_rows = int(shard_rows)
        self.buffers: dict[str, list[dict]] = defaultdict(list)
        self.shard_idx: dict[str, int] = defaultdict(int)
        for split in ["train", "val", "test"]:
            split_dir = self.out_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for old_file in split_dir.glob("*.parquet"):
                old_file.unlink()

    def add(self, split: str, row: dict) -> None:
        out = dict(row)
        out["split"] = split
        buf = self.buffers[split]
        buf.append(out)
        if len(buf) >= self.shard_rows:
            self.flush(split)

    def flush(self, split: str) -> None:
        buf = self.buffers[split]
        if not buf:
            return
        out_path = self.out_root / split / f"part-{self.shard_idx[split]:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(buf), out_path)
        self.buffers[split] = []
        self.shard_idx[split] += 1

    def close(self) -> None:
        for split in list(self.buffers):
            self.flush(split)


def load_small_split_by_user(root: Path, split: str, batch_size: int) -> dict[int, list[dict]]:
    files = parquet_files(root / split)
    out: dict[int, list[dict]] = defaultdict(list)
    for row in tqdm(iter_rows(files, batch_size), desc=f"[load {split}]", unit="row"):
        out[int(row.get("user_id"))].append(row)
    return dict(out)


def main() -> None:
    args = parse_args()
    transition_root = Path(args.transition_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    source_splits = [item.strip() for item in args.source_splits.split(",") if item.strip()]
    if "train" not in source_splits:
        raise ValueError("source_splits must include train so user streams can be read chronologically.")

    side_rows: dict[str, dict[int, list[dict]]] = {}
    for split in source_splits:
        if split == "train":
            continue
        side_rows[split] = load_small_split_by_user(transition_root, split, args.batch_size)

    writer = ShardedWriter(out_root, args.shard_rows)
    train_files = parquet_files(transition_root / "train")
    if not train_files:
        raise FileNotFoundError(f"No train parquet files found under {transition_root / 'train'}")

    counts = Counter()
    users = 0
    current_user = None
    current_rows: list[dict] = []

    def emit_user(user_id: int, train_rows: list[dict]) -> bool:
        nonlocal users
        if user_id is None:
            return True
        rows = list(train_rows)
        for split_map in side_rows.values():
            rows.extend(split_map.get(int(user_id), []))
        rows = sort_user_rows(rows)
        plan = assign_seq_splits(len(rows), args.train_frac, args.val_frac)
        for split, row in zip(plan, rows):
            writer.add(split, row)
            counts[split] += 1
        counts["source_rows"] += len(rows)
        users += 1
        if args.max_users and users >= args.max_users:
            return False
        return True

    pbar = tqdm(iter_rows(train_files, args.batch_size), desc="[stream train users]", unit="row")
    keep_running = True
    for row in pbar:
        user_id = int(row.get("user_id"))
        if current_user is None:
            current_user = user_id
        if user_id != current_user:
            keep_running = emit_user(current_user, current_rows)
            current_rows = []
            current_user = user_id
            pbar.set_postfix(users=users, train=counts["train"], val=counts["val"], test=counts["test"], refresh=False)
            if not keep_running:
                break
        current_rows.append(row)
    if keep_running and current_rows:
        emit_user(current_user, current_rows)
    writer.close()

    meta = {
        "transition_root": str(transition_root),
        "out_root": str(out_root),
        "source_splits": source_splits,
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "test_frac": float(1.0 - args.train_frac - args.val_frac),
        "users_written": int(users),
        "counts": {key: int(value) for key, value in counts.items()},
        "note": "Predictor-only chronological split over already-built session/consecutive-item steps.",
    }
    (out_root / "predictor_seq_split.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
