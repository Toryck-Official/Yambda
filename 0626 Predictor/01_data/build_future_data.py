from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
RESPONSE_NAMES = ("listen", "like", "dislike", "unlike", "undislike")
REGRET_TO_ID = {"none": 0, "low_play": 1, "dislike": 2, "unlike": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt session-run transitions for the future predictor.")
    parser.add_argument("--transition_root", required=True)
    parser.add_argument("--out_dir", default=str(ROOT / "artifacts" / "future_data"))
    parser.add_argument("--dense2orig_npy", required=True)
    parser.add_argument("--dense_item2sid_npy", default="")
    parser.add_argument("--history_len", type=int, default=50)
    parser.add_argument("--future_horizon", type=int, default=5)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--shard_rows", type=int, default=50000)
    parser.add_argument("--write_needed_items", action="store_true")
    return parser.parse_args()


def pad_left(values: list, length: int, pad_value):
    selected = list(values)[-length:]
    return [pad_value] * (length - len(selected)) + selected


def list_split_files(root: Path, split: str) -> list[Path]:
    return sorted((root / split).glob("*.parquet"))


def dense_to_orig(dense_ids: list[int], dense2orig: np.ndarray) -> list[int]:
    result = []
    for dense_id in dense_ids:
        idx = int(dense_id)
        result.append(int(dense2orig[idx]) if 0 < idx < len(dense2orig) else 0)
    return result


def lookup_sid(dense_id: int, dense_item2sid: np.ndarray | None) -> list[int]:
    if dense_item2sid is None or dense_id <= 0 or dense_id >= dense_item2sid.shape[0]:
        return []
    return [int(value) for value in dense_item2sid[dense_id].tolist()]


def response_targets(row: dict) -> list[float]:
    targets = row.get("response_targets")
    if targets is not None:
        return [float(value) for value in targets]
    return [
        float(int(row.get("n_listen", 0)) > 0),
        float(row.get("has_like", 0)),
        float(row.get("has_dislike", 0)),
        float(row.get("has_unlike", 0)),
        float(row.get("has_undislike", 0)),
    ]


def history_field(row: dict, name: str, length: int, pad_value):
    return pad_left(row.get(name) or [], length, pad_value)


def adapt_row(
    row: dict,
    split: str,
    args: argparse.Namespace,
    dense2orig: np.ndarray,
    dense_item2sid: np.ndarray | None,
) -> dict:
    history_dense = [int(value) for value in row.get("history_item_ids", [])]
    next_history_dense = [int(value) for value in row.get("next_history_item_ids", [])]
    history_orig = dense_to_orig(history_dense, dense2orig)
    next_history_orig = dense_to_orig(next_history_dense, dense2orig)
    target_dense = int(row["target_dense_item_id"])
    target_orig = int(row.get("target_orig_item_id") or 0)
    if target_orig <= 0 and 0 < target_dense < len(dense2orig):
        target_orig = int(dense2orig[target_dense])
    regret_type = str(row.get("regret_type", "none"))
    targets = response_targets(row)

    return {
        "transition_id": int(row.get("transition_id", 0)),
        "user_id": int(row["user_id"]),
        "session_id": int(row.get("session_id", 0)),
        "session_step_idx": int(row.get("session_step_idx", 0)),
        "user_step_idx": int(row.get("user_step_idx", 0)),
        "split": split,
        "history_item_ids": pad_left(history_orig, args.history_len, 0),
        "history_dense_item_ids": pad_left(history_dense, args.history_len, 0),
        "history_feedbacks": history_field(row, "history_feedbacks", args.history_len, 0.0),
        "history_event_type_ids": history_field(row, "history_event_type_ids", args.history_len, 0),
        "history_response_targets": history_field(
            row, "history_response_targets", args.history_len, [0.0] * len(RESPONSE_NAMES)
        ),
        "history_play_ratios": history_field(row, "history_play_ratios", args.history_len, 0.0),
        "history_play_excesses": history_field(row, "history_play_excesses", args.history_len, 0.0),
        "history_is_organic": history_field(row, "history_is_organic", args.history_len, 0),
        "history_time_gap_seconds": history_field(row, "history_time_gap_seconds", args.history_len, 0.0),
        "history_same_session": history_field(row, "history_same_session", args.history_len, 0),
        "next_history_item_ids": pad_left(next_history_orig, args.history_len, 0),
        "next_history_dense_item_ids": pad_left(next_history_dense, args.history_len, 0),
        "next_history_feedbacks": history_field(row, "next_history_feedbacks", args.history_len, 0.0),
        "next_history_event_type_ids": history_field(row, "next_history_event_type_ids", args.history_len, 0),
        "next_history_response_targets": history_field(
            row, "next_history_response_targets", args.history_len, [0.0] * len(RESPONSE_NAMES)
        ),
        "next_history_play_ratios": history_field(row, "next_history_play_ratios", args.history_len, 0.0),
        "next_history_play_excesses": history_field(row, "next_history_play_excesses", args.history_len, 0.0),
        "next_history_is_organic": history_field(row, "next_history_is_organic", args.history_len, 0),
        "next_history_time_gap_seconds": history_field(
            row, "next_history_time_gap_seconds", args.history_len, 0.0
        ),
        "next_history_same_session": history_field(row, "next_history_same_session", args.history_len, 0),
        "target_item_id": target_orig,
        "target_dense_item_id": target_dense,
        "target_sid": lookup_sid(target_dense, dense_item2sid),
        "response_targets": targets,
        "played_ratio": float(row.get("played_ratio", row.get("max_play_ratio", 0.0))),
        "played_ratio_clipped": float(
            row.get("played_ratio_clipped", np.clip(row.get("max_play_ratio", 0.0), 0.0, 1.0))
        ),
        "play_excess": float(row.get("play_excess", max(float(row.get("max_play_ratio", 0.0)) - 1.0, 0.0))),
        "reward_v2": float(row.get("reward_scaled", 0.0)),
        "reward_raw_v2": float(row.get("reward_raw", 0.0)),
        "regret_type_id": int(REGRET_TO_ID.get(regret_type, 0)),
        "regret_strength": float(row.get("regret_strength", 0.0)),
        "future_return": float(row.get("future_return", row.get("reward_scaled", 0.0))),
        "future_regret_any": int(row.get("future_regret_any", regret_type != "none")),
        "future_horizon": int(row.get("future_horizon_actual", args.future_horizon)),
        "bootstrap_mask": float(row.get("bootstrap_mask", 0.0)),
        "is_organic": int(row.get("is_organic", 0)),
        "track_length_seconds": int(row.get("track_length_seconds", 0)),
    }


class ShardWriter:
    def __init__(self, root: Path, shard_rows: int) -> None:
        self.root = root
        self.shard_rows = int(shard_rows)
        self.buffers = {split: [] for split in SPLITS}
        self.indices = {split: 0 for split in SPLITS}
        for split in SPLITS:
            split_dir = root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for old_shard in split_dir.glob("part-*.parquet"):
                old_shard.unlink()

    def add(self, split: str, row: dict) -> None:
        self.buffers[split].append(row)
        if len(self.buffers[split]) >= self.shard_rows:
            self.flush(split)

    def flush(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return
        path = self.root / split / f"part-{self.indices[split]:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        self.buffers[split] = []
        self.indices[split] += 1

    def close(self) -> None:
        for split in SPLITS:
            self.flush(split)


def main() -> None:
    args = parse_args()
    transition_root = Path(args.transition_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dense2orig = np.load(args.dense2orig_npy, mmap_mode="r")
    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r") if args.dense_item2sid_npy else None
    writer = ShardWriter(out_dir, args.shard_rows)
    counts = Counter()
    response_positive = {split: np.zeros(len(RESPONSE_NAMES), dtype=np.int64) for split in SPLITS}
    needed_items: set[int] = set()
    rows_written = 0

    for split in SPLITS:
        files = list_split_files(transition_root, split)
        if not files:
            raise FileNotFoundError(f"No session-run parquet files found for split={split}: {transition_root}")
        for path in files:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=1024):
                for source_row in batch.to_pylist():
                    if args.max_rows and counts[split] >= args.max_rows:
                        break
                    row = adapt_row(source_row, split, args, dense2orig, dense_item2sid)
                    if row["target_item_id"] <= 0 or (dense_item2sid is not None and not row["target_sid"]):
                        continue
                    writer.add(split, row)
                    counts[split] += 1
                    rows_written += 1
                    response_positive[split] += np.asarray(row["response_targets"], dtype=np.int64)
                    if args.write_needed_items:
                        needed_items.add(int(row["target_item_id"]))
                        needed_items.update(int(value) for value in row["history_item_ids"] if int(value) > 0)
                        needed_items.update(int(value) for value in row["next_history_item_ids"] if int(value) > 0)
                if args.max_rows and counts[split] >= args.max_rows:
                    break
            if args.max_rows and counts[split] >= args.max_rows:
                break
    writer.close()

    if args.write_needed_items:
        np.save(out_dir / "needed_item_ids.npy", np.asarray(sorted(needed_items), dtype=np.uint32))
    meta = {
        "source": "session_run",
        "transition_root": str(transition_root),
        "history_len": int(args.history_len),
        "future_horizon": int(args.future_horizon),
        "reward_version": "v2",
        "response_names": list(RESPONSE_NAMES),
        "response_positive_counts": {split: values.tolist() for split, values in response_positive.items()},
        "counts": dict(counts),
        "rows_written": int(rows_written),
        "needed_items": len(needed_items),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
