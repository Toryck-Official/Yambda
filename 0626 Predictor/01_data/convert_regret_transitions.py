from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

RAW_EVENTS = ["listen", "like", "dislike", "unlike", "undislike"]
RESPONSE_TO_INDEX = {name: idx for idx, name in enumerate(RAW_EVENTS)}
REGRET_TO_ID = {"none": 0, "low_play": 1, "dislike": 2, "unlike": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 0408 Regret step-level transitions to 0626 Predictor future_data."
    )
    parser.add_argument(
        "--transition_root",
        default="/root/autodl-tmp/0408Yambda/Regret/artifacts/current/data",
        help="0408 Regret transition root containing train/val/test parquet shards.",
    )
    parser.add_argument(
        "--mapping_root",
        default="/root/autodl-tmp/0408Yambda/Regret/artifacts/mappings/raw_rqkmeans",
        help="Directory with dense_item2sid.npy and dense2orig_item_id.npy.",
    )
    parser.add_argument(
        "--out_dir",
        default="/root/autodl-tmp/0626/0626 Predictor/01_data/processed/future_data",
    )
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--history_len", type=int, default=50)
    parser.add_argument("--future_horizon", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--reward_column", default="reward_scaled")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--shard_rows", type=int, default=100000)
    parser.add_argument("--max_rows", type=int, default=0, help="Debug limit per split. 0 means full split.")
    return parser.parse_args()


def parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.parquet"))


def pad_left(values: Iterable, length: int, fill):
    seq = list(values or [])[-length:]
    return [fill] * (length - len(seq)) + seq


def as_int(value, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def as_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def play_bucket_id(played_ratio_norm: float) -> int:
    if played_ratio_norm <= 0.0:
        return 0
    if played_ratio_norm <= 0.2:
        return 1
    if played_ratio_norm <= 0.8:
        return 2
    if played_ratio_norm <= 1.0:
        return 3
    return 4


def dense_to_orig(ids: list[int], dense2orig: np.ndarray) -> list[int]:
    out: list[int] = []
    n_items = int(dense2orig.shape[0])
    for item in ids:
        dense_id = int(item)
        if 0 < dense_id < n_items:
            out.append(int(dense2orig[dense_id]))
        else:
            out.append(0)
    return out


def response_targets(row: dict) -> tuple[list[float], int]:
    targets = [
        float(as_int(row.get("n_listen")) > 0),
        float(as_int(row.get("effective_like")) > 0),
        float(as_int(row.get("effective_dislike")) > 0),
        float(as_int(row.get("effective_unlike")) > 0),
        float(as_int(row.get("effective_undislike")) > 0),
    ]
    for event in ("dislike", "unlike", "like", "undislike", "listen"):
        idx = RESPONSE_TO_INDEX[event]
        if targets[idx] > 0:
            return targets, idx
    return targets, RESPONSE_TO_INDEX["listen"]


def sort_user_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (as_int(row.get("user_step_idx")), as_int(row.get("step_time"))))


class ShardedWriter:
    def __init__(self, out_split: Path, shard_rows: int) -> None:
        self.out_split = out_split
        self.shard_rows = int(shard_rows)
        self.out_split.mkdir(parents=True, exist_ok=True)
        self.buffer: list[dict] = []
        self.shard_idx = 0
        self.rows_written = 0

    def write(self, row: dict) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        path = self.out_split / f"part-{self.shard_idx:05d}.parquet"
        table = pa.Table.from_pylist(self.buffer)
        pq.write_table(table, path, compression="zstd")
        self.rows_written += len(self.buffer)
        self.buffer.clear()
        self.shard_idx += 1


def convert_user_rows(
    rows: list[dict],
    writer: ShardedWriter,
    *,
    split: str,
    history_len: int,
    future_horizon: int,
    gamma: float,
    reward_column: str,
    dense2orig: np.ndarray,
    dense_item2sid: np.ndarray,
    needed_items: set[int],
) -> None:
    ordered = sort_user_rows(rows)
    rewards = [as_float(row.get(reward_column, row.get("reward_scaled", row.get("paper_effective_reward")))) for row in ordered]
    regrets = [str(row.get("regret_type") or "none") for row in ordered]
    n_sid = int(dense_item2sid.shape[0])
    for idx, row in enumerate(ordered):
        history_dense = [int(item) for item in pad_left(row.get("history_item_ids") or [], history_len, 0)]
        history_orig = dense_to_orig(history_dense, dense2orig)
        history_feedbacks = [float(item) for item in pad_left(row.get("history_feedbacks") or [], history_len, 0.0)]
        history_event_types = [int(item) for item in pad_left(row.get("history_event_type_ids") or [], history_len, 0)]

        target_dense = as_int(row.get("target_dense_item_id"))
        target_orig = as_int(row.get("target_orig_item_id"))
        if target_orig <= 0 and 0 < target_dense < dense2orig.shape[0]:
            target_orig = int(dense2orig[target_dense])
        if 0 < target_dense < n_sid:
            target_sid = [int(item) for item in dense_item2sid[target_dense].tolist()]
        else:
            target_sid = []

        targets, response_target = response_targets(row)
        played_ratio = as_float(row.get("max_play_ratio"))
        clipped = float(np.clip(played_ratio, 0.0, 1.0))
        reward = as_float(row.get(reward_column, row.get("reward_scaled", row.get("paper_effective_reward"))))
        future_return = 0.0
        future_regret_any = 0.0
        for offset, future_reward in enumerate(rewards[idx : idx + future_horizon]):
            future_return += (gamma**offset) * float(future_reward)
            if regrets[idx + offset] != "none":
                future_regret_any = 1.0

        for item in history_orig:
            if item > 0:
                needed_items.add(item)
        if target_orig > 0:
            needed_items.add(target_orig)

        writer.write(
            {
                "split": split,
                "user_id": as_int(row.get("user_id")),
                "position": as_int(row.get("user_step_idx")),
                "step_time": as_int(row.get("step_time")),
                "history_item_ids": history_orig,
                "history_dense_item_ids": history_dense,
                "history_feedbacks": history_feedbacks,
                "history_event_type_ids": history_event_types,
                "target_item_id": target_orig,
                "target_dense_item_id": target_dense,
                "target_sid": target_sid,
                "response_target": response_target,
                "response_targets": targets,
                "played_ratio": played_ratio,
                "played_ratio_clipped": clipped,
                "play_bucket_id": play_bucket_id(clipped),
                "reward_v2": reward,
                "paper_base_reward": as_float(row.get("paper_base_reward")),
                "paper_effective_reward": as_float(row.get("paper_effective_reward")),
                "reward_scaled": as_float(row.get("reward_scaled")),
                "regret_type": str(row.get("regret_type") or "none"),
                "regret_type_id": int(REGRET_TO_ID.get(str(row.get("regret_type") or "none"), 0)),
                "regret_strength": as_float(row.get("regret_strength")),
                "future_return": float(future_return),
                "future_regret_any": float(future_regret_any),
            }
        )


def convert_split(
    split: str,
    *,
    transition_root: Path,
    out_dir: Path,
    history_len: int,
    future_horizon: int,
    gamma: float,
    reward_column: str,
    batch_size: int,
    shard_rows: int,
    max_rows: int,
    dense2orig: np.ndarray,
    dense_item2sid: np.ndarray,
    needed_items: set[int],
) -> int:
    split_path = transition_root / split
    files = parquet_files(split_path)
    if not files:
        raise FileNotFoundError(f"No parquet files found for split {split}: {split_path}")
    out_split = out_dir / split
    for old_file in out_split.glob("*.parquet"):
        old_file.unlink()
    writer = ShardedWriter(out_split, shard_rows=shard_rows)
    columns = [
        "user_id",
        "user_step_idx",
        "step_time",
        "history_item_ids",
        "history_feedbacks",
        "history_event_type_ids",
        "target_orig_item_id",
        "target_dense_item_id",
        "n_listen",
        "max_play_ratio",
        "effective_like",
        "effective_dislike",
        "effective_unlike",
        "effective_undislike",
        "reward_scaled",
        "paper_base_reward",
        "paper_effective_reward",
        reward_column,
        "regret_type",
        "regret_strength",
    ]
    columns = list(dict.fromkeys(columns))
    current_user = None
    user_rows: list[dict] = []
    read_rows = 0
    progress = tqdm(files, desc=f"[convert:{split}]", unit="file")
    for file_path in progress:
        pf = pq.ParquetFile(file_path)
        present = [col for col in columns if col in pf.schema_arrow.names]
        for batch in pf.iter_batches(columns=present, batch_size=batch_size):
            for row in batch.to_pylist():
                user_id = as_int(row.get("user_id"))
                if current_user is None:
                    current_user = user_id
                if user_id != current_user:
                    convert_user_rows(
                        user_rows,
                        writer,
                        split=split,
                        history_len=history_len,
                        future_horizon=future_horizon,
                        gamma=gamma,
                        reward_column=reward_column,
                        dense2orig=dense2orig,
                        dense_item2sid=dense_item2sid,
                        needed_items=needed_items,
                    )
                    user_rows = []
                    current_user = user_id
                user_rows.append(row)
                read_rows += 1
                if max_rows and read_rows >= max_rows:
                    break
            if max_rows and read_rows >= max_rows:
                break
        progress.set_postfix(read=read_rows, written=writer.rows_written + len(writer.buffer))
        if max_rows and read_rows >= max_rows:
            break
    if user_rows:
        convert_user_rows(
            user_rows,
            writer,
            split=split,
            history_len=history_len,
            future_horizon=future_horizon,
            gamma=gamma,
            reward_column=reward_column,
            dense2orig=dense2orig,
            dense_item2sid=dense_item2sid,
            needed_items=needed_items,
        )
    writer.flush()
    return writer.rows_written


def main() -> None:
    args = parse_args()
    transition_root = Path(args.transition_root)
    mapping_root = Path(args.mapping_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dense2orig = np.load(mapping_root / "dense2orig_item_id.npy", mmap_mode="r")
    dense_item2sid = np.load(mapping_root / "dense_item2sid.npy", mmap_mode="r")
    needed_items: set[int] = set()
    counts: dict[str, int] = {}
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        counts[split] = convert_split(
            split,
            transition_root=transition_root,
            out_dir=out_dir,
            history_len=args.history_len,
            future_horizon=args.future_horizon,
            gamma=args.gamma,
            reward_column=args.reward_column,
            batch_size=args.batch_size,
            shard_rows=args.shard_rows,
            max_rows=args.max_rows,
            dense2orig=dense2orig,
            dense_item2sid=dense_item2sid,
            needed_items=needed_items,
        )

    needed_path = out_dir / "needed_item_ids.npy"
    np.save(needed_path, np.asarray(sorted(needed_items), dtype=np.int64))
    meta = {
        "source": "0408 Regret step-level transitions",
        "transition_root": str(transition_root),
        "mapping_root": str(mapping_root),
        "history_len": args.history_len,
        "future_horizon": args.future_horizon,
        "gamma": args.gamma,
        "reward_column": args.reward_column,
        "splits": counts,
        "needed_items": int(len(needed_items)),
        "dense_item2sid_npy": str(mapping_root / "dense_item2sid.npy"),
        "dense2orig_item_id_npy": str(mapping_root / "dense2orig_item_id.npy"),
        "embed_store": str(mapping_root),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
