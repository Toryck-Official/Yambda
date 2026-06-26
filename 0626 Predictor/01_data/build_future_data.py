from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_data"))

from reward import EVENT_TO_ID, RAW_EVENTS, REGRET_TO_ID, RESPONSE_TO_INDEX, RewardConfig, history_signal, play_bucket_id, summarize_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build action-conditioned future predictor rows from Yambda sequences.")
    parser.add_argument("--multi_event", default="/Users/Toryck/Coding/DATASET/Yambda/sequential/50m/multi_event.parquet")
    parser.add_argument("--out_dir", default=str(ROOT / "artifacts" / "future_data"))
    parser.add_argument("--history_len", type=int, default=50)
    parser.add_argument("--future_horizon", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--min_history", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_users", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--shard_rows", type=int, default=50000)
    parser.add_argument("--write_needed_items", action="store_true")
    parser.add_argument("--split_mode", choices=["user", "row"], default="user")
    parser.add_argument("--orig2dense_npy", default="")
    parser.add_argument("--dense_item2sid_npy", default="")
    parser.add_argument("--keep_missing_sid", action="store_true")
    return parser.parse_args()


def split_name(user_id: int) -> str:
    bucket = (int(user_id) * 9973) % 1000
    if bucket < 900:
        return "train"
    if bucket < 950:
        return "val"
    return "test"


def row_split_name(row_idx: int) -> str:
    bucket = int(row_idx) % 20
    if bucket < 16:
        return "train"
    if bucket < 18:
        return "val"
    return "test"


def pad_left(values: list, length: int, pad_value):
    values = list(values)[-length:]
    if len(values) < length:
        values = [pad_value] * (length - len(values)) + values
    return values


def event_dict(event_type: str, timestamp: int, played_ratio_pct: int, raw_pos: int) -> dict:
    play_pct = 0 if played_ratio_pct is None else int(played_ratio_pct)
    return {
        "event_type": str(event_type),
        "timestamp": int(timestamp),
        "played_ratio_norm": float(play_pct) / 100.0 if event_type == "listen" else 0.0,
        "raw_pos": int(raw_pos),
    }


def event_reward(event_type: str, timestamp: int, played_ratio_pct: int, raw_pos: int, cfg: RewardConfig) -> dict:
    return summarize_events([event_dict(event_type, timestamp, played_ratio_pct, raw_pos)], cfg)


def safe_play_pct(value) -> int:
    return 0 if value is None else int(value)


class ShardWriter:
    def __init__(self, out_dir: Path, shard_rows: int) -> None:
        self.out_dir = out_dir
        self.shard_rows = int(shard_rows)
        self.buffers = {"train": [], "val": [], "test": []}
        self.shard_idx = {"train": 0, "val": 0, "test": 0}
        for split in self.buffers:
            (self.out_dir / split).mkdir(parents=True, exist_ok=True)

    def write(self, split: str, row: dict) -> None:
        buf = self.buffers[split]
        buf.append(row)
        if len(buf) >= self.shard_rows:
            self.flush(split)

    def flush(self, split: str) -> None:
        buf = self.buffers[split]
        if not buf:
            return
        path = self.out_dir / split / f"part-{self.shard_idx[split]:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(buf), path)
        self.buffers[split] = []
        self.shard_idx[split] += 1

    def close(self) -> None:
        for split in list(self.buffers):
            self.flush(split)


def map_dense(orig_item_id: int, orig2dense: np.ndarray | None) -> int:
    if orig2dense is None:
        return 0
    if orig_item_id < 0 or orig_item_id >= len(orig2dense):
        return 0
    dense_id = int(orig2dense[orig_item_id])
    return dense_id if dense_id > 0 else 0


def lookup_sid(dense_id: int, dense_item2sid: np.ndarray | None) -> list[int]:
    if dense_item2sid is None or dense_id <= 0 or dense_id >= dense_item2sid.shape[0]:
        return []
    return [int(x) for x in dense_item2sid[dense_id].tolist()]


def make_row(
    row: dict,
    pos: int,
    row_idx: int,
    args: argparse.Namespace,
    cfg: RewardConfig,
    orig2dense: np.ndarray | None,
    dense_item2sid: np.ndarray | None,
) -> dict:
    uid = int(row["uid"])
    timestamps = row["timestamp"]
    item_ids = row["item_id"]
    event_types = row["event_type"]
    played = row["played_ratio_pct"]

    h0 = max(0, pos - args.history_len)
    hist_items = [int(x) for x in item_ids[h0:pos]]
    hist_dense = [map_dense(item_id, orig2dense) for item_id in hist_items]
    hist_events = [str(x) for x in event_types[h0:pos]]
    hist_play = [safe_play_pct(x) for x in played[h0:pos]]
    hist_feedbacks = [
        history_signal(event_type, float(play_pct) / 100.0 if event_type == "listen" else 0.0)
        for event_type, play_pct in zip(hist_events, hist_play)
    ]
    hist_event_ids = [EVENT_TO_ID.get(event_type, 0) for event_type in hist_events]

    target_event = str(event_types[pos])
    target_item = int(item_ids[pos])
    target_dense = map_dense(target_item, orig2dense)
    target_play_pct = safe_play_pct(played[pos])
    target_play_ratio = float(target_play_pct) / 100.0 if target_event == "listen" else 0.0
    current = event_reward(target_event, int(timestamps[pos]), target_play_pct, pos, cfg)

    rewards = []
    regrets = []
    for offset in range(args.future_horizon):
        j = pos + offset
        if j >= len(item_ids):
            break
        summary = event_reward(str(event_types[j]), int(timestamps[j]), safe_play_pct(played[j]), j, cfg)
        rewards.append(float(summary["reward_scaled"]))
        regrets.append(int(summary["regret_type_id"]))
    future_return = float(sum((args.gamma**idx) * reward for idx, reward in enumerate(rewards)))
    future_regret_any = int(any(item != REGRET_TO_ID["none"] for item in regrets))
    response_targets = [0.0] * len(RAW_EVENTS)
    response_targets[int(RESPONSE_TO_INDEX[target_event])] = 1.0

    return {
        "user_id": uid,
        "position": int(pos),
        "split": split_name(uid) if args.split_mode == "user" else row_split_name(row_idx),
        "history_item_ids": pad_left(hist_items, args.history_len, 0),
        "history_dense_item_ids": pad_left(hist_dense, args.history_len, 0),
        "history_feedbacks": pad_left(hist_feedbacks, args.history_len, 0.0),
        "history_event_type_ids": pad_left(hist_event_ids, args.history_len, 0),
        "target_item_id": target_item,
        "target_dense_item_id": int(target_dense),
        "target_sid": lookup_sid(target_dense, dense_item2sid),
        "target_event_type": target_event,
        "response_target": int(RESPONSE_TO_INDEX[target_event]),
        "response_targets": response_targets,
        "played_ratio": float(target_play_ratio),
        "played_ratio_clipped": float(np.clip(target_play_ratio, 0.0, 1.0)),
        "play_bucket_id": int(play_bucket_id(target_play_ratio)),
        "reward_v2": float(current["reward_scaled"]),
        "reward_raw_v2": float(current["reward_raw"]),
        "regret_type_id": int(current["regret_type_id"]),
        "regret_strength": float(current["regret_strength"]),
        "future_return": future_return,
        "future_regret_any": future_regret_any,
        "future_horizon": int(args.future_horizon),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = RewardConfig(version="v2")
    orig2dense = np.load(args.orig2dense_npy, mmap_mode="r") if args.orig2dense_npy else None
    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r") if args.dense_item2sid_npy else None
    writer = ShardWriter(out_dir, args.shard_rows)
    needed_items: set[int] = set()
    counts = Counter()

    pf = pq.ParquetFile(args.multi_event)
    columns = ["uid", "timestamp", "item_id", "played_ratio_pct", "event_type"]
    users_seen = 0
    rows_written = 0
    for batch in pf.iter_batches(columns=columns, batch_size=128):
        for user_row in batch.to_pylist():
            if args.max_users and users_seen >= args.max_users:
                break
            users_seen += 1
            n_events = len(user_row["item_id"])
            start = min(max(args.min_history, 1), n_events)
            for pos in range(start, n_events, max(args.stride, 1)):
                if args.max_rows and rows_written >= args.max_rows:
                    break
                row = make_row(user_row, pos, rows_written, args, cfg, orig2dense, dense_item2sid)
                if dense_item2sid is not None and not args.keep_missing_sid:
                    if int(row["target_dense_item_id"]) <= 0 or not row["target_sid"]:
                        continue
                split = row["split"]
                writer.write(split, row)
                counts[split] += 1
                rows_written += 1
                if args.write_needed_items:
                    needed_items.add(int(row["target_item_id"]))
                    needed_items.update(int(x) for x in row["history_item_ids"] if int(x) > 0)
            if args.max_rows and rows_written >= args.max_rows:
                break
        if (args.max_users and users_seen >= args.max_users) or (args.max_rows and rows_written >= args.max_rows):
            break
    writer.close()

    if args.write_needed_items:
        np.save(out_dir / "needed_item_ids.npy", np.asarray(sorted(needed_items), dtype=np.uint32))
    meta = {
        "multi_event": str(args.multi_event),
        "history_len": args.history_len,
        "future_horizon": args.future_horizon,
        "gamma": args.gamma,
        "reward_version": "v2",
        "raw_events": RAW_EVENTS,
        "counts": dict(counts),
        "users_seen": users_seen,
        "rows_written": rows_written,
        "needed_items": len(needed_items),
        "split_mode": args.split_mode,
        "orig2dense_npy": str(args.orig2dense_npy) if args.orig2dense_npy else "",
        "dense_item2sid_npy": str(args.dense_item2sid_npy) if args.dense_item2sid_npy else "",
        "keep_missing_sid": bool(args.keep_missing_sid),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
