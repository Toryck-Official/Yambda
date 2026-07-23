from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset


def list_parquet_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted(root.glob("*.parquet"))


class EmbedStore:
    def __init__(self, root: str | Path) -> None:
        root = Path(root)
        self.item_ids = np.load(root / "item_ids.npy", mmap_mode="r")
        self.embeds = np.load(root / "item_embeds.npy", mmap_mode="r")
        self.dim = int(self.embeds.shape[1])

    def lookup(self, ids) -> np.ndarray:
        arr = np.asarray(ids, dtype=np.uint32)
        flat = arr.reshape(-1)
        out = np.zeros((flat.shape[0], self.dim), dtype=np.float32)
        mask = flat > 0
        if mask.any():
            wanted = flat[mask]
            pos = np.searchsorted(self.item_ids, wanted)
            in_range = pos < len(self.item_ids)
            valid = np.zeros_like(in_range, dtype=bool)
            valid[in_range] = self.item_ids[pos[in_range]] == wanted[in_range]
            if valid.any():
                out_indices = np.flatnonzero(mask)[valid]
                out[out_indices] = self.embeds[pos[valid]]
        return out.reshape(*arr.shape, self.dim)


class FutureIterableDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: str | Path,
        split: str = "train",
        max_rows: int = 0,
        batch_read_size: int = 2048,
    ) -> None:
        super().__init__()
        self.files = list_parquet_files(Path(parquet_path) / split)
        if not self.files:
            self.files = list_parquet_files(parquet_path)
        if not self.files:
            raise FileNotFoundError(f"No parquet files found for {parquet_path}")
        self.max_rows = int(max_rows)
        self.batch_read_size = int(batch_read_size)

    def __iter__(self) -> Iterable[dict]:
        emitted = 0
        columns = [
            "transition_id",
            "user_id",
            "session_id",
            "session_step_idx",
            "user_step_idx",
            "history_item_ids",
            "history_dense_item_ids",
            "history_feedbacks",
            "history_event_type_ids",
            "history_response_targets",
            "history_play_ratios",
            "history_play_excesses",
            "history_is_organic",
            "history_time_gap_seconds",
            "history_same_session",
            "next_history_item_ids",
            "next_history_dense_item_ids",
            "next_history_feedbacks",
            "next_history_event_type_ids",
            "next_history_response_targets",
            "next_history_play_ratios",
            "next_history_play_excesses",
            "next_history_is_organic",
            "next_history_time_gap_seconds",
            "next_history_same_session",
            "target_item_id",
            "target_dense_item_id",
            "target_sid",
            "response_targets",
            "played_ratio",
            "played_ratio_clipped",
            "play_excess",
            "reward_v2",
            "regret_type_id",
            "regret_strength",
            "future_return",
            "future_regret_any",
            "bootstrap_mask",
            "is_organic",
            "track_length_seconds",
        ]
        for file_path in self.files:
            pf = pq.ParquetFile(file_path)
            present = [col for col in columns if col in pf.schema_arrow.names]
            for batch in pf.iter_batches(columns=present, batch_size=self.batch_read_size):
                for row in batch.to_pylist():
                    yield row
                    emitted += 1
                    if self.max_rows and emitted >= self.max_rows:
                        return


def collate_future(rows: list[dict], store: EmbedStore) -> dict[str, torch.Tensor]:
    history_ids = np.asarray([row["history_item_ids"] for row in rows], dtype=np.uint32)
    next_history_ids = np.asarray([row["next_history_item_ids"] for row in rows], dtype=np.uint32)
    target_ids = np.asarray([row["target_item_id"] for row in rows], dtype=np.uint32)
    history_features = store.lookup(history_ids)
    next_history_features = store.lookup(next_history_ids)
    action_features = store.lookup(target_ids)
    history_feedbacks = np.asarray([row["history_feedbacks"] for row in rows], dtype=np.float32)
    history_event_type_ids = np.asarray([row["history_event_type_ids"] for row in rows], dtype=np.int64)
    history_mask = (history_ids > 0).astype(np.float32)
    history_len = history_ids.shape[1]

    def sequence(name: str, dtype, default):
        values = []
        for row in rows:
            value = row.get(name)
            values.append(value if value is not None else [default] * history_len)
        return np.asarray(values, dtype=dtype)

    history_responses = sequence("history_response_targets", np.float32, [0.0] * 5)
    history_play_ratios = sequence("history_play_ratios", np.float32, 0.0)
    history_play_excesses = sequence("history_play_excesses", np.float32, 0.0)
    history_is_organic = sequence("history_is_organic", np.int64, 0)
    history_time_gaps = sequence("history_time_gap_seconds", np.float32, 0.0)
    history_same_session = sequence("history_same_session", np.int64, 0)
    next_history_feedbacks = sequence("next_history_feedbacks", np.float32, 0.0)
    next_history_event_type_ids = sequence("next_history_event_type_ids", np.int64, 0)
    next_history_responses = sequence("next_history_response_targets", np.float32, [0.0] * 5)
    next_history_play_ratios = sequence("next_history_play_ratios", np.float32, 0.0)
    next_history_play_excesses = sequence("next_history_play_excesses", np.float32, 0.0)
    next_history_is_organic = sequence("next_history_is_organic", np.int64, 0)
    next_history_time_gaps = sequence("next_history_time_gap_seconds", np.float32, 0.0)
    next_history_same_session = sequence("next_history_same_session", np.int64, 0)
    next_history_mask = (next_history_ids > 0).astype(np.float32)

    history_dense = np.asarray(
        [row.get("history_dense_item_ids", [0] * history_ids.shape[1]) for row in rows],
        dtype=np.int64,
    )
    target_dense = np.asarray([row.get("target_dense_item_id", 0) for row in rows], dtype=np.int64)
    sid_lists = [row.get("target_sid") or [] for row in rows]
    sid_len = max((len(item) for item in sid_lists), default=0)
    target_sid = np.zeros((len(rows), sid_len), dtype=np.int64)
    for idx, sid in enumerate(sid_lists):
        if sid:
            target_sid[idx, : len(sid)] = np.asarray(sid, dtype=np.int64)
    response_targets = [row.get("response_targets", [0.0] * 5) for row in rows]

    batch = {
        "history_item_ids": torch.tensor(history_ids.astype(np.int64), dtype=torch.long),
        "history_dense_item_ids": torch.tensor(history_dense, dtype=torch.long),
        "history_features": torch.tensor(history_features, dtype=torch.float32),
        "history_feedbacks": torch.tensor(history_feedbacks, dtype=torch.float32),
        "history_event_type_ids": torch.tensor(history_event_type_ids, dtype=torch.long),
        "history_response_targets": torch.tensor(history_responses, dtype=torch.float32),
        "history_play_ratios": torch.tensor(history_play_ratios, dtype=torch.float32),
        "history_play_excesses": torch.tensor(history_play_excesses, dtype=torch.float32),
        "history_is_organic": torch.tensor(history_is_organic, dtype=torch.long),
        "history_time_gap_seconds": torch.tensor(history_time_gaps, dtype=torch.float32),
        "history_same_session": torch.tensor(history_same_session, dtype=torch.long),
        "history_mask": torch.tensor(history_mask, dtype=torch.float32),
        "next_history_item_ids": torch.tensor(next_history_ids.astype(np.int64), dtype=torch.long),
        "next_history_features": torch.tensor(next_history_features, dtype=torch.float32),
        "next_history_feedbacks": torch.tensor(next_history_feedbacks, dtype=torch.float32),
        "next_history_event_type_ids": torch.tensor(next_history_event_type_ids, dtype=torch.long),
        "next_history_response_targets": torch.tensor(next_history_responses, dtype=torch.float32),
        "next_history_play_ratios": torch.tensor(next_history_play_ratios, dtype=torch.float32),
        "next_history_play_excesses": torch.tensor(next_history_play_excesses, dtype=torch.float32),
        "next_history_is_organic": torch.tensor(next_history_is_organic, dtype=torch.long),
        "next_history_time_gap_seconds": torch.tensor(next_history_time_gaps, dtype=torch.float32),
        "next_history_same_session": torch.tensor(next_history_same_session, dtype=torch.long),
        "next_history_mask": torch.tensor(next_history_mask, dtype=torch.float32),
        "target_item_id": torch.tensor(target_ids.astype(np.int64), dtype=torch.long),
        "target_dense_item_id": torch.tensor(target_dense, dtype=torch.long),
        "action_features": torch.tensor(action_features, dtype=torch.float32),
        "response_targets": torch.tensor(response_targets, dtype=torch.float32),
        "played_ratio": torch.tensor([row["played_ratio"] for row in rows], dtype=torch.float32),
        "played_ratio_clipped": torch.tensor([row["played_ratio_clipped"] for row in rows], dtype=torch.float32),
        "play_excess": torch.tensor([row.get("play_excess", 0.0) for row in rows], dtype=torch.float32),
        "reward": torch.tensor([row["reward_v2"] for row in rows], dtype=torch.float32),
        "regret_type_id": torch.tensor([row["regret_type_id"] for row in rows], dtype=torch.long),
        "regret_strength": torch.tensor([row["regret_strength"] for row in rows], dtype=torch.float32),
        "future_return": torch.tensor([row["future_return"] for row in rows], dtype=torch.float32),
        "future_regret_any": torch.tensor([row["future_regret_any"] for row in rows], dtype=torch.float32),
        "bootstrap_mask": torch.tensor([row.get("bootstrap_mask", 0.0) for row in rows], dtype=torch.float32),
        "is_organic": torch.tensor([row.get("is_organic", 0) for row in rows], dtype=torch.long),
        "track_length_seconds": torch.tensor(
            [row.get("track_length_seconds", 0) for row in rows], dtype=torch.float32
        ),
    }
    if sid_len > 0:
        batch["target_sid"] = torch.tensor(target_sid, dtype=torch.long)
    return batch
