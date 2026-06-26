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
            "history_item_ids",
            "history_dense_item_ids",
            "history_feedbacks",
            "history_event_type_ids",
            "target_item_id",
            "target_dense_item_id",
            "target_sid",
            "response_target",
            "response_targets",
            "played_ratio",
            "played_ratio_clipped",
            "play_bucket_id",
            "reward_v2",
            "regret_type_id",
            "regret_strength",
            "future_return",
            "future_regret_any",
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
    target_ids = np.asarray([row["target_item_id"] for row in rows], dtype=np.uint32)
    history_features = store.lookup(history_ids)
    action_features = store.lookup(target_ids)
    history_feedbacks = np.asarray([row["history_feedbacks"] for row in rows], dtype=np.float32)
    history_event_type_ids = np.asarray([row["history_event_type_ids"] for row in rows], dtype=np.int64)
    history_mask = (history_ids > 0).astype(np.float32)

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
    response_targets = []
    for row in rows:
        if row.get("response_targets") is not None:
            response_targets.append(row["response_targets"])
        else:
            one_hot = [0.0] * 5
            one_hot[int(row["response_target"])] = 1.0
            response_targets.append(one_hot)

    batch = {
        "history_item_ids": torch.tensor(history_ids.astype(np.int64), dtype=torch.long),
        "history_dense_item_ids": torch.tensor(history_dense, dtype=torch.long),
        "history_features": torch.tensor(history_features, dtype=torch.float32),
        "history_feedbacks": torch.tensor(history_feedbacks, dtype=torch.float32),
        "history_event_type_ids": torch.tensor(history_event_type_ids, dtype=torch.long),
        "history_mask": torch.tensor(history_mask, dtype=torch.float32),
        "target_item_id": torch.tensor(target_ids.astype(np.int64), dtype=torch.long),
        "target_dense_item_id": torch.tensor(target_dense, dtype=torch.long),
        "action_features": torch.tensor(action_features, dtype=torch.float32),
        "response_target": torch.tensor([row["response_target"] for row in rows], dtype=torch.long),
        "response_targets": torch.tensor(response_targets, dtype=torch.float32),
        "played_ratio": torch.tensor([row["played_ratio"] for row in rows], dtype=torch.float32),
        "played_ratio_clipped": torch.tensor([row["played_ratio_clipped"] for row in rows], dtype=torch.float32),
        "play_bucket_id": torch.tensor([row["play_bucket_id"] for row in rows], dtype=torch.long),
        "reward": torch.tensor([row["reward_v2"] for row in rows], dtype=torch.float32),
        "regret_type_id": torch.tensor([row["regret_type_id"] for row in rows], dtype=torch.long),
        "regret_strength": torch.tensor([row["regret_strength"] for row in rows], dtype=torch.float32),
        "future_return": torch.tensor([row["future_return"] for row in rows], dtype=torch.float32),
        "future_regret_any": torch.tensor([row["future_regret_any"] for row in rows], dtype=torch.float32),
    }
    if sid_len > 0:
        batch["target_sid"] = torch.tensor(target_sid, dtype=torch.long)
    return batch
