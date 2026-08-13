"""Full-history target-group datasets for Minimal SNMPP Pilot."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class FullHistoryTargetDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: str | Path, target_key: str) -> None:
        archive = np.load(Path(path), allow_pickle=False)
        self.event_timestamp = archive["event_timestamp"]
        self.event_feedback = archive["event_feedback"]
        self.event_sid = archive["event_sid"]
        self.group_global_id = archive["group_global_id"]
        self.group_uid = archive["group_uid"]
        self.group_timestamp = archive["group_timestamp"]
        self.group_event_offsets = archive["group_event_offsets"]
        self.group_feedback_counts = archive["group_feedback_counts"]
        self.targets = archive[target_key].astype(np.int64)
        self.target_key = target_key
        starts = np.flatnonzero(np.r_[True, self.group_uid[1:] != self.group_uid[:-1]])
        marker = np.zeros(len(self.group_uid), dtype=np.int64)
        marker[starts] = starts
        self.group_user_start = np.maximum.accumulate(marker)
        if np.any(self.targets <= self.group_user_start[self.targets]):
            raise ValueError("every target must have a previous group")
        self.history_event_counts = (
            self.group_event_offsets[self.targets]
            - self.group_event_offsets[self.group_user_start[self.targets]]
        ).astype(np.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target_group = int(self.targets[index])
        user_group_start = int(self.group_user_start[target_group])
        history_start = int(self.group_event_offsets[user_group_start])
        history_end = int(self.group_event_offsets[target_group])
        user_start_timestamp = int(self.group_timestamp[user_group_start])
        history_times = (
            self.event_timestamp[history_start:history_end].astype(np.float64)
            - user_start_timestamp
        ) / 3600.0
        previous_time = (
            int(self.group_timestamp[target_group - 1]) - user_start_timestamp
        ) / 3600.0
        target_time = (
            int(self.group_timestamp[target_group]) - user_start_timestamp
        ) / 3600.0
        return {
            "history_times": torch.from_numpy(history_times.astype(np.float32)),
            "history_feedback": torch.from_numpy(
                self.event_feedback[history_start:history_end].astype(np.int64)
            ),
            "history_sid": torch.from_numpy(
                self.event_sid[history_start:history_end].astype(np.int64)
            ),
            "previous_time": torch.tensor(previous_time, dtype=torch.float32),
            "target_time": torch.tensor(target_time, dtype=torch.float32),
            "target_feedback_counts": torch.from_numpy(
                self.group_feedback_counts[target_group].astype(np.float32)
            ),
            "uid": torch.tensor(int(self.group_uid[target_group]), dtype=torch.int64),
            "target_group_id": torch.tensor(
                int(self.group_global_id[target_group]), dtype=torch.int64
            ),
            "previous_group_size": torch.tensor(
                int(self.group_feedback_counts[target_group - 1].sum()), dtype=torch.int64
            ),
            "target_group_size": torch.tensor(
                int(self.group_feedback_counts[target_group].sum()), dtype=torch.int64
            ),
            "history_event_count": torch.tensor(history_end - history_start, dtype=torch.int64),
            "gap_seconds": torch.tensor(
                int(self.group_timestamp[target_group])
                - int(self.group_timestamp[target_group - 1]),
                dtype=torch.int64,
            ),
        }


def collate(samples: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch = len(samples)
    maximum = max(len(sample["history_times"]) for sample in samples)
    times = torch.zeros(batch, maximum)
    feedback = torch.zeros(batch, maximum, dtype=torch.long)
    sid = torch.zeros(batch, maximum, 4, dtype=torch.long)
    mask = torch.zeros(batch, maximum, dtype=torch.bool)
    for row, sample in enumerate(samples):
        length = len(sample["history_times"])
        times[row, :length] = sample["history_times"]
        feedback[row, :length] = sample["history_feedback"]
        sid[row, :length] = sample["history_sid"]
        mask[row, :length] = True
    output = {
        "history_times": times,
        "history_feedback": feedback,
        "history_sid": sid,
        "history_mask": mask,
    }
    for key in samples[0]:
        if key not in {"history_times", "history_feedback", "history_sid"}:
            output[key] = torch.stack([sample[key] for sample in samples])
    return output


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths: np.ndarray, batch_size: int, seed: int, shuffle: bool) -> None:
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        order = np.argsort(lengths, kind="stable")
        self.batches = [
            order[start : start + batch_size].tolist()
            for start in range(0, len(order), batch_size)
        ]
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        indices = np.arange(len(self.batches))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(indices)
        self.epoch += 1
        for index in indices:
            yield self.batches[int(index)]

    def __len__(self) -> int:
        return len(self.batches)


def make_loader(
    dataset: FullHistoryTargetDataset,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    sampler = LengthBucketBatchSampler(
        dataset.history_event_counts, batch_size=batch_size, seed=seed, shuffle=shuffle
    )
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=0)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
