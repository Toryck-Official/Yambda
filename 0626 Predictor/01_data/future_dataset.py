from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

RAW_EVENTS = ["listen", "like", "dislike", "unlike", "undislike"]
RESPONSE_TO_INDEX = {name: idx for idx, name in enumerate(RAW_EVENTS)}
REGRET_TO_ID = {"none": 0, "low_play": 1, "dislike": 2, "unlike": 3}


def list_parquet_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted(root.glob("*.parquet"))


def _as_int(value, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _as_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _pad_left(values: Iterable, length: int, fill):
    seq = list(values or [])[-length:]
    return [fill] * (length - len(seq)) + seq


def _play_bucket_id(played_ratio_norm: float) -> int:
    if played_ratio_norm <= 0.0:
        return 0
    if played_ratio_norm <= 0.2:
        return 1
    if played_ratio_norm <= 0.8:
        return 2
    if played_ratio_norm <= 1.0:
        return 3
    return 4


def _positive_play_reward(row: dict) -> float:
    n_listen = _as_int(row.get("n_listen"))
    play = float(np.clip(_as_float(row.get("max_play_ratio")), 0.0, 1.0))
    play_reward = play if n_listen > 0 else 0.0
    reward = (
        play_reward
        + 0.8 * float(_as_int(row.get("effective_like")) > 0)
        - 1.2 * float(_as_int(row.get("effective_dislike")) > 0)
        - 0.6 * float(_as_int(row.get("effective_unlike")) > 0)
        + 0.2 * float(_as_int(row.get("effective_undislike")) > 0)
    )
    return float(np.clip(reward, -2.0, 2.0))


def _explicit_negative_regret_type(row: dict) -> str:
    if _as_int(row.get("effective_dislike")) > 0:
        return "dislike"
    if _as_int(row.get("effective_unlike")) > 0:
        return "unlike"
    return "none"


def _response_targets(row: dict) -> tuple[list[float], int]:
    targets = [
        float(_as_int(row.get("n_listen")) > 0),
        float(_as_int(row.get("has_like")) > 0),
        float(_as_int(row.get("has_dislike")) > 0),
        float(_as_int(row.get("has_unlike")) > 0),
        float(_as_int(row.get("has_undislike")) > 0),
    ]
    for event in ("dislike", "unlike", "like", "undislike", "listen"):
        idx = RESPONSE_TO_INDEX[event]
        if targets[idx] > 0:
            return targets, idx
    return targets, RESPONSE_TO_INDEX["listen"]


def _stable_user_bucket(user_id: int, mod: int) -> int:
    # SplitMix64-style integer mixing; raw Yambda user ids can share the same low digits.
    x = (int(user_id) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 31)
    return int(x % max(int(mod), 1))


class EmbedStore:
    def __init__(self, root: str | Path) -> None:
        root = Path(root)
        self.root = root
        dense_features = root / "dense_item_features.npy"
        if dense_features.exists():
            self.id_mode = "dense"
            self.item_ids = None
            self.embeds = np.load(dense_features, mmap_mode="r")
        else:
            self.id_mode = "orig"
            self.item_ids = np.load(root / "item_ids.npy", mmap_mode="r")
            self.embeds = np.load(root / "item_embeds.npy", mmap_mode="r")
        self.dim = int(self.embeds.shape[1])

    def lookup(self, ids) -> np.ndarray:
        arr = np.asarray(ids, dtype=np.int64)
        flat = arr.reshape(-1)
        out = np.zeros((flat.shape[0], self.dim), dtype=np.float32)
        if self.id_mode == "dense":
            mask = (flat > 0) & (flat < self.embeds.shape[0])
            if mask.any():
                out[np.flatnonzero(mask)] = self.embeds[flat[mask]]
            return out.reshape(*arr.shape, self.dim)

        mask = flat > 0
        if mask.any():
            assert self.item_ids is not None
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
        mapping_root: str | Path | None = None,
        history_len: int = 50,
        future_horizon: int = 5,
        gamma: float = 0.9,
        reward_column: str = "reward_scaled",
        reward_mode: str = "positive_play",
        regret_mode: str = "explicit_negative",
        user_sample_mod: int = 0,
        user_sample_bucket: int = 0,
        max_users: int = 0,
    ) -> None:
        super().__init__()
        self.files = list_parquet_files(Path(parquet_path) / split)
        if not self.files:
            self.files = list_parquet_files(parquet_path)
        if not self.files:
            raise FileNotFoundError(f"No parquet files found for {parquet_path}")
        self.max_rows = int(max_rows)
        self.batch_read_size = int(batch_read_size)
        self.mapping_root = Path(mapping_root) if mapping_root is not None else None
        self.history_len = int(history_len)
        self.future_horizon = int(future_horizon)
        self.gamma = float(gamma)
        self.reward_column = str(reward_column)
        self.reward_mode = str(reward_mode)
        self.regret_mode = str(regret_mode)
        self.user_sample_mod = int(user_sample_mod or 0)
        self.user_sample_bucket = int(user_sample_bucket or 0)
        self.max_users = int(max_users or 0)
        if self.user_sample_mod < 0:
            raise ValueError("user_sample_mod must be non-negative.")
        if self.user_sample_mod > 0 and not (0 <= self.user_sample_bucket < self.user_sample_mod):
            raise ValueError("user_sample_bucket must satisfy 0 <= bucket < mod.")
        self._dense2orig = None
        self._dense_item2sid = None

    def _schema_mode(self, file_path: Path) -> str:
        names = set(pq.ParquetFile(file_path).schema_arrow.names)
        if {"target_item_id", "target_sid"}.issubset(names):
            return "future"
        if {"target_orig_item_id", "target_dense_item_id"}.issubset(names):
            return "regret_transition"
        raise RuntimeError(f"Unsupported parquet schema in {file_path}: {sorted(names)[:20]}")

    @property
    def dense2orig(self) -> np.ndarray:
        if self._dense2orig is None:
            if self.mapping_root is None:
                raise RuntimeError("mapping_root is required when reading Regret transition parquet directly.")
            self._dense2orig = np.load(self.mapping_root / "dense2orig_item_id.npy", mmap_mode="r")
        return self._dense2orig

    @property
    def dense_item2sid(self) -> np.ndarray:
        if self._dense_item2sid is None:
            if self.mapping_root is None:
                raise RuntimeError("mapping_root is required when reading Regret transition parquet directly.")
            self._dense_item2sid = np.load(self.mapping_root / "dense_item2sid.npy", mmap_mode="r")
        return self._dense_item2sid

    def _dense_to_orig(self, ids: list[int]) -> list[int]:
        dense2orig = self.dense2orig
        out: list[int] = []
        for item in ids:
            dense_id = int(item)
            if 0 < dense_id < dense2orig.shape[0]:
                out.append(int(dense2orig[dense_id]))
            else:
                out.append(0)
        return out

    def _row_reward(self, row: dict) -> float:
        if self.reward_mode == "source":
            return _as_float(row.get(self.reward_column, row.get("reward_scaled", row.get("paper_effective_reward"))))
        if self.reward_mode == "positive_play":
            return _positive_play_reward(row)
        raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")

    def _row_regret_type(self, row: dict) -> str:
        if self.regret_mode == "source":
            return str(row.get("regret_type") or "none")
        if self.regret_mode == "explicit_negative":
            return _explicit_negative_regret_type(row)
        raise ValueError(f"Unsupported regret_mode: {self.regret_mode}")

    def _keep_user(self, user_id: int) -> bool:
        if self.user_sample_mod <= 0:
            return True
        return _stable_user_bucket(user_id, self.user_sample_mod) == self.user_sample_bucket

    def _convert_user_rows(self, rows: list[dict]) -> Iterable[dict]:
        ordered = sorted(rows, key=lambda row: (_as_int(row.get("user_step_idx")), _as_int(row.get("step_time"))))
        rewards = [self._row_reward(row) for row in ordered]
        regrets = [self._row_regret_type(row) for row in ordered]
        sid_table = self.dense_item2sid
        for idx, row in enumerate(ordered):
            history_dense = [int(item) for item in _pad_left(row.get("history_item_ids") or [], self.history_len, 0)]
            history_orig = self._dense_to_orig(history_dense)
            next_history_dense = [int(item) for item in _pad_left(row.get("next_history_item_ids") or [], self.history_len, 0)]
            if not any(next_history_dense):
                next_history_dense = history_dense
            next_history_orig = self._dense_to_orig(next_history_dense)
            target_dense = _as_int(row.get("target_dense_item_id"))
            target_orig = _as_int(row.get("target_orig_item_id"))
            if target_orig <= 0 and 0 < target_dense < self.dense2orig.shape[0]:
                target_orig = int(self.dense2orig[target_dense])
            target_sid = [int(item) for item in sid_table[target_dense].tolist()] if 0 < target_dense < sid_table.shape[0] else []
            response_targets, response_target = _response_targets(row)
            played_ratio = _as_float(row.get("max_play_ratio"))
            clipped = float(np.clip(played_ratio, 0.0, 1.0))
            reward = rewards[idx]
            future_return = 0.0
            future_regret_any = 0.0
            for offset, future_reward in enumerate(rewards[idx : idx + self.future_horizon]):
                future_return += (self.gamma**offset) * float(future_reward)
                if regrets[idx + offset] != "none":
                    future_regret_any = 1.0
            yield {
                "history_item_ids": history_orig,
                "history_dense_item_ids": history_dense,
                "next_history_item_ids": next_history_orig,
                "next_history_dense_item_ids": next_history_dense,
                "history_feedbacks": [float(item) for item in _pad_left(row.get("history_feedbacks") or [], self.history_len, 0.0)],
                "history_event_type_ids": [int(item) for item in _pad_left(row.get("history_event_type_ids") or [], self.history_len, 0)],
                "next_history_feedbacks": [float(item) for item in _pad_left(row.get("next_history_feedbacks") or [], self.history_len, 0.0)],
                "next_history_event_type_ids": [int(item) for item in _pad_left(row.get("next_history_event_type_ids") or [], self.history_len, 0)],
                "target_item_id": target_orig,
                "target_dense_item_id": target_dense,
                "target_sid": target_sid,
                "response_target": response_target,
                "response_targets": response_targets,
                "played_ratio": played_ratio,
                "played_ratio_clipped": clipped,
                "play_bucket_id": _play_bucket_id(clipped),
                "reward_v2": reward,
                "regret_type_id": int(REGRET_TO_ID.get(regrets[idx], 0)),
                "regret_strength": _as_float(row.get("regret_strength")),
                "future_return": float(future_return),
                "future_regret_any": float(future_regret_any),
            }

    def _iter_future_data(self) -> Iterable[dict]:
        columns = [
            "user_id",
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
            "bootstrap_mask",
        ]
        emitted = 0
        selected_users: set[int] = set()
        for file_path in self.files:
            pf = pq.ParquetFile(file_path)
            present = [col for col in columns if col in pf.schema_arrow.names]
            for batch in pf.iter_batches(columns=present, batch_size=self.batch_read_size):
                for row in batch.to_pylist():
                    if "user_id" in row:
                        user_id = _as_int(row.get("user_id"))
                        if not self._keep_user(user_id):
                            continue
                        if self.max_users > 0 and user_id not in selected_users:
                            if len(selected_users) >= self.max_users:
                                return
                            selected_users.add(user_id)
                    yield row
                    emitted += 1
                    if self.max_rows and emitted >= self.max_rows:
                        return

    def _iter_regret_transitions(self) -> Iterable[dict]:
        columns = [
            "user_id",
            "user_step_idx",
            "step_time",
            "history_item_ids",
            "history_feedbacks",
            "history_event_type_ids",
            "next_history_item_ids",
            "next_history_feedbacks",
            "next_history_event_type_ids",
            "target_orig_item_id",
            "target_dense_item_id",
            "n_listen",
            "max_play_ratio",
            "effective_like",
            "effective_dislike",
            "effective_unlike",
            "effective_undislike",
            "reward_scaled",
            "paper_effective_reward",
            self.reward_column,
            "regret_type",
            "regret_strength",
        ]
        columns = list(dict.fromkeys(columns))
        emitted = 0
        selected_users = 0
        current_user = None
        user_rows: list[dict] = []
        for file_path in self.files:
            pf = pq.ParquetFile(file_path)
            present = [col for col in columns if col in pf.schema_arrow.names]
            for batch in pf.iter_batches(columns=present, batch_size=self.batch_read_size):
                for row in batch.to_pylist():
                    user_id = _as_int(row.get("user_id"))
                    if current_user is None:
                        current_user = user_id
                    if user_id != current_user:
                        if user_rows and self._keep_user(int(current_user)):
                            if self.max_users > 0 and selected_users >= self.max_users:
                                return
                            selected_users += 1
                            for converted in self._convert_user_rows(user_rows):
                                yield converted
                                emitted += 1
                                if self.max_rows and emitted >= self.max_rows:
                                    return
                        user_rows = []
                        current_user = user_id
                    user_rows.append(row)
        if user_rows and self._keep_user(int(current_user)):
            if self.max_users > 0 and selected_users >= self.max_users:
                return
            for converted in self._convert_user_rows(user_rows):
                yield converted
                emitted += 1
                if self.max_rows and emitted >= self.max_rows:
                    return

    def __iter__(self) -> Iterable[dict]:
        mode = self._schema_mode(self.files[0])
        if mode == "future":
            yield from self._iter_future_data()
        else:
            yield from self._iter_regret_transitions()


def collate_future(rows: list[dict], store: EmbedStore) -> dict[str, torch.Tensor]:
    history_ids = np.asarray([row["history_item_ids"] for row in rows], dtype=np.uint32)
    target_ids = np.asarray([row["target_item_id"] for row in rows], dtype=np.uint32)
    history_dense = np.asarray(
        [row.get("history_dense_item_ids", [0] * history_ids.shape[1]) for row in rows],
        dtype=np.int64,
    )
    next_history_ids = np.asarray(
        [row.get("next_history_item_ids", row["history_item_ids"]) for row in rows],
        dtype=np.uint32,
    )
    next_history_dense = np.asarray(
        [row.get("next_history_dense_item_ids", row.get("history_dense_item_ids", [0] * history_ids.shape[1])) for row in rows],
        dtype=np.int64,
    )
    target_dense = np.asarray([row.get("target_dense_item_id", 0) for row in rows], dtype=np.int64)
    if store.id_mode == "dense":
        history_features = store.lookup(history_dense)
        next_history_features = store.lookup(next_history_dense)
        action_features = store.lookup(target_dense)
    else:
        history_features = store.lookup(history_ids)
        next_history_features = store.lookup(next_history_ids)
        action_features = store.lookup(target_ids)
    history_feedbacks = np.asarray([row["history_feedbacks"] for row in rows], dtype=np.float32)
    history_event_type_ids = np.asarray([row["history_event_type_ids"] for row in rows], dtype=np.int64)
    next_history_feedbacks = np.asarray([row.get("next_history_feedbacks", row["history_feedbacks"]) for row in rows], dtype=np.float32)
    next_history_event_type_ids = np.asarray([row.get("next_history_event_type_ids", row["history_event_type_ids"]) for row in rows], dtype=np.int64)
    history_mask = (history_dense > 0).astype(np.float32)
    next_history_mask = (next_history_dense > 0).astype(np.float32)
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
    next_history_responses = sequence("next_history_response_targets", np.float32, [0.0] * 5)
    next_history_play_ratios = sequence("next_history_play_ratios", np.float32, 0.0)
    next_history_play_excesses = sequence("next_history_play_excesses", np.float32, 0.0)
    next_history_is_organic = sequence("next_history_is_organic", np.int64, 0)
    next_history_time_gaps = sequence("next_history_time_gap_seconds", np.float32, 0.0)
    next_history_same_session = sequence("next_history_same_session", np.int64, 0)
    history_has_rich = np.asarray(
        [float(any(row.get(name) is not None for name in (
            "history_response_targets", "history_play_ratios", "history_play_excesses",
            "history_is_organic", "history_time_gap_seconds", "history_same_session",
        ))) for row in rows],
        dtype=np.float32,
    )
    next_history_has_rich = np.asarray(
        [float(any(row.get(name) is not None for name in (
            "next_history_response_targets", "next_history_play_ratios", "next_history_play_excesses",
            "next_history_is_organic", "next_history_time_gap_seconds", "next_history_same_session",
        ))) for row in rows],
        dtype=np.float32,
    )

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
        "history_response_targets": torch.tensor(history_responses, dtype=torch.float32),
        "history_play_ratios": torch.tensor(history_play_ratios, dtype=torch.float32),
        "history_play_excesses": torch.tensor(history_play_excesses, dtype=torch.float32),
        "history_is_organic": torch.tensor(history_is_organic, dtype=torch.long),
        "history_time_gap_seconds": torch.tensor(history_time_gaps, dtype=torch.float32),
        "history_same_session": torch.tensor(history_same_session, dtype=torch.long),
        "history_has_rich_features": torch.tensor(history_has_rich, dtype=torch.float32),
        "history_mask": torch.tensor(history_mask, dtype=torch.float32),
        "next_history_item_ids": torch.tensor(next_history_ids.astype(np.int64), dtype=torch.long),
        "next_history_dense_item_ids": torch.tensor(next_history_dense, dtype=torch.long),
        "next_history_features": torch.tensor(next_history_features, dtype=torch.float32),
        "next_history_feedbacks": torch.tensor(next_history_feedbacks, dtype=torch.float32),
        "next_history_event_type_ids": torch.tensor(next_history_event_type_ids, dtype=torch.long),
        "next_history_response_targets": torch.tensor(next_history_responses, dtype=torch.float32),
        "next_history_play_ratios": torch.tensor(next_history_play_ratios, dtype=torch.float32),
        "next_history_play_excesses": torch.tensor(next_history_play_excesses, dtype=torch.float32),
        "next_history_is_organic": torch.tensor(next_history_is_organic, dtype=torch.long),
        "next_history_time_gap_seconds": torch.tensor(next_history_time_gaps, dtype=torch.float32),
        "next_history_same_session": torch.tensor(next_history_same_session, dtype=torch.long),
        "next_history_has_rich_features": torch.tensor(next_history_has_rich, dtype=torch.float32),
        "next_history_mask": torch.tensor(next_history_mask, dtype=torch.float32),
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
        "bootstrap_mask": torch.tensor([row.get("bootstrap_mask", 1.0) for row in rows], dtype=torch.float32),
    }
    if sid_len > 0:
        batch["target_sid"] = torch.tensor(target_sid, dtype=torch.long)
    return batch
