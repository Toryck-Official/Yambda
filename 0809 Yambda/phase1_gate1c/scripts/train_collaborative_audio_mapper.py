#!/usr/bin/env python3
"""Train simple MLPs from train-only collaborative representation to audio space."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "phase1_gate1c" / "configs" / "experiment.json"
DEFAULT_RUNS = ROOT / "phase1_gate1c" / "artifacts" / "collaborative_runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    return p.parse_args()


class Mapper(nn.Module):
    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def batches(indices: np.ndarray, size: int, rng=None):
    order = indices.copy()
    if rng is not None: rng.shuffle(order)
    for start in range(0, len(order), size): yield order[start : start + size]


def make_input(embedding, unique_relation, event_relation, indices, scalar_mean, scalar_std):
    vectors = np.asarray(embedding[indices], dtype=np.float32)
    scalar = np.concatenate([
        np.log1p(unique_relation[:, indices].T).astype(np.float32),
        np.log1p(event_relation[:, indices].T).astype(np.float32),
    ], axis=1)
    scalar = (scalar - scalar_mean) / scalar_std
    return np.concatenate([vectors, scalar], axis=1)


@torch.no_grad()
def evaluate(model, embedding, unique_relation, event_relation, truth, indices, scalar_mean, scalar_std, batch_size, device):
    model.eval(); cosine = 0.0; output = []
    for index in batches(indices, batch_size):
        x = torch.from_numpy(make_input(embedding, unique_relation, event_relation, index, scalar_mean, scalar_std)).to(device)
        y = torch.from_numpy(np.asarray(truth[index], dtype=np.float32)).to(device)
        prediction = model(x)
        cosine += float((prediction * y).sum())
        output.append(prediction.cpu().numpy().astype(np.float32))
    return cosine / len(indices), np.concatenate(output)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    paths = {key: Path(value) for key, value in config["paths"].items()}
    graph = paths["graph"]
    item_ids = np.load(graph / "explicit_item_id.uint32.npy", mmap_mode="r")
    lookup = np.full(int(item_ids.max()) + 1, -1, dtype=np.int32)
    lookup[item_ids] = np.arange(len(item_ids), dtype=np.int32)
    unique_users = np.load(graph / "unique_train_user_count.uint32.npy", mmap_mode="r")
    unique_relation = np.load(graph / "unique_user_count_by_relation.uint32.npy", mmap_mode="r")
    event_relation = np.load(graph / "event_count_by_relation.uint32.npy", mmap_mode="r")
    with np.load(paths["gate1b_sample"], allow_pickle=False) as z:
        sample_item = z["item_id"]
        split = z["split"]
    position = lookup[sample_item]
    if np.any(position < 0): raise RuntimeError("mapper sample outside explicit universe")
    truth = np.load(paths["gate1b_arrays"] / "truth.float32.npy", mmap_mode="r")
    train = np.flatnonzero((split == 0) & (unique_users[position] > 0))
    validation = np.flatnonzero((split == 1) & (unique_users[position] > 0))
    test = np.flatnonzero((split == 2) & (unique_users[position] > 0))
    mapper_config = config["mapper"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for raw_seed in config["seeds"]:
        seed = int(raw_seed); seed_all(seed)
        run = args.runs / f"seed_{seed}"
        embedding = np.load(run / "item_collaborative_embedding.float16.npy", mmap_mode="r")
        sample_position = position
        scalar_train = np.concatenate([
            np.log1p(unique_relation[:, sample_position[train]].T).astype(np.float32),
            np.log1p(event_relation[:, sample_position[train]].T).astype(np.float32),
        ], axis=1)
        scalar_mean = scalar_train.mean(axis=0)
        scalar_std = np.maximum(scalar_train.std(axis=0), 1e-6)
        model = Mapper(72, int(mapper_config["hidden_dim"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(mapper_config["learning_rate"]), weight_decay=float(mapper_config["weight_decay"]))
        rng = np.random.default_rng(seed + 901)
        best = -np.inf; best_state = None; patience = 0; trace = []
        for epoch in range(1, int(mapper_config["epochs"]) + 1):
            model.train(); loss_sum = 0.0
            for index in batches(train, int(mapper_config["batch_size"]), rng):
                graph_index = sample_position[index]
                x = torch.from_numpy(make_input(embedding, unique_relation, event_relation, graph_index, scalar_mean, scalar_std)).to(device)
                y = torch.from_numpy(np.asarray(truth[index], dtype=np.float32)).to(device)
                prediction = model(x)
                loss = (1.0 - (prediction * y).sum(dim=1)).mean()
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                loss_sum += float(loss.detach()) * len(index)
            # evaluate expects graph indices for input but sample indices for truth;
            # build directly here to keep both index spaces explicit.
            model.eval(); val_cos = 0.0
            with torch.no_grad():
                for index in batches(validation, int(mapper_config["batch_size"])):
                    x = torch.from_numpy(make_input(embedding, unique_relation, event_relation, sample_position[index], scalar_mean, scalar_std)).to(device)
                    y = torch.from_numpy(np.asarray(truth[index], dtype=np.float32)).to(device)
                    val_cos += float((model(x) * y).sum())
            val_cos /= len(validation)
            trace.append({"epoch": epoch, "train_cosine_loss": loss_sum / len(train), "validation_cosine": val_cos})
            print(f"seed={seed} mapper epoch={epoch} val_cos={val_cos:.6f}", flush=True)
            if val_cos > best + 1e-6:
                best = val_cos; best_state = copy.deepcopy(model.state_dict()); patience = 0
            else:
                patience += 1
                if patience >= int(mapper_config["patience"]): break
        assert best_state is not None; model.load_state_dict(best_state); model.eval()
        prediction_parts = []
        with torch.no_grad():
            for index in batches(test, int(mapper_config["batch_size"])):
                x = torch.from_numpy(make_input(embedding, unique_relation, event_relation, sample_position[index], scalar_mean, scalar_std)).to(device)
                prediction_parts.append(model(x).cpu().numpy().astype(np.float32))
        prediction = np.concatenate(prediction_parts)
        np.save(run / "mapper_test_sample_index.int64.npy", test.astype(np.int64), allow_pickle=False)
        np.save(run / "collaborative_audio_test_prediction.float32.npy", prediction, allow_pickle=False)
        torch.save({"state_dict": model.state_dict(), "input_dim": 72, "hidden_dim": int(mapper_config["hidden_dim"])}, run / "collaborative_audio_mapper.pt")
        report = {
            "seed": seed,
            "mapper_train_items_with_train_edges": int(len(train)),
            "mapper_validation_items_with_train_edges": int(len(validation)),
            "mapper_test_items_with_train_edges": int(len(test)),
            "audio_labels_used": {"graph_training": False, "mapper_train": True, "mapper_validation_for_early_stopping": True, "mapper_test_input": False},
            "scalar_standardization_fit_on_mapper_train_only": True,
            "trace": trace,
        }
        (run / "mapper_run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        del model, optimizer, prediction, embedding
        torch.cuda.empty_cache()
        print(f"completed mapper seed {seed}", flush=True)


if __name__ == "__main__":
    main()
