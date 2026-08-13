#!/usr/bin/env python3
"""Train the simple train-only four-relation collaborative baseline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "phase1_gate1c" / "configs" / "experiment.json"
DEFAULT_OUTPUT = ROOT / "phase1_gate1c" / "artifacts" / "collaborative_runs"
MARKS = ("like", "dislike", "unlike", "undislike")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


class FourRelationMF(nn.Module):
    def __init__(self, users: int, items: int, dim: int) -> None:
        super().__init__()
        self.user = nn.Embedding(users, dim, sparse=True)
        self.item = nn.Embedding(items, dim, sparse=True)
        self.relation = nn.Parameter(torch.ones(4, dim))
        nn.init.normal_(self.user.weight, std=0.02)
        nn.init.normal_(self.item.weight, std=0.02)
        with torch.no_grad():
            self.relation.add_(0.01 * torch.randn_like(self.relation))

    def score(self, user: torch.Tensor, item: torch.Tensor, relation: int) -> torch.Tensor:
        return (self.user(user) * self.item(item) * self.relation[relation]).sum(dim=1)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    graph_root = Path(config["paths"]["graph"])
    graph_manifest = json.loads((graph_root / "manifest.json").read_text())
    model_config = config["graph_model"]
    item_ids = np.load(graph_root / "explicit_item_id.uint32.npy", mmap_mode="r")
    train_users = np.load(graph_root / "unique_train_user_count.uint32.npy", mmap_mode="r")
    train_items = np.flatnonzero(train_users > 0).astype(np.int64)
    edge_arrays = []
    edge_counts = []
    for mark in MARKS:
        report = graph_manifest["relations"][mark]
        count = int(report["unique_user_item_relation_edges"])
        users = np.memmap(report["uid_file"], mode="r", dtype=np.uint32, shape=(count,))
        items = np.memmap(report["item_position_file"], mode="r", dtype=np.uint32, shape=(count,))
        edge_arrays.append((users, items))
        edge_counts.append(count)
    probabilities = np.asarray(edge_counts, dtype=np.float64)
    probabilities /= probabilities.sum()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Gate 1C graph baseline requires the enabled GPU")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for raw_seed in config["seeds"]:
        seed = int(raw_seed)
        seed_all(seed)
        run_dir = args.output_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model = FourRelationMF(1_000_001, len(item_ids), int(model_config["embedding_dim"])).to(device)
        sparse_optimizer = torch.optim.SparseAdam(
            [model.user.weight, model.item.weight], lr=float(model_config["sparse_learning_rate"])
        )
        relation_optimizer = torch.optim.Adam(
            [model.relation], lr=float(model_config["relation_learning_rate"])
        )
        rng = np.random.default_rng(seed + 301)
        trace = []
        running_loss = 0.0
        running_accuracy = 0.0
        batch_size = int(model_config["batch_size"])
        for step in range(1, int(model_config["training_steps"]) + 1):
            relation = int(rng.choice(4, p=probabilities))
            users_mm, items_mm = edge_arrays[relation]
            edge_index = rng.integers(0, edge_counts[relation], size=batch_size, dtype=np.int64)
            user = torch.from_numpy(np.asarray(users_mm[edge_index], dtype=np.int64)).to(device)
            positive = torch.from_numpy(np.asarray(items_mm[edge_index], dtype=np.int64)).to(device)
            negative_index = rng.integers(0, len(train_items), size=batch_size, dtype=np.int64)
            negative = torch.from_numpy(train_items[negative_index]).to(device)
            collision = negative == positive
            while bool(collision.any()):
                replacements = rng.integers(0, len(train_items), size=int(collision.sum()), dtype=np.int64)
                negative[collision] = torch.from_numpy(train_items[replacements]).to(device)
                collision = negative == positive
            pos_score = model.score(user, positive, relation)
            neg_score = model.score(user, negative, relation)
            ranking = F.softplus(neg_score - pos_score).mean()
            regularization = float(model_config["l2_weight"]) * (
                model.user(user).square().mean()
                + model.item(positive).square().mean()
                + model.item(negative).square().mean()
                + model.relation[relation].square().mean()
            )
            loss = ranking + regularization
            sparse_optimizer.zero_grad(set_to_none=True)
            relation_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            sparse_optimizer.step(); relation_optimizer.step()
            running_loss += float(ranking.detach())
            running_accuracy += float((pos_score > neg_score).float().mean())
            if step % 100 == 0:
                row = {
                    "step": step,
                    "mean_bpr_loss_last_100": running_loss / 100,
                    "pairwise_accuracy_last_100": running_accuracy / 100,
                }
                trace.append(row)
                print(f"seed={seed} step={step} loss={row['mean_bpr_loss_last_100']:.5f} acc={row['pairwise_accuracy_last_100']:.4f}", flush=True)
                running_loss = 0.0; running_accuracy = 0.0
        with torch.no_grad():
            item_embedding = model.item.weight.detach().cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(item_embedding, axis=1, keepdims=True)
            item_embedding[train_users > 0] /= np.maximum(norms[train_users > 0], 1e-12)
            item_embedding[train_users == 0] = 0.0
            np.save(run_dir / "item_collaborative_embedding.float16.npy", item_embedding.astype(np.float16), allow_pickle=False)
            np.save(run_dir / "relation_embedding.float32.npy", model.relation.detach().cpu().numpy().astype(np.float32), allow_pickle=False)
        run_report = {
            "seed": seed,
            "model": model_config,
            "data_contract": graph_manifest["data_contract"],
            "relation_edge_counts": {mark: edge_counts[i] for i, mark in enumerate(MARKS)},
            "relation_sampling_probabilities": {mark: float(probabilities[i]) for i, mark in enumerate(MARKS)},
            "trace": trace,
            "output": str((run_dir / "item_collaborative_embedding.float16.npy").resolve()),
            "audio_embedding_used_in_graph_training": False,
        }
        (run_dir / "run.json").write_text(json.dumps(run_report, ensure_ascii=False, indent=2) + "\n")
        del model, sparse_optimizer, relation_optimizer, item_embedding
        torch.cuda.empty_cache()
        print(f"completed graph seed {seed}", flush=True)


if __name__ == "__main__":
    main()

