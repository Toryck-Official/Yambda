#!/usr/bin/env python3
"""Honest held-out graph-task sanity check for the Gate 1C DistMult encoder.

The holdout and all candidates are derived exclusively from the global train
period. Audio embeddings and validation/test-period interactions are never read.
Metrics use one true held-out item plus sampled, filtered candidate items and
therefore are explicitly named sampled Hits/MRR rather than full-catalog metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "phase1_gate1c" / "artifacts" / "train_four_relation_graph"
OUTPUT = ROOT / "phase1_gate1c" / "artifacts" / "distmult_graph_sanity"
MARKS = ("like", "dislike", "unlike", "undislike")
SEEDS = (2026, 2027, 2028)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=GRAPH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--holdout-per-relation", type=int, default=5_000)
    parser.add_argument("--negative-candidates", type=int, default=1_000)
    parser.add_argument("--steps", type=int, default=1_200)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class DistMult(nn.Module):
    def __init__(self, users: int, items: int, dim: int) -> None:
        super().__init__()
        self.user = nn.Embedding(users, dim, sparse=True)
        self.item = nn.Embedding(items, dim, sparse=True)
        self.relation = nn.Parameter(torch.ones(4, dim))
        nn.init.normal_(self.user.weight, std=0.02)
        nn.init.normal_(self.item.weight, std=0.02)
        with torch.no_grad():
            self.relation.add_(0.01 * torch.randn_like(self.relation))

    def score(self, users: torch.Tensor, items: torch.Tensor, relation: int) -> torch.Tensor:
        return (self.user(users) * self.item(items) * self.relation[relation]).sum(dim=-1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_member_sorted(values: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    position = np.searchsorted(sorted_reference, values)
    valid = position < len(sorted_reference)
    result = np.zeros(len(values), dtype=bool)
    result[valid] = sorted_reference[position[valid]] == values[valid]
    return result


def build_holdout(edge_arrays, item_relation_degree, per_relation: int, seed: int):
    """Sample eligible transductive edges with fixed seed.

    Eligibility requires at least two edges for the user under the same
    relation and at least two users for the item under that relation. Removing
    one edge therefore does not create an unseen user-relation or item-relation.
    """
    rng = np.random.default_rng(seed)
    holdout = []
    user_offsets = []
    details = {}
    for relation, (users, items) in enumerate(edge_arrays):
        user_degree = np.bincount(users, minlength=1_000_001)
        offsets = np.empty(len(user_degree) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(user_degree, out=offsets[1:])
        user_offsets.append(offsets)
        selected: set[int] = set()
        attempts = 0
        while len(selected) < per_relation:
            draw = rng.integers(0, len(users), size=max(50_000, 4 * (per_relation - len(selected))), dtype=np.int64)
            eligible = (user_degree[np.asarray(users[draw], dtype=np.int64)] >= 2) & (
                item_relation_degree[relation, np.asarray(items[draw], dtype=np.int64)] >= 2
            )
            selected.update(int(value) for value in draw[eligible])
            attempts += len(draw)
            if attempts > 50_000_000 and len(selected) < per_relation:
                raise RuntimeError(f"insufficient eligible holdout edges for {MARKS[relation]}")
        pool = np.fromiter(selected, dtype=np.int64)
        index = np.sort(rng.choice(pool, size=per_relation, replace=False))
        holdout.append(index)
        details[MARKS[relation]] = {
            "count": int(len(index)),
            "minimum_user_relation_degree_before_holdout": int(user_degree[np.asarray(users[index], dtype=np.int64)].min()),
            "minimum_item_relation_degree_before_holdout": int(item_relation_degree[relation, np.asarray(items[index], dtype=np.int64)].min()),
            "sampling_attempts": int(attempts),
        }
        print(
            f"holdout {MARKS[relation]}: selected={len(index):,} attempts={attempts:,}",
            flush=True,
        )
    return holdout, user_offsets, details


def observed_items(items: np.ndarray, offsets: np.ndarray, uid: int) -> np.ndarray:
    left = int(offsets[uid])
    right = int(offsets[uid + 1])
    return np.asarray(items[left:right], dtype=np.int64)


def build_candidates(edge_arrays, holdout, user_offsets, train_items, negatives: int, seed: int):
    """Build fixed candidates and filter all observed positives for (u,r)."""
    rng = np.random.default_rng(seed)
    query_user = []
    query_item = []
    query_relation = []
    candidate_parts = []
    for relation, ((users, items), indices, offsets) in enumerate(zip(edge_arrays, holdout, user_offsets)):
        for edge_index in indices:
            uid = int(users[edge_index])
            target = int(items[edge_index])
            known = observed_items(items, offsets, uid)
            chosen: list[np.ndarray] = []
            count = 0
            while count < negatives:
                draw = train_items[rng.integers(0, len(train_items), size=2 * (negatives - count), dtype=np.int64)]
                draw = draw[~is_member_sorted(draw, known)]
                if len(draw):
                    draw = np.unique(draw)
                    chosen.append(draw)
                    count += len(draw)
            negative = np.concatenate(chosen)[:negatives]
            query_user.append(uid)
            query_item.append(target)
            query_relation.append(relation)
            candidate_parts.append(negative.astype(np.int32, copy=False))
        print(
            f"candidates {MARKS[relation]}: queries={len(indices):,} negatives={negatives:,}",
            flush=True,
        )
    return {
        "user": np.asarray(query_user, dtype=np.int64),
        "target": np.asarray(query_item, dtype=np.int64),
        "relation": np.asarray(query_relation, dtype=np.int64),
        "negative": np.stack(candidate_parts),
    }


def metric_from_rank(rank: np.ndarray) -> dict:
    return {
        "queries": int(len(rank)),
        "hits_at_1": float(np.mean(rank <= 1)),
        "hits_at_5": float(np.mean(rank <= 5)),
        "hits_at_10": float(np.mean(rank <= 10)),
        "mrr": float(np.mean(1.0 / rank)),
        "mean_rank": float(rank.mean()),
        "median_rank": float(np.median(rank)),
    }


@torch.no_grad()
def evaluate_model(model: DistMult, candidates: dict, eval_batch_size: int, device: torch.device):
    model.eval()
    ranks = np.empty(len(candidates["user"]), dtype=np.int32)
    for relation in range(4):
        relation_rows = np.flatnonzero(candidates["relation"] == relation)
        for start in range(0, len(relation_rows), eval_batch_size):
            rows = relation_rows[start : start + eval_batch_size]
            users = torch.from_numpy(candidates["user"][rows]).to(device)
            targets = torch.from_numpy(candidates["target"][rows]).to(device)
            negatives = torch.from_numpy(candidates["negative"][rows].astype(np.int64)).to(device)
            positive_score = model.score(users, targets, relation)
            user_vector = model.user(users) * model.relation[relation]
            negative_score = (user_vector[:, None, :] * model.item(negatives)).sum(dim=-1)
            ranks[rows] = 1 + (negative_score >= positive_score[:, None]).sum(dim=1).cpu().numpy()
    result = {"overall": metric_from_rank(ranks), "by_relation": {}}
    for relation, mark in enumerate(MARKS):
        result["by_relation"][mark] = metric_from_rank(ranks[candidates["relation"] == relation])
    return result, ranks


def popularity_baseline(candidates: dict, item_relation_degree: np.ndarray):
    ranks = np.empty(len(candidates["user"]), dtype=np.int32)
    for relation in range(4):
        rows = np.flatnonzero(candidates["relation"] == relation)
        positive = item_relation_degree[relation, candidates["target"][rows]]
        negative = item_relation_degree[relation, candidates["negative"][rows]]
        ranks[rows] = 1 + (negative >= positive[:, None]).sum(axis=1)
    return {
        "overall": metric_from_rank(ranks),
        "by_relation": {mark: metric_from_rank(ranks[candidates["relation"] == relation]) for relation, mark in enumerate(MARKS)},
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.graph / "manifest.json").read_text())
    if manifest["data_contract"]["listen_used"] or manifest["data_contract"]["validation_or_test_edges_used"]:
        raise RuntimeError("graph data contract violates Gate 1C")
    item_ids = np.load(args.graph / "explicit_item_id.uint32.npy", mmap_mode="r")
    unique_train_users = np.load(args.graph / "unique_train_user_count.uint32.npy", mmap_mode="r")
    train_items = np.flatnonzero(unique_train_users > 0).astype(np.int64)
    item_relation_degree = np.load(args.graph / "unique_user_count_by_relation.uint32.npy", mmap_mode="r")
    edge_arrays = []
    edge_counts = []
    for mark in MARKS:
        row = manifest["relations"][mark]
        count = int(row["unique_user_item_relation_edges"])
        users = np.memmap(row["uid_file"], mode="r", dtype=np.uint32, shape=(count,))
        items = np.memmap(row["item_position_file"], mode="r", dtype=np.uint32, shape=(count,))
        if np.any(users[1:] < users[:-1]):
            raise RuntimeError(f"{mark} edge file is not sorted by user")
        edge_arrays.append((users, items))
        edge_counts.append(count)
    probabilities = np.asarray(edge_counts, dtype=np.float64)
    probabilities /= probabilities.sum()
    holdout, user_offsets, holdout_details = build_holdout(
        edge_arrays, item_relation_degree, args.holdout_per_relation, seed=4103
    )
    candidates = build_candidates(
        edge_arrays, holdout, user_offsets, train_items, args.negative_candidates, seed=4201
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "fixed_sampled_candidates.npz",
        user=candidates["user"].astype(np.uint32),
        target=candidates["target"].astype(np.uint32),
        relation=candidates["relation"].astype(np.uint8),
        negative=candidates["negative"].astype(np.uint32),
        **{f"holdout_edge_index_{mark}": holdout[i] for i, mark in enumerate(MARKS)},
    )
    popularity = popularity_baseline(candidates, item_relation_degree)
    candidate_count = args.negative_candidates + 1
    random_baseline = {
        "candidate_count": candidate_count,
        "hits_at_1": 1 / candidate_count,
        "hits_at_5": 5 / candidate_count,
        "hits_at_10": 10 / candidate_count,
        "mrr": sum(1 / rank for rank in range(1, candidate_count + 1)) / candidate_count,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal graph sanity check requires enabled GPU")
    runs = []
    for seed in args.seeds:
        seed_all(seed)
        model = DistMult(1_000_001, len(item_ids), args.embedding_dim).to(device)
        sparse_optimizer = torch.optim.SparseAdam([model.user.weight, model.item.weight], lr=0.02)
        relation_optimizer = torch.optim.Adam([model.relation], lr=0.005)
        rng = np.random.default_rng(seed + 301)
        trace = []
        running_loss = running_accuracy = 0.0
        for step in range(1, args.steps + 1):
            relation = int(rng.choice(4, p=probabilities))
            users_mm, items_mm = edge_arrays[relation]
            edge_index = rng.integers(0, edge_counts[relation], size=args.batch_size, dtype=np.int64)
            excluded = is_member_sorted(edge_index, holdout[relation])
            while excluded.any():
                edge_index[excluded] = rng.integers(0, edge_counts[relation], size=int(excluded.sum()), dtype=np.int64)
                excluded = is_member_sorted(edge_index, holdout[relation])
            user = torch.from_numpy(np.asarray(users_mm[edge_index], dtype=np.int64)).to(device)
            positive = torch.from_numpy(np.asarray(items_mm[edge_index], dtype=np.int64)).to(device)
            negative = torch.from_numpy(train_items[rng.integers(0, len(train_items), size=args.batch_size, dtype=np.int64)]).to(device)
            collision = negative == positive
            while bool(collision.any()):
                negative[collision] = torch.from_numpy(train_items[rng.integers(0, len(train_items), size=int(collision.sum()), dtype=np.int64)]).to(device)
                collision = negative == positive
            positive_score = model.score(user, positive, relation)
            negative_score = model.score(user, negative, relation)
            ranking = F.softplus(negative_score - positive_score).mean()
            regularization = 1e-6 * (
                model.user(user).square().mean() + model.item(positive).square().mean()
                + model.item(negative).square().mean() + model.relation[relation].square().mean()
            )
            loss = ranking + regularization
            sparse_optimizer.zero_grad(set_to_none=True)
            relation_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            sparse_optimizer.step()
            relation_optimizer.step()
            running_loss += float(ranking.detach())
            running_accuracy += float((positive_score > negative_score).float().mean())
            if step % 100 == 0 or step == args.steps:
                divisor = 100 if step % 100 == 0 else args.steps % 100
                trace.append({
                    "step": step,
                    "sampled_train_bpr_loss": running_loss / divisor,
                    "sampled_train_pairwise_accuracy": running_accuracy / divisor,
                })
                print(f"seed={seed} step={step} train_acc={trace[-1]['sampled_train_pairwise_accuracy']:.4f}", flush=True)
                running_loss = running_accuracy = 0.0
        metrics, ranks = evaluate_model(model, candidates, args.eval_batch_size, device)
        run = {
            "seed": seed,
            "training_trace": trace,
            "sampled_link_prediction": metrics,
            "relation_vector_cosine": (
                F.normalize(model.relation.detach(), dim=1) @ F.normalize(model.relation.detach(), dim=1).T
            ).cpu().numpy().tolist(),
        }
        runs.append(run)
        np.save(args.output / f"rank_seed_{seed}.int32.npy", ranks, allow_pickle=False)
        print(json.dumps({"seed": seed, "metrics": metrics}, ensure_ascii=False), flush=True)
        del model, sparse_optimizer, relation_optimizer
        torch.cuda.empty_cache()
    summary = {}
    for scope in ("overall",):
        summary[scope] = {}
        for metric in ("hits_at_1", "hits_at_5", "hits_at_10", "mrr", "mean_rank", "median_rank"):
            values = [run["sampled_link_prediction"][scope][metric] for run in runs]
            summary[scope][metric] = {
                "mean": float(np.mean(values)),
                "std_across_seeds": float(np.std(values, ddof=1)),
                "values": values,
            }
    by_relation = {}
    for mark in MARKS:
        by_relation[mark] = {}
        for metric in ("hits_at_1", "hits_at_5", "hits_at_10", "mrr", "mean_rank", "median_rank"):
            values = [run["sampled_link_prediction"]["by_relation"][mark][metric] for run in runs]
            by_relation[mark][metric] = {
                "mean": float(np.mean(values)),
                "std_across_seeds": float(np.std(values, ddof=1)),
                "values": values,
            }
    report = {
        "status": "complete_distmult_graph_task_sanity",
        "data_contract": {
            "source": str((args.graph / "manifest.json").resolve()),
            "global_train_period_only": True,
            "listen_used": False,
            "validation_or_test_period_interactions_used": False,
            "audio_embeddings_used": False,
            "relations_preserved": list(MARKS),
            "holdout_seen_during_training": False,
            "transductive_eligibility": "before holdout: user has >=2 edges under relation and item has >=2 users under relation",
        },
        "candidate_protocol": {
            "name": f"sampled-{candidate_count}-candidate filtered ranking",
            "true_items_per_query": 1,
            "negative_candidates_per_query": args.negative_candidates,
            "negative_source": "uniform items with global-train collaborative evidence",
            "filter": "remove every observed item for the same user and feedback relation",
            "not_full_catalog": True,
        },
        "holdout": holdout_details,
        "training": {
            "model": "Four-Relation DistMult Matrix Factorization",
            "embedding_dim": args.embedding_dim,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seeds": args.seeds,
            "negative_note": "random contrast only; not interpreted as true user rejection",
        },
        "baselines": {"random_expected": random_baseline, "relation_item_popularity": popularity},
        "runs": runs,
        "three_seed_summary": {"overall": summary["overall"], "by_relation": by_relation},
        "boundaries": {"gate2_started": False, "final_sid_assigned": False, "snmpp_or_hpn_trained": False},
    }
    atomic_json(args.output / "metrics.json", report)
    print(json.dumps({"output": str((args.output / 'metrics.json').resolve()), "summary": report["three_seed_summary"], "baselines": report["baselines"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
