from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "01_data"), str(ROOT / "02_model")])

from future_dataset import EmbedStore, FutureIterableDataset, collate_future, list_parquet_files
from hpn import FutureHPNPolicy, score_sid_candidates
from hpn_candidates import SidCandidatePool
from progress_utils import estimate_total_batches, format_float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose HPN recall bottlenecks without retraining.")
    parser.add_argument("--data_dir", default=str(ROOT / "01_data" / "processed" / "predictor_seq_data"))
    parser.add_argument("--embed_store", default=str(ROOT / "01_data" / "processed" / "raw_rqkmeans"))
    parser.add_argument("--dense_item2sid_npy", default=str(ROOT / "01_data" / "processed" / "raw_rqkmeans" / "dense_item2sid.npy"))
    parser.add_argument("--hpn_ckpt", default=str(ROOT / "artifacts" / "hpn" / "hpn.pt"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_rows", type=int, default=20000)
    parser.add_argument("--max_index_items", type=int, default=200000)
    parser.add_argument("--candidate_chunk_size", type=int, default=32768)
    parser.add_argument("--max_candidates", type=int, default=32)
    parser.add_argument("--oracle_negatives", type=int, default=99)
    parser.add_argument("--semantic_negatives", type=int, default=31)
    parser.add_argument("--prefix_top_paths", type=int, default=32)
    parser.add_argument("--prefix_branch_k", type=int, default=16)
    parser.add_argument("--popularity_rows", type=int, default=200000)
    parser.add_argument("--level_weight_sets", default="1,1,1,1;1,0.7,0.4,0.2;1,0.5,0.2,0.1;1,1,0,0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_hpn(path: str, item_dim: int, device: torch.device) -> FutureHPNPolicy:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = FutureHPNPolicy(
        item_dim=item_dim,
        d_model=int(cfg.get("d_model", 128)),
        max_seq_len=50,
        n_layer=int(cfg.get("n_layer", 2)),
        n_head=int(cfg.get("n_head", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        sid_levels=int(cfg.get("sid_levels", 4)),
        sid_vocab_size=int(cfg.get("sid_vocab_size", 256)),
        state_pooling=str(cfg.get("state_pooling", "last")),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def parse_weight_sets(text: str, levels: int) -> list[tuple[str, list[float]]]:
    out: list[tuple[str, list[float]]] = []
    for raw in text.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        weights = [float(item.strip()) for item in raw.split(",") if item.strip()]
        while len(weights) < levels:
            weights.append(0.0)
        weights = weights[:levels]
        name = ",".join(f"{w:g}" for w in weights)
        out.append((name, weights))
    return out or [("1,1,1,1", [1.0] * levels)]


def weighted_score_dense_ids(
    dense_item2sid: np.ndarray,
    dense_ids: np.ndarray,
    sid_logits: list[torch.Tensor],
    weights: list[float],
    device: torch.device,
) -> torch.Tensor:
    if dense_ids.size == 0:
        return torch.empty((sid_logits[0].shape[0], 0), dtype=torch.float32, device=device)
    sid_np = np.asarray(dense_item2sid[dense_ids], dtype=np.int64)[:, : len(sid_logits)]
    sid = torch.as_tensor(sid_np, dtype=torch.long, device=device)
    scores = torch.zeros((sid_logits[0].shape[0], sid.shape[0]), dtype=torch.float32, device=device)
    denom = 0.0
    for level, logits in enumerate(sid_logits):
        weight = float(weights[level]) if level < len(weights) else 0.0
        if weight == 0.0:
            continue
        log_probs = F.log_softmax(logits, dim=-1)
        token = sid[:, level].clamp(min=0, max=log_probs.shape[-1] - 1)
        scores = scores + weight * log_probs.index_select(dim=1, index=token)
        denom += abs(weight)
    return scores / max(denom, 1e-6)


def topk_weighted_pool(
    pool: SidCandidatePool,
    sid_logits: list[torch.Tensor],
    weights: list[float],
    max_candidates: int,
    device: torch.device,
) -> np.ndarray:
    batch_size = int(sid_logits[0].shape[0])
    top_scores = torch.full((batch_size, 0), -1e9, dtype=torch.float32, device=device)
    top_dense = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
    for start in range(0, pool.candidate_count, pool.chunk_size):
        dense_ids = pool.candidate_dense_ids[start : start + pool.chunk_size]
        scores = weighted_score_dense_ids(pool.dense_item2sid, dense_ids, sid_logits, weights, device)
        chunk_k = min(max_candidates, scores.shape[1])
        chunk_scores, chunk_pos = torch.topk(scores, k=chunk_k, dim=1)
        dense_tensor = torch.as_tensor(dense_ids, dtype=torch.long, device=device)
        chunk_dense = dense_tensor.unsqueeze(0).expand(batch_size, -1).gather(1, chunk_pos)
        merged_scores = torch.cat([top_scores, chunk_scores], dim=1)
        merged_dense = torch.cat([top_dense, chunk_dense], dim=1)
        keep_k = min(max_candidates, merged_scores.shape[1])
        top_scores, keep_pos = torch.topk(merged_scores, k=keep_k, dim=1)
        top_dense = merged_dense.gather(1, keep_pos)
    return top_dense.detach().cpu().numpy().astype(np.int64, copy=False)


class PrefixNegativeSampler:
    def __init__(self, dense_item2sid: np.ndarray, sid_levels: int, sid_vocab_size: int, prefix_len: int = 2, seed: int = 2026) -> None:
        self.sid_levels = int(sid_levels)
        self.sid_vocab_size = int(sid_vocab_size)
        self.prefix_len = max(1, min(int(prefix_len), self.sid_levels))
        self.rng = np.random.default_rng(int(seed))
        sid = np.asarray(dense_item2sid[1:, : self.sid_levels])
        valid = np.all(sid >= 0, axis=1)
        self.sid_valid = np.asarray(sid[valid], dtype=np.int64)
        codes = self._codes(self.sid_valid)
        order = np.argsort(codes, kind="mergesort")
        self.codes = codes[order]
        self.order = order.astype(np.int64, copy=False)

    def _codes(self, sid: np.ndarray) -> np.ndarray:
        sid = np.asarray(sid, dtype=np.int64)
        if sid.ndim == 1:
            sid = sid.reshape(1, -1)
        code = np.zeros((sid.shape[0],), dtype=np.int64)
        for level in range(self.prefix_len):
            code = code * self.sid_vocab_size + sid[:, level]
        return code

    def sample_batch(self, target_sid: np.ndarray, negatives: int) -> np.ndarray:
        target_sid = np.asarray(target_sid, dtype=np.int64)[:, : self.sid_levels]
        out = np.zeros((target_sid.shape[0], negatives + 1, self.sid_levels), dtype=np.int64)
        out[:, 0, :] = target_sid
        for row, target in enumerate(target_sid):
            target_key = tuple(int(x) for x in target.tolist())
            used = {target_key}
            code = int(self._codes(target)[0])
            lo = int(np.searchsorted(self.codes, code, side="left"))
            hi = int(np.searchsorted(self.codes, code, side="right"))
            col = 1
            attempts = 0
            while col <= negatives and attempts < max(negatives * 32, 128):
                attempts += 1
                if hi > lo:
                    local = int(self.order[int(self.rng.integers(lo, hi))])
                else:
                    local = int(self.rng.integers(0, self.sid_valid.shape[0]))
                cand = self.sid_valid[local]
                key = tuple(int(x) for x in cand.tolist())
                if key in used:
                    continue
                used.add(key)
                out[row, col, :] = cand
                col += 1
            while col <= negatives:
                cand = self.sid_valid[int(self.rng.integers(0, self.sid_valid.shape[0]))]
                out[row, col, :] = cand
                col += 1
        return out


def update_rank_metrics(prefix: str, scores: torch.Tensor, totals: dict[str, float]) -> None:
    order = scores.argsort(dim=1, descending=True)
    ranks = (order == 0).float().argmax(dim=1) + 1
    n = scores.shape[0]
    totals[f"{prefix}_top1"] += float((ranks == 1).sum().detach().cpu())
    totals[f"{prefix}_top5"] += float((ranks <= min(5, scores.shape[1])).sum().detach().cpu())
    totals[f"{prefix}_top10"] += float((ranks <= min(10, scores.shape[1])).sum().detach().cpu())
    totals[f"{prefix}_mrr_sum"] += float((1.0 / ranks.float()).sum().detach().cpu())
    totals[f"{prefix}_mean_rank_sum"] += float(ranks.float().sum().detach().cpu())
    totals[f"{prefix}_n"] += float(n)


def prefix_beam_hits(sid_logits: list[torch.Tensor], target_sid: torch.Tensor, top_paths: int, branch_k: int) -> dict[int, int]:
    hits = {level + 1: 0 for level in range(len(sid_logits))}
    batch_size = int(target_sid.shape[0])
    for row in range(batch_size):
        beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
        for level, logits in enumerate(sid_logits):
            log_probs = F.log_softmax(logits[row], dim=-1)
            vals, idxs = torch.topk(log_probs, k=min(branch_k, log_probs.shape[-1]))
            new_beams: list[tuple[tuple[int, ...], float]] = []
            for path, score in beams:
                for val, idx in zip(vals.tolist(), idxs.tolist()):
                    new_beams.append((path + (int(idx),), float(score + val)))
            new_beams.sort(key=lambda item: item[1], reverse=True)
            beams = new_beams[:top_paths]
            target_prefix = tuple(int(x) for x in target_sid[row, : level + 1].detach().cpu().tolist())
            if any(path == target_prefix for path, _ in beams):
                hits[level + 1] += 1
    return hits


def build_popular_topk(data_dir: str | Path, split: str, topk: int, max_rows: int) -> list[int]:
    counts: Counter[int] = Counter()
    seen = 0
    for file_path in list_parquet_files(Path(data_dir) / split):
        pf = pq.ParquetFile(file_path)
        if "target_dense_item_id" not in pf.schema_arrow.names:
            continue
        for batch in pf.iter_batches(columns=["target_dense_item_id"], batch_size=8192):
            arr = np.asarray(batch.column(0).to_pylist(), dtype=np.int64)
            for item in arr.tolist():
                if item > 0:
                    counts[int(item)] += 1
            seen += int(arr.shape[0])
            if max_rows > 0 and seen >= max_rows:
                return [item for item, _ in counts.most_common(topk)]
    return [item for item, _ in counts.most_common(topk)]


def finalize_rank_metrics(prefix: str, totals: dict[str, float]) -> dict[str, float]:
    n = max(totals.get(f"{prefix}_n", 0.0), 1.0)
    return {
        "top1": totals.get(f"{prefix}_top1", 0.0) / n,
        "top5": totals.get(f"{prefix}_top5", 0.0) / n,
        "top10": totals.get(f"{prefix}_top10", 0.0) / n,
        "mrr": totals.get(f"{prefix}_mrr_sum", 0.0) / n,
        "mean_rank": totals.get(f"{prefix}_mean_rank_sum", 0.0) / n,
        "examples": int(n),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    store = EmbedStore(args.embed_store)
    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
    sid_levels = int(dense_item2sid.shape[1])
    hpn = load_hpn(args.hpn_ckpt, store.dim, device)
    pool = SidCandidatePool(dense_item2sid, max_candidate_items=args.max_index_items, chunk_size=args.candidate_chunk_size)
    pool_set = set(map(int, pool.candidate_dense_ids.tolist()))
    valid_dense = np.nonzero(np.all(np.asarray(dense_item2sid[1:]) >= 0, axis=1))[0].astype(np.int64) + 1
    rng = np.random.default_rng(int(args.seed))
    semantic_sampler = PrefixNegativeSampler(dense_item2sid, sid_levels=sid_levels, sid_vocab_size=256, prefix_len=2, seed=args.seed)
    weight_sets = parse_weight_sets(args.level_weight_sets, sid_levels)
    popular_top = set(build_popular_topk(args.data_dir, "train", args.max_candidates, args.popularity_rows))

    dataset = FutureIterableDataset(args.data_dir, split=args.split, max_rows=args.max_rows, mapping_root=store.root)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=lambda rows: collate_future(rows, store))
    total_batches = estimate_total_batches(args.data_dir, args.split, args.batch_size, args.max_rows)

    n = 0
    target_in_pool = 0
    popular_hits = 0
    recent_hits = 0
    greedy_prefix_hits = {level + 1: 0 for level in range(sid_levels)}
    beam_prefix_hits = {level + 1: 0 for level in range(sid_levels)}
    token_rank_hits = {level + 1: {1: 0, 5: 0, 10: 0} for level in range(sid_levels)}
    weighted_hits = {name: 0 for name, _ in weight_sets}
    weighted_recalled_ranks: dict[str, list[int]] = {name: [] for name, _ in weight_sets}
    rank_totals: dict[str, float] = Counter()

    pbar = tqdm(loader, total=total_batches, desc=f"[diagnose hpn {args.split}]", unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            batch = {key: value.to(device) for key, value in batch.items()}
            out = hpn(batch)
            sid_logits = out["sid_logits"]
            if not isinstance(sid_logits, list):
                raise TypeError("HPN sid_logits must be a list.")
            target_sid = batch["target_sid"].long()
            target_dense = batch["target_dense_item_id"].detach().cpu().numpy().astype(np.int64)
            batch_size = int(target_dense.shape[0])
            n += batch_size

            target_in_pool += sum(int(item) in pool_set for item in target_dense.tolist())
            popular_hits += sum(int(item) in popular_top for item in target_dense.tolist())
            hist = batch["history_dense_item_ids"].detach().cpu().numpy().astype(np.int64)
            for row, item in enumerate(target_dense.tolist()):
                recent = [int(x) for x in hist[row].tolist() if int(x) > 0][-args.max_candidates:]
                if int(item) in set(recent):
                    recent_hits += 1

            pred_tokens = torch.stack([logits.argmax(dim=-1) for logits in sid_logits], dim=1)
            prefix_equal = torch.ones(batch_size, dtype=torch.bool, device=device)
            for level in range(sid_levels):
                logits = sid_logits[level]
                target_l = target_sid[:, level].clamp(min=0, max=logits.shape[-1] - 1)
                true_score = logits.gather(1, target_l.unsqueeze(1)).squeeze(1)
                rank = (logits > true_score.unsqueeze(1)).sum(dim=1) + 1
                for k in (1, 5, 10):
                    token_rank_hits[level + 1][k] += int((rank <= k).sum().detach().cpu())
                prefix_equal &= pred_tokens[:, level] == target_l
                greedy_prefix_hits[level + 1] += int(prefix_equal.sum().detach().cpu())

            beam_hits = prefix_beam_hits(sid_logits, target_sid, args.prefix_top_paths, args.prefix_branch_k)
            for level, count in beam_hits.items():
                beam_prefix_hits[level] += int(count)

            for name, weights in weight_sets:
                cand_dense = topk_weighted_pool(pool, sid_logits, weights, args.max_candidates, device)
                for row, item in enumerate(target_dense.tolist()):
                    hits = np.where(cand_dense[row] == int(item))[0]
                    if hits.size > 0:
                        weighted_hits[name] += 1
                        weighted_recalled_ranks[name].append(int(hits[0]) + 1)

            # Oracle random negatives: true target is injected at candidate index 0.
            neg_dense = rng.choice(valid_dense, size=(batch_size, args.oracle_negatives), replace=True)
            target_col = target_sid.detach().cpu().numpy()[:, None, :]
            random_sid = np.concatenate([target_col, dense_item2sid[neg_dense][:, :, :sid_levels]], axis=1)
            random_scores = score_sid_candidates(sid_logits, torch.as_tensor(random_sid, dtype=torch.long, device=device))
            update_rank_metrics("oracle_random", random_scores, rank_totals)

            semantic_sid = semantic_sampler.sample_batch(target_sid.detach().cpu().numpy(), args.semantic_negatives)
            semantic_scores = score_sid_candidates(sid_logits, torch.as_tensor(semantic_sid, dtype=torch.long, device=device))
            update_rank_metrics("oracle_semantic_prefix2", semantic_scores, rank_totals)

            pbar.set_postfix(
                pool_hit=format_float(target_in_pool / max(n, 1)),
                hpn=format_float(weighted_hits[weight_sets[0][0]] / max(n, 1)),
                examples=n,
                refresh=False,
            )

    token_metrics = {}
    for level in range(1, sid_levels + 1):
        token_metrics[f"level_{level}"] = {f"top{k}": token_rank_hits[level][k] / max(n, 1) for k in (1, 5, 10)}

    weighted_metrics = {}
    for name, _weights in weight_sets:
        ranks = weighted_recalled_ranks[name]
        weighted_metrics[name] = {
            "recall_at_k": weighted_hits[name] / max(n, 1),
            "hits": int(weighted_hits[name]),
            "mean_rank_if_recalled": float(np.mean(ranks)) if ranks else None,
        }

    target_in_pool_rate = target_in_pool / max(n, 1)
    result = {
        "config": {
            "data_dir": str(args.data_dir),
            "split": args.split,
            "max_rows": args.max_rows,
            "hpn_ckpt": str(args.hpn_ckpt),
            "candidate_pool_size": pool.candidate_count,
            "max_candidates": args.max_candidates,
            "oracle_negatives": args.oracle_negatives,
            "semantic_negatives": args.semantic_negatives,
            "popular_train_rows": args.popularity_rows,
        },
        "examples": n,
        "coverage": {
            "target_in_candidate_pool": target_in_pool,
            "target_in_candidate_pool_rate": target_in_pool_rate,
            "random_expected_recall_at_k_within_pool": target_in_pool_rate * min(args.max_candidates, max(pool.candidate_count, 1)) / max(pool.candidate_count, 1),
        },
        "baselines": {
            "popular_topk_recall": popular_hits / max(n, 1),
            "recent_history_topk_recall": recent_hits / max(n, 1),
        },
        "hpn_token_rank": token_metrics,
        "hpn_greedy_prefix_exact": {f"prefix_{level}": greedy_prefix_hits[level] / max(n, 1) for level in range(1, sid_levels + 1)},
        "hpn_beam_prefix_recall": {
            f"prefix_{level}_top{args.prefix_top_paths}": beam_prefix_hits[level] / max(n, 1) for level in range(1, sid_levels + 1)
        },
        "hpn_catalog_recall_by_level_weights": weighted_metrics,
        "oracle_candidate_ranking": {
            "random_negatives": finalize_rank_metrics("oracle_random", rank_totals),
            "semantic_prefix2_negatives": finalize_rank_metrics("oracle_semantic_prefix2", rank_totals),
            "random_top1_baseline": 1.0 / (args.oracle_negatives + 1),
            "semantic_top1_random_baseline": 1.0 / (args.semantic_negatives + 1),
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
