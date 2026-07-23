from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "01_data"), str(ROOT / "02_model")])

from future_dataset import EmbedStore, FutureIterableDataset, collate_future
from hpn import FutureHPNPolicy, beam_decode_sid_paths
from predictor import FuturePredictor
from soft_state import SoftStateBuilder
from value import ValueHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HPN top-k candidates reranked by future predictor and value.")
    parser.add_argument("--data_dir", default=str(ROOT / "artifacts" / "future_data"))
    parser.add_argument("--embed_store", default=str(ROOT / "artifacts" / "embed_store"))
    parser.add_argument("--dense_item2sid_npy", required=True)
    parser.add_argument("--dense2orig_npy", required=True)
    parser.add_argument("--hpn_ckpt", default=str(ROOT / "artifacts" / "hpn" / "hpn.pt"))
    parser.add_argument("--predictor_ckpt", default=str(ROOT / "artifacts" / "predictor" / "future_predictor.pt"))
    parser.add_argument("--value_ckpt", default=str(ROOT / "artifacts" / "value" / "future_value.pt"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--top_sid_paths", type=int, default=32)
    parser.add_argument("--branch_k", type=int, default=16)
    parser.add_argument("--max_candidates", type=int, default=64)
    parser.add_argument("--max_items_per_sid", type=int, default=4)
    parser.add_argument("--max_index_items", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def masked_norm(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.float()
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * valid).sum(dim=1, keepdim=True) / denom
    var = (((x - mean) * valid) ** 2).sum(dim=1, keepdim=True) / denom
    return (x - mean) / var.sqrt().clamp_min(1e-6)


class SidPathIndex:
    def __init__(self, dense_item2sid: np.ndarray, max_items_per_sid: int = 4, max_index_items: int = 0) -> None:
        self.path_to_dense: dict[tuple[int, ...], list[int]] = {}
        stop = dense_item2sid.shape[0] if max_index_items <= 0 else min(dense_item2sid.shape[0], max_index_items + 1)
        for dense_id in range(1, stop):
            sid = dense_item2sid[dense_id]
            if np.any(sid < 0):
                continue
            path = tuple(int(x) for x in sid.tolist())
            for prefix_len in range(0, len(path) + 1):
                key = path[:prefix_len]
                bucket = self.path_to_dense.setdefault(key, [])
                if len(bucket) < max_items_per_sid:
                    bucket.append(int(dense_id))

    def candidates_for_paths(self, paths: list[list[int]], scores: list[float], max_candidates: int) -> tuple[list[int], list[float]]:
        dense_ids: list[int] = []
        hpn_scores: list[float] = []
        used = set()
        for path, score in zip(paths, scores):
            clean_path = tuple(int(x) for x in path)
            for prefix_len in range(len(clean_path), -1, -1):
                bucket = self.path_to_dense.get(clean_path[:prefix_len], [])
                for dense_id in bucket:
                    if dense_id in used:
                        continue
                    dense_ids.append(dense_id)
                    hpn_scores.append(float(score))
                    used.add(dense_id)
                    if len(dense_ids) >= max_candidates:
                        return dense_ids, hpn_scores
                if bucket:
                    break
        return dense_ids, hpn_scores


def load_hpn(path: str, item_dim: int, device: torch.device) -> FutureHPNPolicy:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = FutureHPNPolicy(
        item_dim=item_dim,
        d_model=int(cfg.get("d_model", 128)),
        max_seq_len=int(cfg.get("max_seq_len", 50)),
        n_layer=int(cfg.get("n_layer", 2)),
        n_head=int(cfg.get("n_head", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        sid_levels=int(cfg.get("sid_levels", 4)),
        sid_vocab_size=int(cfg.get("sid_vocab_size", 256)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_predictor(path: str, item_dim: int, device: torch.device) -> FuturePredictor:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = FuturePredictor(
        item_dim=item_dim,
        d_model=int(cfg.get("d_model", 128)),
        max_seq_len=int(cfg.get("max_seq_len", 50)),
        n_layer=int(cfg.get("n_layer", 2)),
        n_head=int(cfg.get("n_head", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    store = EmbedStore(args.embed_store)
    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
    dense2orig = np.load(args.dense2orig_npy, mmap_mode="r")
    index = SidPathIndex(dense_item2sid, max_items_per_sid=args.max_items_per_sid, max_index_items=args.max_index_items)

    hpn = load_hpn(args.hpn_ckpt, store.dim, device)
    predictor = load_predictor(args.predictor_ckpt, store.dim, device)
    value_ckpt = torch.load(args.value_ckpt, map_location="cpu")
    d_model = int(value_ckpt.get("predictor_config", {}).get("d_model", 128))
    soft_state = SoftStateBuilder(item_dim=store.dim, d_model=d_model).to(device)
    value_head = ValueHead(d_model=d_model).to(device)
    soft_state.load_state_dict(value_ckpt["soft_state"])
    value_head.load_state_dict(value_ckpt["value_head"])
    soft_state.eval()
    value_head.eval()

    dataset = FutureIterableDataset(args.data_dir, split=args.split, max_rows=args.max_rows)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=lambda rows: collate_future(rows, store))

    n = 0
    recall = 0
    top1 = 0
    rank_sum = 0.0
    empty = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            hpn_out = hpn(batch)
            paths_t, path_scores_t = beam_decode_sid_paths(hpn_out["sid_logits"], args.top_sid_paths, args.branch_k)
            candidate_dense = []
            hpn_scores = []
            masks = []
            for row_idx in range(paths_t.shape[0]):
                paths = paths_t[row_idx].detach().cpu().tolist()
                scores = path_scores_t[row_idx].detach().cpu().tolist()
                dense_ids, scores_out = index.candidates_for_paths(paths, scores, args.max_candidates)
                if not dense_ids:
                    empty += 1
                    dense_ids = [0]
                    scores_out = [-1e9]
                mask = [1] * len(dense_ids)
                while len(dense_ids) < args.max_candidates:
                    dense_ids.append(0)
                    scores_out.append(-1e9)
                    mask.append(0)
                candidate_dense.append(dense_ids[: args.max_candidates])
                hpn_scores.append(scores_out[: args.max_candidates])
                masks.append(mask[: args.max_candidates])

            candidate_dense_np = np.asarray(candidate_dense, dtype=np.int64)
            candidate_orig = np.zeros_like(candidate_dense_np)
            valid_dense = (candidate_dense_np > 0) & (candidate_dense_np < len(dense2orig))
            candidate_orig[valid_dense] = dense2orig[candidate_dense_np[valid_dense]]
            candidate_features = torch.tensor(store.lookup(candidate_orig.astype(np.uint32)), dtype=torch.float32, device=device)
            hpn_score = torch.tensor(hpn_scores, dtype=torch.float32, device=device)
            mask = torch.tensor(masks, dtype=torch.bool, device=device)

            pred = predictor(batch, candidate_features)
            soft = soft_state(pred["state_emb"], candidate_features, pred)
            next_value = value_head(soft["next_state_emb"].reshape(-1, d_model)).view(candidate_features.shape[0], candidate_features.shape[1])
            regret_risk = pred["regret_probs"][..., 1:].sum(dim=-1)
            future_score = pred["predicted_reward"] + float(args.gamma) * next_value - float(args.eta) * regret_risk
            final_logit = masked_norm(hpn_score.clamp_min(-1e4), mask) + masked_norm(future_score, mask)
            final_logit = final_logit.masked_fill(~mask, -1e9)

            target_dense = batch["target_dense_item_id"].detach().cpu().numpy()
            order = final_logit.argsort(dim=1, descending=True).detach().cpu().numpy()
            for i in range(candidate_dense_np.shape[0]):
                n += 1
                target = int(target_dense[i])
                hits = np.where(candidate_dense_np[i] == target)[0]
                if hits.size == 0:
                    continue
                recall += 1
                rank_pos = int(np.where(order[i] == hits[0])[0][0]) + 1
                rank_sum += rank_pos
                if rank_pos == 1:
                    top1 += 1

    metrics = {
        "examples": n,
        "empty_candidate_rows": empty,
        "hpn_recall": recall / max(n, 1),
        "future_top1_given_all": top1 / max(n, 1),
        "future_top1_given_recalled": top1 / max(recall, 1),
        "mean_rank_given_recalled": rank_sum / max(recall, 1),
        "sid_paths_indexed": len(index.path_to_dense),
        "max_candidates": args.max_candidates,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
