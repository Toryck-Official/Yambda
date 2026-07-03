from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "01_data"), str(ROOT / "02_model")])

from future_dataset import EmbedStore, FutureIterableDataset, collate_future
from progress_utils import estimate_total_batches, format_float
from predictor import FuturePredictor


RESPONSE_NAMES = ["listen", "like", "dislike", "unlike", "undislike"]
REGRET_NAMES = ["none", "low_play", "dislike", "unlike"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose action-conditioned predictor contrast on oracle candidates.")
    parser.add_argument("--data_dir", default=str(ROOT / "01_data" / "processed" / "predictor_seq_data"))
    parser.add_argument("--embed_store", default=str(ROOT / "01_data" / "processed" / "raw_rqkmeans"))
    parser.add_argument("--ckpt", default=str(ROOT / "artifacts" / "predictor_weight_10p_e1" / "future_predictor.pt"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_rows", type=int, default=20000)
    parser.add_argument("--candidate_k", type=int, default=8)
    parser.add_argument(
        "--candidate_negative_mode",
        choices=["inbatch", "history", "history_inbatch", "semantic_inbatch", "history_semantic_inbatch"],
        default="history_inbatch",
        help="Negative action source for oracle candidate contrast.",
    )
    parser.add_argument("--semantic_prefix_level", type=int, default=2, help="Prefix length for semantic in-batch negatives.")
    parser.add_argument("--seed", type=int, default=2026)
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


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_predictor(path: str, item_dim: int, device: torch.device) -> FuturePredictor:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = FuturePredictor(
        item_dim=int(ckpt.get("item_dim", item_dim)),
        d_model=int(cfg.get("d_model", 128)),
        max_seq_len=50,
        n_layer=int(cfg.get("n_layer", 2)),
        n_head=int(cfg.get("n_head", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        state_pooling=str(cfg.get("state_pooling", "last")),
        sid_levels=int(cfg.get("sid_levels", 4)),
        sid_vocab_size=int(cfg.get("sid_vocab_size", 256)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model


def _roll_action_features(action_features: torch.Tensor, shift: int) -> torch.Tensor:
    batch_size = int(action_features.shape[0])
    if batch_size <= 1:
        return action_features.clone()
    return torch.roll(action_features, shifts=int(shift) % batch_size, dims=0)


def _history_negative_features(batch: dict[str, torch.Tensor], slot: int, fallback: torch.Tensor) -> tuple[torch.Tensor, int]:
    history_features = batch.get("history_features")
    history_dense = batch.get("history_dense_item_ids")
    target_dense = batch.get("target_dense_item_id")
    if history_features is None or history_dense is None or target_dense is None:
        return fallback, 0
    negative = fallback.clone()
    used = 0
    valid = (history_dense > 0) & (history_dense != target_dense.unsqueeze(1))
    for row_idx in range(int(history_dense.shape[0])):
        positions = torch.nonzero(valid[row_idx], as_tuple=False).flatten()
        if int(positions.numel()) == 0:
            continue
        pick = positions[-1 - ((int(slot) - 1) % int(positions.numel()))]
        negative[row_idx] = history_features[row_idx, pick]
        used += 1
    return negative, used


def _semantic_inbatch_negative_features(
    batch: dict[str, torch.Tensor],
    action_features: torch.Tensor,
    slot: int,
    prefix_level: int,
    fallback: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    target_sid = batch.get("target_sid")
    if target_sid is None or target_sid.dim() != 2 or int(target_sid.shape[0]) <= 1:
        return fallback, 0
    prefix = max(1, min(int(prefix_level), int(target_sid.shape[1])))
    negative = fallback.clone()
    used = 0
    sid_prefix = target_sid[:, :prefix]
    for row_idx in range(int(target_sid.shape[0])):
        same = (sid_prefix == sid_prefix[row_idx].unsqueeze(0)).all(dim=1)
        same[row_idx] = False
        candidates = torch.nonzero(same, as_tuple=False).flatten()
        if int(candidates.numel()) == 0:
            continue
        pick = candidates[(int(slot) - 1) % int(candidates.numel())]
        negative[row_idx] = action_features[pick]
        used += 1
    return negative, used


def make_action_conditioned_candidates(
    batch: dict[str, torch.Tensor],
    candidate_k: int,
    negative_mode: str,
    semantic_prefix_level: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    candidate_k = max(int(candidate_k), 1)
    action_features = batch["action_features"]
    parts = [action_features]
    history_used = 0
    semantic_used = 0
    fallback_used = 0
    total_negatives = int(action_features.shape[0]) * max(candidate_k - 1, 0)

    for slot in range(1, candidate_k):
        fallback = _roll_action_features(action_features, slot)
        negative = fallback
        if negative_mode == "history":
            negative, used = _history_negative_features(batch, slot, fallback)
            history_used += used
            fallback_used += int(action_features.shape[0]) - used
        elif negative_mode == "history_inbatch":
            if slot % 2 == 1:
                negative, used = _history_negative_features(batch, slot, fallback)
                history_used += used
                fallback_used += int(action_features.shape[0]) - used
            else:
                fallback_used += int(action_features.shape[0])
        elif negative_mode == "semantic_inbatch":
            negative, used = _semantic_inbatch_negative_features(batch, action_features, slot, semantic_prefix_level, fallback)
            semantic_used += used
            fallback_used += int(action_features.shape[0]) - used
        elif negative_mode == "history_semantic_inbatch":
            pattern = slot % 3
            if pattern == 1:
                negative, used = _history_negative_features(batch, slot, fallback)
                history_used += used
                fallback_used += int(action_features.shape[0]) - used
            elif pattern == 2:
                negative, used = _semantic_inbatch_negative_features(batch, action_features, slot, semantic_prefix_level, fallback)
                semantic_used += used
                fallback_used += int(action_features.shape[0]) - used
            else:
                fallback_used += int(action_features.shape[0])
        else:
            fallback_used += int(action_features.shape[0])
        parts.append(negative)

    denom = max(float(total_negatives), 1.0)
    return torch.stack(parts, dim=1), {
        "history_negative_share": float(history_used) / denom,
        "semantic_negative_share": float(semantic_used) / denom,
        "fallback_negative_share": float(fallback_used) / denom,
    }

def expected_formula_reward(response_probs: torch.Tensor, play_ratio: torch.Tensor) -> torch.Tensor:
    listen = response_probs[..., 0]
    like = response_probs[..., 1]
    dislike = response_probs[..., 2]
    unlike = response_probs[..., 3]
    undislike = response_probs[..., 4]
    play = play_ratio.clamp(0.0, 1.0) * listen
    reward = play + 0.8 * like - 1.2 * dislike - 0.6 * unlike + 0.2 * undislike
    return reward.clamp(-2.0, 2.0)


def update_rank_stats(score: torch.Tensor, stats: dict[str, float]) -> None:
    order = score.argsort(dim=1, descending=True)
    ranks = (order == 0).nonzero(as_tuple=False)[:, 1] + 1
    batch_n = int(score.shape[0])
    true_score = score[:, 0]
    neg_score = score[:, 1:].mean(dim=1)
    pairwise_win = (true_score.unsqueeze(1) > score[:, 1:]).float().mean(dim=1)

    stats["n"] += batch_n
    stats["top1"] += float((ranks == 1).sum().detach().cpu())
    stats["top3"] += float((ranks <= 3).sum().detach().cpu())
    stats["rank_sum"] += float(ranks.float().sum().detach().cpu())
    stats["rr_sum"] += float((1.0 / ranks.float()).sum().detach().cpu())
    stats["true_sum"] += float(true_score.sum().detach().cpu())
    stats["neg_sum"] += float(neg_score.sum().detach().cpu())
    stats["win_sum"] += float(pairwise_win.sum().detach().cpu())


def finish_rank_stats(stats: dict[str, float]) -> dict[str, float]:
    n = max(stats["n"], 1.0)
    true_mean = stats["true_sum"] / n
    neg_mean = stats["neg_sum"] / n
    return {
        "top1": stats["top1"] / n,
        "top3": stats["top3"] / n,
        "mrr": stats["rr_sum"] / n,
        "mean_rank": stats["rank_sum"] / n,
        "true_mean": true_mean,
        "negative_mean": neg_mean,
        "true_minus_negative": true_mean - neg_mean,
        "pairwise_true_gt_negative": stats["win_sum"] / n,
    }


def new_rank_stats() -> dict[str, float]:
    return {
        "n": 0.0,
        "top1": 0.0,
        "top3": 0.0,
        "rank_sum": 0.0,
        "rr_sum": 0.0,
        "true_sum": 0.0,
        "neg_sum": 0.0,
        "win_sum": 0.0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    store = EmbedStore(args.embed_store)
    model = load_predictor(args.ckpt, store.dim, device)

    dataset = FutureIterableDataset(args.data_dir, split=args.split, max_rows=args.max_rows, mapping_root=store.root)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=lambda rows: collate_future(rows, store))
    total_batches = estimate_total_batches(args.data_dir, args.split, args.batch_size, args.max_rows)

    rank_stats = {
        "candidate_logit": new_rank_stats(),
        "predicted_reward": new_rank_stats(),
        "predicted_future_return": new_rank_stats(),
        "future_regret_safe_score": new_rank_stats(),
        "expected_formula_reward": new_rank_stats(),
        "play_ratio": new_rank_stats(),
        "positive_response_score": new_rank_stats(),
        "negative_response_risk": new_rank_stats(),
        "low_regret_score": new_rank_stats(),
    }
    for name in RESPONSE_NAMES:
        rank_stats[f"response_prob_{name}"] = new_rank_stats()
    for name in REGRET_NAMES:
        rank_stats[f"regret_prob_{name}"] = new_rank_stats()

    n = 0
    source_share_sums = {"history_negative_share": 0.0, "semantic_negative_share": 0.0, "fallback_negative_share": 0.0}
    source_share_batches = 0
    response_true_sum = torch.zeros(5, dtype=torch.float64)
    response_neg_sum = torch.zeros(5, dtype=torch.float64)
    regret_true_sum = torch.zeros(4, dtype=torch.float64)
    regret_neg_sum = torch.zeros(4, dtype=torch.float64)

    pbar = tqdm(loader, total=total_batches, desc=f"[predictor contrast {args.split}]", unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            batch = {key: value.to(device) for key, value in batch.items()}
            candidates, source_stats = make_action_conditioned_candidates(
                batch,
                args.candidate_k,
                args.candidate_negative_mode,
                args.semantic_prefix_level,
            )
            for key in source_share_sums:
                source_share_sums[key] += float(source_stats[key])
            source_share_batches += 1
            out = model(batch, candidates)
            response = out["response_probs"]
            regret = out["regret_probs"]
            predicted_reward = out["predicted_reward"]
            play_ratio = out["predicted_play_ratio"]
            expected_reward = expected_formula_reward(response, play_ratio)
            positive_score = response[..., 0] + response[..., 1] + response[..., 4]
            negative_risk = response[..., 2] + response[..., 3]
            low_regret_score = -regret[..., 1:].sum(dim=-1)

            scores = {
                "candidate_logit": out["candidate_logit"],
                "predicted_reward": predicted_reward,
                "predicted_future_return": out.get("predicted_future_return", predicted_reward),
                "future_regret_safe_score": -out.get("future_regret_prob", predicted_reward.new_zeros(predicted_reward.shape)),
                "expected_formula_reward": expected_reward,
                "play_ratio": play_ratio,
                "positive_response_score": positive_score,
                "negative_response_risk": -negative_risk,
                "low_regret_score": low_regret_score,
            }
            for idx, name in enumerate(RESPONSE_NAMES):
                scores[f"response_prob_{name}"] = response[..., idx]
            for idx, name in enumerate(REGRET_NAMES):
                scores[f"regret_prob_{name}"] = regret[..., idx]

            for name, score in scores.items():
                update_rank_stats(score, rank_stats[name])

            batch_n = int(response.shape[0])
            n += batch_n
            response_true_sum += response[:, 0, :].detach().cpu().double().sum(dim=0)
            response_neg_sum += response[:, 1:, :].detach().cpu().double().mean(dim=1).sum(dim=0)
            regret_true_sum += regret[:, 0, :].detach().cpu().double().sum(dim=0)
            regret_neg_sum += regret[:, 1:, :].detach().cpu().double().mean(dim=1).sum(dim=0)

            main_stats = finish_rank_stats(rank_stats["expected_formula_reward"])
            pbar.set_postfix(
                top1=format_float(main_stats["top1"]),
                mrr=format_float(main_stats["mrr"]),
                diff=format_float(main_stats["true_minus_negative"]),
                examples=n,
                refresh=False,
            )

    response_means = {}
    for idx, name in enumerate(RESPONSE_NAMES):
        true_mean = float(response_true_sum[idx] / max(n, 1))
        neg_mean = float(response_neg_sum[idx] / max(n, 1))
        response_means[name] = {
            "true_mean": true_mean,
            "negative_mean": neg_mean,
            "true_minus_negative": true_mean - neg_mean,
        }
    regret_means = {}
    for idx, name in enumerate(REGRET_NAMES):
        true_mean = float(regret_true_sum[idx] / max(n, 1))
        neg_mean = float(regret_neg_sum[idx] / max(n, 1))
        regret_means[name] = {
            "true_mean": true_mean,
            "negative_mean": neg_mean,
            "true_minus_negative": true_mean - neg_mean,
        }

    metrics = {
        "examples": n,
        "candidate_k": int(args.candidate_k),
        "candidate_negative_mode": args.candidate_negative_mode,
        "semantic_prefix_level": int(args.semantic_prefix_level),
        "history_negative_share": source_share_sums["history_negative_share"] / max(source_share_batches, 1),
        "semantic_negative_share": source_share_sums["semantic_negative_share"] / max(source_share_batches, 1),
        "fallback_negative_share": source_share_sums["fallback_negative_share"] / max(source_share_batches, 1),
        "oracle_true_index": 0,
        "random_top1_baseline": 1.0 / max(int(args.candidate_k), 1),
        "random_top3_baseline": min(3, int(args.candidate_k)) / max(int(args.candidate_k), 1),
        "score_metrics": {name: finish_rank_stats(stats) for name, stats in rank_stats.items()},
        "response_prob_means": response_means,
        "regret_prob_means": regret_means,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
