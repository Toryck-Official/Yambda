from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "01_data"), str(ROOT / "02_model"), str(ROOT / "04_eval")])

from future_dataset import EmbedStore, FutureIterableDataset, collate_future
from progress_utils import estimate_total_batches, format_float
from predictor import FuturePredictor
from soft_state import SoftStateBuilder
from value import ValueHead


REGRET_CLASSES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose predictor+value with oracle candidates.")
    parser.add_argument("--data_dir", default=str(ROOT / "01_data" / "processed" / "predictor_seq_data"))
    parser.add_argument("--embed_store", default=str(ROOT / "01_data" / "processed" / "raw_rqkmeans"))
    parser.add_argument("--predictor_ckpt", default=str(ROOT / "artifacts" / "predictor_weight_10p_e1" / "future_predictor.pt"))
    parser.add_argument("--predictor_ckpts", default="")
    parser.add_argument("--value_ckpt", default=str(ROOT / "artifacts" / "value_bellman_10p_inbatch" / "future_value.pt"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_rows", type=int, default=20000)
    parser.add_argument("--candidate_k", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--bayes_samples", type=int, default=4)
    parser.add_argument("--reward_sample_mode", choices=["sampled_formula", "predictor_mean"], default="sampled_formula")
    parser.add_argument("--sample_regret_from_response", type=int, default=1)
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
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def resolve_predictor_ckpts(args: argparse.Namespace, value_ckpt: dict) -> list[str]:
    if args.predictor_ckpts:
        ckpts = [item.strip() for item in args.predictor_ckpts.split(",") if item.strip()]
    elif value_ckpt.get("predictor_ckpts"):
        ckpts = [str(item) for item in value_ckpt["predictor_ckpts"]]
    else:
        ckpts = [str(args.predictor_ckpt)]
    if not ckpts:
        raise RuntimeError("No predictor checkpoint was provided.")
    return ckpts


def make_oracle_inbatch_candidates(action_features: torch.Tensor, candidate_k: int) -> torch.Tensor:
    candidate_k = max(int(candidate_k), 1)
    parts = [action_features]
    batch_size = action_features.shape[0]
    for shift in range(1, candidate_k):
        parts.append(torch.roll(action_features, shifts=shift % max(batch_size, 1), dims=0))
    return torch.stack(parts, dim=1)


def sample_one_hot(probs: torch.Tensor) -> torch.Tensor:
    n_class = probs.shape[-1]
    flat = probs.reshape(-1, n_class).clamp_min(1e-8)
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    idx = torch.multinomial(flat, 1, replacement=True).squeeze(-1)
    one_hot = F.one_hot(idx, num_classes=n_class).to(dtype=probs.dtype)
    return one_hot.view(*probs.shape)


def sample_multi_label(probs: torch.Tensor) -> torch.Tensor:
    return torch.bernoulli(probs.clamp(0.0, 1.0))


def response_to_reward(response: torch.Tensor, play_ratio: torch.Tensor) -> torch.Tensor:
    listen = response[..., 0]
    like = response[..., 1]
    dislike = response[..., 2]
    unlike = response[..., 3]
    undislike = response[..., 4]
    play = play_ratio.clamp(0.0, 1.0) * listen
    reward = play + 0.8 * like - 1.2 * dislike - 0.6 * unlike + 0.2 * undislike
    return reward.clamp(-2.0, 2.0)


def response_to_regret_probs(response: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros(response.shape[:-1], dtype=torch.long, device=response.device)
    labels = torch.where(response[..., 2] > 0.5, torch.full_like(labels, 2), labels)
    labels = torch.where((labels == 0) & (response[..., 3] > 0.5), torch.full_like(labels, 3), labels)
    return F.one_hot(labels, num_classes=REGRET_CLASSES).to(dtype=response.dtype)


def score_candidates(
    predictors: list[FuturePredictor],
    batch: dict[str, torch.Tensor],
    candidate_features: torch.Tensor,
    soft_state: SoftStateBuilder,
    value_head: ValueHead,
    gamma: float,
    eta: float,
    bayes_samples: int,
    reward_sample_mode: str,
    sample_regret_from_response: bool,
) -> dict[str, torch.Tensor]:
    preds = [predictor(batch, candidate_features) for predictor in predictors]
    state_emb = preds[0]["state_emb"]
    scores = []
    rewards = []
    next_values = []
    regret_risks = []

    for sample_idx in range(max(int(bayes_samples), 1)):
        pred = preds[sample_idx % len(preds)]
        if reward_sample_mode == "sampled_formula":
            response = sample_multi_label(pred["response_probs"])
            reward = response_to_reward(response, pred["predicted_play_ratio"])
            regret_probs = response_to_regret_probs(response) if sample_regret_from_response else sample_one_hot(pred["regret_probs"])
        elif reward_sample_mode == "predictor_mean":
            response = pred["response_probs"]
            reward = pred["predicted_reward"]
            regret_probs = pred["regret_probs"]
        else:
            raise ValueError(f"Unsupported reward_sample_mode: {reward_sample_mode}")

        soft = soft_state(
            state_emb,
            candidate_features,
            {
                "response_probs": response,
                "regret_probs": regret_probs,
                "predicted_play_ratio": pred["predicted_play_ratio"],
                "predicted_reward": reward,
            },
        )
        d_model = soft["next_state_emb"].shape[-1]
        next_value = value_head(soft["next_state_emb"].reshape(-1, d_model)).view(
            candidate_features.shape[0],
            candidate_features.shape[1],
        )
        regret_risk = regret_probs[..., 1:].sum(dim=-1)
        scores.append(reward + float(gamma) * next_value - float(eta) * regret_risk)
        rewards.append(reward)
        next_values.append(next_value)
        regret_risks.append(regret_risk)

    return {
        "score": torch.stack(scores, dim=0).mean(dim=0),
        "reward": torch.stack(rewards, dim=0).mean(dim=0),
        "next_value": torch.stack(next_values, dim=0).mean(dim=0),
        "regret_risk": torch.stack(regret_risks, dim=0).mean(dim=0),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    store = EmbedStore(args.embed_store)

    value_ckpt = torch.load(args.value_ckpt, map_location="cpu")
    predictor_ckpts = resolve_predictor_ckpts(args, value_ckpt)
    predictors = [load_predictor(path, store.dim, device) for path in predictor_ckpts]
    d_model = int(value_ckpt.get("predictor_config", {}).get("d_model", 128))

    soft_state = SoftStateBuilder(item_dim=store.dim, d_model=d_model).to(device)
    value_head = ValueHead(d_model=d_model).to(device)
    soft_state.load_state_dict(value_ckpt["soft_state"])
    value_head.load_state_dict(value_ckpt["value_head"])
    soft_state.eval()
    value_head.eval()

    dataset = FutureIterableDataset(args.data_dir, split=args.split, max_rows=args.max_rows, mapping_root=store.root)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=lambda rows: collate_future(rows, store))
    total_batches = estimate_total_batches(args.data_dir, args.split, args.batch_size, args.max_rows)

    n = 0
    top1 = 0
    top3 = 0
    rank_sum = 0.0
    reciprocal_rank_sum = 0.0
    true_score_sum = 0.0
    neg_score_sum = 0.0
    true_reward_sum = 0.0
    neg_reward_sum = 0.0
    true_next_value_sum = 0.0
    neg_next_value_sum = 0.0
    true_regret_sum = 0.0
    neg_regret_sum = 0.0

    pbar = tqdm(loader, total=total_batches, desc=f"[oracle value {args.split}]", unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            batch = {key: value.to(device) for key, value in batch.items()}
            candidates = make_oracle_inbatch_candidates(batch["action_features"], args.candidate_k)
            scored = score_candidates(
                predictors,
                batch,
                candidates,
                soft_state,
                value_head,
                gamma=args.gamma,
                eta=args.eta,
                bayes_samples=args.bayes_samples,
                reward_sample_mode=args.reward_sample_mode,
                sample_regret_from_response=bool(args.sample_regret_from_response),
            )
            score = scored["score"]
            order = score.argsort(dim=1, descending=True)
            ranks = (order == 0).nonzero(as_tuple=False)[:, 1] + 1
            batch_n = int(score.shape[0])
            n += batch_n
            top1 += int((ranks == 1).sum().detach().cpu())
            top3 += int((ranks <= 3).sum().detach().cpu())
            rank_sum += float(ranks.float().sum().detach().cpu())
            reciprocal_rank_sum += float((1.0 / ranks.float()).sum().detach().cpu())

            true_score_sum += float(score[:, 0].sum().detach().cpu())
            neg_score_sum += float(score[:, 1:].mean(dim=1).sum().detach().cpu())
            true_reward_sum += float(scored["reward"][:, 0].sum().detach().cpu())
            neg_reward_sum += float(scored["reward"][:, 1:].mean(dim=1).sum().detach().cpu())
            true_next_value_sum += float(scored["next_value"][:, 0].sum().detach().cpu())
            neg_next_value_sum += float(scored["next_value"][:, 1:].mean(dim=1).sum().detach().cpu())
            true_regret_sum += float(scored["regret_risk"][:, 0].sum().detach().cpu())
            neg_regret_sum += float(scored["regret_risk"][:, 1:].mean(dim=1).sum().detach().cpu())

            pbar.set_postfix(
                top1=format_float(top1 / max(n, 1)),
                mrr=format_float(reciprocal_rank_sum / max(n, 1)),
                rank=format_float(rank_sum / max(n, 1)),
                examples=n,
                refresh=False,
            )

    metrics = {
        "examples": n,
        "candidate_k": int(args.candidate_k),
        "oracle_true_index": 0,
        "top1": top1 / max(n, 1),
        "top3": top3 / max(n, 1),
        "mrr": reciprocal_rank_sum / max(n, 1),
        "mean_rank": rank_sum / max(n, 1),
        "true_score_mean": true_score_sum / max(n, 1),
        "negative_score_mean": neg_score_sum / max(n, 1),
        "true_minus_negative_score": (true_score_sum - neg_score_sum) / max(n, 1),
        "true_reward_mean": true_reward_sum / max(n, 1),
        "negative_reward_mean": neg_reward_sum / max(n, 1),
        "true_next_value_mean": true_next_value_sum / max(n, 1),
        "negative_next_value_mean": neg_next_value_sum / max(n, 1),
        "true_regret_risk_mean": true_regret_sum / max(n, 1),
        "negative_regret_risk_mean": neg_regret_sum / max(n, 1),
        "predictor_count": len(predictors),
        "bayes_samples": int(args.bayes_samples),
        "reward_sample_mode": args.reward_sample_mode,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
