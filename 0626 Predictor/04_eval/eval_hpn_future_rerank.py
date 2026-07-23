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
sys.path.extend([str(ROOT / "01_data"), str(ROOT / "02_model")])

from future_dataset import EmbedStore, FutureIterableDataset, collate_future
from progress_utils import estimate_total_batches, format_float
from hpn import FutureHPNPolicy
from hpn_candidates import SidPathIndex, build_hpn_candidate_batch
from predictor import FuturePredictor
from soft_state import SoftStateBuilder
from value import ValueHead

REGRET_CLASSES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HPN top-k candidates reranked by future predictor and value.")
    parser.add_argument("--data_dir", default=str(ROOT / "artifacts" / "future_data"))
    parser.add_argument("--embed_store", default=str(ROOT / "artifacts" / "embed_store"))
    parser.add_argument("--dense_item2sid_npy", required=True)
    parser.add_argument("--dense2orig_npy", required=True)
    parser.add_argument("--hpn_ckpt", default=str(ROOT / "artifacts" / "hpn" / "hpn.pt"))
    parser.add_argument("--predictor_ckpt", default=str(ROOT / "artifacts" / "predictor" / "future_predictor.pt"))
    parser.add_argument("--predictor_ckpts", default="")
    parser.add_argument("--predictor_manifest", default="")
    parser.add_argument("--value_ckpt", default=str(ROOT / "artifacts" / "value" / "future_value.pt"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--top_sid_paths", type=int, default=32)
    parser.add_argument("--branch_k", type=int, default=16)
    parser.add_argument("--max_candidates", type=int, default=64)
    parser.add_argument("--max_items_per_sid", type=int, default=4, help="Deprecated with direct SID item scoring; kept for CLI compatibility.")
    parser.add_argument("--max_index_items", type=int, default=0, help="Limit HPN candidate pool size for smoke runs; 0 means all valid items.")
    parser.add_argument("--candidate_chunk_size", type=int, default=65536, help="Chunk size for direct HPN item-pool SID scoring.")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--bayes_samples", type=int, default=1)
    parser.add_argument("--reward_sample_mode", choices=["sampled_formula", "predictor_mean"], default="sampled_formula")
    parser.add_argument("--hpn_score_weight", type=float, default=1.0)
    parser.add_argument("--future_score_weight", type=float, default=1.0)
    parser.add_argument("--candidate_logit_weight", type=float, default=0.0)
    parser.add_argument("--semantic_prefix_weight", type=float, default=0.0)
    parser.add_argument("--semantic_level_weights", default="")
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
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_norm(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.float()
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * valid).sum(dim=1, keepdim=True) / denom
    var = (((x - mean) * valid) ** 2).sum(dim=1, keepdim=True) / denom
    return (x - mean) / var.sqrt().clamp_min(1e-6)


def parse_semantic_level_weights(text: str, levels: int) -> list[float]:
    if text.strip():
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    else:
        values = [1.0, 0.7, 0.5, 0.3]
    if not values:
        values = [1.0]
    while len(values) < levels:
        values.append(values[-1])
    return values[:levels]


def candidate_sid_tensor(dense_item2sid: np.ndarray, candidate_dense_np: np.ndarray, device: torch.device) -> torch.Tensor:
    levels = int(dense_item2sid.shape[1])
    candidate_sid_np = np.zeros((*candidate_dense_np.shape, levels), dtype=np.int64)
    valid = (candidate_dense_np > 0) & (candidate_dense_np < dense_item2sid.shape[0])
    if np.any(valid):
        candidate_sid_np[valid] = dense_item2sid[candidate_dense_np[valid]]
    return torch.tensor(candidate_sid_np, dtype=torch.long, device=device)


def semantic_prefix_score_from_logits(
    sid_logits: torch.Tensor,
    candidate_sid: torch.Tensor,
    level_weights: list[float],
) -> torch.Tensor:
    score = torch.zeros(candidate_sid.shape[:2], dtype=sid_logits.dtype, device=sid_logits.device)
    weight_sum = 0.0
    n_levels = min(sid_logits.shape[1], candidate_sid.shape[2], len(level_weights))
    for level in range(n_levels):
        weight = float(level_weights[level])
        log_probs = F.log_softmax(sid_logits[:, level, :], dim=-1)
        token = candidate_sid[:, :, level].clamp(0, log_probs.shape[-1] - 1)
        score = score + weight * log_probs.gather(1, token)
        weight_sum += abs(weight)
    return score / max(weight_sum, 1e-6)



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
        state_pooling=str(cfg.get("state_pooling", "last")),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_predictor(path: str, item_dim: int, device: torch.device) -> FuturePredictor:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = FuturePredictor(
        item_dim=int(ckpt.get("item_dim", item_dim)),
        d_model=int(cfg.get("d_model", 128)),
        max_seq_len=int(cfg.get("max_seq_len", 50)),
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


def resolve_predictor_ckpts(args: argparse.Namespace, value_ckpt: dict) -> list[str]:
    if args.predictor_manifest:
        manifest = json.loads(Path(args.predictor_manifest).read_text(encoding="utf-8"))
        ckpts = [str(item) for item in manifest.get("predictor_ckpts", [])]
    elif args.predictor_ckpts:
        ckpts = [item.strip() for item in args.predictor_ckpts.split(",") if item.strip()]
    elif value_ckpt.get("predictor_ckpts"):
        ckpts = [str(item) for item in value_ckpt["predictor_ckpts"]]
    else:
        ckpts = [str(args.predictor_ckpt)]
    if not ckpts:
        raise RuntimeError("No predictor checkpoints were provided.")
    return ckpts


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


def bayesian_future_score(
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
) -> torch.Tensor:
    preds = [predictor(batch, candidate_features) for predictor in predictors]
    primary_state = preds[0]["state_emb"]
    scores = []
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
        pred_for_soft = {
            "response_probs": response,
            "regret_probs": regret_probs,
            "predicted_play_ratio": pred["predicted_play_ratio"],
            "predicted_reward": reward,
        }
        soft = soft_state(primary_state, candidate_features, pred_for_soft)
        d_model = soft["next_state_emb"].shape[-1]
        next_value = value_head(soft["next_state_emb"].reshape(-1, d_model)).view(candidate_features.shape[0], candidate_features.shape[1])
        regret_risk = regret_probs[..., 1:].sum(dim=-1)
        scores.append(reward + float(gamma) * next_value - float(eta) * regret_risk)
    return torch.stack(scores, dim=0).mean(dim=0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    store = EmbedStore(args.embed_store)
    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
    dense2orig = np.load(args.dense2orig_npy, mmap_mode="r")
    index = SidPathIndex(dense_item2sid, max_items_per_sid=args.max_items_per_sid, max_index_items=args.max_index_items, chunk_size=args.candidate_chunk_size)

    hpn = load_hpn(args.hpn_ckpt, store.dim, device)
    semantic_level_weights = parse_semantic_level_weights(args.semantic_level_weights, int(dense_item2sid.shape[1]))
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

    n = 0
    recall = 0
    top1 = 0
    rank_sum = 0.0
    empty = 0
    total_batches = estimate_total_batches(args.data_dir, args.split, args.batch_size, args.max_rows)
    pbar = tqdm(loader, total=total_batches, desc=f"[eval rerank {args.split}]", unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            batch = {key: value.to(device) for key, value in batch.items()}
            candidate_out = build_hpn_candidate_batch(
                batch=batch,
                hpn=hpn,
                index=index,
                dense2orig=dense2orig,
                store=store,
                device=device,
                max_candidates=args.max_candidates,
                top_sid_paths=args.top_sid_paths,
                branch_k=args.branch_k,
                fallback_to_logged=False,
            )
            candidate_features = candidate_out["candidate_features"]
            candidate_dense_np = candidate_out["candidate_dense"]
            hpn_score = candidate_out["hpn_scores"]
            mask = candidate_out["candidate_mask"]
            empty += int(candidate_out["empty_rows"])

            components = []
            if abs(float(args.hpn_score_weight)) > 0.0:
                components.append(float(args.hpn_score_weight) * masked_norm(hpn_score.clamp_min(-1e4), mask))

            if abs(float(args.future_score_weight)) > 0.0:
                future_score = bayesian_future_score(
                    predictors,
                    batch,
                    candidate_features,
                    soft_state,
                    value_head,
                    gamma=args.gamma,
                    eta=args.eta,
                    bayes_samples=args.bayes_samples,
                    reward_sample_mode=args.reward_sample_mode,
                    sample_regret_from_response=bool(args.sample_regret_from_response),
                )
                components.append(float(args.future_score_weight) * masked_norm(future_score, mask))

            if abs(float(args.candidate_logit_weight)) > 0.0 or abs(float(args.semantic_prefix_weight)) > 0.0:
                rank_pred = predictors[0](batch, candidate_features)
                if abs(float(args.candidate_logit_weight)) > 0.0:
                    components.append(float(args.candidate_logit_weight) * masked_norm(rank_pred["candidate_logit"], mask))
                if abs(float(args.semantic_prefix_weight)) > 0.0:
                    candidate_sid = candidate_sid_tensor(dense_item2sid, candidate_dense_np, device)
                    semantic_score = semantic_prefix_score_from_logits(rank_pred["sid_logits"], candidate_sid, semantic_level_weights)
                    components.append(float(args.semantic_prefix_weight) * masked_norm(semantic_score, mask))

            if not components:
                raise RuntimeError("At least one rerank score weight must be non-zero.")
            final_logit = torch.stack(components, dim=0).sum(dim=0).masked_fill(~mask, -1e9)

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
            pbar.set_postfix(
                recall=format_float(recall / max(n, 1)),
                top1=format_float(top1 / max(n, 1)),
                empty=empty,
                examples=n,
                refresh=False,
            )

    metrics = {
        "examples": n,
        "empty_candidate_rows": empty,
        "hpn_recall": recall / max(n, 1),
        "future_top1_given_all": top1 / max(n, 1),
        "future_top1_given_recalled": top1 / max(recall, 1),
        "mean_rank_given_recalled": rank_sum / max(recall, 1),
        "candidate_pool_size": index.candidate_count,
        "max_candidates": args.max_candidates,
        "predictor_count": len(predictors),
        "bayes_samples": int(args.bayes_samples),
        "reward_sample_mode": args.reward_sample_mode,
        "score_weights": {
            "hpn_score": float(args.hpn_score_weight),
            "future_score": float(args.future_score_weight),
            "candidate_logit": float(args.candidate_logit_weight),
            "semantic_prefix": float(args.semantic_prefix_weight),
        },
        "semantic_level_weights": semantic_level_weights,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
