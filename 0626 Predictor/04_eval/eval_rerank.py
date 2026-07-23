from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "01_data"), str(ROOT / "02_model")])

from future_dataset import EmbedStore, FutureIterableDataset, collate_future
from progress_utils import estimate_total_batches, format_float
from predictor import FuturePredictor
from soft_state import SoftStateBuilder
from value import CandidateScorer, ValueHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate future-aware reranking with in-batch candidates.")
    parser.add_argument("--data_dir", default=str(ROOT / "artifacts" / "future_data"))
    parser.add_argument("--embed_store", default=str(ROOT / "artifacts" / "embed_store"))
    parser.add_argument("--predictor_ckpt", default=str(ROOT / "artifacts" / "predictor" / "future_predictor.pt"))
    parser.add_argument("--value_ckpt", default=str(ROOT / "artifacts" / "value" / "future_value.pt"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--candidate_k", type=int, default=8)
    parser.add_argument("--max_rows", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.9)
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


def make_inbatch_candidates(action_features: torch.Tensor, candidate_k: int) -> torch.Tensor:
    parts = [action_features]
    for shift in range(1, max(candidate_k, 1)):
        parts.append(torch.roll(action_features, shifts=shift % max(action_features.shape[0], 1), dims=0))
    return torch.stack(parts[:candidate_k], dim=1)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    store = EmbedStore(args.embed_store)

    pred_ckpt = torch.load(args.predictor_ckpt, map_location="cpu")
    pred_cfg = pred_ckpt.get("config", {})
    predictor = FuturePredictor(
        item_dim=int(pred_ckpt.get("item_dim", store.dim)),
        d_model=int(pred_cfg.get("d_model", 128)),
        max_seq_len=int(pred_cfg.get("max_seq_len", 50)),
        n_layer=int(pred_cfg.get("n_layer", 2)),
        n_head=int(pred_cfg.get("n_head", 4)),
        dropout=float(pred_cfg.get("dropout", 0.1)),
        state_pooling=str(pred_cfg.get("state_pooling", "last")),
        sid_levels=int(pred_cfg.get("sid_levels", 4)),
        sid_vocab_size=int(pred_cfg.get("sid_vocab_size", 256)),
    ).to(device)
    predictor.load_state_dict(pred_ckpt["model_state"], strict=False)
    predictor.eval()

    value_ckpt = torch.load(args.value_ckpt, map_location="cpu")
    d_model = int(pred_cfg.get("d_model", 128))
    soft_state = SoftStateBuilder(item_dim=store.dim, d_model=d_model).to(device)
    value_head = ValueHead(d_model=d_model).to(device)
    soft_state.load_state_dict(value_ckpt["soft_state"])
    value_head.load_state_dict(value_ckpt["value_head"])
    soft_state.eval()
    value_head.eval()
    scorer = CandidateScorer(gamma=args.gamma).to(device)

    dataset = FutureIterableDataset(args.data_dir, split=args.split, max_rows=args.max_rows, mapping_root=store.root)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=lambda rows: collate_future(rows, store))

    logged_top1 = 0
    logged_rank_sum = 0.0
    n_examples = 0
    future_score_mean = 0.0
    total_batches = estimate_total_batches(args.data_dir, args.split, args.batch_size, args.max_rows)
    pbar = tqdm(loader, total=total_batches, desc=f"[eval inbatch-rerank {args.split}]", unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            batch = {key: value.to(device) for key, value in batch.items()}
            candidates = make_inbatch_candidates(batch["action_features"], args.candidate_k)
            pred = predictor(batch, candidates)
            soft = soft_state(pred["state_emb"], candidates, pred)
            next_value = value_head(soft["next_state_emb"].reshape(-1, d_model)).view(candidates.shape[0], candidates.shape[1])
            hpn_logit = torch.zeros_like(pred["predicted_reward"])
            scored = scorer(hpn_logit, pred["predicted_reward"], next_value, pred["regret_probs"])
            order = scored["final_logit"].argsort(dim=1, descending=True)
            ranks = (order == 0).nonzero(as_tuple=False)[:, 1] + 1
            logged_top1 += int((ranks == 1).sum().cpu())
            logged_rank_sum += float(ranks.float().sum().cpu())
            future_score_mean += float(scored["future_score"].mean().cpu()) * int(ranks.numel())
            n_examples += int(ranks.numel())
            pbar.set_postfix(
                top1=format_float(logged_top1 / max(n_examples, 1)),
                mean_rank=format_float(logged_rank_sum / max(n_examples, 1)),
                examples=n_examples,
                refresh=False,
            )

    metrics = {
        "logged_action_top1_rate": logged_top1 / max(n_examples, 1),
        "logged_action_mean_rank": logged_rank_sum / max(n_examples, 1),
        "future_score_mean": future_score_mean / max(n_examples, 1),
        "examples": n_examples,
        "candidate_k": args.candidate_k,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
