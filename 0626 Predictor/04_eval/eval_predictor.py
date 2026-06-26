from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "01_data"), str(ROOT / "02_model")])

from future_dataset import EmbedStore, FutureIterableDataset, collate_future
from predictor import FuturePredictor, predictor_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate future predictor heads.")
    parser.add_argument("--data_dir", default=str(ROOT / "artifacts" / "future_data"))
    parser.add_argument("--embed_store", default=str(ROOT / "artifacts" / "embed_store"))
    parser.add_argument("--ckpt", default=str(ROOT / "artifacts" / "predictor" / "future_predictor.pt"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_rows", type=int, default=2000)
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


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    store = EmbedStore(args.embed_store)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = FuturePredictor(
        item_dim=int(ckpt.get("item_dim", store.dim)),
        d_model=int(cfg.get("d_model", 128)),
        max_seq_len=50,
        n_layer=int(cfg.get("n_layer", 2)),
        n_head=int(cfg.get("n_head", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = FutureIterableDataset(args.data_dir, split=args.split, max_rows=args.max_rows)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=lambda rows: collate_future(rows, store))

    totals: dict[str, float] = {}
    n_batches = 0
    n_examples = 0
    response_correct = 0
    response_exact = 0
    response_tp = 0.0
    response_fp = 0.0
    response_fn = 0.0
    regret_correct = 0
    reward_abs_error = 0.0
    play_abs_error = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(batch)
            losses = predictor_loss(out, batch)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            response_pred = out["response_logits"].argmax(dim=-1)
            response_multi = (out["response_probs"] >= 0.5).float()
            response_target_multi = batch["response_targets"].float()
            regret_pred = out["regret_logits"].argmax(dim=-1)
            response_correct += int((response_pred == batch["response_target"]).sum().cpu())
            response_exact += int((response_multi == response_target_multi).all(dim=1).sum().cpu())
            response_tp += float((response_multi * response_target_multi).sum().cpu())
            response_fp += float((response_multi * (1.0 - response_target_multi)).sum().cpu())
            response_fn += float(((1.0 - response_multi) * response_target_multi).sum().cpu())
            regret_correct += int((regret_pred == batch["regret_type_id"]).sum().cpu())
            reward_abs_error += float((out["predicted_reward"] - batch["reward"]).abs().sum().cpu())
            play_abs_error += float((out["predicted_play_ratio"] - batch["played_ratio"]).abs().sum().cpu())
            n_examples += int(response_pred.numel())
            n_batches += 1

    metrics = {key: value / max(n_batches, 1) for key, value in totals.items()}
    precision = response_tp / max(response_tp + response_fp, 1e-8)
    recall = response_tp / max(response_tp + response_fn, 1e-8)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    metrics.update(
        {
            "response_acc": response_correct / max(n_examples, 1),
            "response_exact_match": response_exact / max(n_examples, 1),
            "response_micro_precision": precision,
            "response_micro_recall": recall,
            "response_micro_f1": f1,
            "regret_acc": regret_correct / max(n_examples, 1),
            "reward_mae": reward_abs_error / max(n_examples, 1),
            "play_mae": play_abs_error / max(n_examples, 1),
            "examples": n_examples,
        }
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
