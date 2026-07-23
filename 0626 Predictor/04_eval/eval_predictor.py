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
        max_seq_len=int(cfg.get("max_seq_len", 50)),
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
    response_exact = 0
    response_tp = torch.zeros(5)
    response_fp = torch.zeros(5)
    response_fn = torch.zeros(5)
    response_brier = 0.0
    regret_correct = 0
    reward_abs_error = 0.0
    play_abs_error = 0.0
    play_examples = 0

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(batch)
            losses = predictor_loss(
                out,
                batch,
                response_weight=float(cfg.get("response_loss_weight", 1.0)),
                play_weight=float(cfg.get("play_loss_weight", 0.1)),
                reward_weight=float(cfg.get("reward_loss_weight", 0.1)),
                regret_weight=float(cfg.get("regret_loss_weight", 0.1)),
            )
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            response_multi = (out["response_probs"] >= 0.5).float()
            response_target_multi = batch["response_targets"].float()
            regret_pred = out["regret_logits"].argmax(dim=-1)
            response_exact += int((response_multi == response_target_multi).all(dim=1).sum().cpu())
            response_tp += (response_multi * response_target_multi).sum(dim=0).cpu()
            response_fp += (response_multi * (1.0 - response_target_multi)).sum(dim=0).cpu()
            response_fn += ((1.0 - response_multi) * response_target_multi).sum(dim=0).cpu()
            response_brier += float(((out["response_probs"] - response_target_multi) ** 2).sum().cpu())
            regret_correct += int((regret_pred == batch["regret_type_id"]).sum().cpu())
            reward_abs_error += float((out["predicted_reward"] - batch["reward"]).abs().sum().cpu())
            listen_mask = response_target_multi[:, 0] > 0.5
            play_abs_error += float(
                (out["predicted_play_ratio"][listen_mask] - batch["played_ratio"][listen_mask]).abs().sum().cpu()
            )
            play_examples += int(listen_mask.sum().cpu())
            n_examples += int(response_multi.shape[0])
            n_batches += 1

    metrics = {key: value / max(n_batches, 1) for key, value in totals.items()}
    total_tp = float(response_tp.sum())
    total_fp = float(response_fp.sum())
    total_fn = float(response_fn.sum())
    precision = total_tp / max(total_tp + total_fp, 1e-8)
    recall = total_tp / max(total_tp + total_fn, 1e-8)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    metrics.update(
        {
            "response_exact_match": response_exact / max(n_examples, 1),
            "response_micro_precision": precision,
            "response_micro_recall": recall,
            "response_micro_f1": f1,
            "response_brier": response_brier / max(n_examples * 5, 1),
            "regret_acc": regret_correct / max(n_examples, 1),
            "reward_mae": reward_abs_error / max(n_examples, 1),
            "play_mae_given_listen": play_abs_error / max(play_examples, 1),
            "examples": n_examples,
        }
    )
    response_names = ["listen", "like", "dislike", "unlike", "undislike"]
    for idx, name in enumerate(response_names):
        class_precision = float(response_tp[idx]) / max(float(response_tp[idx] + response_fp[idx]), 1e-8)
        class_recall = float(response_tp[idx]) / max(float(response_tp[idx] + response_fn[idx]), 1e-8)
        metrics[f"{name}_precision"] = class_precision
        metrics[f"{name}_recall"] = class_recall
        metrics[f"{name}_f1"] = 2.0 * class_precision * class_recall / max(class_precision + class_recall, 1e-8)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
