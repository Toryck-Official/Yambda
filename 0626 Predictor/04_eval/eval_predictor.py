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
        state_pooling=str(cfg.get("state_pooling", "last")),
        sid_levels=int(cfg.get("sid_levels", 4)),
        sid_vocab_size=int(cfg.get("sid_vocab_size", 256)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    dataset = FutureIterableDataset(args.data_dir, split=args.split, max_rows=args.max_rows, mapping_root=store.root)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=lambda rows: collate_future(rows, store))

    totals: dict[str, float] = {}
    n_batches = 0
    n_examples = 0
    response_correct = 0
    response_exact = 0
    response_tp = 0.0
    response_fp = 0.0
    response_fn = 0.0
    response_tp_by_class = torch.zeros(5, dtype=torch.float64)
    response_fp_by_class = torch.zeros(5, dtype=torch.float64)
    response_fn_by_class = torch.zeros(5, dtype=torch.float64)
    regret_correct = 0
    regret_correct_by_class = torch.zeros(4, dtype=torch.float64)
    regret_count_by_class = torch.zeros(4, dtype=torch.float64)
    reward_abs_error = 0.0
    play_abs_error = 0.0
    future_return_abs_error = 0.0
    future_regret_correct = 0
    future_regret_tp = 0.0
    future_regret_fp = 0.0
    future_regret_fn = 0.0
    sid_correct_by_level: torch.Tensor | None = None
    sid_prefix_correct_by_level: torch.Tensor | None = None
    sid_examples = 0

    total_batches = estimate_total_batches(args.data_dir, args.split, args.batch_size, args.max_rows)
    pbar = tqdm(loader, total=total_batches, desc=f"[eval predictor {args.split}]", unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(batch)
            losses = predictor_loss(out, batch, future_return_weight=1.0, future_regret_weight=1.0)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            response_pred = out["response_logits"].argmax(dim=-1)
            response_multi = (out["response_probs"] >= 0.5).float()
            response_target_multi = batch["response_targets"].float()
            regret_pred = out["regret_logits"].argmax(dim=-1)
            response_correct += int((response_pred == batch["response_target"]).sum().cpu())
            response_exact += int((response_multi == response_target_multi).all(dim=1).sum().cpu())
            tp_vec = (response_multi * response_target_multi).sum(dim=0).detach().cpu().double()
            fp_vec = (response_multi * (1.0 - response_target_multi)).sum(dim=0).detach().cpu().double()
            fn_vec = ((1.0 - response_multi) * response_target_multi).sum(dim=0).detach().cpu().double()
            response_tp_by_class += tp_vec
            response_fp_by_class += fp_vec
            response_fn_by_class += fn_vec
            response_tp += float(tp_vec.sum())
            response_fp += float(fp_vec.sum())
            response_fn += float(fn_vec.sum())
            regret_match = regret_pred == batch["regret_type_id"]
            regret_correct += int(regret_match.sum().cpu())
            for cls in range(regret_count_by_class.numel()):
                cls_mask = batch["regret_type_id"] == cls
                regret_count_by_class[cls] += float(cls_mask.sum().detach().cpu())
                regret_correct_by_class[cls] += float((regret_match & cls_mask).sum().detach().cpu())
            reward_abs_error += float((out["predicted_reward"] - batch["reward"]).abs().sum().cpu())
            play_abs_error += float((out["predicted_play_ratio"] - batch["played_ratio"]).abs().sum().cpu())
            if "predicted_future_return" in out and "future_return" in batch:
                future_return_abs_error += float((out["predicted_future_return"] - batch["future_return"]).abs().sum().cpu())
            if "future_regret_prob" in out and "future_regret_any" in batch:
                future_target = batch["future_regret_any"].float() > 0.5
                future_pred = out["future_regret_prob"] >= 0.5
                future_regret_correct += int((future_pred == future_target).sum().cpu())
                future_regret_tp += float((future_target & future_pred).sum().cpu())
                future_regret_fp += float(((~future_target) & future_pred).sum().cpu())
                future_regret_fn += float((future_target & (~future_pred)).sum().cpu())
            if "sid_logits" in out and "target_sid" in batch:
                levels = min(int(out["sid_logits"].shape[1]), int(batch["target_sid"].shape[1]))
                if levels > 0:
                    sid_pred = out["sid_logits"][:, :levels, :].argmax(dim=-1)
                    sid_target = batch["target_sid"][:, :levels].long().clamp(min=0, max=out["sid_logits"].shape[-1] - 1)
                    sid_match = sid_pred == sid_target
                    prefix_match = sid_match.long().cumprod(dim=1).bool()
                    if sid_correct_by_level is None:
                        sid_correct_by_level = torch.zeros(levels, dtype=torch.float64)
                        sid_prefix_correct_by_level = torch.zeros(levels, dtype=torch.float64)
                    sid_correct_by_level[:levels] += sid_match.detach().cpu().double().sum(dim=0)
                    sid_prefix_correct_by_level[:levels] += prefix_match.detach().cpu().double().sum(dim=0)
                    sid_examples += int(sid_match.shape[0])
            n_examples += int(response_pred.numel())
            n_batches += 1
            pbar.set_postfix(
                loss=format_float(losses["loss"].detach().cpu()),
                response_acc=format_float(response_correct / max(n_examples, 1)),
                regret_acc=format_float(regret_correct / max(n_examples, 1)),
                examples=n_examples,
                refresh=False,
            )

    metrics = {key: value / max(n_batches, 1) for key, value in totals.items()}
    precision = response_tp / max(response_tp + response_fp, 1e-8)
    recall = response_tp / max(response_tp + response_fn, 1e-8)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    response_names = ["listen", "like", "dislike", "unlike", "undislike"]
    response_by_class = {}
    for idx, name in enumerate(response_names):
        tp = float(response_tp_by_class[idx])
        fp = float(response_fp_by_class[idx])
        fn = float(response_fn_by_class[idx])
        cls_precision = tp / max(tp + fp, 1e-8)
        cls_recall = tp / max(tp + fn, 1e-8)
        cls_f1 = 2.0 * cls_precision * cls_recall / max(cls_precision + cls_recall, 1e-8)
        response_by_class[name] = {
            "precision": cls_precision,
            "recall": cls_recall,
            "f1": cls_f1,
            "support": int(tp + fn),
            "predicted_positive": int(tp + fp),
        }

    regret_names = ["none", "low_play", "dislike", "unlike"]
    regret_by_class = {}
    for idx, name in enumerate(regret_names):
        support = float(regret_count_by_class[idx])
        correct = float(regret_correct_by_class[idx])
        regret_by_class[name] = {
            "recall": correct / max(support, 1e-8),
            "support": int(support),
        }

    future_precision = future_regret_tp / max(future_regret_tp + future_regret_fp, 1e-8)
    future_recall = future_regret_tp / max(future_regret_tp + future_regret_fn, 1e-8)
    future_f1 = 2.0 * future_precision * future_recall / max(future_precision + future_recall, 1e-8)

    if sid_correct_by_level is not None and sid_prefix_correct_by_level is not None:
        for idx in range(int(sid_correct_by_level.numel())):
            metrics[f"sid_acc_l{idx + 1}"] = float(sid_correct_by_level[idx] / max(sid_examples, 1))
            metrics[f"sid_prefix_acc_l{idx + 1}"] = float(sid_prefix_correct_by_level[idx] / max(sid_examples, 1))

    metrics.update(
        {
            "response_acc_primary": response_correct / max(n_examples, 1),
            "response_exact_match": response_exact / max(n_examples, 1),
            "response_micro_precision": precision,
            "response_micro_recall": recall,
            "response_micro_f1": f1,
            "response_by_class": response_by_class,
            "regret_acc": regret_correct / max(n_examples, 1),
            "regret_by_class": regret_by_class,
            "reward_mae": reward_abs_error / max(n_examples, 1),
            "play_mae": play_abs_error / max(n_examples, 1),
            "future_return_mae": future_return_abs_error / max(n_examples, 1),
            "future_regret_acc": future_regret_correct / max(n_examples, 1),
            "future_regret_precision": future_precision,
            "future_regret_recall": future_recall,
            "future_regret_f1": future_f1,
            "examples": n_examples,
        }
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
