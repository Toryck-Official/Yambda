from __future__ import annotations

import argparse
import json
import math
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
from progress_utils import estimate_total_batches, format_float, split_row_count
from predictor import FuturePredictor, predictor_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the action-conditioned future predictor.")
    parser.add_argument("--data_dir", default=str(ROOT / "artifacts" / "future_data"))
    parser.add_argument("--embed_store", default=str(ROOT / "artifacts" / "embed_store"))
    parser.add_argument("--out_dir", default=str(ROOT / "artifacts" / "predictor"))
    parser.add_argument("--init_ckpt", default="", help="Optional predictor checkpoint to continue training from.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_train_rows", type=int, default=5000)
    parser.add_argument("--max_val_rows", type=int, default=1000)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--state_pooling", choices=["last", "mean", "last_mean"], default="last")
    parser.add_argument("--sid_levels", type=int, default=4)
    parser.add_argument("--sid_vocab_size", type=int, default=256)
    parser.add_argument("--sid_loss_weight", type=float, default=0.0, help="Optional next semantic-ID token prediction loss weight; 0 disables it.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--response_pos_weight",
        default="",
        help="Optional comma-separated BCE positive weights for listen,like,dislike,unlike,undislike.",
    )
    parser.add_argument(
        "--response_class_weight",
        default="",
        help="Optional comma-separated BCE element weights for listen,like,dislike,unlike,undislike.",
    )
    parser.add_argument(
        "--regret_class_weight",
        default="",
        help="Optional comma-separated CE class weights for none,low_play,dislike,unlike.",
    )
    parser.add_argument("--response_weight", type=float, default=1.0)
    parser.add_argument("--play_weight", type=float, default=1.0)
    parser.add_argument("--reward_weight", type=float, default=1.0)
    parser.add_argument("--regret_weight", type=float, default=1.0)
    parser.add_argument("--future_return_weight", type=float, default=1.0, help="Auxiliary loss weight for discounted future reward prediction.")
    parser.add_argument("--future_regret_weight", type=float, default=1.0, help="Auxiliary loss weight for future explicit-regret probability prediction.")
    parser.add_argument("--future_regret_pos_weight", type=float, default=1.0, help="BCE positive weight for rare future-regret labels.")
    parser.add_argument("--contrastive_weight", type=float, default=0.0, help="Optional formula-reward in-batch contrastive loss weight; 0 disables it.")
    parser.add_argument("--contrastive_candidate_k", type=int, default=8)
    parser.add_argument("--contrastive_temperature", type=float, default=0.2)
    parser.add_argument("--candidate_ce_weight", type=float, default=0.0, help="Optional direct next-item candidate CE loss weight; 0 disables it.")
    parser.add_argument("--candidate_ce_k", type=int, default=8)
    parser.add_argument("--candidate_ce_temperature", type=float, default=1.0)
    parser.add_argument(
        "--candidate_negative_mode",
        choices=["inbatch", "history", "history_inbatch", "semantic_inbatch", "history_semantic_inbatch"],
        default="history_inbatch",
        help="Negative action source for contrastive/candidate losses.",
    )
    parser.add_argument("--semantic_prefix_level", type=int, default=2, help="Prefix length for semantic in-batch negatives.")
    parser.add_argument("--train_user_sample_mod", type=int, default=0, help="Keep train users whose stable hash bucket equals bucket; 0 disables sampling.")
    parser.add_argument("--train_user_sample_bucket", type=int, default=0)
    parser.add_argument("--train_max_users", type=int, default=0, help="Optional cap on selected train users after user-level sampling.")
    parser.add_argument("--save_each_epoch", type=int, default=1, help="Save future_predictor_epoch{n}.pt after every epoch when nonzero.")
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


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_weight_list(text: str, expected: int, device: torch.device) -> torch.Tensor | None:
    if not text:
        return None
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} weights, got {len(values)} from {text!r}")
    return torch.tensor(values, dtype=torch.float32, device=device)


def estimate_sampled_total_batches(
    data_dir: str,
    split: str,
    batch_size: int,
    max_rows: int,
    user_sample_mod: int = 0,
    max_users: int = 0,
) -> int | None:
    if max_rows > 0:
        return estimate_total_batches(data_dir, split, batch_size, max_rows)
    if max_users > 0:
        return None
    if user_sample_mod > 0:
        rows = split_row_count(data_dir, split)
        if rows is None:
            return None
        return int(math.ceil((rows / max(user_sample_mod, 1)) / max(int(batch_size), 1)))
    return estimate_total_batches(data_dir, split, batch_size, max_rows)


def make_loader(
    data_dir: str,
    split: str,
    store: EmbedStore,
    batch_size: int,
    max_rows: int,
    user_sample_mod: int = 0,
    user_sample_bucket: int = 0,
    max_users: int = 0,
) -> DataLoader:
    dataset = FutureIterableDataset(
        data_dir,
        split=split,
        max_rows=max_rows,
        mapping_root=store.root,
        user_sample_mod=user_sample_mod,
        user_sample_bucket=user_sample_bucket,
        max_users=max_users,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=lambda rows: collate_future(rows, store),
    )


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
    batch_size = int(history_dense.shape[0])
    # Histories are left padded; take recent valid items first, then cycle backward.
    for row_idx in range(batch_size):
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
    batch_size = int(target_sid.shape[0])
    prefix = max(1, min(int(prefix_level), int(target_sid.shape[1])))
    negative = fallback.clone()
    used = 0
    sid_prefix = target_sid[:, :prefix]
    for row_idx in range(batch_size):
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
    negative_mode: str = "inbatch",
    semantic_prefix_level: int = 2,
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
        source = "fallback"
        if negative_mode == "history":
            negative, used = _history_negative_features(batch, slot, fallback)
            history_used += used
            fallback_used += int(action_features.shape[0]) - used
            source = "history"
        elif negative_mode == "history_inbatch":
            if slot % 2 == 1:
                negative, used = _history_negative_features(batch, slot, fallback)
                history_used += used
                fallback_used += int(action_features.shape[0]) - used
                source = "history"
            else:
                fallback_used += int(action_features.shape[0])
        elif negative_mode == "semantic_inbatch":
            negative, used = _semantic_inbatch_negative_features(batch, action_features, slot, semantic_prefix_level, fallback)
            semantic_used += used
            fallback_used += int(action_features.shape[0]) - used
            source = "semantic"
        elif negative_mode == "history_semantic_inbatch":
            pattern = slot % 3
            if pattern == 1:
                negative, used = _history_negative_features(batch, slot, fallback)
                history_used += used
                fallback_used += int(action_features.shape[0]) - used
                source = "history"
            elif pattern == 2:
                negative, used = _semantic_inbatch_negative_features(batch, action_features, slot, semantic_prefix_level, fallback)
                semantic_used += used
                fallback_used += int(action_features.shape[0]) - used
                source = "semantic"
            else:
                fallback_used += int(action_features.shape[0])
        else:
            fallback_used += int(action_features.shape[0])
        parts.append(negative)
    candidates = torch.stack(parts, dim=1)
    denom = max(float(total_negatives), 1.0)
    return candidates, {
        "history_negative_share": float(history_used) / denom,
        "semantic_negative_share": float(semantic_used) / denom,
        "fallback_negative_share": float(fallback_used) / denom,
    }

def semantic_sid_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "sid_logits" not in out or "target_sid" not in batch:
        zero = out["state_emb"].new_tensor(0.0)
        return {"loss": zero, "acc": [], "active": False}
    logits = out["sid_logits"]
    target = batch["target_sid"].long().to(logits.device)
    levels = min(int(logits.shape[1]), int(target.shape[1]))
    if levels <= 0:
        zero = logits.new_tensor(0.0)
        return {"loss": zero, "acc": [], "active": False}
    losses = []
    acc = []
    for level in range(levels):
        level_logits = logits[:, level, :]
        level_target = target[:, level].clamp(min=0, max=level_logits.shape[-1] - 1)
        losses.append(torch.nn.functional.cross_entropy(level_logits, level_target))
        acc.append((level_logits.argmax(dim=-1) == level_target).float().mean().detach())
    return {"loss": torch.stack(losses).mean(), "acc": acc, "active": True}


def expected_formula_reward(out: dict[str, torch.Tensor]) -> torch.Tensor:
    response = out["response_probs"]
    listen = response[..., 0]
    like = response[..., 1]
    dislike = response[..., 2]
    unlike = response[..., 3]
    undislike = response[..., 4]
    play = out["predicted_play_ratio"].clamp(0.0, 1.0) * listen
    reward = play + 0.8 * like - 1.2 * dislike - 0.6 * unlike + 0.2 * undislike
    return reward.clamp(-2.0, 2.0)


def contrastive_candidate_loss(
    model: FuturePredictor,
    batch: dict[str, torch.Tensor],
    candidate_k: int,
    temperature: float,
    negative_mode: str,
    semantic_prefix_level: int,
) -> dict[str, torch.Tensor]:
    candidates, source_stats = make_action_conditioned_candidates(batch, candidate_k, negative_mode, semantic_prefix_level)
    out = model(batch, candidates)
    scores = expected_formula_reward(out) / max(float(temperature), 1e-6)
    target = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    loss = torch.nn.functional.cross_entropy(scores, target)
    order = scores.argsort(dim=1, descending=True)
    ranks = (order == 0).float().argmax(dim=1) + 1
    return {
        "loss": loss,
        "top1": (ranks == 1).float().mean().detach(),
        "mrr": (1.0 / ranks.float()).mean().detach(),
        **{key: scores.new_tensor(value) for key, value in source_stats.items()},
    }


def candidate_ce_loss(
    model: FuturePredictor,
    batch: dict[str, torch.Tensor],
    candidate_k: int,
    temperature: float,
    negative_mode: str,
    semantic_prefix_level: int,
) -> dict[str, torch.Tensor]:
    candidates, source_stats = make_action_conditioned_candidates(batch, candidate_k, negative_mode, semantic_prefix_level)
    out = model(batch, candidates)
    scores = out["candidate_logit"] / max(float(temperature), 1e-6)
    target = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    loss = torch.nn.functional.cross_entropy(scores, target)
    order = scores.argsort(dim=1, descending=True)
    ranks = (order == 0).float().argmax(dim=1) + 1
    return {
        "loss": loss,
        "top1": (ranks == 1).float().mean().detach(),
        "mrr": (1.0 / ranks.float()).mean().detach(),
        **{key: scores.new_tensor(value) for key, value in source_stats.items()},
    }


def run_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    train: bool,
    desc: str,
    total_batches: int | None,
    response_pos_weight: torch.Tensor | None,
    response_class_weight: torch.Tensor | None,
    regret_class_weight: torch.Tensor | None,
    future_regret_pos_weight: torch.Tensor | None,
    response_weight: float,
    play_weight: float,
    reward_weight: float,
    regret_weight: float,
    future_return_weight: float,
    future_regret_weight: float,
    contrastive_weight: float,
    contrastive_candidate_k: int,
    contrastive_temperature: float,
    candidate_ce_weight: float,
    candidate_ce_k: int,
    candidate_ce_temperature: float,
    candidate_negative_mode: str,
    semantic_prefix_level: int,
    sid_loss_weight: float,
) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {}
    n_batches = 0
    correct_response = 0
    response_tp = 0.0
    response_fp = 0.0
    response_fn = 0.0
    explicit_regret_tp = 0.0
    explicit_regret_fn = 0.0
    future_regret_tp = 0.0
    future_regret_fp = 0.0
    future_regret_fn = 0.0
    future_return_abs_error = 0.0
    contrastive_top1_sum = 0.0
    contrastive_mrr_sum = 0.0
    candidate_top1_sum = 0.0
    candidate_mrr_sum = 0.0
    candidate_history_share_sum = 0.0
    candidate_semantic_share_sum = 0.0
    candidate_fallback_share_sum = 0.0
    sid_acc_sums: list[float] = []
    sid_active_batches = 0
    n_examples = 0
    pbar = tqdm(loader, total=total_batches, desc=desc, unit="batch", dynamic_ncols=True)
    for batch in pbar:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            out = model(batch)
            losses = predictor_loss(
                out,
                batch,
                response_weight=response_weight,
                play_weight=play_weight,
                reward_weight=reward_weight,
                regret_weight=regret_weight,
                future_return_weight=future_return_weight,
                future_regret_weight=future_regret_weight,
                future_regret_pos_weight=future_regret_pos_weight,
                response_pos_weight=response_pos_weight,
                response_class_weight=response_class_weight,
                regret_class_weight=regret_class_weight,
            )
            if float(contrastive_weight) > 0.0:
                contrast = contrastive_candidate_loss(
                    model,
                    batch,
                    candidate_k=contrastive_candidate_k,
                    temperature=contrastive_temperature,
                    negative_mode=candidate_negative_mode,
                    semantic_prefix_level=semantic_prefix_level,
                )
                losses["contrastive_loss"] = contrast["loss"].detach()
                losses["loss"] = losses["loss"] + float(contrastive_weight) * contrast["loss"]
                contrastive_top1_sum += float(contrast["top1"].cpu())
                contrastive_mrr_sum += float(contrast["mrr"].cpu())
            if float(candidate_ce_weight) > 0.0:
                candidate = candidate_ce_loss(
                    model,
                    batch,
                    candidate_k=candidate_ce_k,
                    temperature=candidate_ce_temperature,
                    negative_mode=candidate_negative_mode,
                    semantic_prefix_level=semantic_prefix_level,
                )
                losses["candidate_ce_loss"] = candidate["loss"].detach()
                losses["loss"] = losses["loss"] + float(candidate_ce_weight) * candidate["loss"]
                candidate_top1_sum += float(candidate["top1"].cpu())
                candidate_mrr_sum += float(candidate["mrr"].cpu())
                candidate_history_share_sum += float(candidate["history_negative_share"].cpu())
                candidate_semantic_share_sum += float(candidate["semantic_negative_share"].cpu())
                candidate_fallback_share_sum += float(candidate["fallback_negative_share"].cpu())
            if float(sid_loss_weight) > 0.0:
                sid = semantic_sid_loss(out, batch)
                if sid["active"]:
                    losses["sid_loss"] = sid["loss"].detach()
                    losses["loss"] = losses["loss"] + float(sid_loss_weight) * sid["loss"]
                    sid_active_batches += 1
                    if not sid_acc_sums:
                        sid_acc_sums = [0.0 for _ in sid["acc"]]
                    for idx, value in enumerate(sid["acc"]):
                        sid_acc_sums[idx] += float(value.cpu())
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        n_batches += 1
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        pred = out["response_logits"].argmax(dim=-1)
        correct_response += int((pred == batch["response_target"]).sum().detach().cpu())
        response_multi = (out["response_probs"] >= 0.5).float()
        response_target_multi = batch["response_targets"].float()
        response_tp += float((response_multi * response_target_multi).sum().detach().cpu())
        response_fp += float((response_multi * (1.0 - response_target_multi)).sum().detach().cpu())
        response_fn += float(((1.0 - response_multi) * response_target_multi).sum().detach().cpu())
        regret_pred = out["regret_logits"].argmax(dim=-1)
        explicit_target = batch["regret_type_id"].long() > 1
        explicit_pred = regret_pred > 1
        explicit_regret_tp += float((explicit_target & explicit_pred).sum().detach().cpu())
        explicit_regret_fn += float((explicit_target & (~explicit_pred)).sum().detach().cpu())
        if "future_regret_prob" in out and "future_regret_any" in batch:
            future_target = batch["future_regret_any"].float() > 0.5
            future_pred = out["future_regret_prob"] >= 0.5
            future_regret_tp += float((future_target & future_pred).sum().detach().cpu())
            future_regret_fp += float(((~future_target) & future_pred).sum().detach().cpu())
            future_regret_fn += float((future_target & (~future_pred)).sum().detach().cpu())
        if "predicted_future_return" in out and "future_return" in batch:
            future_return_abs_error += float((out["predicted_future_return"] - batch["future_return"].float()).abs().sum().detach().cpu())
        n_examples += int(pred.numel())
        response_precision = response_tp / max(response_tp + response_fp, 1e-8)
        response_recall = response_tp / max(response_tp + response_fn, 1e-8)
        response_f1 = 2.0 * response_precision * response_recall / max(response_precision + response_recall, 1e-8)
        explicit_recall = explicit_regret_tp / max(explicit_regret_tp + explicit_regret_fn, 1e-8)
        pbar.set_postfix(
            loss=format_float(losses["loss"].detach().cpu()),
            resp_f1=format_float(response_f1),
            expneg_rec=format_float(explicit_recall),
            future_mae=format_float(future_return_abs_error / max(n_examples, 1)),
            c_top1=format_float(contrastive_top1_sum / max(n_batches, 1)) if float(contrastive_weight) > 0.0 else 0,
            cand_top1=format_float(candidate_top1_sum / max(n_batches, 1)) if float(candidate_ce_weight) > 0.0 else 0,
            hist_neg=format_float(candidate_history_share_sum / max(n_batches, 1)) if float(candidate_ce_weight) > 0.0 else 0,
            sid_l1=format_float(sid_acc_sums[0] / max(sid_active_batches, 1)) if sid_acc_sums else 0,
            examples=n_examples,
            refresh=False,
        )
    if n_batches == 0:
        return {"loss": 0.0}
    metrics = {key: value / n_batches for key, value in totals.items()}
    metrics["response_acc_primary"] = correct_response / max(n_examples, 1)
    response_precision = response_tp / max(response_tp + response_fp, 1e-8)
    response_recall = response_tp / max(response_tp + response_fn, 1e-8)
    metrics["response_micro_precision"] = response_precision
    metrics["response_micro_recall"] = response_recall
    metrics["response_micro_f1"] = 2.0 * response_precision * response_recall / max(response_precision + response_recall, 1e-8)
    metrics["explicit_regret_recall"] = explicit_regret_tp / max(explicit_regret_tp + explicit_regret_fn, 1e-8)
    future_precision = future_regret_tp / max(future_regret_tp + future_regret_fp, 1e-8)
    future_recall = future_regret_tp / max(future_regret_tp + future_regret_fn, 1e-8)
    metrics["future_regret_precision"] = future_precision
    metrics["future_regret_recall"] = future_recall
    metrics["future_regret_f1"] = 2.0 * future_precision * future_recall / max(future_precision + future_recall, 1e-8)
    metrics["future_return_mae"] = future_return_abs_error / max(n_examples, 1)
    if float(contrastive_weight) > 0.0:
        metrics["contrastive_top1"] = contrastive_top1_sum / max(n_batches, 1)
        metrics["contrastive_mrr"] = contrastive_mrr_sum / max(n_batches, 1)
    if float(candidate_ce_weight) > 0.0:
        metrics["candidate_ce_top1"] = candidate_top1_sum / max(n_batches, 1)
        metrics["candidate_ce_mrr"] = candidate_mrr_sum / max(n_batches, 1)
        metrics["candidate_history_negative_share"] = candidate_history_share_sum / max(n_batches, 1)
        metrics["candidate_semantic_negative_share"] = candidate_semantic_share_sum / max(n_batches, 1)
        metrics["candidate_fallback_negative_share"] = candidate_fallback_share_sum / max(n_batches, 1)
    if float(sid_loss_weight) > 0.0 and sid_acc_sums:
        for idx, value in enumerate(sid_acc_sums, start=1):
            metrics[f"sid_acc_l{idx}"] = value / max(sid_active_batches, 1)
    metrics["examples"] = n_examples
    return metrics


def save_checkpoint(path: Path, model: FuturePredictor, args: argparse.Namespace, item_dim: int) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(args),
            "item_dim": item_dim,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    response_pos_weight = parse_weight_list(args.response_pos_weight, 5, device)
    response_class_weight = parse_weight_list(args.response_class_weight, 5, device)
    regret_class_weight = parse_weight_list(args.regret_class_weight, 4, device)
    future_regret_pos_weight = torch.tensor(float(args.future_regret_pos_weight), dtype=torch.float32, device=device)
    store = EmbedStore(args.embed_store)
    model = FuturePredictor(
        item_dim=store.dim,
        d_model=args.d_model,
        max_seq_len=50,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
        state_pooling=args.state_pooling,
        sid_levels=args.sid_levels,
        sid_vocab_size=args.sid_vocab_size,
    ).to(device)
    if args.init_ckpt:
        init = torch.load(args.init_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(init["model_state"], strict=False)
        print(json.dumps({"init_ckpt": args.init_ckpt, "missing": missing, "unexpected": unexpected}, ensure_ascii=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loader = make_loader(
            args.data_dir,
            "train",
            store,
            args.batch_size,
            args.max_train_rows,
            user_sample_mod=args.train_user_sample_mod,
            user_sample_bucket=args.train_user_sample_bucket,
            max_users=args.train_max_users,
        )
        val_loader = make_loader(args.data_dir, "val", store, args.batch_size, args.max_val_rows)
        train_total = estimate_sampled_total_batches(
            args.data_dir,
            "train",
            args.batch_size,
            args.max_train_rows,
            user_sample_mod=args.train_user_sample_mod,
            max_users=args.train_max_users,
        )
        val_total = estimate_total_batches(args.data_dir, "val", args.batch_size, args.max_val_rows)
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, train=True,
            desc=f"[predictor epoch {epoch}/{args.epochs} train]", total_batches=train_total,
            response_pos_weight=response_pos_weight, response_class_weight=response_class_weight,
            regret_class_weight=regret_class_weight, future_regret_pos_weight=future_regret_pos_weight,
            response_weight=args.response_weight, play_weight=args.play_weight,
            reward_weight=args.reward_weight, regret_weight=args.regret_weight,
            future_return_weight=args.future_return_weight,
            future_regret_weight=args.future_regret_weight,
            contrastive_weight=args.contrastive_weight,
            contrastive_candidate_k=args.contrastive_candidate_k,
            contrastive_temperature=args.contrastive_temperature,
            candidate_ce_weight=args.candidate_ce_weight,
            candidate_ce_k=args.candidate_ce_k,
            candidate_ce_temperature=args.candidate_ce_temperature,
            candidate_negative_mode=args.candidate_negative_mode,
            semantic_prefix_level=args.semantic_prefix_level,
            sid_loss_weight=args.sid_loss_weight,
        )
        val_metrics = run_epoch(
            model, val_loader, optimizer, device, train=False,
            desc=f"[predictor epoch {epoch}/{args.epochs} val]", total_batches=val_total,
            response_pos_weight=response_pos_weight, response_class_weight=response_class_weight,
            regret_class_weight=regret_class_weight, future_regret_pos_weight=future_regret_pos_weight,
            response_weight=args.response_weight, play_weight=args.play_weight,
            reward_weight=args.reward_weight, regret_weight=args.regret_weight,
            future_return_weight=args.future_return_weight,
            future_regret_weight=args.future_regret_weight,
            contrastive_weight=args.contrastive_weight,
            contrastive_candidate_k=args.contrastive_candidate_k,
            contrastive_temperature=args.contrastive_temperature,
            candidate_ce_weight=args.candidate_ce_weight,
            candidate_ce_k=args.candidate_ce_k,
            candidate_ce_temperature=args.candidate_ce_temperature,
            candidate_negative_mode=args.candidate_negative_mode,
            semantic_prefix_level=args.semantic_prefix_level,
            sid_loss_weight=args.sid_loss_weight,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if args.save_each_epoch:
            save_checkpoint(out_dir / f"future_predictor_epoch{epoch}.pt", model, args, store.dim)

    save_checkpoint(out_dir / "future_predictor.pt", model, args, store.dim)
    (out_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
