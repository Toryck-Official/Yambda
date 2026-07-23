from __future__ import annotations

import argparse
import json
import math
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
from progress_utils import estimate_total_batches, format_float, split_row_count
from hpn import FutureHPNPolicy
from hpn_candidates import SidPathIndex, build_hpn_candidate_batch
from predictor import FuturePredictor
from soft_state import SoftStateBuilder
from value import ValueHead, bellman_value_loss


REGRET_CLASSES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train value head with Bayesian Bellman candidate max targets.")
    parser.add_argument("--data_dir", default=str(ROOT / "artifacts" / "future_data"))
    parser.add_argument("--embed_store", default=str(ROOT / "artifacts" / "embed_store"))
    parser.add_argument("--predictor_ckpt", default=str(ROOT / "artifacts" / "predictor" / "future_predictor.pt"))
    parser.add_argument("--predictor_ckpts", default="", help="Comma-separated predictor checkpoints. Overrides --predictor_ckpt when set.")
    parser.add_argument("--predictor_manifest", default="", help="JSON manifest written by train_predictor_ensemble.py.")
    parser.add_argument("--out_dir", default=str(ROOT / "artifacts" / "value"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--candidate_k", type=int, default=8)
    parser.add_argument("--sample_m", type=int, default=1, help="Number of prediction-oracle samples N2 for the Bayesian Bellman target.")
    parser.add_argument("--bayes_samples", type=int, default=0, help="Alias for --sample_m; if >0 it overrides sample_m.")
    parser.add_argument("--reward_sample_mode", choices=["sampled_formula", "predictor_mean"], default="sampled_formula")
    parser.add_argument("--sample_regret_from_response", type=int, default=1)
    parser.add_argument("--soft_state_loss_weight", type=float, default=1.0)
    parser.add_argument("--target_tau", type=float, default=0.01)
    parser.add_argument("--target_update_interval", type=int, default=1)
    parser.add_argument("--candidate_source", choices=["inbatch", "hpn"], default="inbatch")
    parser.add_argument("--hpn_ckpt", default="")
    parser.add_argument("--dense_item2sid_npy", default="")
    parser.add_argument("--dense2orig_npy", default="")
    parser.add_argument("--top_sid_paths", type=int, default=32)
    parser.add_argument("--branch_k", type=int, default=16)
    parser.add_argument("--max_items_per_sid", type=int, default=4, help="Deprecated with direct SID item scoring; kept for CLI compatibility.")
    parser.add_argument("--max_index_items", type=int, default=0, help="Limit HPN candidate pool size for smoke runs; 0 means all valid items.")
    parser.add_argument("--candidate_chunk_size", type=int, default=65536, help="Chunk size for direct HPN item-pool SID scoring.")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--max_train_rows", type=int, default=5000)
    parser.add_argument("--max_val_rows", type=int, default=1000)
    parser.add_argument("--train_user_sample_mod", type=int, default=0, help="Keep train users whose stable hash bucket equals bucket; 0 disables sampling.")
    parser.add_argument("--train_user_sample_bucket", type=int, default=0)
    parser.add_argument("--train_max_users", type=int, default=0)
    parser.add_argument("--save_each_epoch", type=int, default=1)
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


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


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


def make_inbatch_candidates(action_features: torch.Tensor, candidate_k: int) -> torch.Tensor:
    batch_size = action_features.shape[0]
    parts = [action_features]
    for shift in range(1, max(candidate_k, 1)):
        parts.append(torch.roll(action_features, shifts=shift % max(batch_size, 1), dims=0))
    return torch.stack(parts[:candidate_k], dim=1)


def resolve_predictor_ckpts(args: argparse.Namespace) -> list[str]:
    if args.predictor_manifest:
        manifest = json.loads(Path(args.predictor_manifest).read_text(encoding="utf-8"))
        ckpts = [str(item) for item in manifest.get("predictor_ckpts", [])]
    elif args.predictor_ckpts:
        ckpts = [item.strip() for item in args.predictor_ckpts.split(",") if item.strip()]
    else:
        ckpts = [str(args.predictor_ckpt)]
    if not ckpts:
        raise RuntimeError("No predictor checkpoints were provided.")
    for ckpt in ckpts:
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"Predictor checkpoint not found: {ckpt}")
    return ckpts


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
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_predictor(path: str, item_dim: int, device: torch.device) -> tuple[FuturePredictor, dict, int]:
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
    return model, cfg, int(ckpt.get("item_dim", item_dim))


def make_candidates(batch, args, hpn_pack, store, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    if args.candidate_source == "inbatch":
        return make_inbatch_candidates(batch["action_features"], args.candidate_k), None, 0
    if hpn_pack is None:
        raise RuntimeError("--candidate_source hpn requires HPN candidate resources.")
    out = build_hpn_candidate_batch(
        batch=batch,
        hpn=hpn_pack["hpn"],
        index=hpn_pack["index"],
        dense2orig=hpn_pack["dense2orig"],
        store=store,
        device=device,
        max_candidates=args.candidate_k,
        top_sid_paths=args.top_sid_paths,
        branch_k=args.branch_k,
        fallback_to_logged=True,
    )
    return out["candidate_features"], out["candidate_mask"], int(out["empty_rows"])


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


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    tau = float(tau)
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def make_next_history_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "history_features": batch["next_history_features"],
        "history_feedbacks": batch["next_history_feedbacks"],
        "history_event_type_ids": batch["next_history_event_type_ids"],
        "history_response_targets": batch["next_history_response_targets"],
        "history_play_ratios": batch["next_history_play_ratios"],
        "history_play_excesses": batch["next_history_play_excesses"],
        "history_is_organic": batch["next_history_is_organic"],
        "history_time_gap_seconds": batch["next_history_time_gap_seconds"],
        "history_same_session": batch["next_history_same_session"],
        "history_has_rich_features": batch["next_history_has_rich_features"],
        "history_mask": batch.get("next_history_mask"),
    }


def predictor_oracle_outputs(predictors: list[FuturePredictor], batch: dict[str, torch.Tensor], candidate_features: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    outs = []
    with torch.no_grad():
        for predictor in predictors:
            outs.append(predictor(batch, candidate_features))
    return outs


def sampled_bayesian_bellman_loss(
    pred_samples: list[dict[str, torch.Tensor]],
    candidate_features: torch.Tensor,
    candidate_mask: torch.Tensor | None,
    soft_state,
    value_head,
    target_value_head,
    gamma: float,
    sample_m: int,
    reward_sample_mode: str,
    sample_regret_from_response: bool,
) -> dict[str, torch.Tensor]:
    sample_m = max(int(sample_m), 1)
    primary = pred_samples[0]
    state_emb = primary["state_emb"]
    state_value = value_head(state_emb)
    next_value_head = target_value_head if target_value_head is not None else value_head
    sample_targets: list[torch.Tensor] = []
    sample_rewards: list[torch.Tensor] = []
    sample_regret_risks: list[torch.Tensor] = []

    for sample_idx in range(sample_m):
        pred = pred_samples[sample_idx % len(pred_samples)]
        if reward_sample_mode == "sampled_formula":
            response = sample_multi_label(pred["response_probs"])
            reward = response_to_reward(response, pred["predicted_play_ratio"])
            if sample_regret_from_response:
                regret_probs = response_to_regret_probs(response)
            else:
                regret_probs = sample_one_hot(pred["regret_probs"])
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
        soft = soft_state(state_emb, candidate_features, pred_for_soft)
        next_value = next_value_head(soft["next_state_emb"].reshape(-1, soft["next_state_emb"].shape[-1])).view(
            candidate_features.shape[0], candidate_features.shape[1]
        )
        target_q = reward + float(gamma) * next_value
        if candidate_mask is not None:
            target_q = target_q.masked_fill(~candidate_mask.bool(), -1e9)
        sample_targets.append(target_q.max(dim=1).values)
        sample_rewards.append(reward.mean())
        sample_regret_risks.append(regret_probs[..., 1:].sum(dim=-1).mean())

    target = torch.stack(sample_targets, dim=1).mean(dim=1).detach()
    loss = F.mse_loss(state_value, target)
    return {
        "loss": loss,
        "target_value": target,
        "sample_reward_mean": torch.stack(sample_rewards).mean().detach(),
        "sample_regret_risk": torch.stack(sample_regret_risks).mean().detach(),
    }


def run_epoch(
    predictors: list[FuturePredictor],
    soft_state,
    value_head,
    target_value_head,
    loader,
    optimizer,
    device: torch.device,
    args,
    train: bool,
    hpn_pack,
    store,
    desc: str,
    total_batches: int | None,
) -> dict[str, float]:
    soft_state.train(train)
    value_head.train(train)
    if target_value_head is not None:
        target_value_head.eval()
    totals = {
        "loss": 0.0,
        "bellman_loss": 0.0,
        "soft_state_loss": 0.0,
        "target_mean": 0.0,
        "state_value_mean": 0.0,
        "sample_reward_mean": 0.0,
        "sample_regret_risk": 0.0,
        "empty_candidate_rows": 0.0,
    }
    n_batches = 0
    n_examples = 0
    pbar = tqdm(loader, total=total_batches, desc=desc, unit="batch", dynamic_ncols=True)
    for batch in pbar:
        batch = move_batch(batch, device)
        candidate_features, candidate_mask, empty_rows = make_candidates(batch, args, hpn_pack, store, device)
        pred_samples = predictor_oracle_outputs(predictors, batch, candidate_features)
        sample_count = int(args.bayes_samples) if int(args.bayes_samples) > 0 else int(args.sample_m)
        losses = sampled_bayesian_bellman_loss(
            pred_samples,
            candidate_features,
            candidate_mask,
            soft_state,
            value_head,
            target_value_head,
            gamma=args.gamma,
            sample_m=sample_count,
            reward_sample_mode=args.reward_sample_mode,
            sample_regret_from_response=bool(args.sample_regret_from_response),
        )
        bellman_loss = losses["loss"]
        soft_state_loss = bellman_loss.new_tensor(0.0)
        if float(args.soft_state_loss_weight) > 0 and "next_history_features" in batch:
            with torch.no_grad():
                logged_pred = predictors[0](batch)
                next_state_target = predictors[0].encode_history(make_next_history_batch(batch))["state_emb"]
            soft_logged = soft_state(logged_pred["state_emb"], batch["action_features"], logged_pred)
            soft_state_loss = F.mse_loss(soft_logged["next_state_emb"], next_state_target)
            losses["loss"] = bellman_loss + float(args.soft_state_loss_weight) * soft_state_loss
        losses["bellman_loss"] = bellman_loss
        losses["soft_state_loss"] = soft_state_loss
        state_value = value_head(pred_samples[0]["state_emb"])
        if train:
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(list(soft_state.parameters()) + list(value_head.parameters()), 1.0)
            optimizer.step()
            if target_value_head is not None and int(args.target_update_interval) > 0 and (n_batches + 1) % int(args.target_update_interval) == 0:
                soft_update(target_value_head, value_head, float(args.target_tau))
        totals["loss"] += float(losses["loss"].detach().cpu())
        totals["bellman_loss"] += float(losses["bellman_loss"].detach().cpu())
        totals["soft_state_loss"] += float(losses["soft_state_loss"].detach().cpu())
        totals["target_mean"] += float(losses["target_value"].mean().detach().cpu())
        totals["state_value_mean"] += float(state_value.mean().detach().cpu())
        totals["sample_reward_mean"] += float(losses["sample_reward_mean"].detach().cpu())
        totals["sample_regret_risk"] += float(losses["sample_regret_risk"].detach().cpu())
        totals["empty_candidate_rows"] += float(empty_rows)
        n_batches += 1
        n_examples += int(batch["target_dense_item_id"].shape[0])
        pbar.set_postfix(
            loss=format_float(losses["loss"].detach().cpu()),
            soft=format_float(losses["soft_state_loss"].detach().cpu()),
            target=format_float(losses["target_value"].mean().detach().cpu()),
            rew=format_float(losses["sample_reward_mean"].detach().cpu()),
            empty=empty_rows,
            examples=n_examples,
            refresh=False,
        )
    metrics = {key: value / max(n_batches, 1) for key, value in totals.items()}
    metrics["examples"] = n_examples
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    store = EmbedStore(args.embed_store)

    predictor_ckpts = resolve_predictor_ckpts(args)
    predictors: list[FuturePredictor] = []
    predictor_cfg: dict = {}
    predictor_item_dim = store.dim
    for idx, ckpt_path in enumerate(predictor_ckpts):
        predictor, cfg, item_dim = load_predictor(ckpt_path, store.dim, device)
        predictors.append(predictor)
        if idx == 0:
            predictor_cfg = cfg
            predictor_item_dim = item_dim
    d_model = int(predictor_cfg.get("d_model", 128))

    hpn_pack = None
    if args.candidate_source == "hpn":
        if not args.hpn_ckpt or not args.dense_item2sid_npy or not args.dense2orig_npy:
            raise RuntimeError("--candidate_source hpn requires --hpn_ckpt, --dense_item2sid_npy, and --dense2orig_npy.")
        dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
        hpn_pack = {
            "hpn": load_hpn(args.hpn_ckpt, store.dim, device),
            "index": SidPathIndex(dense_item2sid, args.max_items_per_sid, args.max_index_items, args.candidate_chunk_size),
            "dense2orig": np.load(args.dense2orig_npy, mmap_mode="r"),
        }

    soft_state = SoftStateBuilder(item_dim=store.dim, d_model=d_model).to(device)
    value_head = ValueHead(d_model=d_model).to(device)
    target_value_head = ValueHead(d_model=d_model).to(device)
    target_value_head.load_state_dict(value_head.state_dict())
    for param in target_value_head.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.AdamW(list(soft_state.parameters()) + list(value_head.parameters()), lr=args.lr, weight_decay=1e-4)

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
            predictors, soft_state, value_head, target_value_head, train_loader, optimizer, device, args, train=True,
            hpn_pack=hpn_pack, store=store, desc=f"[value epoch {epoch}/{args.epochs} train]", total_batches=train_total
        )
        val_metrics = run_epoch(
            predictors, soft_state, value_head, target_value_head, val_loader, optimizer, device, args, train=False,
            hpn_pack=hpn_pack, store=store, desc=f"[value epoch {epoch}/{args.epochs} val]", total_batches=val_total
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if args.save_each_epoch:
            torch.save(
                {
                    "soft_state": soft_state.state_dict(),
                    "value_head": value_head.state_dict(),
                    "config": vars(args),
                    "predictor_config": predictor_cfg,
                    "predictor_ckpts": predictor_ckpts,
                    "predictor_item_dim": predictor_item_dim,
                    "item_dim": store.dim,
                },
                out_dir / f"future_value_epoch{epoch}.pt",
            )

    torch.save(
        {
            "soft_state": soft_state.state_dict(),
            "value_head": value_head.state_dict(),
            "config": vars(args),
            "predictor_config": predictor_cfg,
            "predictor_ckpts": predictor_ckpts,
            "predictor_item_dim": predictor_item_dim,
            "item_dim": store.dim,
        },
        out_dir / "future_value.pt",
    )
    (out_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
