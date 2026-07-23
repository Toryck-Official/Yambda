from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "01_data"), str(ROOT / "02_model")])

from future_dataset import EmbedStore, FutureIterableDataset, collate_future
from hpn import FutureHPNPolicy
from hpn_candidates import SidPathIndex, build_hpn_candidate_batch
from predictor import FuturePredictor
from soft_state import SoftStateBuilder
from value import ValueHead, bellman_value_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train value head with Bellman-style candidate max targets.")
    parser.add_argument("--data_dir", default=str(ROOT / "artifacts" / "future_data"))
    parser.add_argument("--embed_store", default=str(ROOT / "artifacts" / "embed_store"))
    parser.add_argument("--predictor_ckpt", default=str(ROOT / "artifacts" / "predictor" / "future_predictor.pt"))
    parser.add_argument("--out_dir", default=str(ROOT / "artifacts" / "value"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--candidate_k", type=int, default=8)
    parser.add_argument("--sample_m", type=int, default=1)
    parser.add_argument("--candidate_source", choices=["inbatch", "hpn"], default="inbatch")
    parser.add_argument("--hpn_ckpt", default="")
    parser.add_argument("--dense_item2sid_npy", default="")
    parser.add_argument("--dense2orig_npy", default="")
    parser.add_argument("--top_sid_paths", type=int, default=32)
    parser.add_argument("--branch_k", type=int, default=16)
    parser.add_argument("--max_items_per_sid", type=int, default=4)
    parser.add_argument("--max_index_items", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--state_loss_weight", type=float, default=1.0)
    parser.add_argument("--candidate_loss_weight", type=float, default=0.1)
    parser.add_argument("--target_tau", type=float, default=0.01)
    parser.add_argument("--max_train_rows", type=int, default=5000)
    parser.add_argument("--max_val_rows", type=int, default=1000)
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


def make_loader(data_dir: str, split: str, store: EmbedStore, batch_size: int, max_rows: int) -> DataLoader:
    dataset = FutureIterableDataset(data_dir, split=split, max_rows=max_rows)
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
    for param in model.parameters():
        param.requires_grad_(False)
    return model


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


def sample_one_hot(probs: torch.Tensor, sample_m: int) -> torch.Tensor:
    batch_size, candidate_k, n_class = probs.shape
    flat = probs.reshape(batch_size * candidate_k, n_class).clamp_min(1e-8)
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    idx = torch.multinomial(flat, int(sample_m), replacement=True)
    one_hot = F.one_hot(idx, num_classes=n_class).to(dtype=probs.dtype)
    return one_hot.view(batch_size, candidate_k, sample_m, n_class).permute(0, 2, 1, 3).reshape(
        batch_size * sample_m, candidate_k, n_class
    )


def sample_multi_label(probs: torch.Tensor, sample_m: int) -> torch.Tensor:
    batch_size, candidate_k, n_class = probs.shape
    rep = probs.unsqueeze(1).expand(batch_size, sample_m, candidate_k, n_class)
    return torch.bernoulli(rep.clamp(0.0, 1.0)).reshape(batch_size * sample_m, candidate_k, n_class)


def repeat_candidate_tensor(x: torch.Tensor, sample_m: int) -> torch.Tensor:
    batch_size, candidate_k = x.shape[:2]
    return x.unsqueeze(1).expand(batch_size, sample_m, *x.shape[1:]).reshape(batch_size * sample_m, candidate_k, *x.shape[2:])


def sampled_bellman_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    candidate_features: torch.Tensor,
    candidate_mask: torch.Tensor | None,
    soft_state,
    value_head,
    target_value_head,
    gamma: float,
    sample_m: int,
) -> dict[str, torch.Tensor]:
    sample_m = max(int(sample_m), 1)
    state_value = value_head(pred["state_emb"])
    if sample_m == 1:
        soft = soft_state(pred["state_emb"], candidate_features, pred)
        next_value = target_value_head(
            soft["next_state_emb"].reshape(-1, soft["next_state_emb"].shape[-1])
        ).view(candidate_features.shape[0], candidate_features.shape[1])
        return bellman_value_loss(
            state_value,
            pred["predicted_reward"],
            next_value,
            gamma=gamma,
            candidate_mask=candidate_mask,
        )

    batch_size, candidate_k, item_dim = candidate_features.shape
    state_rep = pred["state_emb"].unsqueeze(1).expand(batch_size, sample_m, -1).reshape(batch_size * sample_m, -1)
    candidate_rep = candidate_features.unsqueeze(1).expand(batch_size, sample_m, candidate_k, item_dim).reshape(
        batch_size * sample_m, candidate_k, item_dim
    )
    pred_sampled = {
        "response_probs": sample_multi_label(pred["response_probs"], sample_m),
        "regret_probs": sample_one_hot(pred["regret_probs"], sample_m),
        "predicted_play_ratio": repeat_candidate_tensor(pred["predicted_play_ratio"], sample_m),
        "predicted_reward": repeat_candidate_tensor(pred["predicted_reward"], sample_m),
    }
    soft = soft_state(state_rep, candidate_rep, pred_sampled)
    next_value = target_value_head(
        soft["next_state_emb"].reshape(-1, soft["next_state_emb"].shape[-1])
    ).view(batch_size, sample_m, candidate_k)
    reward_rep = pred_sampled["predicted_reward"].view(batch_size, sample_m, candidate_k)
    target_q = reward_rep + float(gamma) * next_value
    if candidate_mask is not None:
        mask_rep = candidate_mask.unsqueeze(1).expand(batch_size, sample_m, candidate_k)
        target_q = target_q.masked_fill(~mask_rep.bool(), -1e9)
    target = target_q.max(dim=2).values.mean(dim=1).detach()
    loss = F.mse_loss(state_value, target)
    return {"loss": loss, "target_value": target}


def update_target(source, target, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.mul_(1.0 - float(tau)).add_(source_param, alpha=float(tau))


def run_epoch(
    predictor,
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
) -> dict[str, float]:
    soft_state.train(train)
    value_head.train(train)
    totals = {
        "loss": 0.0,
        "value_loss": 0.0,
        "state_loss": 0.0,
        "candidate_loss": 0.0,
        "target_mean": 0.0,
        "state_value_mean": 0.0,
        "empty_candidate_rows": 0.0,
    }
    n_batches = 0
    for batch in loader:
        batch = move_batch(batch, device)
        candidate_features, candidate_mask, empty_rows = make_candidates(batch, args, hpn_pack, store, device)
        with torch.no_grad():
            pred = predictor(batch, candidate_features)
            logged_pred = predictor(batch)
            next_state = predictor.state_encoder(batch, prefix="next_history")["state_emb"]
            next_target_value = target_value_head(next_state)
        candidate_losses = sampled_bellman_loss(
            pred,
            batch,
            candidate_features,
            candidate_mask,
            soft_state,
            value_head,
            target_value_head,
            gamma=args.gamma,
            sample_m=args.sample_m,
        )
        soft_logged = soft_state(logged_pred["state_emb"], batch["action_features"], logged_pred)
        state_loss = F.mse_loss(soft_logged["next_state_emb"], next_state.detach())
        state_value = value_head(logged_pred["state_emb"])
        td_target = (
            batch["reward"]
            + float(args.gamma) * batch["bootstrap_mask"] * next_target_value
        ).detach()
        mc_target = batch["future_return"].detach()
        value_loss = 0.5 * F.mse_loss(state_value, td_target) + 0.5 * F.mse_loss(state_value, mc_target)
        loss = (
            value_loss
            + float(args.state_loss_weight) * state_loss
            + float(args.candidate_loss_weight) * candidate_losses["loss"]
        )
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(soft_state.parameters()) + list(value_head.parameters()), 1.0)
            optimizer.step()
            update_target(value_head, target_value_head, args.target_tau)
        totals["loss"] += float(loss.detach().cpu())
        totals["value_loss"] += float(value_loss.detach().cpu())
        totals["state_loss"] += float(state_loss.detach().cpu())
        totals["candidate_loss"] += float(candidate_losses["loss"].detach().cpu())
        totals["target_mean"] += float(td_target.mean().detach().cpu())
        totals["state_value_mean"] += float(state_value.mean().detach().cpu())
        totals["empty_candidate_rows"] += float(empty_rows)
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    store = EmbedStore(args.embed_store)

    ckpt = torch.load(args.predictor_ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})
    predictor = FuturePredictor(
        item_dim=int(ckpt.get("item_dim", store.dim)),
        d_model=int(cfg.get("d_model", 128)),
        max_seq_len=int(cfg.get("max_seq_len", 50)),
        n_layer=int(cfg.get("n_layer", 2)),
        n_head=int(cfg.get("n_head", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    predictor.load_state_dict(ckpt["model_state"])
    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad_(False)

    hpn_pack = None
    if args.candidate_source == "hpn":
        if not args.hpn_ckpt or not args.dense_item2sid_npy or not args.dense2orig_npy:
            raise RuntimeError("--candidate_source hpn requires --hpn_ckpt, --dense_item2sid_npy, and --dense2orig_npy.")
        dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
        hpn_pack = {
            "hpn": load_hpn(args.hpn_ckpt, store.dim, device),
            "index": SidPathIndex(dense_item2sid, args.max_items_per_sid, args.max_index_items),
            "dense2orig": np.load(args.dense2orig_npy, mmap_mode="r"),
        }

    soft_state = SoftStateBuilder(item_dim=store.dim, d_model=int(cfg.get("d_model", 128))).to(device)
    value_head = ValueHead(d_model=int(cfg.get("d_model", 128))).to(device)
    target_value_head = copy.deepcopy(value_head).to(device).eval()
    for param in target_value_head.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.AdamW(list(soft_state.parameters()) + list(value_head.parameters()), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loader = make_loader(args.data_dir, "train", store, args.batch_size, args.max_train_rows)
        val_loader = make_loader(args.data_dir, "val", store, args.batch_size, args.max_val_rows)
        train_metrics = run_epoch(
            predictor,
            soft_state,
            value_head,
            target_value_head,
            train_loader,
            optimizer,
            device,
            args,
            train=True,
            hpn_pack=hpn_pack,
            store=store,
        )
        val_metrics = run_epoch(
            predictor,
            soft_state,
            value_head,
            target_value_head,
            val_loader,
            optimizer,
            device,
            args,
            train=False,
            hpn_pack=hpn_pack,
            store=store,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    torch.save(
        {
            "soft_state": soft_state.state_dict(),
            "value_head": value_head.state_dict(),
            "config": vars(args),
            "predictor_config": cfg,
            "item_dim": store.dim,
        },
        out_dir / "future_value.pt",
    )
    (out_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
