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
from hpn import FutureHPNPolicy, hpn_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HPN on future_data target SID labels.")
    parser.add_argument("--data_dir", default=str(ROOT / "artifacts" / "future_data"))
    parser.add_argument("--embed_store", default=str(ROOT / "artifacts" / "embed_store"))
    parser.add_argument("--out_dir", default=str(ROOT / "artifacts" / "hpn"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_train_rows", type=int, default=5000)
    parser.add_argument("--max_val_rows", type=int, default=1000)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sid_levels", type=int, default=4)
    parser.add_argument("--sid_vocab_size", type=int, default=256)
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


def make_loader(data_dir: str, split: str, store: EmbedStore, batch_size: int, max_rows: int) -> DataLoader:
    dataset = FutureIterableDataset(data_dir, split=split, max_rows=max_rows)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=lambda rows: collate_future(rows, store))


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(model, loader, optimizer, device: torch.device, train: bool) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = move_batch(batch, device)
        if "target_sid" not in batch or batch["target_sid"].numel() == 0:
            raise RuntimeError("future_data has no target_sid. Rebuild with --orig2dense_npy and --dense_item2sid_npy.")
        with torch.set_grad_enabled(train):
            out = model(batch)
            losses = hpn_loss(out, batch["target_sid"])
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    store = EmbedStore(args.embed_store)
    model = FutureHPNPolicy(
        item_dim=store.dim,
        d_model=args.d_model,
        max_seq_len=50,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
        sid_levels=args.sid_levels,
        sid_vocab_size=args.sid_vocab_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loader = make_loader(args.data_dir, "train", store, args.batch_size, args.max_train_rows)
        val_loader = make_loader(args.data_dir, "val", store, args.batch_size, args.max_val_rows)
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    torch.save({"model_state": model.state_dict(), "config": vars(args), "item_dim": store.dim}, out_dir / "hpn.pt")
    (out_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
