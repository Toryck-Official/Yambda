from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = Path(__file__).resolve().parent
sys.path.extend([str(ROOT / "01_data"), str(BASELINE_DIR)])

from baseline_models import SASRecSIDPolicy, sid_ce_loss  # noqa: E402
from future_dataset import EmbedStore, FutureIterableDataset, collate_future  # noqa: E402
from progress_utils import estimate_total_batches, format_float  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train plain SASRec-SID baseline.")
    parser.add_argument("--data_dir", default=str(ROOT / "01_data" / "processed" / "predictor_seq_data"))
    parser.add_argument("--embed_store", default=str(ROOT / "01_data" / "processed" / "raw_rqkmeans"))
    parser.add_argument("--out_dir", default=str(BASELINE_DIR / "artifacts" / "sasrec_sid"))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_train_rows", type=int, default=1000000)
    parser.add_argument("--max_val_rows", type=int, default=100000)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--state_pooling", choices=["last", "mean", "last_mean"], default="last_mean")
    parser.add_argument("--sid_levels", type=int, default=4)
    parser.add_argument("--sid_vocab_size", type=int, default=256)
    parser.add_argument("--use_history_feedback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_history_event_type", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name if name != "cuda" or torch.cuda.is_available() else "cpu")
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


def make_loader(data_dir: str, split: str, store: EmbedStore, batch_size: int, max_rows: int) -> DataLoader:
    dataset = FutureIterableDataset(data_dir, split=split, max_rows=max_rows, mapping_root=store.root)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=lambda rows: collate_future(rows, store))


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: SASRecSIDPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
    total_batches: int | None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    n_batches = 0
    n_examples = 0
    pbar = tqdm(loader, total=total_batches, desc=desc, unit="batch", dynamic_ncols=True)
    for batch in pbar:
        batch = move_batch(batch, device)
        if "target_sid" not in batch or batch["target_sid"].numel() == 0:
            raise RuntimeError("future data has no target_sid.")
        with torch.set_grad_enabled(train):
            out = model(batch)
            losses = sid_ce_loss(out["sid_logits"], batch["target_sid"])  # type: ignore[arg-type]
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        n_batches += 1
        n_examples += int(batch["target_sid"].shape[0])
        pbar.set_postfix(
            loss=format_float(losses["loss"].detach().cpu()),
            full=format_float(losses["full_path_acc"].detach().cpu()),
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
    model = SASRecSIDPolicy(
        item_dim=store.dim,
        d_model=args.d_model,
        max_seq_len=50,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
        sid_levels=args.sid_levels,
        sid_vocab_size=args.sid_vocab_size,
        state_pooling=args.state_pooling,
        use_history_feedback=args.use_history_feedback,
        use_history_event_type=args.use_history_event_type,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loader = make_loader(args.data_dir, "train", store, args.batch_size, args.max_train_rows)
        val_loader = make_loader(args.data_dir, "val", store, args.batch_size, args.max_val_rows)
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            desc=f"[sasrec epoch {epoch}/{args.epochs} train]",
            total_batches=estimate_total_batches(args.data_dir, "train", args.batch_size, args.max_train_rows),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device,
            desc=f"[sasrec epoch {epoch}/{args.epochs} val]",
            total_batches=estimate_total_batches(args.data_dir, "val", args.batch_size, args.max_val_rows),
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    torch.save(
        {
            "model_type": "sasrec_sid",
            "model_state": model.state_dict(),
            "config": vars(args),
            "item_dim": store.dim,
        },
        out_dir / "sasrec_sid.pt",
    )
    (out_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] model saved to {out_dir / 'sasrec_sid.pt'}")


if __name__ == "__main__":
    main()

