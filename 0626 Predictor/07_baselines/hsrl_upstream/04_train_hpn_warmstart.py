#!/usr/bin/env python3
"""Optional HPN warm-start training entry kept for baseline reproducibility."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


BASELINE_DIR = Path(__file__).resolve().parents[1]
PREDICTOR_ROOT = BASELINE_DIR.parent
WORKSPACE_ROOT = PREDICTOR_ROOT.parent
PROJECT_ROOT = Path(os.environ.get("HSRL_PROJECT_ROOT", str(WORKSPACE_ROOT / "HSRL")))
DEFAULT_YAMBA_DATA_DIR = Path(os.environ.get("YAMBA_DATA_DIR", str(PROJECT_ROOT / "../0330Yambda/data")))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from adapter.bootstrap import install_hsrl_adapter  # noqa: E402

install_hsrl_adapter()

from model.policy.SIDPolicy_credit import SIDPolicy_credit  # type: ignore  # noqa: E402
from utils import set_random_seed  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional warm-start HPN on Yambda split TSV files")
    parser.add_argument("--train_file", type=str, default=str(PROJECT_ROOT / "artifacts/processed/train.tsv"))
    parser.add_argument("--val_file", type=str, default=str(PROJECT_ROOT / "artifacts/processed/val.tsv"))
    parser.add_argument("--embeddings_parquet", type=str, default=str(DEFAULT_YAMBA_DATA_DIR / "embeddings.parquet"))
    parser.add_argument("--orig2dense_npy", type=str, default=str(PROJECT_ROOT / "artifacts/mappings/yambda_orig2dense_item_id.npy"))
    parser.add_argument("--dense_item2sid_npy", type=str, default=str(PROJECT_ROOT / "artifacts/mappings/yambda_dense_item2sid.npy"))
    parser.add_argument("--embedding_column", type=str, default="normalized_embed", choices=["embed", "normalized_embed"])
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_val_rows", type=int, default=0)
    parser.add_argument("--train_positive_only", action="store_true")
    parser.add_argument("--min_train_reward", type=float, default=-1e9)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--n_worker", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--sasrec_n_layer", type=int, default=2)
    parser.add_argument("--sasrec_d_model", type=int, default=64)
    parser.add_argument("--sasrec_d_forward", type=int, default=128)
    parser.add_argument("--sasrec_n_head", type=int, default=4)
    parser.add_argument("--sasrec_dropout", type=float, default=0.1)
    parser.add_argument("--sid_temp", type=float, default=1.0)
    parser.add_argument("--save_path", type=str, default=str(PROJECT_ROOT / "artifacts/models/hpn_warmstart.pt"))
    parser.add_argument("--save_meta", type=str, default=str(PROJECT_ROOT / "artifacts/models/hpn_warmstart.meta.json"))
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


def parse_list_cell(cell: object) -> list[int]:
    if isinstance(cell, list):
        return [int(x) for x in cell]
    if isinstance(cell, str):
        return [] if cell == "" else [int(x) for x in ast.literal_eval(cell)]
    if pd.isna(cell):
        return []
    return [int(cell)]


def load_split_df(path: Path, max_rows: int) -> pd.DataFrame:
    nrows = None if max_rows <= 0 else max_rows
    return pd.read_table(path, sep="\t", engine="python", nrows=nrows)


def collect_needed_dense_ids(df: pd.DataFrame) -> set[int]:
    needed: set[int] = set()
    for _, row in df.iterrows():
        for iid in parse_list_cell(row["user_mid_history"]):
            if iid > 0:
                needed.add(iid)
        target = int(row["target_dense_item_id"])
        if target > 0:
            needed.add(target)
    return needed


def build_feature_cache(
    parquet_path: Path,
    embedding_column: str,
    orig2dense: np.ndarray,
    needed_dense_ids: set[int],
    embedding_dim: int,
) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    feature_cache: dict[int, np.ndarray] = {0: np.zeros(embedding_dim, dtype=np.float32)}
    remaining = set(int(x) for x in needed_dense_ids if x > 0)
    if not remaining:
        return feature_cache, {"needed": 0, "cached": 0, "missing": 0}

    pf = pq.ParquetFile(parquet_path)
    rows_seen = 0
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=4096, columns=["item_id", embedding_column]), start=1):
        pyd = batch.to_pydict()
        for orig_item_id, vec in zip(pyd["item_id"], pyd[embedding_column]):
            rows_seen += 1
            orig_item_id = int(orig_item_id)
            if orig_item_id >= len(orig2dense):
                continue
            dense_item_id = int(orig2dense[orig_item_id])
            if dense_item_id in remaining:
                feature_cache[dense_item_id] = np.asarray(vec, dtype=np.float32)
                remaining.remove(dense_item_id)
        if batch_idx % 50 == 0:
            print(f"[cache] scanned_rows={rows_seen:,}, cached={len(feature_cache)-1:,}, remaining={len(remaining):,}")
        if not remaining:
            break

    return feature_cache, {
        "needed": int(len(needed_dense_ids)),
        "cached": int(len(feature_cache) - 1),
        "missing": int(len(remaining)),
    }


@dataclass
class SampleRecord:
    history_ids: list[int]
    target_dense_item_id: int
    target_sid: list[int]
    reward: float
    feedback_label: int


class YambdaHPNDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        dense_item2sid: np.ndarray,
        feature_cache: dict[int, np.ndarray],
        max_seq_len: int,
        positive_only: bool = False,
        min_reward: float = -1e9,
    ) -> None:
        self.feature_cache = feature_cache
        self.records: list[SampleRecord] = []
        for _, row in df.iterrows():
            reward = float(row["user_clicks"])
            feedback_label = int(row.get("feedback_label", 0))
            if positive_only and feedback_label <= 0:
                continue
            if reward < min_reward:
                continue

            history_ids = parse_list_cell(row["user_mid_history"])[-max_seq_len:]
            if len(history_ids) < max_seq_len:
                history_ids = [0] * (max_seq_len - len(history_ids)) + history_ids

            target_dense_item_id = int(row["target_dense_item_id"])
            target_sid = dense_item2sid[target_dense_item_id].tolist()
            if any(int(x) < 0 for x in target_sid):
                continue
            self.records.append(
                SampleRecord(
                    history_ids=history_ids,
                    target_dense_item_id=target_dense_item_id,
                    target_sid=[int(x) for x in target_sid],
                    reward=reward,
                    feedback_label=feedback_label,
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        history_features = np.stack(
            [self.feature_cache.get(iid, self.feature_cache[0]) for iid in rec.history_ids],
            axis=0,
        ).astype(np.float32)
        return {
            "history_features": torch.tensor(history_features, dtype=torch.float32),
            "target_sid": torch.tensor(rec.target_sid, dtype=torch.long),
            "reward": torch.tensor(rec.reward, dtype=torch.float32),
            "feedback_label": torch.tensor(rec.feedback_label, dtype=torch.float32),
        }


class DummyYambdaEnvSpec:
    def __init__(self, n_item: int, item_dim: int, max_seq_len: int) -> None:
        self.action_space = {
            "item_id": ("nominal", n_item),
            "item_feature": ("continuous", item_dim, "normal"),
        }
        self.observation_space = {
            "history": ("sequence", max_seq_len, ("continuous", item_dim)),
        }


def run_epoch(
    model: SIDPolicy_credit,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    is_train: bool,
) -> dict[str, float]:
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0
    total_token_correct: list[int] | None = None
    total_full_correct = 0
    sid_levels = model.sid_levels
    label = "train" if is_train else "val"
    pbar = tqdm(loader, desc=f"[epoch {epoch}] {label}", ncols=80)
    for batch in pbar:
        history_features = batch["history_features"].to(device)
        target_sid = batch["target_sid"].to(device)
        output = model({"history_features": history_features})
        sid_logits = output["sid_logits"]
        losses = []
        full_correct_mask = torch.ones(target_sid.shape[0], dtype=torch.bool, device=device)
        if total_token_correct is None:
            total_token_correct = [0] * sid_levels
        for level in range(sid_levels):
            logits_l = sid_logits[level]
            target_l = target_sid[:, level]
            losses.append(F.cross_entropy(logits_l, target_l))
            correct_l = torch.argmax(logits_l, dim=-1) == target_l
            total_token_correct[level] += int(correct_l.sum().item())
            full_correct_mask &= correct_l
        loss = sum(losses) / len(losses)
        if "reg" in output:
            loss = loss + 1e-6 * output["reg"]
        if is_train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_size = int(target_sid.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        total_full_correct += int(full_correct_mask.sum().item())
        pbar.set_postfix_str(f"loss={loss.item():.4f} acc={total_full_correct/max(total_samples,1):.4f}", refresh=True)
    pbar.close()
    token_correct = total_token_correct or [0] * sid_levels
    return {
        "loss": total_loss / max(total_samples, 1),
        "full_path_acc": total_full_correct / max(total_samples, 1),
        **{f"token_acc_l{level+1}": correct / max(total_samples, 1) for level, correct in enumerate(token_correct)},
        "n_sample": float(total_samples),
    }


def infer_sid_spec(dense_item2sid: np.ndarray) -> tuple[int, int]:
    valid = dense_item2sid[1:]
    sid_levels = int(valid.shape[1])
    vocab_sizes = [int(valid[:, i].max()) + 1 for i in range(sid_levels)]
    if len(set(vocab_sizes)) != 1:
        raise ValueError(f"Inconsistent per-level SID vocab sizes: {vocab_sizes}")
    return sid_levels, int(vocab_sizes[0])


def main() -> None:
    args = parse_args()
    save_path = Path(args.save_path)
    save_meta = Path(args.save_meta)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_meta.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    set_random_seed(args.seed)
    print(f"[device] using {device}")

    train_df = load_split_df(Path(args.train_file), args.max_train_rows)
    val_df = load_split_df(Path(args.val_file), args.max_val_rows)
    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
    sid_levels, sid_vocab_size = infer_sid_spec(dense_item2sid)
    needed_dense_ids = collect_needed_dense_ids(train_df) | collect_needed_dense_ids(val_df)
    orig2dense = np.load(args.orig2dense_npy, mmap_mode="r")
    feature_cache, cache_stats = build_feature_cache(
        Path(args.embeddings_parquet),
        args.embedding_column,
        orig2dense,
        needed_dense_ids,
        args.embedding_dim,
    )
    print(f"[cache] stats={cache_stats}")

    train_dataset = YambdaHPNDataset(
        train_df,
        dense_item2sid=dense_item2sid,
        feature_cache=feature_cache,
        max_seq_len=args.max_seq_len,
        positive_only=args.train_positive_only,
        min_reward=args.min_train_reward,
    )
    val_dataset = YambdaHPNDataset(
        val_df,
        dense_item2sid=dense_item2sid,
        feature_cache=feature_cache,
        max_seq_len=args.max_seq_len,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty.")

    env_spec = DummyYambdaEnvSpec(
        n_item=dense_item2sid.shape[0] - 1,
        item_dim=args.embedding_dim,
        max_seq_len=args.max_seq_len,
    )
    policy_args = SimpleNamespace(
        sasrec_n_layer=args.sasrec_n_layer,
        sasrec_d_model=args.sasrec_d_model,
        sasrec_d_forward=args.sasrec_d_forward,
        sasrec_n_head=args.sasrec_n_head,
        sasrec_dropout=args.sasrec_dropout,
        sid_levels=sid_levels,
        sid_vocab_sizes=sid_vocab_size,
        sid_temp=args.sid_temp,
    )
    model = SIDPolicy_credit(policy_args, env_spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker)

    history = []
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, epoch, True)
        val_metrics = run_epoch(model, val_loader, device, None, epoch, False)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "policy_args": vars(policy_args),
                    "sid_levels": sid_levels,
                    "sid_vocab_size": sid_vocab_size,
                },
                save_path,
            )
        print(
            f"[epoch {epoch}] train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f}"
        )

    meta = {
        "train_file": args.train_file,
        "val_file": args.val_file,
        "dense_item2sid_npy": args.dense_item2sid_npy,
        "sid_levels": sid_levels,
        "sid_vocab_size": sid_vocab_size,
        "cache_stats": cache_stats,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_val_loss": best_val_loss,
        "history": history,
        "save_path": str(save_path),
        "note": "Recreated optional warm-start script; Regret experiments should not depend on this stage.",
    }
    save_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] training meta saved to {save_meta}")


if __name__ == "__main__":
    main()
