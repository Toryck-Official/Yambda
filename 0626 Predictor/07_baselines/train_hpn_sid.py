from __future__ import annotations

import argparse
import json
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
from progress_utils import estimate_total_batches, format_float
from hpn import FutureHPNPolicy, hpn_loss, score_sid_candidates


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
    parser.add_argument("--state_pooling", choices=["last", "mean", "last_mean"], default="last_mean")
    parser.add_argument("--sid_levels", type=int, default=4)
    parser.add_argument("--sid_vocab_size", type=int, default=256)
    parser.add_argument("--dense_item2sid_npy", default="", help="dense_item_id -> SID mapping used for semantic hard negatives.")
    parser.add_argument("--candidate_ce_weight", type=float, default=1.0, help="Candidate ranking loss weight.")
    parser.add_argument("--candidate_ce_k", type=int, default=32, help="Final candidate SID paths, target first.")
    parser.add_argument("--candidate_ce_temperature", type=float, default=1.0)
    parser.add_argument("--candidate_negative_mode", choices=["inbatch", "semantic_hard"], default="semantic_hard")
    parser.add_argument("--candidate_pool_k", type=int, default=128, help="Sampled candidate pool before selecting current-model hard negatives.")
    parser.add_argument("--hard_prefix_levels", default="3,2,1", help="SID prefix lengths used for semantic negatives, e.g. 3,2,1.")
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


def parse_prefix_levels(text: str, sid_levels: int) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if 1 <= value <= sid_levels:
            values.append(value)
    values = sorted(set(values), reverse=True)
    return values or [min(3, sid_levels), min(2, sid_levels), 1]


def make_loader(data_dir: str, split: str, store: EmbedStore, batch_size: int, max_rows: int) -> DataLoader:
    dataset = FutureIterableDataset(data_dir, split=split, max_rows=max_rows, mapping_root=store.root)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=lambda rows: collate_future(rows, store))


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


class SidHardNegativeSampler:
    """Sample semantically close SID negatives, then HPN selects the hardest ones.

    The prefix index makes candidates share the same coarse SID prefix as the
    target whenever possible. The final CE loss still scores them with current
    HPN logits, so the negatives are hard because the current HPN ranks them high.
    """

    def __init__(
        self,
        dense_item2sid: np.ndarray,
        sid_vocab_size: int,
        sid_levels: int,
        prefix_levels: list[int],
        seed: int = 2026,
    ) -> None:
        self.sid_vocab_size = int(sid_vocab_size)
        self.sid_levels = int(sid_levels)
        self.prefix_levels = [level for level in prefix_levels if 1 <= int(level) <= self.sid_levels]
        self.rng = np.random.default_rng(int(seed))
        sid = np.asarray(dense_item2sid)
        sid_slice = sid[1:, : self.sid_levels]
        valid = np.all(sid_slice >= 0, axis=1)
        self.sid_valid = np.asarray(sid_slice[valid], dtype=np.int64)
        if self.sid_valid.size == 0:
            raise RuntimeError("No valid SID rows found for hard negative sampling.")
        self.prefix_indexes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for level in self.prefix_levels:
            codes = self._codes(self.sid_valid, level)
            order = np.argsort(codes, kind="mergesort")
            self.prefix_indexes[int(level)] = (codes[order], order.astype(np.int64, copy=False))

    def _codes(self, sid: np.ndarray, prefix_len: int) -> np.ndarray:
        sid = np.asarray(sid, dtype=np.int64)
        if sid.ndim == 1:
            code = np.int64(0)
            for level in range(prefix_len):
                code = code * self.sid_vocab_size + np.int64(sid[level])
            return np.asarray([code], dtype=np.int64)
        code = np.zeros((sid.shape[0],), dtype=np.int64)
        for level in range(prefix_len):
            code = code * self.sid_vocab_size + sid[:, level].astype(np.int64, copy=False)
        return code

    def _append_unique(self, out: list[np.ndarray], used: set[tuple[int, ...]], target_key: tuple[int, ...], sid: np.ndarray) -> bool:
        key = tuple(int(x) for x in sid.tolist())
        if key == target_key or key in used:
            return False
        used.add(key)
        out.append(np.asarray(sid, dtype=np.int64))
        return True

    def _sample_prefix(self, target: np.ndarray, prefix_len: int, count: int, used: set[tuple[int, ...]]) -> list[np.ndarray]:
        if count <= 0 or prefix_len not in self.prefix_indexes:
            return []
        codes, order = self.prefix_indexes[prefix_len]
        code = int(self._codes(target, prefix_len)[0])
        lo = int(np.searchsorted(codes, code, side="left"))
        hi = int(np.searchsorted(codes, code, side="right"))
        if hi <= lo:
            return []
        target_key = tuple(int(x) for x in target.tolist())
        picks: list[np.ndarray] = []
        max_attempts = max(count * 16, 32)
        for _ in range(max_attempts):
            if len(picks) >= count:
                break
            local = int(order[int(self.rng.integers(lo, hi))])
            cand = self.sid_valid[local]
            if self._append_unique(picks, used, target_key, cand):
                continue
        return picks

    def _sample_random(self, target: np.ndarray, count: int, used: set[tuple[int, ...]]) -> list[np.ndarray]:
        target_key = tuple(int(x) for x in target.tolist())
        picks: list[np.ndarray] = []
        max_attempts = max(count * 16, 64)
        for _ in range(max_attempts):
            if len(picks) >= count:
                break
            cand = self.sid_valid[int(self.rng.integers(0, self.sid_valid.shape[0]))]
            self._append_unique(picks, used, target_key, cand)
        return picks

    def sample_batch(self, target_sid: np.ndarray, pool_k: int) -> np.ndarray:
        target_sid = np.asarray(target_sid, dtype=np.int64)[:, : self.sid_levels]
        batch_size = int(target_sid.shape[0])
        pool_k = max(int(pool_k), 1)
        out = np.zeros((batch_size, pool_k, self.sid_levels), dtype=np.int64)
        out[:, 0, :] = target_sid
        for row_idx, target in enumerate(target_sid):
            used: set[tuple[int, ...]] = {tuple(int(x) for x in target.tolist())}
            candidates: list[np.ndarray] = []
            needed = pool_k - 1
            for level_idx, prefix_len in enumerate(self.prefix_levels):
                remaining_levels = max(len(self.prefix_levels) - level_idx, 1)
                take = max(1, needed // remaining_levels) if needed > 0 else 0
                prefix_picks = self._sample_prefix(target, prefix_len, take, used)
                candidates.extend(prefix_picks)
                needed = pool_k - 1 - len(candidates)
                if needed <= 0:
                    break
            if len(candidates) < pool_k - 1:
                candidates.extend(self._sample_random(target, pool_k - 1 - len(candidates), used))
            if len(candidates) < pool_k - 1:
                # Degenerate fallback for tiny catalogs: repeat random rows, but never the target if possible.
                candidates.extend(self._sample_random(target, pool_k - 1 - len(candidates), set()))
            for col, sid in enumerate(candidates[: pool_k - 1], start=1):
                out[row_idx, col, :] = sid[: self.sid_levels]
        return out


def make_inbatch_candidate_sid(target_sid: torch.Tensor, candidate_k: int) -> torch.Tensor:
    candidate_k = max(int(candidate_k), 1)
    parts = [target_sid]
    batch_size = int(target_sid.shape[0])
    for shift in range(1, candidate_k):
        parts.append(torch.roll(target_sid, shifts=shift % max(batch_size, 1), dims=0))
    return torch.stack(parts[:candidate_k], dim=1)


def select_current_hard_candidates(
    sid_logits: list[torch.Tensor],
    candidate_pool_sid: torch.Tensor,
    final_k: int,
) -> torch.Tensor:
    final_k = max(int(final_k), 1)
    if candidate_pool_sid.shape[1] <= final_k:
        return candidate_pool_sid[:, :final_k, :]
    with torch.no_grad():
        pool_scores = score_sid_candidates(sid_logits, candidate_pool_sid)
        neg_scores = pool_scores[:, 1:]
        keep_neg = min(final_k - 1, neg_scores.shape[1])
        _, hard_pos = torch.topk(neg_scores, k=keep_neg, dim=1)
        hard_pos = hard_pos + 1
    target_pos = torch.zeros((candidate_pool_sid.shape[0], 1), dtype=torch.long, device=candidate_pool_sid.device)
    gather_pos = torch.cat([target_pos, hard_pos], dim=1)
    gather_pos = gather_pos.unsqueeze(-1).expand(-1, -1, candidate_pool_sid.shape[-1])
    return candidate_pool_sid.gather(1, gather_pos)


def hpn_candidate_ce_loss(
    out: dict[str, torch.Tensor | list[torch.Tensor]],
    target_sid: torch.Tensor,
    candidate_k: int,
    temperature: float,
    sampler: SidHardNegativeSampler | None,
    candidate_pool_k: int,
) -> dict[str, torch.Tensor]:
    sid_logits = out["sid_logits"]
    if not isinstance(sid_logits, list):
        raise TypeError("sid_logits must be a list of tensors.")
    target_sid = target_sid.long()
    if sampler is None:
        candidate_sid = make_inbatch_candidate_sid(target_sid, candidate_k)
        pool_size = candidate_sid.shape[1]
    else:
        pool_k = max(int(candidate_pool_k), int(candidate_k))
        pool_np = sampler.sample_batch(target_sid.detach().cpu().numpy(), pool_k)
        pool_sid = torch.as_tensor(pool_np, dtype=torch.long, device=target_sid.device)
        candidate_sid = select_current_hard_candidates(sid_logits, pool_sid, candidate_k)
        pool_size = pool_sid.shape[1]
    scores = score_sid_candidates(sid_logits, candidate_sid) / max(float(temperature), 1e-6)
    labels = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    loss = F.cross_entropy(scores, labels)
    order = scores.argsort(dim=1, descending=True)
    ranks = (order == 0).float().argmax(dim=1) + 1
    return {
        "loss": loss,
        "top1": (ranks == 1).float().mean().detach(),
        "mrr": (1.0 / ranks.float()).mean().detach(),
        "pool_size": torch.tensor(float(pool_size), dtype=loss.dtype, device=loss.device),
    }


def run_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    train: bool,
    desc: str,
    total_batches: int | None,
    candidate_ce_weight: float,
    candidate_ce_k: int,
    candidate_ce_temperature: float,
    sampler: SidHardNegativeSampler | None,
    candidate_pool_k: int,
) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {}
    n_batches = 0
    n_examples = 0
    pbar = tqdm(loader, total=total_batches, desc=desc, unit="batch", dynamic_ncols=True)
    for batch in pbar:
        batch = move_batch(batch, device)
        if "target_sid" not in batch or batch["target_sid"].numel() == 0:
            raise RuntimeError("future_data has no target_sid. Rebuild with --orig2dense_npy and --dense_item2sid_npy.")
        with torch.set_grad_enabled(train):
            out = model(batch)
            losses = hpn_loss(out, batch["target_sid"])
            sid_ce = losses["loss"]
            losses["sid_ce_loss"] = sid_ce.detach()
            if float(candidate_ce_weight) > 0:
                cand = hpn_candidate_ce_loss(
                    out,
                    batch["target_sid"],
                    candidate_ce_k,
                    candidate_ce_temperature,
                    sampler=sampler,
                    candidate_pool_k=candidate_pool_k,
                )
                total_loss = sid_ce + float(candidate_ce_weight) * cand["loss"]
                losses["candidate_ce_loss"] = cand["loss"].detach()
                losses["candidate_ce_top1"] = cand["top1"]
                losses["candidate_ce_mrr"] = cand["mrr"]
                losses["candidate_pool_size"] = cand["pool_size"].detach()
                losses["loss"] = total_loss
            else:
                losses["candidate_ce_loss"] = sid_ce.new_tensor(0.0)
                losses["candidate_ce_top1"] = sid_ce.new_tensor(0.0)
                losses["candidate_ce_mrr"] = sid_ce.new_tensor(0.0)
                losses["candidate_pool_size"] = sid_ce.new_tensor(0.0)
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        n_batches += 1
        n_examples += int(batch["target_dense_item_id"].shape[0])
        pbar.set_postfix(
            loss=format_float(losses["loss"].detach().cpu()),
            sid=format_float(losses["sid_ce_loss"].detach().cpu()),
            cand=format_float(losses["candidate_ce_loss"].detach().cpu()),
            c_top1=format_float(losses["candidate_ce_top1"].detach().cpu()),
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
    model = FutureHPNPolicy(
        item_dim=store.dim,
        d_model=args.d_model,
        max_seq_len=50,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
        sid_levels=args.sid_levels,
        sid_vocab_size=args.sid_vocab_size,
        state_pooling=args.state_pooling,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    sampler = None
    if float(args.candidate_ce_weight) > 0 and args.candidate_negative_mode == "semantic_hard":
        dense_item2sid_path = Path(args.dense_item2sid_npy) if args.dense_item2sid_npy else Path(args.embed_store) / "dense_item2sid.npy"
        dense_item2sid = np.load(dense_item2sid_path, mmap_mode="r")
        prefix_levels = parse_prefix_levels(args.hard_prefix_levels, args.sid_levels)
        sampler = SidHardNegativeSampler(
            dense_item2sid=dense_item2sid,
            sid_vocab_size=args.sid_vocab_size,
            sid_levels=args.sid_levels,
            prefix_levels=prefix_levels,
            seed=args.seed,
        )
        print(json.dumps({
            "hard_negative_sampler": "semantic_hard",
            "dense_item2sid_npy": str(dense_item2sid_path),
            "prefix_levels": prefix_levels,
            "candidate_pool_k": int(args.candidate_pool_k),
            "candidate_ce_k": int(args.candidate_ce_k),
        }, ensure_ascii=False))

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loader = make_loader(args.data_dir, "train", store, args.batch_size, args.max_train_rows)
        val_loader = make_loader(args.data_dir, "val", store, args.batch_size, args.max_val_rows)
        train_total = estimate_total_batches(args.data_dir, "train", args.batch_size, args.max_train_rows)
        val_total = estimate_total_batches(args.data_dir, "val", args.batch_size, args.max_val_rows)
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            desc=f"[hpn epoch {epoch}/{args.epochs} train]",
            total_batches=train_total,
            candidate_ce_weight=args.candidate_ce_weight,
            candidate_ce_k=args.candidate_ce_k,
            candidate_ce_temperature=args.candidate_ce_temperature,
            sampler=sampler,
            candidate_pool_k=args.candidate_pool_k,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            train=False,
            desc=f"[hpn epoch {epoch}/{args.epochs} val]",
            total_batches=val_total,
            candidate_ce_weight=args.candidate_ce_weight,
            candidate_ce_k=args.candidate_ce_k,
            candidate_ce_temperature=args.candidate_ce_temperature,
            sampler=sampler,
            candidate_pool_k=args.candidate_pool_k,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    torch.save({"model_state": model.state_dict(), "config": vars(args), "item_dim": store.dim}, out_dir / "hpn.pt")
    (out_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
