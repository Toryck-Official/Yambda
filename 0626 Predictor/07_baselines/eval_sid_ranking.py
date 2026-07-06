from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
sys.path.extend([str(ROOT / "01_data"), str(ROOT / "02_model"), str(BASELINE_DIR)])

from baseline_models import SASRecSIDPolicy, score_sid_candidates  # noqa: E402
from future_dataset import EmbedStore, FutureIterableDataset, collate_future  # noqa: E402
from hpn import FutureHPNPolicy  # noqa: E402
from progress_utils import estimate_total_batches  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SID candidate ranking for offline baselines.")
    parser.add_argument("--model_type", choices=["sasrec_sid", "hpn_sid", "hsrl_sid"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default=str(ROOT / "01_data" / "processed" / "predictor_seq_data"))
    parser.add_argument("--embed_store", default=str(ROOT / "01_data" / "processed" / "raw_rqkmeans"))
    parser.add_argument("--dense_item2sid_npy", default=str(ROOT / "01_data" / "processed" / "raw_rqkmeans" / "dense_item2sid.npy"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_rows", type=int, default=50000)
    parser.add_argument("--candidate_k", type=int, default=100)
    parser.add_argument("--semantic_prefix", type=int, default=2)
    parser.add_argument("--k_list", default="1,5,10,20")
    parser.add_argument("--hsrl_project_root", default=str(WORKSPACE_ROOT / "HSRL"))
    parser.add_argument("--hsrl_bootstrap_root", default=str(WORKSPACE_ROOT))
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--d_forward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sid_temp", type=float, default=1.0)
    parser.add_argument("--save_meta", default=str(BASELINE_DIR / "artifacts" / "evals" / "sid_ranking.meta.json"))
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


class SidNegativeSampler:
    def __init__(self, dense_item2sid: np.ndarray, sid_levels: int, sid_vocab_size: int, semantic_prefix: int, seed: int) -> None:
        sid = np.asarray(dense_item2sid[1:, :sid_levels])
        valid = np.all(sid >= 0, axis=1)
        self.sid_valid = np.asarray(sid[valid], dtype=np.int64)
        if self.sid_valid.size == 0:
            raise RuntimeError("No valid SID rows found for eval negatives.")
        self.sid_levels = int(sid_levels)
        self.sid_vocab_size = int(sid_vocab_size)
        self.semantic_prefix = max(0, min(int(semantic_prefix), self.sid_levels))
        self.rng = np.random.default_rng(int(seed))
        self.codes: np.ndarray | None = None
        self.order: np.ndarray | None = None
        if self.semantic_prefix > 0:
            codes = self._codes(self.sid_valid)
            order = np.argsort(codes, kind="mergesort")
            self.codes = codes[order]
            self.order = order.astype(np.int64, copy=False)

    def _codes(self, sid: np.ndarray) -> np.ndarray:
        sid = np.asarray(sid, dtype=np.int64)
        if sid.ndim == 1:
            sid = sid.reshape(1, -1)
        code = np.zeros((sid.shape[0],), dtype=np.int64)
        for level in range(self.semantic_prefix):
            code = code * self.sid_vocab_size + sid[:, level]
        return code

    def _sample_one(self, target: np.ndarray, used: set[tuple[int, ...]]) -> np.ndarray:
        if self.codes is not None and self.order is not None and self.semantic_prefix > 0:
            code = int(self._codes(target)[0])
            lo = int(np.searchsorted(self.codes, code, side="left"))
            hi = int(np.searchsorted(self.codes, code, side="right"))
        else:
            lo, hi = 0, 0
        for _ in range(128):
            if hi > lo:
                local = int(self.order[int(self.rng.integers(lo, hi))])
            else:
                local = int(self.rng.integers(0, self.sid_valid.shape[0]))
            cand = self.sid_valid[local]
            key = tuple(int(x) for x in cand.tolist())
            if key not in used:
                used.add(key)
                return cand
        cand = self.sid_valid[int(self.rng.integers(0, self.sid_valid.shape[0]))]
        used.add(tuple(int(x) for x in cand.tolist()))
        return cand

    def sample(self, target_sid: np.ndarray, candidate_k: int) -> np.ndarray:
        target_sid = np.asarray(target_sid, dtype=np.int64)[:, : self.sid_levels]
        candidate_k = max(int(candidate_k), 1)
        out = np.zeros((target_sid.shape[0], candidate_k, self.sid_levels), dtype=np.int64)
        out[:, 0, :] = target_sid
        for row_idx, target in enumerate(target_sid):
            used = {tuple(int(x) for x in target.tolist())}
            for col in range(1, candidate_k):
                out[row_idx, col, :] = self._sample_one(target, used)
        return out


class HSRLEnvSpec:
    def __init__(self, n_item: int, item_dim: int, max_seq_len: int) -> None:
        self.action_space = {
            "item_id": ("nominal", n_item),
            "item_feature": ("continuous", item_dim, "normal"),
        }
        self.observation_space = {
            "history": ("sequence", max_seq_len, ("continuous", item_dim)),
        }


def load_state_dict_compatible(model: torch.nn.Module, checkpoint: str, device: torch.device) -> tuple[int, int, int]:
    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    else:
        state = ckpt
    own = model.state_dict()
    compatible = {}
    skipped = 0
    for key, value in state.items():
        if key in own and tuple(own[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped += 1
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return len(compatible), len(missing), skipped + len(unexpected)


def load_model(args: argparse.Namespace, item_dim: int, sid_levels: int, sid_vocab_size: int, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if args.model_type == "sasrec_sid":
        model = SASRecSIDPolicy(
            item_dim=item_dim,
            d_model=int(cfg.get("d_model", args.d_model)),
            max_seq_len=50,
            n_layer=int(cfg.get("n_layer", args.n_layer)),
            n_head=int(cfg.get("n_head", args.n_head)),
            dropout=float(cfg.get("dropout", args.dropout)),
            sid_levels=int(cfg.get("sid_levels", sid_levels)),
            sid_vocab_size=int(cfg.get("sid_vocab_size", sid_vocab_size)),
            state_pooling=str(cfg.get("state_pooling", "last_mean")),
            use_history_feedback=bool(cfg.get("use_history_feedback", False)),
            use_history_event_type=bool(cfg.get("use_history_event_type", False)),
        ).to(device)
    elif args.model_type == "hpn_sid":
        model = FutureHPNPolicy(
            item_dim=item_dim,
            d_model=int(cfg.get("d_model", args.d_model)),
            max_seq_len=50,
            n_layer=int(cfg.get("n_layer", args.n_layer)),
            n_head=int(cfg.get("n_head", args.n_head)),
            dropout=float(cfg.get("dropout", args.dropout)),
            sid_levels=int(cfg.get("sid_levels", sid_levels)),
            sid_vocab_size=int(cfg.get("sid_vocab_size", sid_vocab_size)),
            state_pooling=str(cfg.get("state_pooling", "last_mean")),
        ).to(device)
    else:
        sys.path.insert(0, str(Path(args.hsrl_bootstrap_root)))
        from adapter.bootstrap import install_hsrl_adapter  # type: ignore

        install_hsrl_adapter(str(Path(args.hsrl_project_root) / "hsrl_core"))
        from model.policy.SIDPolicy_credit import SIDPolicy_credit  # type: ignore

        hsrl_args = SimpleNamespace(
            sasrec_n_layer=args.n_layer,
            sasrec_d_model=args.d_model,
            sasrec_d_forward=args.d_forward,
            sasrec_n_head=args.n_head,
            sasrec_dropout=args.dropout,
            sid_levels=sid_levels,
            sid_vocab_sizes=sid_vocab_size,
            sid_temp=args.sid_temp,
            sara_eta=0.0,
            sara_layer_weights="1,1,1,1",
        )
        env = HSRLEnvSpec(n_item=int(np.load(args.dense_item2sid_npy, mmap_mode="r").shape[0] - 1), item_dim=item_dim, max_seq_len=50)
        model = SIDPolicy_credit(hsrl_args, env).to(device)
    loaded, missing, skipped = load_state_dict_compatible(model, args.checkpoint, device)
    print(f"[model] type={args.model_type} loaded={loaded} missing={missing} skipped_or_unexpected={skipped}")
    model.eval()
    return model


def make_loader(data_dir: str, split: str, store: EmbedStore, batch_size: int, max_rows: int) -> DataLoader:
    dataset = FutureIterableDataset(data_dir, split=split, max_rows=max_rows, mapping_root=store.root)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=lambda rows: collate_future(rows, store))


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def update_metrics(scores: torch.Tensor, k_list: list[int], totals: dict[str, float]) -> None:
    order = scores.argsort(dim=1, descending=True)
    ranks = (order == 0).float().argmax(dim=1) + 1
    n = int(scores.shape[0])
    totals["n"] = totals.get("n", 0.0) + float(n)
    totals["mrr_sum"] = totals.get("mrr_sum", 0.0) + float((1.0 / ranks.float()).sum().detach().cpu())
    totals["mean_rank_sum"] = totals.get("mean_rank_sum", 0.0) + float(ranks.float().sum().detach().cpu())
    for k in k_list:
        kk = min(int(k), int(scores.shape[1]))
        hit = (ranks <= kk).float()
        totals[f"hr@{k}"] = totals.get(f"hr@{k}", 0.0) + float(hit.sum().detach().cpu())
        ndcg = torch.where(ranks <= kk, 1.0 / torch.log2(ranks.float() + 1.0), torch.zeros_like(ranks.float()))
        totals[f"ndcg@{k}"] = totals.get(f"ndcg@{k}", 0.0) + float(ndcg.sum().detach().cpu())


def finalize(totals: dict[str, float], k_list: list[int]) -> dict[str, float]:
    n = max(float(totals.get("n", 0.0)), 1.0)
    out = {
        "n": int(totals.get("n", 0.0)),
        "mrr": totals.get("mrr_sum", 0.0) / n,
        "mean_rank": totals.get("mean_rank_sum", 0.0) / n,
    }
    for k in k_list:
        out[f"hr@{k}"] = totals.get(f"hr@{k}", 0.0) / n
        out[f"ndcg@{k}"] = totals.get(f"ndcg@{k}", 0.0) / n
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    store = EmbedStore(args.embed_store)
    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
    sid_levels = int(dense_item2sid.shape[1])
    sid_vocab_size = int(np.asarray(dense_item2sid[1:]).max()) + 1
    model = load_model(args, store.dim, sid_levels, sid_vocab_size, device)
    sampler = SidNegativeSampler(dense_item2sid, sid_levels, sid_vocab_size, args.semantic_prefix, args.seed)
    k_list = [int(x) for x in str(args.k_list).split(",") if x.strip()]
    loader = make_loader(args.data_dir, args.split, store, args.batch_size, args.max_rows)
    total_batches = estimate_total_batches(args.data_dir, args.split, args.batch_size, args.max_rows)
    totals: dict[str, float] = {"n": 0.0}
    pbar = tqdm(loader, total=total_batches, desc=f"[eval {args.model_type} {args.split}]", unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            batch = move_batch(batch, device)
            if args.model_type == "hsrl_sid":
                out = model({"history_features": batch["history_features"]})
            else:
                out = model(batch)
            target_sid = batch["target_sid"].detach().cpu().numpy().astype(np.int64)
            candidate_sid_np = sampler.sample(target_sid, args.candidate_k)
            candidate_sid = torch.as_tensor(candidate_sid_np, dtype=torch.long, device=device)
            scores = score_sid_candidates(out["sid_logits"], candidate_sid)  # type: ignore[arg-type]
            update_metrics(scores, k_list, totals)
            metrics = finalize(totals, k_list)
            pbar.set_postfix(mrr=f"{metrics['mrr']:.4f}", hr10=f"{metrics.get('hr@10', 0.0):.4f}", refresh=False)
    metrics = finalize(totals, k_list)
    meta = {
        "args": vars(args),
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "sid_levels": sid_levels,
        "sid_vocab_size": sid_vocab_size,
        "candidate_k": int(args.candidate_k),
        "metrics": metrics,
    }
    save_path = Path(args.save_meta)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[done] meta saved to {save_path}")


if __name__ == "__main__":
    main()
