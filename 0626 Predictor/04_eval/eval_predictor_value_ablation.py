from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "01_data"), str(ROOT / "02_model")])

from future_dataset import EmbedStore, FutureIterableDataset, collate_future
from predictor import FuturePredictor


RESPONSE_NAMES = ["listen", "like", "dislike", "unlike", "undislike"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline ablation: does predictor scoring select higher-value candidates?")
    parser.add_argument("--data_dir", default=str(ROOT / "01_data" / "processed" / "predictor_seq_data"))
    parser.add_argument("--embed_store", default=str(ROOT / "01_data" / "processed" / "raw_rqkmeans"))
    parser.add_argument("--ckpt", default=str(ROOT / "artifacts" / "predictor_actionfix_1m" / "future_predictor.pt"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_rows", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--candidate_k", type=int, default=8)
    parser.add_argument(
        "--candidate_negative_mode",
        choices=["inbatch", "history", "history_inbatch", "semantic_inbatch", "history_semantic_inbatch"],
        default="semantic_inbatch",
    )
    parser.add_argument("--semantic_prefix_level", type=int, default=1)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="")
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
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def as_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def as_int(value, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def sid_prefix(row: dict, level: int) -> tuple[int, ...]:
    sid = row.get("target_sid") or []
    if not sid:
        return tuple()
    prefix = max(1, min(int(level), len(sid)))
    return tuple(int(x) for x in sid[:prefix])


def batched(iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_predictor(path: str, item_dim: int, device: torch.device) -> FuturePredictor:
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
    return model


def fallback_index(row_idx: int, slot: int, batch_size: int) -> int:
    return (int(row_idx) - int(slot)) % max(int(batch_size), 1)


def history_pick(row: dict, target_dense: int, slot: int) -> tuple[int, float] | None:
    history_dense = [as_int(x) for x in (row.get("history_dense_item_ids") or [])]
    history_feedbacks = [as_float(x) for x in (row.get("history_feedbacks") or [])]
    valid = [idx for idx, item in enumerate(history_dense) if item > 0 and item != target_dense]
    if not valid:
        return None
    pos = valid[-1 - ((int(slot) - 1) % len(valid))]
    feedback = history_feedbacks[pos] if pos < len(history_feedbacks) else 0.0
    return history_dense[pos], feedback


def semantic_pick(rows: list[dict], row_idx: int, slot: int, prefix_level: int) -> int | None:
    target = sid_prefix(rows[row_idx], prefix_level)
    if not target:
        return None
    candidates = [
        idx for idx, row in enumerate(rows)
        if idx != row_idx and sid_prefix(row, prefix_level) == target
    ]
    if not candidates:
        return None
    return candidates[(int(slot) - 1) % len(candidates)]


def choose_negative(rows: list[dict], row_idx: int, slot: int, mode: str, prefix_level: int) -> tuple[str, int, float, float, float]:
    row = rows[row_idx]
    target_dense = as_int(row.get("target_dense_item_id"))
    batch_size = len(rows)

    def fallback() -> tuple[str, int, float, float, float]:
        neg_idx = fallback_index(row_idx, slot, batch_size)
        neg = rows[neg_idx]
        return (
            "fallback",
            as_int(neg.get("target_dense_item_id")),
            as_float(neg.get("reward_v2")),
            as_float(neg.get("future_return")),
            as_float(neg.get("future_regret_any")),
        )

    if mode == "history":
        picked = history_pick(row, target_dense, slot)
        if picked is not None:
            dense, reward = picked
            return "history", dense, reward, float("nan"), float("nan")
        return fallback()

    if mode == "history_inbatch":
        if slot % 2 == 1:
            picked = history_pick(row, target_dense, slot)
            if picked is not None:
                dense, reward = picked
                return "history", dense, reward, float("nan"), float("nan")
        return fallback()

    if mode == "semantic_inbatch":
        idx = semantic_pick(rows, row_idx, slot, prefix_level)
        if idx is not None:
            neg = rows[idx]
            return (
                "semantic",
                as_int(neg.get("target_dense_item_id")),
                as_float(neg.get("reward_v2")),
                as_float(neg.get("future_return")),
                as_float(neg.get("future_regret_any")),
            )
        return fallback()

    if mode == "history_semantic_inbatch":
        pattern = slot % 3
        if pattern == 1:
            picked = history_pick(row, target_dense, slot)
            if picked is not None:
                dense, reward = picked
                return "history", dense, reward, float("nan"), float("nan")
        if pattern == 2:
            idx = semantic_pick(rows, row_idx, slot, prefix_level)
            if idx is not None:
                neg = rows[idx]
                return (
                    "semantic",
                    as_int(neg.get("target_dense_item_id")),
                    as_float(neg.get("reward_v2")),
                    as_float(neg.get("future_return")),
                    as_float(neg.get("future_regret_any")),
                )
        return fallback()

    return fallback()


def build_candidate_arrays(rows: list[dict], candidate_k: int, mode: str, prefix_level: int):
    n = len(rows)
    dense = np.zeros((n, candidate_k), dtype=np.int64)
    reward = np.zeros((n, candidate_k), dtype=np.float32)
    future_return = np.full((n, candidate_k), np.nan, dtype=np.float32)
    future_regret = np.full((n, candidate_k), np.nan, dtype=np.float32)
    sources = []
    for row_idx, row in enumerate(rows):
        dense[row_idx, 0] = as_int(row.get("target_dense_item_id"))
        reward[row_idx, 0] = as_float(row.get("reward_v2"))
        future_return[row_idx, 0] = as_float(row.get("future_return"))
        future_regret[row_idx, 0] = as_float(row.get("future_regret_any"))
        row_sources = ["target"]
        for slot in range(1, candidate_k):
            source, neg_dense, neg_reward, neg_future, neg_regret = choose_negative(rows, row_idx, slot, mode, prefix_level)
            dense[row_idx, slot] = neg_dense
            reward[row_idx, slot] = neg_reward
            future_return[row_idx, slot] = neg_future
            future_regret[row_idx, slot] = neg_regret
            row_sources.append(source)
        sources.append(row_sources)
    return dense, reward, future_return, future_regret, sources


def expected_formula_reward(response_probs: torch.Tensor, play_ratio: torch.Tensor) -> torch.Tensor:
    listen = response_probs[..., 0]
    like = response_probs[..., 1]
    dislike = response_probs[..., 2]
    unlike = response_probs[..., 3]
    undislike = response_probs[..., 4]
    play = play_ratio.clamp(0.0, 1.0) * listen
    reward = play + 0.8 * like - 1.2 * dislike - 0.6 * unlike + 0.2 * undislike
    return reward.clamp(-2.0, 2.0)


def cosine_scores(batch: dict[str, torch.Tensor], candidate_features: torch.Tensor) -> dict[str, torch.Tensor]:
    history = batch["history_features"]
    mask = batch["history_mask"].unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    mean_hist = (history * mask).sum(dim=1) / denom
    valid = batch["history_mask"] > 0
    rev_idx = torch.flip(valid, dims=[1]).float().argmax(dim=1)
    last_pos = valid.shape[1] - 1 - rev_idx
    last_hist = history[torch.arange(history.shape[0], device=history.device), last_pos]
    cand = F.normalize(candidate_features, dim=-1)
    return {
        "mean_history_cosine": (cand * F.normalize(mean_hist, dim=-1).unsqueeze(1)).sum(dim=-1),
        "last_history_cosine": (cand * F.normalize(last_hist, dim=-1).unsqueeze(1)).sum(dim=-1),
    }


@dataclass
class DecisionStats:
    rows: int = 0
    selected_label_sum: float = 0.0
    target_label_sum: float = 0.0
    random_label_sum: float = 0.0
    best_label_sum: float = 0.0
    best_hit_sum: float = 0.0
    regret_sum: float = 0.0

    def update_index(self, labels: np.ndarray, selected_idx: int) -> None:
        finite = np.isfinite(labels)
        if not finite[selected_idx] or not finite[0]:
            return
        valid_labels = labels[finite]
        best = float(valid_labels.max())
        selected = float(labels[selected_idx])
        target = float(labels[0])
        self.rows += 1
        self.selected_label_sum += selected
        self.target_label_sum += target
        self.random_label_sum += float(valid_labels.mean())
        self.best_label_sum += best
        self.best_hit_sum += float(abs(selected - best) <= 1e-8)
        self.regret_sum += best - selected

    def update_random(self, labels: np.ndarray) -> None:
        finite = np.isfinite(labels)
        if not finite[0] or not finite.any():
            return
        valid_labels = labels[finite]
        best = float(valid_labels.max())
        best_count = int(np.sum(np.abs(valid_labels - best) <= 1e-8))
        selected = float(valid_labels.mean())
        self.rows += 1
        self.selected_label_sum += selected
        self.target_label_sum += float(labels[0])
        self.random_label_sum += selected
        self.best_label_sum += best
        self.best_hit_sum += best_count / max(int(valid_labels.shape[0]), 1)
        self.regret_sum += best - selected

    def finish(self) -> dict[str, float]:
        n = max(self.rows, 1)
        return {
            "rows": self.rows,
            "selected_label_mean": self.selected_label_sum / n,
            "target_label_mean": self.target_label_sum / n,
            "random_label_mean": self.random_label_sum / n,
            "best_label_mean": self.best_label_sum / n,
            "best_hit_rate": self.best_hit_sum / n,
            "best_minus_selected_mean": self.regret_sum / n,
            "selected_minus_random_mean": (self.selected_label_sum - self.random_label_sum) / n,
            "selected_minus_target_mean": (self.selected_label_sum - self.target_label_sum) / n,
        }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    store = EmbedStore(args.embed_store)
    model = load_predictor(args.ckpt, store.dim, device)
    dataset = FutureIterableDataset(args.data_dir, split=args.split, max_rows=args.max_rows, mapping_root=store.root)

    score_names = [
        "random",
        "logged_target",
        "candidate_logit",
        "predicted_reward",
        "predicted_future_return",
        "expected_formula_reward",
        "safe_future_return",
        "positive_response_score",
        "mean_history_cosine",
        "last_history_cosine",
    ]
    label_names = ["reward", "future_return", "non_future_regret"]
    stats = {label: {score: DecisionStats() for score in score_names} for label in label_names}
    source_counts = {"history": 0, "semantic": 0, "fallback": 0, "target": 0}
    rows_seen = 0

    pbar = tqdm(
        batched(iter(dataset), args.batch_size),
        desc=f"[predictor value ablation {args.split}]",
        unit="batch",
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for rows in pbar:
            batch = collate_future(rows, store)
            batch = {key: value.to(device) for key, value in batch.items()}
            dense, reward_label, future_label, regret_label, sources = build_candidate_arrays(
                rows,
                max(int(args.candidate_k), 1),
                args.candidate_negative_mode,
                args.semantic_prefix_level,
            )
            for row_sources in sources:
                for source in row_sources:
                    source_counts[source] = source_counts.get(source, 0) + 1
            candidate_features = torch.tensor(store.lookup(dense), dtype=torch.float32, device=device)
            out = model(batch, candidate_features)
            response = out["response_probs"]
            predicted_reward = out["predicted_reward"]
            predicted_future = out.get("predicted_future_return", predicted_reward)
            expected_reward = expected_formula_reward(response, out["predicted_play_ratio"])
            future_regret_prob = out.get("future_regret_prob", predicted_reward.new_zeros(predicted_reward.shape))
            cos = cosine_scores(batch, candidate_features)
            scores = {
                "candidate_logit": out["candidate_logit"].detach().cpu().numpy(),
                "predicted_reward": predicted_reward.detach().cpu().numpy(),
                "predicted_future_return": predicted_future.detach().cpu().numpy(),
                "expected_formula_reward": expected_reward.detach().cpu().numpy(),
                "safe_future_return": (predicted_future - float(args.eta) * future_regret_prob).detach().cpu().numpy(),
                "positive_response_score": (response[..., 0] + response[..., 1] + response[..., 4]).detach().cpu().numpy(),
                "mean_history_cosine": cos["mean_history_cosine"].detach().cpu().numpy(),
                "last_history_cosine": cos["last_history_cosine"].detach().cpu().numpy(),
            }
            labels = {
                "reward": reward_label,
                "future_return": future_label,
                "non_future_regret": 1.0 - regret_label,
            }
            for row_idx in range(len(rows)):
                for label_name, label_matrix in labels.items():
                    row_labels = label_matrix[row_idx]
                    stats[label_name]["random"].update_random(row_labels)
                    stats[label_name]["logged_target"].update_index(row_labels, 0)
                    for score_name, score_matrix in scores.items():
                        selected_idx = int(np.nanargmax(score_matrix[row_idx]))
                        stats[label_name][score_name].update_index(row_labels, selected_idx)
            rows_seen += len(rows)
            pbar.set_postfix(rows=rows_seen, refresh=False)

    total_sources = max(sum(source_counts.values()), 1)
    result = {
        "split": args.split,
        "max_rows": int(args.max_rows),
        "rows": rows_seen,
        "ckpt": args.ckpt,
        "candidate_k": int(args.candidate_k),
        "candidate_negative_mode": args.candidate_negative_mode,
        "semantic_prefix_level": int(args.semantic_prefix_level),
        "source_share": {key: value / total_sources for key, value in source_counts.items()},
        "metrics": {
            label: {score: value.finish() for score, value in score_stats.items()}
            for label, score_stats in stats.items()
        },
        "notes": [
            "This is an offline diagnostic, not a true counterfactual policy evaluation.",
            "future_return and non_future_regret metrics skip history negatives because those labels are unavailable for historical proxy items.",
            "selected_minus_random_mean > 0 indicates the score selects candidates with higher logged label than random within the same candidate set.",
        ],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
