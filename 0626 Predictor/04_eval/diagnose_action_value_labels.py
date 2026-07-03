from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "01_data")])

from future_dataset import FutureIterableDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether current candidate negatives provide learnable action-value labels."
    )
    parser.add_argument("--data_dir", default=str(ROOT / "01_data" / "processed" / "predictor_seq_data"))
    parser.add_argument("--mapping_root", default=str(ROOT / "01_data" / "processed" / "raw_rqkmeans"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_rows", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--candidate_k", type=int, default=8)
    parser.add_argument(
        "--candidate_negative_mode",
        choices=["inbatch", "history", "history_inbatch", "semantic_inbatch", "history_semantic_inbatch"],
        default="history_semantic_inbatch",
    )
    parser.add_argument("--semantic_prefix_level", type=int, default=1)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def batched(iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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


@dataclass
class PairStats:
    count: int = 0
    target_sum: float = 0.0
    negative_sum: float = 0.0
    gap_sum: float = 0.0
    target_gt: int = 0
    target_eq: int = 0
    target_lt: int = 0

    def update(self, target: float, negative: float) -> None:
        self.count += 1
        self.target_sum += float(target)
        self.negative_sum += float(negative)
        gap = float(target) - float(negative)
        self.gap_sum += gap
        if gap > 1e-8:
            self.target_gt += 1
        elif gap < -1e-8:
            self.target_lt += 1
        else:
            self.target_eq += 1

    def finish(self) -> dict[str, float]:
        n = max(self.count, 1)
        return {
            "count": self.count,
            "target_mean": self.target_sum / n,
            "negative_mean": self.negative_sum / n,
            "target_minus_negative_mean": self.gap_sum / n,
            "target_gt_negative_rate": self.target_gt / n,
            "target_eq_negative_rate": self.target_eq / n,
            "target_lt_negative_rate": self.target_lt / n,
        }


@dataclass
class RankStats:
    count: int = 0
    top1_sum: float = 0.0
    top3_sum: float = 0.0
    rank_sum: float = 0.0
    rr_sum: float = 0.0
    tie_count: int = 0
    tie_size_sum: float = 0.0

    def update(self, scores: list[float]) -> None:
        if not scores:
            return
        target = float(scores[0])
        eps = 1e-8
        greater = sum(1 for score in scores if float(score) > target + eps)
        equal = sum(1 for score in scores if abs(float(score) - target) <= eps)
        equal = max(equal, 1)
        first_rank = greater + 1
        last_rank = greater + equal
        # Tie-aware expected metrics: if candidates have the same score as the
        # target, assume their relative order is random instead of always giving
        # index 0 the win.
        rank = (first_rank + last_rank) / 2.0
        top1 = (1.0 / equal) if greater == 0 else 0.0
        top3 = max(0, min(3, last_rank) - first_rank + 1) / equal
        rr = sum(1.0 / r for r in range(first_rank, last_rank + 1)) / equal
        self.count += 1
        self.top1_sum += top1
        self.top3_sum += top3
        self.rank_sum += rank
        self.rr_sum += rr
        if equal > 1:
            self.tie_count += 1
            self.tie_size_sum += equal

    def finish(self) -> dict[str, float]:
        n = max(self.count, 1)
        return {
            "count": self.count,
            "top1": self.top1_sum / n,
            "top3": self.top3_sum / n,
            "mrr": self.rr_sum / n,
            "mean_rank": self.rank_sum / n,
            "target_tie_rate": self.tie_count / n,
            "target_tie_size_mean_when_tied": self.tie_size_sum / max(self.tie_count, 1),
        }


@dataclass
class SourceStats:
    reward: PairStats = field(default_factory=PairStats)
    future_return: PairStats = field(default_factory=PairStats)
    future_regret_any: PairStats = field(default_factory=PairStats)

    def finish(self) -> dict[str, dict[str, float]]:
        return {
            "reward": self.reward.finish(),
            "future_return": self.future_return.finish(),
            "future_regret_any": self.future_regret_any.finish(),
        }


def fallback_index(row_idx: int, slot: int, batch_size: int) -> int:
    return (int(row_idx) - int(slot)) % max(int(batch_size), 1)


def history_pick(row: dict, target_dense: int, slot: int) -> tuple[float, int] | None:
    history_dense = [as_int(x) for x in (row.get("history_dense_item_ids") or [])]
    history_feedbacks = [as_float(x) for x in (row.get("history_feedbacks") or [])]
    valid = [idx for idx, item in enumerate(history_dense) if item > 0 and item != target_dense]
    if not valid:
        return None
    pos = valid[-1 - ((int(slot) - 1) % len(valid))]
    feedback = history_feedbacks[pos] if pos < len(history_feedbacks) else 0.0
    return feedback, history_dense[pos]


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


def choose_negative(
    rows: list[dict],
    row_idx: int,
    slot: int,
    mode: str,
    prefix_level: int,
) -> tuple[str, float, float | None, float | None, int]:
    row = rows[row_idx]
    target_dense = as_int(row.get("target_dense_item_id"))
    batch_size = len(rows)

    def fallback() -> tuple[str, float, float, float, int]:
        neg_idx = fallback_index(row_idx, slot, batch_size)
        neg = rows[neg_idx]
        return (
            "fallback",
            as_float(neg.get("reward_v2")),
            as_float(neg.get("future_return")),
            as_float(neg.get("future_regret_any")),
            as_int(neg.get("target_dense_item_id")),
        )

    if mode == "history":
        picked = history_pick(row, target_dense, slot)
        if picked is not None:
            feedback, dense = picked
            return "history", feedback, None, None, dense
        return fallback()

    if mode == "history_inbatch":
        if slot % 2 == 1:
            picked = history_pick(row, target_dense, slot)
            if picked is not None:
                feedback, dense = picked
                return "history", feedback, None, None, dense
        return fallback()

    if mode == "semantic_inbatch":
        idx = semantic_pick(rows, row_idx, slot, prefix_level)
        if idx is not None:
            neg = rows[idx]
            return (
                "semantic",
                as_float(neg.get("reward_v2")),
                as_float(neg.get("future_return")),
                as_float(neg.get("future_regret_any")),
                as_int(neg.get("target_dense_item_id")),
            )
        return fallback()

    if mode == "history_semantic_inbatch":
        pattern = slot % 3
        if pattern == 1:
            picked = history_pick(row, target_dense, slot)
            if picked is not None:
                feedback, dense = picked
                return "history", feedback, None, None, dense
        elif pattern == 2:
            idx = semantic_pick(rows, row_idx, slot, prefix_level)
            if idx is not None:
                neg = rows[idx]
                return (
                    "semantic",
                    as_float(neg.get("reward_v2")),
                    as_float(neg.get("future_return")),
                    as_float(neg.get("future_regret_any")),
                    as_int(neg.get("target_dense_item_id")),
                )
        return fallback()

    return fallback()


def main() -> None:
    args = parse_args()
    dataset = FutureIterableDataset(
        args.data_dir,
        split=args.split,
        max_rows=args.max_rows,
        mapping_root=args.mapping_root,
    )
    source_stats: dict[str, SourceStats] = {
        "history": SourceStats(),
        "semantic": SourceStats(),
        "fallback": SourceStats(),
        "all": SourceStats(),
    }
    oracle_reward_rank = RankStats()
    oracle_future_return_rank = RankStats()
    oracle_non_regret_rank = RankStats()
    target_in_history = 0
    target_rows = 0
    source_counts = {"history": 0, "semantic": 0, "fallback": 0}
    same_dense_negative = 0

    pbar = tqdm(
        batched(iter(dataset), args.batch_size),
        desc=f"[diagnose action labels {args.split}]",
        unit="batch",
        dynamic_ncols=True,
    )
    for rows in pbar:
        for row_idx, row in enumerate(rows):
            target_rows += 1
            target_dense = as_int(row.get("target_dense_item_id"))
            history_dense = {as_int(x) for x in (row.get("history_dense_item_ids") or []) if as_int(x) > 0}
            if target_dense in history_dense:
                target_in_history += 1

            target_reward = as_float(row.get("reward_v2"))
            target_future = as_float(row.get("future_return"))
            target_regret = as_float(row.get("future_regret_any"))

            reward_scores = [target_reward]
            future_scores = [target_future]
            non_regret_scores = [1.0 - target_regret]
            for slot in range(1, max(int(args.candidate_k), 1)):
                source, neg_reward, neg_future, neg_future_regret, neg_dense = choose_negative(
                    rows,
                    row_idx,
                    slot,
                    args.candidate_negative_mode,
                    args.semantic_prefix_level,
                )
                source_counts[source] += 1
                same_dense_negative += int(neg_dense == target_dense and target_dense > 0)
                source_stats[source].reward.update(target_reward, neg_reward)
                source_stats["all"].reward.update(target_reward, neg_reward)
                reward_scores.append(neg_reward)
                if neg_future is not None:
                    source_stats[source].future_return.update(target_future, neg_future)
                    source_stats["all"].future_return.update(target_future, neg_future)
                    future_scores.append(neg_future)
                if neg_future_regret is not None:
                    source_stats[source].future_regret_any.update(target_regret, neg_future_regret)
                    source_stats["all"].future_regret_any.update(target_regret, neg_future_regret)
                    non_regret_scores.append(1.0 - neg_future_regret)

            oracle_reward_rank.update(reward_scores)
            if len(future_scores) > 1:
                oracle_future_return_rank.update(future_scores)
            if len(non_regret_scores) > 1:
                oracle_non_regret_rank.update(non_regret_scores)

        processed = min(target_rows, int(args.max_rows) if args.max_rows else target_rows)
        pbar.set_postfix(rows=processed, refresh=False)

    total_neg = max(sum(source_counts.values()), 1)
    result = {
        "split": args.split,
        "max_rows": int(args.max_rows),
        "rows": target_rows,
        "candidate_k": int(args.candidate_k),
        "candidate_negative_mode": args.candidate_negative_mode,
        "semantic_prefix_level": int(args.semantic_prefix_level),
        "negative_source_share": {key: value / total_neg for key, value in source_counts.items()},
        "target_in_history_rate": target_in_history / max(target_rows, 1),
        "same_dense_negative_rate": same_dense_negative / total_neg,
        "pair_label_stats": {key: value.finish() for key, value in source_stats.items()},
        "oracle_rank_by_logged_reward": oracle_reward_rank.finish(),
        "oracle_rank_by_logged_future_return": oracle_future_return_rank.finish(),
        "oracle_rank_by_non_future_regret": oracle_non_regret_rank.finish(),
        "notes": [
            "history negatives use history_feedbacks as reward proxy; future_return/future_regret are unavailable for history negatives.",
            "semantic/fallback negatives use the negative row's logged label, not a true counterfactual label under the current state.",
            "If oracle rank by logged reward/future_return is near random, next-item candidate CE is not aligned with value improvement.",
        ],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
