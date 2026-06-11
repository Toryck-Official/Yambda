#!/usr/bin/env python3
"""
Yambda-HSRL artifact diagnostics.

This script is intentionally standalone and path-driven, so it can be copied to
the cloud run directory and executed after filling in artifact paths.

Main checks:
- SID quality: code usage, entropy, full-path collisions, prefix collisions,
  optional embedding-nearest-neighbor SID locality.
- Split TSV sanity: row counts, reward distribution, feedback counts, history
  length distribution, duplicate sequence_id check if requested.
- Training/eval meta summaries: HPN warmstart, UserResponse, DDPG report,
  candidate-ranking metrics with corrected random baselines.

Example:
  python 08_diagnose_artifacts.py \
    --dense_item2sid_npy artifacts/mappings/yambda_dense_item2sid.npy \
    --embeddings_parquet /path/to/embeddings.parquet \
    --split_meta artifacts/processed/split.meta.json \
    --train_tsv artifacts/processed/train.tsv \
    --val_tsv artifacts/processed/val.tsv \
    --test_tsv artifacts/processed/test.tsv \
    --hpn_meta artifacts/models/hpn_warmstart.meta.json \
    --urm_meta artifacts/env/yambda_user_env.meta.json \
    --sid_report artifacts/models/yambda_sid.report \
    --candidate_ranking_meta artifacts/models/candidate_ranking.meta.json \
    --out_json artifacts/diagnostics/yambda_artifact_diagnostics.json \
    --out_md artifacts/diagnostics/yambda_artifact_diagnostics.md
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Yambda-HSRL artifacts")

    parser.add_argument("--dense_item2sid_npy", default="", help="dense item id -> SID npy, shape [n_item+1, n_levels]")
    parser.add_argument("--embeddings_parquet", default="", help="Yambda embeddings.parquet, optional for SID locality")
    parser.add_argument("--embedding_column", default="normalized_embed", choices=["normalized_embed", "embed"])
    parser.add_argument(
        "--locality_rows",
        type=int,
        default=10000,
        help="Use first N embedding rows for nearest-neighbor SID locality. 0 disables.",
    )
    parser.add_argument("--locality_anchors", type=int, default=1000, help="Number of random anchors for NN locality")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--split_meta", default="", help="split.meta.json")
    parser.add_argument("--train_tsv", default="", help="train.tsv")
    parser.add_argument("--val_tsv", default="", help="val.tsv")
    parser.add_argument("--test_tsv", default="", help="test.tsv")
    parser.add_argument("--scan_tsv_max_rows", type=int, default=0, help="0 means full scan")
    parser.add_argument(
        "--exact_sequence_unique",
        action="store_true",
        help="Track every sequence_id to detect duplicates. Can use substantial RAM for huge train.tsv.",
    )

    parser.add_argument("--hpn_meta", default="", help="hpn_warmstart.meta.json")
    parser.add_argument("--urm_meta", default="", help="yambda_user_env.meta.json")
    parser.add_argument("--sid_meta", default="", help="yambda_sid.meta.json")
    parser.add_argument("--sid_report", default="", help="yambda_sid.report")
    parser.add_argument("--candidate_ranking_meta", default="", help="candidate_ranking.meta.json")

    parser.add_argument("--out_json", default="", help="Output JSON path")
    parser.add_argument("--out_md", default="", help="Output Markdown summary path")
    return parser.parse_args()


def path_or_none(path: str) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.exists() else None


def entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / float(total)
    return float(-(p * np.log2(p)).sum())


def gini_from_counts(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(np.float64))
    if x.sum() <= 0:
        return 0.0
    n = x.size
    return float((2.0 * np.arange(1, n + 1).dot(x) / (n * x.sum())) - (n + 1) / n)


def summarize_sid(dense_item2sid_npy: str) -> dict[str, Any]:
    p = path_or_none(dense_item2sid_npy)
    if p is None:
        return {"status": "skipped", "reason": "dense_item2sid_npy missing or not provided"}

    sid = np.load(p, mmap_mode="r")
    valid = np.asarray(sid[1:])
    n_item, n_levels = valid.shape
    vocab_sizes = [int(valid[:, level].max()) + 1 for level in range(n_levels)]

    per_level = []
    for level in range(n_levels):
        vocab = vocab_sizes[level]
        counts = np.bincount(valid[:, level].astype(np.int64), minlength=vocab)
        used = int((counts > 0).sum())
        entropy_bits = entropy_from_counts(counts)
        max_entropy_bits = math.log2(vocab) if vocab > 1 else 0.0
        top_idx = np.argsort(counts)[-10:][::-1]
        per_level.append(
            {
                "level": level + 1,
                "vocab_size": int(vocab),
                "used_codes": used,
                "utilization": used / max(vocab, 1),
                "entropy_bits": entropy_bits,
                "entropy_ratio": entropy_bits / max_entropy_bits if max_entropy_bits > 0 else 0.0,
                "gini": gini_from_counts(counts),
                "max_bucket_count": int(counts.max()),
                "max_bucket_share": float(counts.max() / max(counts.sum(), 1)),
                "min_nonzero_bucket_count": int(counts[counts > 0].min()) if used else 0,
                "top_tokens": [
                    {"token": int(i), "count": int(counts[i]), "share": float(counts[i] / max(counts.sum(), 1))}
                    for i in top_idx
                ],
            }
        )

    full_unique, full_counts = np.unique(valid, axis=0, return_counts=True)
    collision_groups = int((full_counts > 1).sum())
    collision_items_excess = int((full_counts[full_counts > 1] - 1).sum())

    prefix_stats = []
    for depth in range(1, n_levels + 1):
        prefix_unique, prefix_counts = np.unique(valid[:, :depth], axis=0, return_counts=True)
        prefix_collision_groups = int((prefix_counts > 1).sum())
        prefix_collision_items_excess = int((prefix_counts[prefix_counts > 1] - 1).sum())
        prefix_stats.append(
            {
                "depth": depth,
                "unique_prefixes": int(prefix_unique.shape[0]),
                "collision_groups": prefix_collision_groups,
                "collision_items_excess": prefix_collision_items_excess,
                "collision_rate_excess_items": prefix_collision_items_excess / max(n_item, 1),
                "max_bucket_count": int(prefix_counts.max()),
                "max_bucket_share": float(prefix_counts.max() / max(n_item, 1)),
            }
        )

    return {
        "status": "ok",
        "path": str(p),
        "n_item": int(n_item),
        "n_levels": int(n_levels),
        "vocab_sizes": vocab_sizes,
        "per_level": per_level,
        "full_sid": {
            "unique_paths": int(full_unique.shape[0]),
            "collision_groups": collision_groups,
            "collision_items_excess": collision_items_excess,
            "collision_rate_excess_items": collision_items_excess / max(n_item, 1),
            "max_collision_bucket_count": int(full_counts.max()),
            "max_collision_bucket_share": float(full_counts.max() / max(n_item, 1)),
        },
        "prefix_stats": prefix_stats,
    }


def read_first_embeddings(parquet_path: Path, embedding_column: str, n_rows: int) -> np.ndarray:
    if pq is None:
        raise RuntimeError("pyarrow is required for embeddings_parquet locality diagnostics")
    pf = pq.ParquetFile(parquet_path)
    chunks = []
    seen = 0
    for batch in pf.iter_batches(batch_size=4096, columns=[embedding_column]):
        arr = np.asarray(batch.column(0).to_pylist(), dtype=np.float32)
        need = n_rows - seen
        if arr.shape[0] > need:
            arr = arr[:need]
        chunks.append(arr)
        seen += int(arr.shape[0])
        if seen >= n_rows:
            break
    if not chunks:
        raise RuntimeError(f"No embeddings read from {parquet_path}")
    return np.vstack(chunks).astype(np.float32)


def summarize_sid_locality(
    dense_item2sid_npy: str,
    embeddings_parquet: str,
    embedding_column: str,
    n_rows: int,
    n_anchors: int,
    seed: int,
) -> dict[str, Any]:
    sid_path = path_or_none(dense_item2sid_npy)
    emb_path = path_or_none(embeddings_parquet)
    if sid_path is None or emb_path is None or n_rows <= 0:
        return {"status": "skipped", "reason": "requires dense_item2sid_npy, embeddings_parquet, and locality_rows > 0"}

    sid = np.load(sid_path, mmap_mode="r")
    max_rows = min(int(n_rows), int(sid.shape[0] - 1))
    emb = read_first_embeddings(emb_path, embedding_column, max_rows)
    max_rows = min(max_rows, emb.shape[0])
    emb = emb[:max_rows]
    sid_sample = np.asarray(sid[1 : max_rows + 1])

    # Assumption: 02_build_item_sid.py assigned dense ids in embeddings.parquet row order.
    # If cloud code changed that assumption, use dense2orig/orig2dense to build an aligned
    # sample before calling this locality calculation.
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    n_anchors = min(int(n_anchors), max_rows)
    rng = np.random.default_rng(seed)
    anchors = rng.choice(max_rows, size=n_anchors, replace=False)

    nearest = []
    for start in range(0, n_anchors, 100):
        anchor_ids = anchors[start : start + 100]
        sims = emb[anchor_ids] @ emb.T
        for row_idx, item_idx in enumerate(anchor_ids):
            sims[row_idx, item_idx] = -np.inf
        nearest.append(np.argmax(sims, axis=1))
    nearest_idx = np.concatenate(nearest)

    a_sid = sid_sample[anchors]
    b_sid = sid_sample[nearest_idx]
    hamming = (a_sid != b_sid).sum(axis=1)

    prefix_lens = []
    for a, b in zip(a_sid, b_sid):
        length = 0
        for x, y in zip(a, b):
            if int(x) != int(y):
                break
            length += 1
        prefix_lens.append(length)
    prefix_lens = np.asarray(prefix_lens)

    n_levels = int(sid_sample.shape[1])
    out = {
        "status": "ok",
        "assumption": "dense_id i corresponds to embeddings.parquet row i-1, matching 02_build_item_sid.py streaming assignment",
        "embedding_pool_rows": int(max_rows),
        "anchor_count": int(n_anchors),
        "mean_sid_hamming_distance_to_embedding_nn": float(hamming.mean()),
        "median_sid_hamming_distance_to_embedding_nn": float(np.median(hamming)),
        "mean_common_prefix_len": float(prefix_lens.mean()),
        "same_full_sid_share": float((hamming == 0).mean()),
    }
    for depth in range(1, n_levels + 1):
        out[f"same_prefix_{depth}_share"] = float((prefix_lens >= depth).mean())
    return out


def parse_list_cell(cell: str) -> list[Any]:
    if cell is None or cell == "" or cell == "[]":
        return []
    try:
        value = ast.literal_eval(cell)
        return value if isinstance(value, list) else [value]
    except Exception:
        return []


def summarize_tsv(path: str, max_rows: int, exact_sequence_unique: bool) -> dict[str, Any]:
    p = path_or_none(path)
    if p is None:
        return {"status": "skipped", "reason": f"{path or '<empty>'} missing or not provided"}

    reward_values = []
    history_lens = []
    next_history_lens = []
    feedback_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    user_counts: Counter[str] = Counter()
    seq_seen = set() if exact_sequence_unique else None
    duplicate_seq = 0
    n = 0

    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n += 1
            if max_rows > 0 and n > max_rows:
                n -= 1
                break

            if seq_seen is not None:
                sid = row.get("sequence_id", "")
                if sid in seq_seen:
                    duplicate_seq += 1
                seq_seen.add(sid)

            try:
                reward_values.append(float(row.get("user_clicks", 0.0)))
            except ValueError:
                pass
            feedback_counts[str(row.get("feedback_label", ""))] += 1
            event_counts[str(row.get("anchor_event_type", ""))] += 1
            user_counts[str(row.get("user_id", ""))] += 1
            history_lens.append(len(parse_list_cell(row.get("user_mid_history", ""))))
            next_history_lens.append(len(parse_list_cell(row.get("next_user_mid_history", ""))))

    rewards = np.asarray(reward_values, dtype=np.float64)
    return {
        "status": "ok",
        "path": str(p),
        "rows_scanned": int(n),
        "note": "rows_scanned may be partial if scan_tsv_max_rows > 0",
        "sequence_id_exact_unique_checked": bool(exact_sequence_unique),
        "duplicate_sequence_id_count": int(duplicate_seq) if exact_sequence_unique else None,
        "unique_sequence_id_count": int(len(seq_seen)) if seq_seen is not None else None,
        "unique_user_count": int(len(user_counts)),
        "feedback_label_counts": dict(feedback_counts.most_common()),
        "anchor_event_type_counts": dict(event_counts.most_common()),
        "reward": {
            "mean": float(rewards.mean()) if rewards.size else None,
            "std": float(rewards.std()) if rewards.size else None,
            "min": float(rewards.min()) if rewards.size else None,
            "p25": float(np.quantile(rewards, 0.25)) if rewards.size else None,
            "p50": float(np.quantile(rewards, 0.50)) if rewards.size else None,
            "p75": float(np.quantile(rewards, 0.75)) if rewards.size else None,
            "max": float(rewards.max()) if rewards.size else None,
            "positive_share": float((rewards > 0).mean()) if rewards.size else None,
            "negative_share": float((rewards < 0).mean()) if rewards.size else None,
        },
        "history_len": {
            "mean": float(np.mean(history_lens)) if history_lens else None,
            "p50": float(np.quantile(history_lens, 0.50)) if history_lens else None,
            "p95": float(np.quantile(history_lens, 0.95)) if history_lens else None,
            "max": int(max(history_lens)) if history_lens else None,
        },
        "next_history_len": {
            "mean": float(np.mean(next_history_lens)) if next_history_lens else None,
            "p50": float(np.quantile(next_history_lens, 0.50)) if next_history_lens else None,
            "p95": float(np.quantile(next_history_lens, 0.95)) if next_history_lens else None,
            "max": int(max(next_history_lens)) if next_history_lens else None,
        },
    }


def load_json(path: str) -> dict[str, Any]:
    p = path_or_none(path)
    if p is None:
        return {"status": "skipped", "reason": f"{path or '<empty>'} missing or not provided"}
    data = json.loads(p.read_text(encoding="utf-8"))
    data["_status"] = "ok"
    data["_path"] = str(p)
    return data


def summarize_split_meta(path: str) -> dict[str, Any]:
    meta = load_json(path)
    if meta.get("_status") != "ok":
        return meta
    stats = meta.get("run_stats", {})
    kept = int(stats.get("total_kept_events", 0) or 0)
    split_counts = stats.get("split_counts", {})
    train = int(split_counts.get("train", 0) or 0)
    warnings = []
    if kept and train > kept:
        warnings.append("train rows > kept events; this is unusual for one-row-per-episode splitting and should be audited")
    return {
        "status": "ok",
        "path": meta["_path"],
        "users_processed": stats.get("users_processed"),
        "total_raw_events": stats.get("total_raw_events"),
        "total_kept_events": stats.get("total_kept_events"),
        "total_missing_mapping": stats.get("total_missing_mapping"),
        "split_counts": split_counts,
        "warnings": warnings,
    }


def summarize_hpn_meta(path: str) -> dict[str, Any]:
    meta = load_json(path)
    if meta.get("_status") != "ok":
        return meta
    hist = meta.get("history", [])
    sid_vocab = int(meta.get("sid_vocab_size", 0) or 0)
    random_token_acc = 1.0 / sid_vocab if sid_vocab > 0 else None
    best = min(hist, key=lambda r: r.get("val", {}).get("loss", float("inf"))) if hist else None
    return {
        "status": "ok",
        "path": meta["_path"],
        "train_samples": meta.get("train_samples"),
        "val_samples": meta.get("val_samples"),
        "best_val_loss": meta.get("best_val_loss"),
        "best_epoch": best.get("epoch") if best else None,
        "random_token_acc": random_token_acc,
        "best_val_metrics": best.get("val") if best else None,
        "config": {
            "train_positive_only": meta.get("train_positive_only"),
            "min_train_reward": meta.get("min_train_reward"),
            "epochs": meta.get("epochs"),
            "batch_size": meta.get("batch_size"),
            "lr": meta.get("lr"),
            "max_seq_len": meta.get("max_seq_len"),
        },
    }


def summarize_urm_meta(path: str) -> dict[str, Any]:
    meta = load_json(path)
    if meta.get("_status") != "ok":
        return meta
    hist = meta.get("history", [])
    best = min(hist, key=lambda r: r.get("val_mse", float("inf"))) if hist else None
    last = hist[-1] if hist else None
    warnings = []
    if best and last and last.get("val_mse", 0) > best.get("val_mse", 0):
        warnings.append("validation MSE worsened after best epoch; possible overfitting")
    return {
        "status": "ok",
        "path": meta["_path"],
        "train_samples": meta.get("train_samples"),
        "val_samples": meta.get("val_samples"),
        "best_val_mse": meta.get("best_val_mse"),
        "best_epoch": best.get("epoch") if best else None,
        "last_epoch": last,
        "warnings": warnings,
    }


def summarize_sid_report(path: str) -> dict[str, Any]:
    p = path_or_none(path)
    if p is None:
        return {"status": "skipped", "reason": "sid_report missing or not provided"}
    text = p.read_text(encoding="utf-8")
    pattern = re.compile(
        r"step: (\d+) @ episode report: \{'average_total_reward': np\.float\d+\(([-0-9.e]+)\).*?"
        r"'reward_variance': np\.float\d+\(([-0-9.e]+)\).*?"
        r"'average_n_step': np\.float\d+\(([-0-9.e]+)\).*?"
        r"@ step loss: \{'critic_loss': np\.float64\(([-0-9.e]+)\), "
        r"'actor_loss': np\.float64\(([-0-9.e]+)\), "
        r"'entropy_loss': np\.float64\(([-0-9.e]+)\), "
        r"'bc_loss': np\.float64\(([-0-9.e]+)\), "
        r"'advantage': np\.float64\(([-0-9.e]+)\)\}"
    )
    rows = []
    for m in pattern.finditer(text):
        rows.append(tuple([int(m.group(1))] + [float(m.group(i)) for i in range(2, 10)]))
    rows = [r for r in rows if r[0] > 0]
    if not rows:
        return {"status": "empty", "path": str(p), "reason": "no step reports parsed"}

    def avg(idx: int, xs: list[tuple]) -> float:
        return float(np.mean([x[idx] for x in xs]))

    first = rows[: min(10, len(rows))]
    last = rows[-min(10, len(rows)) :]
    best = max(rows, key=lambda x: x[1])
    warnings = []
    if avg(1, last) <= avg(1, first):
        warnings.append("last-window reward is not higher than first-window reward; no clear RL improvement")
    if all(abs(x[7]) < 1e-12 for x in rows):
        warnings.append("bc_loss is zero for all parsed reports; BC term likely inactive")
    return {
        "status": "ok",
        "path": str(p),
        "n_reports": len(rows),
        "first_window_avg_reward": avg(1, first),
        "last_window_avg_reward": avg(1, last),
        "first_window_avg_critic_loss": avg(4, first),
        "last_window_avg_critic_loss": avg(4, last),
        "first_window_avg_entropy": avg(6, first),
        "last_window_avg_entropy": avg(6, last),
        "best_reward_report": {
            "step": best[0],
            "average_total_reward": best[1],
            "reward_variance": best[2],
            "average_n_step": best[3],
            "critic_loss": best[4],
            "actor_loss": best[5],
            "entropy_loss": best[6],
            "bc_loss": best[7],
            "advantage": best[8],
        },
        "last_report": {
            "step": rows[-1][0],
            "average_total_reward": rows[-1][1],
            "reward_variance": rows[-1][2],
            "average_n_step": rows[-1][3],
            "critic_loss": rows[-1][4],
            "actor_loss": rows[-1][5],
            "entropy_loss": rows[-1][6],
            "bc_loss": rows[-1][7],
            "advantage": rows[-1][8],
        },
        "warnings": warnings,
    }


def random_ranking_baselines(candidate_size: int, k_list: list[int]) -> dict[str, float]:
    out = {
        "mean_rank": (candidate_size + 1) / 2.0,
        "mrr": sum(1.0 / r for r in range(1, candidate_size + 1)) / candidate_size,
    }
    for k in k_list:
        kk = min(k, candidate_size)
        out[f"hr@{k}"] = kk / candidate_size
        out[f"ndcg@{k}"] = sum(1.0 / math.log2(r + 1) for r in range(1, kk + 1)) / candidate_size
    return out


def summarize_candidate_ranking(path: str) -> dict[str, Any]:
    meta = load_json(path)
    if meta.get("_status") != "ok":
        return meta
    metrics = meta.get("metrics", {})
    candidate_size = int(meta.get("candidate_size", 0) or 0)
    k_list = []
    for key in metrics:
        if key.startswith("hr@"):
            try:
                k_list.append(int(key.split("@", 1)[1]))
            except ValueError:
                pass
    k_list = sorted(set(k_list))
    baselines = random_ranking_baselines(candidate_size, k_list) if candidate_size > 0 else {}
    return {
        "status": "ok",
        "path": meta["_path"],
        "candidate_size": candidate_size,
        "num_negatives": meta.get("num_negatives"),
        "actor_checkpoint": meta.get("actor_checkpoint"),
        "actor_loaded": meta.get("actor_loaded"),
        "metrics": metrics,
        "correct_random_baselines": baselines,
        "note": "This is random-negative candidate ranking, not full-catalog or hard-negative evaluation.",
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = ["# Yambda-HSRL Artifact Diagnostics", ""]

    sid = report.get("sid_quality", {})
    lines += ["## SID Quality", ""]
    if sid.get("status") == "ok":
        lines.append(f"- Items: {sid['n_item']:,}; levels: {sid['n_levels']}; vocab_sizes: {sid['vocab_sizes']}")
        full = sid["full_sid"]
        lines.append(
            f"- Full SID unique paths: {full['unique_paths']:,}; "
            f"excess collision rate: {full['collision_rate_excess_items']:.4%}; "
            f"max bucket: {full['max_collision_bucket_count']}"
        )
        lines.append("")
        lines.append("| Level | Used | Utilization | Entropy bits | Entropy ratio | Max bucket share | Gini |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for row in sid["per_level"]:
            lines.append(
                f"| {row['level']} | {row['used_codes']}/{row['vocab_size']} | {row['utilization']:.2%} | "
                f"{row['entropy_bits']:.4f} | {row['entropy_ratio']:.2%} | "
                f"{row['max_bucket_share']:.4%} | {row['gini']:.4f} |"
            )
    else:
        lines.append(f"- Skipped: {sid.get('reason', sid.get('status'))}")

    loc = report.get("sid_locality", {})
    lines += ["", "## SID Locality", ""]
    if loc.get("status") == "ok":
        lines.append(
            f"- Pool rows: {loc['embedding_pool_rows']:,}; anchors: {loc['anchor_count']:,}; "
            f"mean Hamming to embedding NN: {loc['mean_sid_hamming_distance_to_embedding_nn']:.3f}"
        )
        prefix_keys = sorted(k for k in loc if k.startswith("same_prefix_"))
        lines.append("- Same-prefix shares: " + ", ".join(f"{k}={loc[k]:.2%}" for k in prefix_keys))
        lines.append(f"- Same full SID share: {loc['same_full_sid_share']:.2%}")
        lines.append(f"- Assumption: {loc['assumption']}")
    else:
        lines.append(f"- Skipped: {loc.get('reason', loc.get('status'))}")

    lines += ["", "## Split", ""]
    split = report.get("split_meta", {})
    if split.get("status") == "ok":
        lines.append(f"- Split counts: {split.get('split_counts')}")
        for w in split.get("warnings", []):
            lines.append(f"- Warning: {w}")
    else:
        lines.append(f"- Split meta skipped: {split.get('reason', split.get('status'))}")
    for name in ["train_tsv", "val_tsv", "test_tsv"]:
        item = report.get(name, {})
        if item.get("status") == "ok":
            lines.append(
                f"- {name}: rows_scanned={item['rows_scanned']:,}, users={item['unique_user_count']:,}, "
                f"reward_mean={item['reward']['mean']}, positive_share={item['reward']['positive_share']}"
            )

    lines += ["", "## Model Artifacts", ""]
    hpn = report.get("hpn", {})
    if hpn.get("status") == "ok":
        lines.append(f"- HPN: best_epoch={hpn['best_epoch']}, best_val_loss={hpn['best_val_loss']}")
    urm = report.get("urm", {})
    if urm.get("status") == "ok":
        lines.append(f"- UserResponse: best_epoch={urm['best_epoch']}, best_val_mse={urm['best_val_mse']}")
        for w in urm.get("warnings", []):
            lines.append(f"- UserResponse warning: {w}")
    sid_report = report.get("sid_report", {})
    if sid_report.get("status") == "ok":
        lines.append(
            f"- RL report: first_reward={sid_report['first_window_avg_reward']:.4f}, "
            f"last_reward={sid_report['last_window_avg_reward']:.4f}, "
            f"best_step={sid_report['best_reward_report']['step']}"
        )
        for w in sid_report.get("warnings", []):
            lines.append(f"- RL warning: {w}")
    cr = report.get("candidate_ranking", {})
    if cr.get("status") == "ok":
        m = cr.get("metrics", {})
        b = cr.get("correct_random_baselines", {})
        lines.append(
            f"- Candidate ranking: mean_rank={m.get('mean_rank')}, mrr={m.get('mrr')}, "
            f"hr@10={m.get('hr@10')}; random_mrr={b.get('mrr')}"
        )
        lines.append(f"- Candidate ranking note: {cr.get('note')}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    report: dict[str, Any] = {
        "sid_quality": summarize_sid(args.dense_item2sid_npy),
        "sid_locality": summarize_sid_locality(
            dense_item2sid_npy=args.dense_item2sid_npy,
            embeddings_parquet=args.embeddings_parquet,
            embedding_column=args.embedding_column,
            n_rows=args.locality_rows,
            n_anchors=args.locality_anchors,
            seed=args.seed,
        ),
        "split_meta": summarize_split_meta(args.split_meta),
        "train_tsv": summarize_tsv(args.train_tsv, args.scan_tsv_max_rows, args.exact_sequence_unique),
        "val_tsv": summarize_tsv(args.val_tsv, args.scan_tsv_max_rows, args.exact_sequence_unique),
        "test_tsv": summarize_tsv(args.test_tsv, args.scan_tsv_max_rows, args.exact_sequence_unique),
        "hpn": summarize_hpn_meta(args.hpn_meta),
        "urm": summarize_urm_meta(args.urm_meta),
        "sid_meta": load_json(args.sid_meta),
        "sid_report": summarize_sid_report(args.sid_report),
        "candidate_ranking": summarize_candidate_ranking(args.candidate_ranking_meta),
    }

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md = build_markdown(report)
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
    else:
        print(md)


if __name__ == "__main__":
    main()
