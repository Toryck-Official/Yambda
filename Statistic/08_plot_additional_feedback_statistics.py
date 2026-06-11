#!/usr/bin/env python3
"""Plot additional Yambda feedback statistics for regret analysis."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


STAT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STAT_ROOT.parent
DATA_ROOT = PROJECT_ROOT.parent / "0330Yambda/data/sequential-50m"
EVENT_DISTRIBUTION_JSON = PROJECT_ROOT / "Regret/artifacts/diagnostics/event_distribution_full.json"

FEEDBACK_FILES = {
    "like": "likes.parquet",
    "dislike": "dislikes.parquet",
    "unlike": "unlikes.parquet",
    "undislike": "undislikes.parquet",
}
FEEDBACK_TYPES = ("like", "dislike", "unlike", "undislike")
KEY_TRANSITIONS = (
    ("like_to_unlike", "Like -> Unlike"),
    ("dislike_to_undislike", "Dislike -> Undislike"),
    ("like_to_dislike", "Like -> Dislike"),
    ("dislike_to_like", "Dislike -> Like"),
)

BLUE = "#7DB6E8"
ORANGE = "#FF9A45"
GREEN = "#8FD18E"
GRAY = "#8A8A8A"
DARK = "#222222"

TIME_BUCKETS = ("<=10s", "10s-1d", "1d-7d", ">7d")
TIME_BUCKET_COLORS = ("#7DB6E8", "#A6CDED", "#FFC083", "#FF9A45")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot additional Yambda feedback statistics")
    parser.add_argument("--data_root", default=str(DATA_ROOT))
    parser.add_argument(
        "--transition_json",
        default=str(STAT_ROOT / "feedback_transition_state_change.json"),
    )
    parser.add_argument(
        "--event_distribution_json",
        default=str(EVENT_DISTRIBUTION_JSON),
    )
    parser.add_argument("--timestamp_unit_seconds", type=float, default=5.0)
    parser.add_argument("--out_dir", default=str(STAT_ROOT))
    return parser.parse_args()


def load_feedback_events(data_root: Path) -> dict[int, dict[int, list[tuple[int, str]]]]:
    by_user_item: dict[int, dict[int, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for event_type, filename in FEEDBACK_FILES.items():
        table = pq.read_table(data_root / filename, columns=["uid", "timestamp", "item_id"])
        pyd = table.to_pydict()
        for uid, timestamps, item_ids in zip(pyd["uid"], pyd["timestamp"], pyd["item_id"], strict=True):
            if timestamps is None or item_ids is None:
                continue
            for timestamp, item_id in zip(timestamps, item_ids, strict=True):
                by_user_item[int(uid)][int(item_id)].append((int(timestamp), event_type))
    return by_user_item


def collect_transition_gaps(data_root: Path, timestamp_unit_seconds: float) -> dict[str, list[float]]:
    by_user_item = load_feedback_events(data_root)
    wanted = {key for key, _ in KEY_TRANSITIONS}
    gaps: dict[str, list[float]] = defaultdict(list)
    for item_histories in by_user_item.values():
        for events in item_histories.values():
            if len(events) < 2:
                continue
            ordered = sorted(events, key=lambda row: (row[0], row[1]))
            for (prev_ts, prev_type), (timestamp, event_type) in zip(ordered, ordered[1:], strict=False):
                if prev_type == event_type:
                    continue
                gap = (timestamp - prev_ts) * timestamp_unit_seconds
                key = f"{prev_type}_to_{event_type}"
                if gap >= 0 and key in wanted:
                    gaps[key].append(float(gap))
    return gaps


def gap_bucket(gap_seconds: float) -> str:
    if gap_seconds <= 10:
        return "<=10s"
    if gap_seconds <= 86400:
        return "10s-1d"
    if gap_seconds <= 604800:
        return "1d-7d"
    return ">7d"


def fmt_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def plot_interval_distribution(gaps: dict[str, list[float]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=220)
    bins = np.linspace(-3, 4.5, 46)
    colors = [BLUE, ORANGE, "#4F9AD2", "#D87928"]
    for (key, label), color in zip(KEY_TRANSITIONS, colors, strict=True):
        values = np.asarray(gaps.get(key, []), dtype=float)
        if values.size == 0:
            continue
        hours = np.maximum(values / 3600.0, 1e-3)
        log_hours = np.log10(hours)
        ax.hist(
            log_hours,
            bins=bins,
            alpha=0.55,
            color=color,
            edgecolor="#4A4A4A",
            linewidth=0.45,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}$(Time Interval) (Hours)", fontsize=15)
    ax.set_ylabel("Count (log scale)", fontsize=15)
    ax.set_title("Time Intervals for Key Feedback Transitions", fontsize=17, pad=12)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    top_ticks = [
        (10 / 3600, "10s"),
        (60 / 3600, "1m"),
        (3600 / 3600, "1h"),
        (86400 / 3600, "1d"),
        (604800 / 3600, "7d"),
        (2592000 / 3600, "30d"),
    ]
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks([math.log10(x) for x, _ in top_ticks], [label for _, label in top_ticks])
    ax_top.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "feedback_key_transition_intervals.png", bbox_inches="tight")


def plot_top_transition_counts(transition_data: dict, out_dir: Path) -> None:
    summary = transition_data["transition_summary"]
    top_items = sorted(
        ((key, value["count"]) for key, value in summary.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    labels = [key.replace("_to_", " -> ").title() for key, _ in top_items]
    values = [count for _, count in top_items]
    colors = [ORANGE if key in {k for k, _ in KEY_TRANSITIONS} else BLUE for key, _ in top_items]

    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=220)
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, edgecolor="#4A4A4A", linewidth=0.7, alpha=0.88)
    ax.set_yscale("log")
    ax.set_ylim(max(min(values) * 0.7, 1), max(values) * 1.65)
    ax.set_ylabel("Count (log scale)", fontsize=15)
    ax.set_title("Top Explicit Feedback Transitions", fontsize=17, pad=12)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.08,
            fmt_count(int(value)),
            ha="center",
            va="bottom",
            fontsize=11,
            color=DARK,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "feedback_top_transition_counts.png", bbox_inches="tight")


def plot_time_bucket_stack(gaps: dict[str, list[float]], out_dir: Path) -> None:
    labels = [label for _, label in KEY_TRANSITIONS]
    counts_by_bucket = []
    shares_by_bucket = []
    for key, _ in KEY_TRANSITIONS:
        counts = Counter(gap_bucket(gap) for gap in gaps.get(key, []))
        total = max(sum(counts.values()), 1)
        counts_by_bucket.append([counts[bucket] for bucket in TIME_BUCKETS])
        shares_by_bucket.append([counts[bucket] / total for bucket in TIME_BUCKETS])
    shares = np.asarray(shares_by_bucket, dtype=float)
    raw_counts = np.asarray(counts_by_bucket, dtype=int)

    fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=220)
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for idx, bucket in enumerate(TIME_BUCKETS):
        ax.bar(
            x,
            shares[:, idx],
            bottom=bottom,
            color=TIME_BUCKET_COLORS[idx],
            edgecolor="white",
            linewidth=1.0,
            label=bucket,
        )
        bottom += shares[:, idx]
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Share", fontsize=15)
    ax.set_title("Short-term Undo vs Long-term Reversal", fontsize=17, pad=28)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, fontsize=11, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    for row_idx, label in enumerate(labels):
        y = 0
        for bucket_idx in range(len(TIME_BUCKETS)):
            share = shares[row_idx, bucket_idx]
            if share >= 0.08:
                ax.text(
                    row_idx,
                    y + share / 2,
                    f"{share * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=DARK,
                )
            y += share
        ax.text(row_idx, 1.015, fmt_count(int(raw_counts[row_idx].sum())), ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "feedback_transition_time_buckets.png", bbox_inches="tight")


def plot_regret_signal_composition(transition_data: dict, event_data: dict, out_dir: Path) -> None:
    listen_bins = event_data["listen_ratio_bins"]
    low_play = (
        listen_bins.get("eq_0", 0)
        + listen_bins.get("(0,0.05]", 0)
        + listen_bins.get("(0.05,0.10]", 0)
        + listen_bins.get("(0.10,0.20]", 0)
    )
    explicit_counts = transition_data["explicit_counts"]
    like_to_dislike = transition_data["transition_summary"]["like_to_dislike"]["count"]
    signals = [
        ("Low-play\nlisten <=20%", int(low_play), GREEN),
        ("Unlike", int(explicit_counts["unlike"]), BLUE),
        ("Dislike", int(explicit_counts["dislike"]), ORANGE),
        ("Like -> Dislike", int(like_to_dislike), "#D87928"),
    ]
    labels = [item[0] for item in signals]
    values = [item[1] for item in signals]
    colors = [item[2] for item in signals]

    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=220)
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, edgecolor="#4A4A4A", linewidth=0.7, alpha=0.88)
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)", fontsize=15)
    ax.set_title("Candidate Regret Signal Composition", fontsize=17, pad=12)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.08,
            fmt_count(int(value)),
            ha="center",
            va="bottom",
            fontsize=12,
            color=DARK,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "regret_signal_composition.png", bbox_inches="tight")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transition_data = json.loads(Path(args.transition_json).read_text(encoding="utf-8"))
    event_data = json.loads(Path(args.event_distribution_json).read_text(encoding="utf-8"))
    gaps = collect_transition_gaps(Path(args.data_root), float(args.timestamp_unit_seconds))
    plot_interval_distribution(gaps, out_dir)
    plot_top_transition_counts(transition_data, out_dir)
    plot_time_bucket_stack(gaps, out_dir)
    plot_regret_signal_composition(transition_data, event_data, out_dir)
    print(f"[done] figures saved to {out_dir}")


if __name__ == "__main__":
    main()
