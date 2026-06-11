#!/usr/bin/env python3
"""Plot a 4x4 explicit feedback transition matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


STAT_ROOT = Path(__file__).resolve().parent
FEEDBACK_TYPES = ("like", "dislike", "unlike", "undislike")
DISPLAY_NAMES = ("Like", "Dislike", "Unlike", "Undislike")
BLUE = "#7DB6E8"
ORANGE = "#FF9A45"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Yambda explicit feedback transition matrix")
    parser.add_argument(
        "--input_json",
        default=str(STAT_ROOT / "feedback_transition_state_machine.json"),
    )
    parser.add_argument(
        "--out_png",
        default=str(STAT_ROOT / "feedback_transition_matrix.png"),
    )
    return parser.parse_args()


def fmt_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    counts_obj = data["transition_matrix_counts"]
    share_obj = data["transition_matrix_row_share"]
    counts = np.array(
        [[counts_obj[src][dst] for dst in FEEDBACK_TYPES] for src in FEEDBACK_TYPES],
        dtype=float,
    )
    row_share = np.array(
        [[share_obj[src][dst] for dst in FEEDBACK_TYPES] for src in FEEDBACK_TYPES],
        dtype=float,
    )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "yambda_blue",
        ["#F8FBFE", "#D7EBFB", BLUE, "#3B86C5"],
    )
    positive_counts = counts[counts > 0]
    norm = mcolors.LogNorm(vmin=max(positive_counts.min(), 1), vmax=positive_counts.max())
    plot_counts = np.ma.masked_where(counts <= 0, counts)

    fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=220)
    cmap.set_bad("#F7FAFC")
    im = ax.imshow(plot_counts, cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(len(DISPLAY_NAMES)), labels=DISPLAY_NAMES)
    ax.set_yticks(np.arange(len(DISPLAY_NAMES)), labels=DISPLAY_NAMES)
    ax.set_xlabel("Current Explicit Feedback", fontsize=15)
    ax.set_ylabel("Previous Explicit Feedback", fontsize=15)
    ax.set_title("State-filtered Explicit Feedback Transition Matrix", fontsize=17, pad=14)
    ax.tick_params(axis="both", labelsize=13)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#444444")

    ax.set_xticks(np.arange(-0.5, len(DISPLAY_NAMES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DISPLAY_NAMES), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx, src in enumerate(FEEDBACK_TYPES):
        for col_idx, dst in enumerate(FEEDBACK_TYPES):
            count = int(counts[row_idx, col_idx])
            if count <= 0:
                ax.text(
                    col_idx,
                    row_idx,
                    "-",
                    ha="center",
                    va="center",
                    fontsize=13.5,
                    color="#A0A0A0",
                )
                continue
            pct = row_share[row_idx, col_idx] * 100.0
            color = "#1F1F1F" if count < 20_000 else "white"
            ax.text(
                col_idx,
                row_idx - 0.08,
                fmt_count(count),
                ha="center",
                va="center",
                fontsize=13.5,
                fontweight="bold",
                color=color,
            )
            ax.text(
                col_idx,
                row_idx + 0.18,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=11.5,
                color=color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Count (log scale)", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout()
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[done] figure saved to {out_path}")


if __name__ == "__main__":
    main()
