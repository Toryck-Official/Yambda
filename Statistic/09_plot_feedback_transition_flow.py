#!/usr/bin/env python3
"""Plot state-filtered explicit feedback transitions as a flow diagram."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STAT_ROOT = Path(__file__).resolve().parent
FEEDBACK_TYPES = ("like", "dislike", "unlike", "undislike")
DISPLAY_NAMES = {
    "like": "Like",
    "dislike": "Dislike",
    "unlike": "Unlike",
    "undislike": "Undislike",
}

POS_LIGHT = "#B9DCF5"
POS_DARK = "#2F7DBA"
NEG_LIGHT = "#FFC48A"
NEG_DARK = "#D45F1E"
DARK = "#222222"
LIGHT_BOX = "#F7FAFC"
POSITIVE_TRANSITIONS = {
    ("dislike", "like"),
    ("dislike", "undislike"),
    ("unlike", "like"),
    ("undislike", "like"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot state-filtered feedback transition flow")
    parser.add_argument(
        "--input_json",
        default=str(STAT_ROOT / "feedback_transition_state_machine.json"),
    )
    parser.add_argument(
        "--out_png",
        default=str(STAT_ROOT / "feedback_transition_flow.png"),
    )
    return parser.parse_args()


def fmt_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{int(max(0, min(1, value)) * 255):02x}" for value in rgb)


def blend(c0: str, c1: str, t: float) -> str:
    a = hex_to_rgb(c0)
    b = hex_to_rgb(c1)
    return rgb_to_hex(tuple(a_i * (1 - t) + b_i * t for a_i, b_i in zip(a, b, strict=True)))


def transition_polarity(src: str, dst: str) -> str:
    return "positive" if (src, dst) in POSITIVE_TRANSITIONS else "negative"


def edge_color(src: str, dst: str, strength: float) -> str:
    strength = max(0.0, min(1.0, strength))
    if transition_polarity(src, dst) == "positive":
        return blend(POS_LIGHT, POS_DARK, strength)
    return blend(NEG_LIGHT, NEG_DARK, strength)


def bezier_points(x0: float, y0: float, x1: float, y1: float, n: int = 140) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 1, n)
    cx0 = x0 + 0.34 * (x1 - x0)
    cx1 = x0 + 0.66 * (x1 - x0)
    x = (1 - t) ** 3 * x0 + 3 * (1 - t) ** 2 * t * cx0 + 3 * (1 - t) * t**2 * cx1 + t**3 * x1
    y = (1 - t) ** 3 * y0 + 3 * (1 - t) ** 2 * t * y0 + 3 * (1 - t) * t**2 * y1 + t**3 * y1
    return x, y


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    counts_obj = data["transition_matrix_counts"]
    row_share_obj = data["transition_matrix_row_share"]

    edges: list[tuple[str, str, int, float]] = []
    for src in FEEDBACK_TYPES:
        for dst in FEEDBACK_TYPES:
            count = int(counts_obj[src][dst])
            if count > 0:
                edges.append((src, dst, count, float(row_share_obj[src][dst])))
    edges.sort(key=lambda item: item[2])

    min_count = min(count for _, _, count, _ in edges)
    max_count = max(count for _, _, count, _ in edges)
    log_min = math.log10(min_count)
    log_max = math.log10(max_count)

    left_y = {
        "like": 0.82,
        "dislike": 0.58,
        "unlike": 0.34,
        "undislike": 0.10,
    }
    right_y = dict(left_y)
    left_x = 0.08
    right_x = 0.66

    fig, ax = plt.subplots(figsize=(11.8, 5.9), dpi=220)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.03, 0.95)
    ax.axis("off")

    ax.text(left_x, 0.92, "Previous Explicit Feedback", ha="center", va="bottom", fontsize=15)
    ax.text(right_x, 0.92, "Current Explicit Feedback", ha="center", va="bottom", fontsize=15)
    ax.set_title("State-filtered Explicit Feedback Transition Flow", fontsize=17, pad=14)

    box_w = 0.18
    box_h = 0.075
    for event_type in FEEDBACK_TYPES:
        for x, y, ha in [(left_x, left_y[event_type], "center"), (right_x, right_y[event_type], "center")]:
            rect = plt.Rectangle(
                (x - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                facecolor=LIGHT_BOX,
                edgecolor="#4A4A4A",
                linewidth=1.0,
                zorder=4,
            )
            ax.add_patch(rect)
            ax.text(x, y, DISPLAY_NAMES[event_type], ha=ha, va="center", fontsize=13.5, zorder=5, color=DARK)

    x0 = left_x + box_w / 2 + 0.018
    x1 = right_x - box_w / 2 - 0.018
    src_offsets = {
        ("like", "unlike"): 0.014,
        ("like", "dislike"): -0.014,
        ("dislike", "like"): 0.014,
        ("dislike", "undislike"): -0.014,
        ("unlike", "like"): 0.014,
        ("unlike", "dislike"): -0.014,
        ("undislike", "like"): 0.014,
        ("undislike", "dislike"): -0.014,
    }
    dst_offsets = {
        ("like", "unlike"): 0.000,
        ("like", "dislike"): 0.014,
        ("dislike", "like"): -0.012,
        ("dislike", "undislike"): 0.000,
        ("unlike", "like"): 0.012,
        ("unlike", "dislike"): -0.014,
        ("undislike", "like"): -0.024,
        ("undislike", "dislike"): 0.000,
    }

    for src, dst, count, row_share in edges:
        y0 = left_y[src] + src_offsets.get((src, dst), 0.0)
        y1 = right_y[dst] + dst_offsets.get((src, dst), 0.0)
        if log_max == log_min:
            width = 5.0
            strength = 0.65
        else:
            strength = (math.log10(count) - log_min) / (log_max - log_min)
            width = 2.0 + 8.5 * strength
        x, y = bezier_points(x0, y0, x1, y1)
        ax.plot(
            x,
            y,
            color=edge_color(src, dst, strength),
            alpha=0.72,
            linewidth=width,
            solid_capstyle="round",
            zorder=2,
        )
    panel_x = 0.785
    ax.text(panel_x, 0.83, "Transitions", ha="left", va="bottom", fontsize=13.5, fontweight="bold", color=DARK)
    ax.plot([panel_x, panel_x + 0.035], [0.805, 0.805], color=POS_DARK, alpha=0.82, linewidth=4.5, solid_capstyle="round")
    ax.text(panel_x + 0.050, 0.805, "Positive", ha="left", va="center", fontsize=9.8, color=DARK)
    ax.plot([panel_x + 0.155, panel_x + 0.190], [0.805, 0.805], color=NEG_DARK, alpha=0.82, linewidth=4.5, solid_capstyle="round")
    ax.text(panel_x + 0.205, 0.805, "Negative", ha="left", va="center", fontsize=9.8, color=DARK)
    display_edges = sorted(edges, key=lambda item: item[2], reverse=True)
    y = 0.755
    for src, dst, count, row_share in display_edges:
        if log_max == log_min:
            strength = 0.65
        else:
            strength = (math.log10(count) - log_min) / (log_max - log_min)
        color = edge_color(src, dst, strength)
        ax.plot([panel_x, panel_x + 0.035], [y, y], color=color, alpha=0.75, linewidth=5.0, solid_capstyle="round")
        ax.text(
            panel_x + 0.050,
            y,
            f"{fmt_count(count)}",
            ha="left",
            va="center",
            fontsize=10.0,
            color=DARK,
        )
        ax.text(
            panel_x + 0.105,
            y,
            f"{DISPLAY_NAMES[src]}->{DISPLAY_NAMES[dst]}",
            ha="left",
            va="center",
            fontsize=10.0,
            color=DARK,
        )
        y -= 0.075

    ax.text(
        0.42,
        -0.005,
        "Line width is log-scaled by transition count.",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[done] figure saved to {out_path}")


if __name__ == "__main__":
    main()
