from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "artifacts" / "evals"
OUT_DIR = ROOT / "artifacts" / "summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


rows = []
for path in sorted(EVAL_DIR.glob("*reward_depth*.meta.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    args = data.get("args", {})
    if "avg_cum_reward" not in metrics:
        continue
    rows.append(
        {
            "model_type": data.get("model_type", args.get("model_type", "")),
            "split": args.get("split", ""),
            "episodes": int(metrics.get("episodes", args.get("num_episodes", 0))),
            "avg_cum_reward": metrics.get("avg_cum_reward", ""),
            "avg_step": metrics.get("avg_step", ""),
            "reward_per_step": metrics.get("reward_per_step", ""),
            "negative_rate": metrics.get("negative_rate", ""),
            "failure_rate": metrics.get("failure_rate", ""),
            "early_stop_rate": metrics.get("early_stop_rate", ""),
            "listen_rate": metrics.get("listen_rate", ""),
            "mean_play_ratio": metrics.get("mean_play_ratio", ""),
            "file": path.name,
        }
    )

rows.sort(key=lambda row: (str(row["split"]), int(row["episodes"]), str(row["model_type"])))
columns = [
    "model_type",
    "split",
    "episodes",
    "avg_cum_reward",
    "avg_step",
    "reward_per_step",
    "negative_rate",
    "failure_rate",
    "early_stop_rate",
    "listen_rate",
    "mean_play_ratio",
    "file",
]

tsv = "\t".join(columns) + "\n"
for row in rows:
    tsv += "\t".join(fmt(row.get(col, "")) for col in columns) + "\n"
(OUT_DIR / "baseline_reward_depth_summary.tsv").write_text(tsv, encoding="utf-8")

md = ["# Baseline Reward/Depth Summary", ""]
if rows:
    md.append("| " + " | ".join(columns) + " |")
    md.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        md.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
else:
    md.append("No reward/depth eval meta files found.")
md.append("")
(OUT_DIR / "baseline_reward_depth_summary.md").write_text("\n".join(md), encoding="utf-8")

print(f"[done] rows={len(rows)}")
print(f"[done] tsv={OUT_DIR / 'baseline_reward_depth_summary.tsv'}")
print(f"[done] md={OUT_DIR / 'baseline_reward_depth_summary.md'}")
