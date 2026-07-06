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
for path in sorted(EVAL_DIR.glob("*.meta.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    args = data.get("args", {})
    rows.append(
        {
            "file": path.name,
            "model_type": data.get("model_type", args.get("model_type", "")),
            "split": args.get("split", ""),
            "max_rows": args.get("max_rows", ""),
            "candidate_k": data.get("candidate_k", args.get("candidate_k", "")),
            "n": metrics.get("n", ""),
            "hr@1": metrics.get("hr@1", ""),
            "hr@5": metrics.get("hr@5", ""),
            "hr@10": metrics.get("hr@10", ""),
            "hr@20": metrics.get("hr@20", ""),
            "ndcg@10": metrics.get("ndcg@10", ""),
            "mrr": metrics.get("mrr", ""),
            "mean_rank": metrics.get("mean_rank", ""),
        }
    )

columns = ["model_type", "split", "max_rows", "candidate_k", "n", "hr@1", "hr@5", "hr@10", "hr@20", "ndcg@10", "mrr", "mean_rank", "file"]
tsv = "\t".join(columns) + "\n"
for row in rows:
    tsv += "\t".join(fmt(row.get(col, "")) for col in columns) + "\n"
(OUT_DIR / "baseline_ranking_summary.tsv").write_text(tsv, encoding="utf-8")

md = ["# Baseline Ranking Summary", ""]
if rows:
    md.append("| " + " | ".join(columns) + " |")
    md.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        md.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
else:
    md.append("No eval meta files found.")
md.append("")
(OUT_DIR / "baseline_ranking_summary.md").write_text("\n".join(md), encoding="utf-8")

print(f"[done] rows={len(rows)}")
print(f"[done] tsv={OUT_DIR / 'baseline_ranking_summary.tsv'}")
print(f"[done] md={OUT_DIR / 'baseline_ranking_summary.md'}")

