from __future__ import annotations

import json
import math
from pathlib import Path


def split_row_count(data_dir: str | Path, split: str) -> int | None:
    root = Path(data_dir)
    candidates = [root / "split.meta.json", root / "predictor_seq_split.meta.json", root / "meta.json"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        split_counts = meta.get("split_counts") or meta.get("splits") or meta.get("counts") or {}
        if split in split_counts:
            try:
                return int(split_counts[split])
            except Exception:
                return None
    return None


def estimate_total_batches(data_dir: str | Path, split: str, batch_size: int, max_rows: int = 0) -> int | None:
    batch_size = max(int(batch_size), 1)
    max_rows = int(max_rows or 0)
    split_rows = split_row_count(data_dir, split)
    if max_rows > 0 and split_rows is not None:
        rows = min(max_rows, split_rows)
    elif max_rows > 0:
        rows = max_rows
    else:
        rows = split_rows
    if rows is None or rows <= 0:
        return None
    return int(math.ceil(rows / batch_size))


def format_float(value: float) -> str:
    return f"{float(value):.4g}"
