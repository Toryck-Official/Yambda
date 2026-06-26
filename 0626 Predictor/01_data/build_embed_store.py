from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a numpy embedding store from Yambda embeddings.parquet.")
    parser.add_argument("--embeddings", default="/Users/Toryck/Coding/DATASET/Yambda/embeddings.parquet")
    parser.add_argument("--out_dir", default=str(ROOT / "artifacts" / "embed_store"))
    parser.add_argument("--column", default="normalized_embed", choices=["embed", "normalized_embed"])
    parser.add_argument("--needed_item_ids", default="")
    parser.add_argument("--max_rows", type=int, default=0)
    return parser.parse_args()


def load_needed(path: str) -> set[int] | None:
    if not path:
        return None
    arr = np.load(path, mmap_mode="r")
    return {int(x) for x in arr.tolist()}


def infer_dim(pf: pq.ParquetFile, column: str) -> int:
    table = pf.read_row_group(0, columns=[column]).slice(0, 1)
    return len(table.column(column).to_pylist()[0])


def save_sorted(out_dir: Path, item_ids: np.ndarray, embeds: np.ndarray, meta: dict) -> None:
    order = np.argsort(item_ids)
    np.save(out_dir / "item_ids.npy", item_ids[order].astype(np.uint32, copy=False))
    np.save(out_dir / "item_embeds.npy", embeds[order].astype(np.float32, copy=False))
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(args.embeddings)
    needed = load_needed(args.needed_item_ids)
    dim = infer_dim(pf, args.column)

    ids: list[int] = []
    vecs: list[list[float]] = []
    emitted = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=["item_id", args.column])
        for row in table.to_pylist():
            item_id = int(row["item_id"])
            if needed is not None and item_id not in needed:
                continue
            ids.append(item_id)
            vecs.append(row[args.column])
            emitted += 1
            if args.max_rows and emitted >= args.max_rows:
                break
        if args.max_rows and emitted >= args.max_rows:
            break

    if not ids:
        raise RuntimeError("No embeddings matched the requested item ids.")

    item_ids = np.asarray(ids, dtype=np.uint32)
    embeds = np.asarray(vecs, dtype=np.float32)
    meta = {
        "source": str(args.embeddings),
        "column": args.column,
        "dim": dim,
        "rows": int(len(item_ids)),
        "subset": needed is not None,
    }
    save_sorted(out_dir, item_ids, embeds, meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
