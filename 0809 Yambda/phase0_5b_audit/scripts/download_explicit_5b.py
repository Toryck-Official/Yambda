#!/usr/bin/env python3
"""Download only the four explicit-feedback Yambda-5B sequential files.

The full sequential release is 87.2 GB and cannot fit on the current data
volume.  Phase 0 needs the four explicit streams in full, while listen and raw
event totals can be read from remote Parquet metadata.  This downloader pins the
official dataset revision and records local SHA-256 fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "phase0_5b_audit" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "phase0_5b_audit" / "artifacts" / "download_manifest.json"
REPOSITORY = "yandex/yambda"
REVISION = "dd6f3a19eef5866e346c3270e098baa641a44948"
FILES = (
    "sequential/5b/likes.parquet",
    "sequential/5b/dislikes.parquet",
    "sequential/5b/unlikes.parquet",
    "sequential/5b/undislikes.parquet",
)
EXPECTED_SIZES = {
    "sequential/5b/likes.parquet": 693_712_976,
    "sequential/5b/dislikes.parquet": 89_065_045,
    "sequential/5b/unlikes.parquet": 193_012_360,
    "sequential/5b/undislikes.parquet": 16_248_355,
}
EXPECTED_SHA256 = {
    "sequential/5b/likes.parquet": "b9190588c72713ee4eee129c3f290633a7314067f8e39e1c84962aac390e300a",
    "sequential/5b/dislikes.parquet": "b068dfb567f4335388706933709c3fb26fb5058496127e813d93b8485b36a96a",
    "sequential/5b/unlikes.parquet": "2f971c757090c17cd027badd82af5df8304484e1742602b92ad92ac53c845c04",
    "sequential/5b/undislikes.parquet": "e05922d5ca39139ebb3c1cc9479b306f498868dce1e45194d93bf81d1ac8bb89",
}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    required = sum(EXPECTED_SIZES.values())
    free = shutil.disk_usage(OUTPUT_ROOT).free
    if free < required + 2 * 1024**3:
        raise RuntimeError(
            f"Need at least {required + 2 * 1024**3:,} free bytes, found {free:,}"
        )

    started = time.time()
    records: dict[str, dict[str, object]] = {}
    for filename in FILES:
        print(f"downloading {filename}", flush=True)
        resolved = Path(
            hf_hub_download(
                repo_id=REPOSITORY,
                filename=filename,
                repo_type="dataset",
                revision=REVISION,
                local_dir=OUTPUT_ROOT,
            )
        ).resolve()
        size = resolved.stat().st_size
        if size != EXPECTED_SIZES[filename]:
            raise RuntimeError(
                f"Unexpected size for {filename}: {size}, expected {EXPECTED_SIZES[filename]}"
            )
        sha256 = sha256_file(resolved)
        if sha256 != EXPECTED_SHA256[filename]:
            raise RuntimeError(
                f"SHA-256 mismatch for {filename}: {sha256}, "
                f"expected {EXPECTED_SHA256[filename]}"
            )
        records[filename] = {
            "path": str(resolved),
            "size_bytes": size,
            "sha256": sha256,
        }
        print(f"verified {filename}: {size:,} bytes", flush=True)

    atomic_json(
        MANIFEST_PATH,
        {
            "format_version": 1,
            "repository": REPOSITORY,
            "revision": REVISION,
            "scope": "four full Yambda-5B sequential explicit-feedback files only",
            "files": records,
            "elapsed_seconds": time.time() - started,
        },
    )
    print(str(MANIFEST_PATH.resolve()), flush=True)


if __name__ == "__main__":
    main()
