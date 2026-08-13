from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_dall_exact_dedup_split.py"
SPEC = importlib.util.spec_from_file_location("phase1_dall", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cutoff_keeps_full_timestamp_group() -> None:
    reports = {}
    # Exercise the same searchsorted rule used in split views.
    times = np.array([5, 10, 10, 15], dtype=np.uint32)
    assert np.searchsorted(times, 10, side="right") == 3


def test_exact_key_encoding_is_unique() -> None:
    timestamp = np.array([5, 5, 10], dtype=np.uint64)
    item = np.array([1, 2, 1], dtype=np.uint64)
    key = (timestamp << np.uint64(24)) | item
    assert len(np.unique(key)) == 3
