from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_train_collaborative_coverage.py"
SPEC = importlib.util.spec_from_file_location("gate1c_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_candidate_cutoff_keeps_full_timestamp() -> None:
    hist = np.array([2, 3, 5], dtype=np.uint64)
    rows = MODULE.candidate_cutoffs(hist, [0.4, 0.5, 0.6], 5)
    assert rows[0]["cutoff_timestamp"] == 5
    assert rows[0]["train_events_including_full_cutoff_timestamp"] == 5
    assert rows[1]["train_events_including_full_cutoff_timestamp"] == 5
    assert rows[2]["cutoff_timestamp"] == 10


def test_threshold_counts() -> None:
    values = np.array([0, 1, 2, 5, 10], dtype=np.uint32)
    assert MODULE.threshold_counts(values, [1, 5, 20]) == {
        "at_least_1": 4,
        "at_least_5": 2,
        "at_least_20": 0,
    }
