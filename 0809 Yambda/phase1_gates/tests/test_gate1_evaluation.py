from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_gate1_proxy.py"
SPEC = importlib.util.spec_from_file_location("evaluate_gate1_proxy", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_normalized_blend_and_single_source_fallback() -> None:
    artist = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    album = np.asarray([[0.0, 1.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result, available = MODULE.normalized_blend(
        artist,
        album,
        np.asarray([True, True, False]),
        np.asarray([True, False, True]),
        album_weight=0.5,
    )
    np.testing.assert_allclose(result[0], np.sqrt(0.5), rtol=1e-6)
    np.testing.assert_allclose(result[1], [1.0, 0.0])
    np.testing.assert_allclose(result[2], [0.0, 1.0])
    np.testing.assert_array_equal(available, [True, True, True])


def test_aggregate_prefix_accuracy() -> None:
    diagnostics = {
        "cosine": np.asarray([1.0, 0.5]),
        "squared_error": np.asarray([0.0, 0.25]),
        "prefix": np.asarray([[True, True, True, True], [True, False, False, False]]),
    }
    metrics = MODULE.aggregate(diagnostics, np.asarray([True, True]))
    assert metrics["prefix_accuracy"]["prefix_acc_at_1"] == 1.0
    assert metrics["prefix_accuracy"]["prefix_acc_at_2"] == 0.5
    assert metrics["prefix_accuracy"]["prefix_acc_at_4"] == 0.5

