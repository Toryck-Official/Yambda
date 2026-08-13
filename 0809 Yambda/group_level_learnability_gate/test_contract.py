#!/usr/bin/env python3
"""Small contract tests independent of model fitting."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


MODULE_PATH = Path(__file__).with_name("run_gate.py")
SPEC = importlib.util.spec_from_file_location("group_gate", MODULE_PATH)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def test_mark_loss_is_event_normalized() -> None:
    logits = torch.zeros(2, 4)
    counts = torch.tensor([[2.0, 0, 0, 0], [0, 1.0, 1.0, 0]])
    assert abs(float(gate.mark_loss(logits, counts)) - np.log(4)) < 1e-6


def test_lognormal_nll_finite_positive_gap() -> None:
    gap = torch.tensor([0.1, 1.0, 100.0])
    nll = gate.lognormal_nll(gap, torch.zeros(3), torch.ones(3))
    assert torch.isfinite(nll).all()


def test_group_feature_is_permutation_invariant() -> None:
    rng = np.random.default_rng(7)
    codebook = rng.normal(size=(4, 256, 128)).astype(np.float32)
    sid = rng.integers(0, 256, size=(7, 4), dtype=np.uint8)
    event_offsets = np.array([0, 7], dtype=np.int64)
    feedback_counts = np.array([[2, 1, 3, 1]], dtype=np.uint16)
    timestamp = np.array([10], dtype=np.uint32)
    a = gate._features_for_group_range(0, 1, event_sid=sid, event_offsets=event_offsets,
                                       feedback_counts=feedback_counts, timestamps=timestamp,
                                       codebooks=codebook)
    p = rng.permutation(7)
    b = gate._features_for_group_range(0, 1, event_sid=sid[p], event_offsets=event_offsets,
                                       feedback_counts=feedback_counts, timestamps=timestamp,
                                       codebooks=codebook)
    assert np.allclose(a, b, atol=2e-5)


if __name__ == "__main__":
    test_mark_loss_is_event_normalized()
    test_lognormal_nll_finite_positive_gap()
    test_group_feature_is_permutation_invariant()
    print("3 contract tests passed")
