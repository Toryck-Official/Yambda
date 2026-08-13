from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, script: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STABILITY = load("gate1b_stability", "calibrate_rq_stability.py")
AUDIT = load("gate1b_audit", "audit_metadata_information.py")


def test_controlled_perturbation_hits_cosine() -> None:
    rng = np.random.default_rng(7)
    truth = rng.normal(size=(1000, 16)).astype(np.float32)
    truth /= np.linalg.norm(truth, axis=1, keepdims=True)
    perturbed = STABILITY.controlled_perturbation(truth, 0.85, rng)
    cosine = np.sum(truth * perturbed, axis=1)
    assert np.allclose(cosine, 0.85, atol=2e-6)
    assert np.allclose(np.linalg.norm(perturbed, axis=1), 1.0, atol=2e-6)


def test_splitmix_is_deterministic_and_sensitive() -> None:
    x = np.array([1, 2, 3, 1], dtype=np.uint64)
    first = AUDIT.splitmix64(x)
    second = AUDIT.splitmix64(x)
    assert np.array_equal(first, second)
    assert first[0] == first[3]
    assert len(np.unique(first[:3])) == 3


def test_bucket_summary_counts_identity_groups() -> None:
    signature = np.array([[1, 2], [1, 2], [3, 4], [5, 6], [5, 6], [5, 6]], dtype=np.uint64)
    report, _, _ = AUDIT.bucket_summary(signature)
    assert report["unique_contexts"] == 3
    assert report["singleton_items"] == 1
    assert report["items_in_nonunique_context"] == 5
    assert report["maximum_context_collision_bucket"] == 3
