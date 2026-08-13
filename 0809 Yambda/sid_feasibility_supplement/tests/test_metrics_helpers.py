import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_sid_feasibility_supplement.py"
SPEC = importlib.util.spec_from_file_location("sid_supplement", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_collision_metrics_counts_items_not_only_buckets():
    codes = np.asarray(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 5], [9, 8, 7, 6]],
        dtype=np.uint8,
    )
    result = MODULE.collision_metrics(codes)
    assert result["unique_semantic_sid_paths"] == 3
    assert result["singleton_items"] == 2
    assert result["collision_items"] == 2
    assert result["collision_item_ratio"] == .5


def test_prefix_consistency_is_cumulative():
    reference = np.asarray([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.uint8)
    candidate = np.asarray([[1, 9, 3, 4], [1, 2, 3, 8]], dtype=np.uint8)
    result = MODULE.prefix_consistency(reference, candidate)
    assert result["token_accuracy"] == [1.0, .5, 1.0, .5]
    assert result["prefix_accuracy"] == [1.0, .5, .5, 0.0]


def test_exact_duplicate_metrics_verifies_rows():
    values = np.asarray([[1, 2], [1, 2], [1, 3], [1, 3], [9, 9]], dtype=np.float32)
    # Helper is specified for 128 dimensions because its vectorized fingerprint
    # has a fixed coefficient contract matching the audio feature schema.
    values = np.pad(values, ((0, 0), (0, 126)))
    result = MODULE.exact_duplicate_metrics(values)
    assert result["duplicate_buckets"] == 2
    assert result["duplicate_items"] == 4
    assert result["duplicate_surplus"] == 2
    assert result["maximum_duplicate_bucket"] == 2


def test_hungarian_alignment_removes_cluster_label_permutation():
    rng = np.random.default_rng(3)
    reference_codebooks = rng.normal(size=(4, 256, 128)).astype(np.float32)
    reference_codes = rng.integers(0, 256, size=(1000, 4), dtype=np.uint8)
    candidate_codebooks = np.empty_like(reference_codebooks)
    candidate_codes = np.empty_like(reference_codes)
    for level in range(4):
        permutation = rng.permutation(256)
        candidate_codebooks[level] = reference_codebooks[level, permutation]
        inverse = np.empty(256, dtype=np.uint8)
        inverse[permutation] = np.arange(256, dtype=np.uint8)
        candidate_codes[:, level] = inverse[reference_codes[:, level]]
    result = MODULE.aligned_prefix_consistency(
        reference_codebooks,
        reference_codes,
        candidate_codebooks,
        candidate_codes,
    )
    assert result["prefix_accuracy"] == [1.0, 1.0, 1.0, 1.0]
