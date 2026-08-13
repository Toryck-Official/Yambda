from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_metadata_support_and_sample.py"
SPEC = importlib.util.spec_from_file_location("prepare_metadata_support", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PROXY_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_gate1_proxy_vectors.py"
PROXY_SPEC = importlib.util.spec_from_file_location("build_gate1_proxy", PROXY_SCRIPT)
assert PROXY_SPEC is not None and PROXY_SPEC.loader is not None
PROXY_MODULE = importlib.util.module_from_spec(PROXY_SPEC)
import sys
sys.modules[PROXY_SPEC.name] = PROXY_MODULE
PROXY_SPEC.loader.exec_module(PROXY_MODULE)


def test_metadata_support_strictly_removes_target(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.parquet"
    pq.write_table(
        pa.table(
            {
                "artist_id": np.asarray([1, 1, 1, 2], dtype=np.uint32),
                "item_id": np.asarray([10, 11, 12, 13], dtype=np.uint32),
            }
        ),
        mapping,
    )
    explicit_lookup = np.zeros(14, dtype=np.int32)
    explicit_lookup[[10, 11, 12, 13]] = [1, 2, 3, 4]
    source_present = np.zeros(14, dtype=bool)
    source_present[[10, 11]] = True
    maximum, relations, report = MODULE.metadata_loo_support(
        mapping,
        "artist_id",
        explicit_lookup,
        source_present,
        explicit_rows=4,
    )
    np.testing.assert_array_equal(maximum, [1, 1, 2, 0])
    np.testing.assert_array_equal(relations, [1, 1, 1, 0])
    assert report["explicit_items_with_strict_loo_support"] == 3


def test_category_allocation_conserves_requested_sample() -> None:
    allocation = MODULE.allocate_category_samples(
        {1: 100_000, 2: 12_000, 3: 80_000}, total=100_000, minimum=10_000
    )
    assert sum(allocation.values()) == 100_000
    assert allocation[2] >= 10_000
    assert all(allocation[key] <= value for key, value in {1: 100_000, 2: 12_000, 3: 80_000}.items())


def test_stratified_choice_conserves_size_and_is_reproducible() -> None:
    positions = np.arange(1000)
    labels = positions % 7
    first = MODULE.stratified_choice(
        positions, labels, requested=333, rng=np.random.default_rng(2026)
    )
    second = MODULE.stratified_choice(
        positions, labels, requested=333, rng=np.random.default_rng(2026)
    )
    assert len(first) == 333
    assert len(np.unique(first)) == 333
    np.testing.assert_array_equal(first, second)


def test_streaming_proxy_strictly_removes_target(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.parquet"
    pq.write_table(
        pa.table(
            {
                "artist_id": np.asarray([1, 1, 1, 2], dtype=np.uint32),
                "item_id": np.asarray([10, 11, 12, 13], dtype=np.uint32),
            }
        ),
        mapping,
        row_group_size=2,
    )
    orig2dense = np.zeros(14, dtype=np.int32)
    orig2dense[[10, 11]] = [1, 2]
    features = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    # The implementation is fixed at the production 128-d contract.
    padded = np.zeros((3, 128), dtype=np.float32)
    padded[:, :2] = features
    result = PROXY_MODULE.metadata_group_predictions_streaming(
        mapping,
        "artist_id",
        np.asarray([10, 11, 12, 13]),
        orig2dense,
        padded,
    )
    np.testing.assert_allclose(result.vectors[0, :2], [0.0, 1.0])
    np.testing.assert_allclose(result.vectors[1, :2], [1.0, 0.0])
    np.testing.assert_allclose(result.vectors[2, :2], np.sqrt(0.5), rtol=1e-6)
    np.testing.assert_array_equal(result.available, [True, True, True, False])
