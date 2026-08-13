from pathlib import Path
import importlib.util


PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_timestamp_order.py"
SPEC = importlib.util.spec_from_file_location("audit_timestamp_order", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_like_unlike_unique_when_pre_like_inactive():
    label, order, posts = MOD.enumerate_orders((0, 2), {0})
    assert label == "uniquely_recoverable"
    assert order == (0, 2)
    assert posts == {0}


def test_like_unlike_has_no_order_constraint_when_pre_like_active():
    label, order, posts = MOD.enumerate_orders((0, 2), {1})
    assert label == "no_order_constraint"
    assert order is None
    assert posts == {0, 1}


def test_like_dislike_has_no_order_constraint():
    label, order, posts = MOD.enumerate_orders((0, 1), {0})
    assert label == "no_order_constraint"
    assert order is None
    assert posts == {3}


def test_unlike_without_active_like_is_inconsistent():
    label, order, posts = MOD.enumerate_orders((2, 1), {0})
    assert label == "inconsistent"
    assert order is None
    assert posts == {2}


def test_group_bins():
    import numpy as np
    got = MOD.bin_index(np.asarray([1, 2, 3, 5, 6, 10, 11, 20, 21, 50, 51, 100, 101, 200, 201, 500, 501, 1000, 1001]))
    assert got.tolist() == [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10]
