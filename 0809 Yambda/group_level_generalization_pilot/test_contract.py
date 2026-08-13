#!/usr/bin/env python3
"""Generalization-pilot selection and no-test contract checks."""

import importlib.util
from pathlib import Path
import sys

import numpy as np


path = Path(__file__).with_name("run_generalization.py")
spec = importlib.util.spec_from_file_location("group_generalization", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_stable_user_order() -> None:
    users = np.array([10, 2, 99, 7], dtype=np.uint32)
    assert np.array_equal(module.stable_user_order(users), module.stable_user_order(users.copy()))


def test_configuration_is_frozen() -> None:
    assert module.EPOCHS == 20
    assert module.BATCH_SIZE == 64
    assert module.LEARNING_RATE == 1e-3
    assert module.SEEDS == [2026, 2027, 2028]


if __name__ == "__main__":
    test_stable_user_order()
    test_configuration_is_frozen()
    print("2 generalization contract tests passed")
