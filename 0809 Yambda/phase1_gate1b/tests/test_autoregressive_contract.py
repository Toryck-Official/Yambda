from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_gate1b_predictors.py"
SPEC = importlib.util.spec_from_file_location("train_gate1b", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_direct_sid_inference_has_no_truth_argument() -> None:
    signature = inspect.signature(MODULE.AutoregressiveSIDPredictor.autoregressive)
    assert list(signature.parameters) == ["self", "context"]
    model = MODULE.AutoregressiveSIDPredictor(12, 16, 0.0)
    prediction, logits = model.autoregressive(torch.randn(5, 12))
    assert prediction.shape == (5, 4)
    assert len(logits) == 4


def test_embedding_predictor_is_unit_normalized() -> None:
    model = MODULE.EmbeddingPredictor(12, 16, 0.0)
    output = model(torch.randn(7, 12))
    assert torch.allclose(output.norm(dim=1), torch.ones(7), atol=1e-5)
