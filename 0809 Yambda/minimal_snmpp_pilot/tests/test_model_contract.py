from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "minimal_snmpp_pilot"))
from model import MinimalSNMPP  # noqa: E402


def make_codebook(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    path = tmp_path / "codebook.npy"
    np.save(path, rng.normal(size=(4, 256, 128)).astype(np.float32))
    return path


def make_batch(order=(0, 2, 3)) -> dict[str, torch.Tensor]:
    history_feedback = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    history_sid = torch.tensor(
        [[[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8], [9, 10, 11, 12]]],
        dtype=torch.long,
    )
    counts = torch.zeros(1, 4)
    for value in order:
        counts[0, value] += 1
    return {
        "history_times": torch.tensor([[0.0, 0.0, 0.5, 0.5]]),
        "history_feedback": history_feedback,
        "history_sid": history_sid,
        "history_mask": torch.ones(1, 4, dtype=torch.bool),
        "previous_time": torch.tensor([0.5]),
        "target_time": torch.tensor([1.25]),
        "target_feedback_counts": counts,
    }


def test_frozen_codebook_and_no_sid_head(tmp_path: Path):
    model = MinimalSNMPP(make_codebook(tmp_path))
    assert "codebook" in dict(model.named_buffers())
    assert "codebook" not in dict(model.named_parameters())
    assert not any("sid_head" in name for name, _ in model.named_modules())
    assert model.integration_q == 64


def test_group_multiset_permutation_invariance_and_finite_gradients(tmp_path: Path):
    torch.manual_seed(2)
    model = MinimalSNMPP(make_codebook(tmp_path))
    left = model.loss(make_batch((0, 2, 3)), deterministic_integral=True)
    right = model.loss(make_batch((3, 0, 2)), deterministic_integral=True)
    assert torch.allclose(left.optimization_loss, right.optimization_loss, atol=1e-7)
    assert torch.allclose(left.target_lambdas, right.target_lambdas, atol=0, rtol=0)
    left.optimization_loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert (left.target_lambdas > 0).all()


def test_no_same_group_history_and_no_history_normalization(tmp_path: Path):
    torch.manual_seed(3)
    model = MinimalSNMPP(make_codebook(tmp_path))
    batch = make_batch()
    original = model.loss(batch, deterministic_integral=True)
    changed_target = {k: v.clone() for k, v in batch.items()}
    changed_target["target_feedback_counts"] = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
    changed = model.loss(changed_target, deterministic_integral=True)
    # Target marks affect mark scoring but cannot enter target-time intensity.
    assert torch.equal(original.target_lambdas, changed.target_lambdas)

    duplicated = {k: v.clone() for k, v in batch.items()}
    for key in ("history_times", "history_feedback", "history_sid", "history_mask"):
        duplicated[key] = torch.cat([duplicated[key], duplicated[key]], dim=1)
    doubled = model.loss(duplicated, deterministic_integral=True)
    assert not torch.allclose(original.target_lambdas, doubled.target_lambdas)


def test_random_integral_changes_but_midpoint_is_deterministic(tmp_path: Path):
    model = MinimalSNMPP(make_codebook(tmp_path))
    a = model.loss(make_batch(), deterministic_integral=True).integral_by_group
    b = model.loss(make_batch(), deterministic_integral=True).integral_by_group
    assert torch.equal(a, b)
    # The stratified points must be random.  A nearly constant initialized
    # intensity may nevertheless make their float32 integral exactly equal.
    random_ratios = [model.stratified_ratios(1, deterministic=False) for _ in range(3)]
    assert not torch.equal(random_ratios[0], random_ratios[1])
