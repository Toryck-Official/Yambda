from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from minimal_snmpp_pilot.model import MinimalSNMPP
from minimal_snmpp_pilot.failure_diagnosis.factorized_model import FactorizedMinimalSNMPP


def codebook(tmp_path: Path) -> Path:
    path = tmp_path / "codebook.npy"
    np.save(path, np.random.default_rng(7).normal(size=(4, 256, 128)).astype(np.float32))
    return path


def batch(order=(0, 1, 2, 3)) -> dict[str, torch.Tensor]:
    counts = torch.zeros(1, 4)
    for mark in order:
        counts[0, mark] += 1
    return {
        "history_times": torch.tensor([[0.0, 0.0, 0.5, 0.5]]),
        "history_feedback": torch.tensor([[0, 1, 2, 3]]),
        "history_sid": torch.tensor([[[1,2,3,4],[4,3,2,1],[5,6,7,8],[9,10,11,12]]]),
        "history_mask": torch.ones(1, 4, dtype=torch.bool),
        "previous_time": torch.tensor([0.5]),
        "target_time": torch.tensor([1.25]),
        "target_feedback_counts": counts,
    }


def outputs(model, sample):
    return model.conditional_outputs(
        sample["target_time"][:, None], sample["history_times"], sample["history_feedback"],
        sample["history_sid"], sample["history_mask"], return_influence=True,
    )


def test_positive_normalized_and_lambda_factorization(tmp_path: Path):
    model = FactorizedMinimalSNMPP(codebook(tmp_path))
    total, q, lambdas, _ = outputs(model, batch())
    assert torch.isfinite(total).all() and (total > 0).all()
    assert torch.isfinite(q).all() and (q >= 0).all()
    assert torch.allclose(q.sum(-1), torch.ones_like(total), atol=1e-7)
    assert torch.allclose(lambdas.sum(-1), total, atol=1e-7)


def test_singleton_marked_likelihood_identity(tmp_path: Path):
    model = FactorizedMinimalSNMPP(codebook(tmp_path))
    sample = batch((2,))
    total, q, lambdas, _ = outputs(model, sample)
    integral = model.loss(sample, deterministic_integral=True).integral_by_group
    factorized = -torch.log(total[:, 0]) - torch.log(q[:, 0, 2]) + integral
    marked = -torch.log(lambdas[:, 0, 2]) + integral
    assert torch.allclose(factorized, marked, atol=1e-7, rtol=1e-7)


def test_permutation_invariance_and_shared_pre_history(tmp_path: Path):
    model = FactorizedMinimalSNMPP(codebook(tmp_path))
    left = model.loss(batch((0,1,2,3)), deterministic_integral=True)
    right = model.loss(batch((3,2,1,0)), deterministic_integral=True)
    assert torch.equal(left.target_lambdas, right.target_lambdas)
    assert torch.allclose(left.optimization_loss, right.optimization_loss, atol=1e-7)
    changed = batch((0,0,0,0))
    changed_output = model.loss(changed, deterministic_integral=True)
    assert torch.equal(left.target_lambdas, changed_output.target_lambdas)


def test_delayed_update_raw_sum_and_no_hidden_order(tmp_path: Path):
    model = FactorizedMinimalSNMPP(codebook(tmp_path))
    sample = batch()
    original = model.loss(sample, deterministic_integral=True)
    duplicated = {key: value.clone() for key, value in sample.items()}
    for key in ("history_times", "history_feedback", "history_sid", "history_mask"):
        duplicated[key] = torch.cat([duplicated[key], duplicated[key]], dim=1)
    changed = model.loss(duplicated, deterministic_integral=True)
    original_context, _ = model.shared_temporal_context(
        sample["target_time"][:, None], sample["history_times"], sample["history_feedback"],
        sample["history_sid"], sample["history_mask"], return_influence=False,
    )
    changed_context, _ = model.shared_temporal_context(
        duplicated["target_time"][:, None], duplicated["history_times"], duplicated["history_feedback"],
        duplicated["history_sid"], duplicated["history_mask"], return_influence=False,
    )
    assert not torch.equal(original_context, changed_context)
    # Only a count multiset is accepted for the target group; no target order exists in the API.
    assert "target_feedback_sequence" not in sample


def test_q64_contract_and_random_vs_midpoint(tmp_path: Path):
    model = FactorizedMinimalSNMPP(codebook(tmp_path))
    assert model.integration_q == 64
    assert torch.equal(model.stratified_ratios(2, deterministic=True), model.stratified_ratios(2, deterministic=True))
    assert not torch.equal(model.stratified_ratios(2, deterministic=False), model.stratified_ratios(2, deterministic=False))


def test_four_mark_head_gradients_are_finite_nonzero(tmp_path: Path):
    model = FactorizedMinimalSNMPP(codebook(tmp_path))
    result = model.loss(batch((0, 0, 1, 2, 3)), deterministic_integral=True)
    result.feedback_loss.backward()
    assert model.mark_head.bias.grad is not None
    assert torch.isfinite(model.mark_head.bias.grad).all()
    assert (model.mark_head.bias.grad.abs() > 0).all()
    assert model.mark_head.weight.grad is not None
    assert torch.isfinite(model.mark_head.weight.grad).all()
    assert (torch.linalg.vector_norm(model.mark_head.weight.grad, dim=1) > 0).all()


def test_only_lightweight_output_capacity_added_and_codebook_frozen(tmp_path: Path):
    path = codebook(tmp_path)
    old = MinimalSNMPP(path)
    new = FactorizedMinimalSNMPP(path)
    old_parameters = sum(p.numel() for p in old.parameters())
    new_parameters = sum(p.numel() for p in new.parameters())
    assert new.head_parameter_count == 25
    assert new_parameters - old_parameters == 21
    assert "codebook" in dict(new.named_buffers())
    assert "codebook" not in dict(new.named_parameters())
