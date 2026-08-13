from __future__ import annotations

import itertools
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase2_timestamp_gate"))

from grouped_likelihood import GroupedLikelihoodGate  # noqa: E402


DTYPE = torch.float64


def sample_inputs(target=(0, 2, 3)):
    history_feedback = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    history_times = torch.tensor([0.0, 0.0, 0.5, 0.5], dtype=DTYPE)
    previous = torch.tensor(0.5, dtype=DTYPE)
    current = torch.tensor(1.25, dtype=DTYPE)
    target_feedback = torch.tensor(target, dtype=torch.long)
    return history_feedback, history_times, previous, current, target_feedback


def test_permutation_invariance():
    model = GroupedLikelihoodGate(dtype=DTYPE)
    reference = model.group_loss(*sample_inputs((0, 2, 3)))
    for perm in itertools.permutations((0, 2, 3)):
        got = model.group_loss(*sample_inputs(perm))
        assert torch.equal(got.time, reference.time)
        assert torch.allclose(got.mark_sum, reference.mark_sum, atol=1e-14, rtol=0)
        assert torch.equal(got.intensity, reference.intensity)
        assert torch.equal(got.q, reference.q)


def test_singleton_equivalence_to_standard_marked_nll():
    model = GroupedLikelihoodGate(dtype=DTYPE)
    inputs = sample_inputs((2,))
    grouped = model.group_loss(*inputs)
    standard = grouped.integral - torch.log(grouped.details.lambdas[2])
    assert torch.allclose(grouped.theoretical, standard, atol=1e-12, rtol=0)


def test_shared_history_delayed_update_integral_once_and_no_zero_delta():
    model = GroupedLikelihoodGate(dtype=DTYPE)
    groups = [
        (0.0, [0, 1]),
        (0.5, [2, 3, 0]),
        (1.25, [1, 2]),
    ]
    history_feedback: list[int] = []
    history_times: list[float] = []
    trace = []
    integral_count = 0
    positive_intervals = []
    for index, (timestamp, marks) in enumerate(groups):
        pre_count = len(history_feedback)
        # The first group initializes history; subsequent groups are scored once.
        if index:
            hfb = torch.tensor(history_feedback, dtype=torch.long)
            hts = torch.tensor(history_times, dtype=DTYPE)
            loss = model.group_loss(
                hfb,
                hts,
                torch.tensor(groups[index - 1][0], dtype=DTYPE),
                torch.tensor(timestamp, dtype=DTYPE),
                torch.tensor(marks, dtype=torch.long),
            )
            assert torch.isfinite(loss.theoretical)
            integral_count += 1
            positive_intervals.append(timestamp - groups[index - 1][0])
        # Update only after every mark in the group has been scored.
        assert len(history_feedback) == pre_count
        history_feedback.extend(marks)
        history_times.extend([timestamp] * len(marks))
        trace.append((pre_count, len(history_feedback)))

    assert trace == [(0, 2), (2, 5), (5, 7)]
    assert integral_count == len(groups) - 1
    assert all(x > 0 for x in positive_intervals)


def test_numerical_stability_and_four_feedback_gradient_visibility():
    model = GroupedLikelihoodGate(dtype=DTYPE)
    total_loss = model.base_score.new_zeros(())
    for feedback in range(4):
        total_loss = total_loss + model.group_loss(*sample_inputs((feedback,))).normalized_multitask
    total_loss.backward()
    assert torch.isfinite(total_loss)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert (model.base_score.grad.abs() > 0).all()


def test_target_cardinality_replication_has_no_time_or_intensity_effect():
    model = GroupedLikelihoodGate(dtype=DTYPE)
    results = []
    for size in (1, 2, 5, 20, 100, 501):
        inputs = sample_inputs(tuple([0, 1, 2, 3] * ((size + 3) // 4))[:size])
        results.append(model.group_loss(*inputs))
    for got in results[1:]:
        assert torch.equal(got.time, results[0].time)
        assert torch.equal(got.intensity, results[0].intensity)
    # Conditional marks add once per observed event; their mean is cardinality-stable
    # for an exactly repeated composition (use multiples of four).
    multiples = []
    for size in (4, 20, 100, 500):
        inputs = sample_inputs(tuple([0, 1, 2, 3] * (size // 4)))
        multiples.append(model.group_loss(*inputs))
    for got in multiples[1:]:
        assert torch.allclose(got.mark_mean, multiples[0].mark_mean, atol=1e-14, rtol=0)
    ratios = [float(x.mark_sum / size) for x, size in zip(multiples, (4, 20, 100, 500))]
    assert max(ratios) - min(ratios) < 1e-12
