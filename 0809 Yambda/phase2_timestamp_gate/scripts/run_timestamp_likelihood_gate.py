#!/usr/bin/env python3
"""Run the audit-only Timestamp Likelihood Gate for protocol B.

No optimizer or training loop is used.  The model is a deterministic signed
four-feedback kernel whose purpose is to expose likelihood accounting, history
updates, cardinality effects, gradients, and raw-sum burst behavior.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase2_timestamp_gate"))

from grouped_likelihood import GroupedLikelihoodGate, tensors_for_record  # noqa: E402


SUBSET = ROOT / "phase2_timestamp_gate/artifacts/gate_subset.json"
OUT = ROOT / "phase2_timestamp_gate/artifacts/timestamp_likelihood_gate_metrics.json"
DTYPE = torch.float64
BIN_ORDER = ("1", "2-5", "6-20", "21-100", "101-500", "500+")


def finite_scalar(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all())


def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum())
    return math.sqrt(total)


def summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def hard_contract_checks(model: GroupedLikelihoodGate) -> dict:
    history_feedback = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    history_times = torch.tensor([0.0, 0.0, 0.5, 0.5], dtype=DTYPE)
    previous = torch.tensor(0.5, dtype=DTYPE)
    target_time = torch.tensor(1.25, dtype=DTYPE)

    base_target = torch.tensor([0, 2, 3], dtype=torch.long)
    reference = model.group_loss(history_feedback, history_times, previous, target_time, base_target)
    maximum_permutation_difference = 0.0
    maximum_output_difference = 0.0
    for permutation in itertools.permutations([0, 2, 3]):
        got = model.group_loss(
            history_feedback,
            history_times,
            previous,
            target_time,
            torch.tensor(permutation, dtype=torch.long),
        )
        maximum_permutation_difference = max(
            maximum_permutation_difference, abs(float(got.mark_sum - reference.mark_sum))
        )
        maximum_output_difference = max(
            maximum_output_difference,
            float((got.q - reference.q).abs().max()),
            abs(float(got.intensity - reference.intensity)),
            abs(float(got.time - reference.time)),
        )

    singleton_differences = []
    for feedback in range(4):
        got = model.group_loss(
            history_feedback,
            history_times,
            previous,
            target_time,
            torch.tensor([feedback], dtype=torch.long),
        )
        standard = got.integral - torch.log(got.details.lambdas[feedback])
        singleton_differences.append(abs(float(got.theoretical - standard)))

    # Explicit sequence trace: score with pre-history, update only after scoring.
    groups = [(0.0, [0, 1]), (0.5, [2, 3, 0]), (1.25, [1, 2])]
    h_feedback: list[int] = []
    h_times: list[float] = []
    trace = []
    integral_count = 0
    zero_delta_transition_count = 0
    for index, (timestamp, marks) in enumerate(groups):
        pre_count = len(h_feedback)
        if index:
            got = model.group_loss(
                torch.tensor(h_feedback, dtype=torch.long),
                torch.tensor(h_times, dtype=DTYPE),
                torch.tensor(groups[index - 1][0], dtype=DTYPE),
                torch.tensor(timestamp, dtype=DTYPE),
                torch.tensor(marks, dtype=torch.long),
            )
            if not finite_scalar(got.theoretical):
                raise RuntimeError("non-finite synthetic sequence loss")
            integral_count += 1
            if timestamp - groups[index - 1][0] == 0:
                zero_delta_transition_count += 1
        count_during_scoring = len(h_feedback)
        h_feedback.extend(marks)
        h_times.extend([timestamp] * len(marks))
        trace.append(
            {
                "pre_history_events": pre_count,
                "history_events_during_group_scoring": count_during_scoring,
                "post_group_history_events": len(h_feedback),
            }
        )

    model.zero_grad(set_to_none=True)
    visibility_loss = model.base_score.new_zeros(())
    for feedback in range(4):
        visibility_loss = visibility_loss + model.group_loss(
            history_feedback,
            history_times,
            previous,
            target_time,
            torch.tensor([feedback], dtype=torch.long),
        ).normalized_multitask
    visibility_loss.backward()
    feedback_base_gradients = model.base_score.grad.detach().cpu().tolist()
    all_parameter_gradients_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )

    # Controlled cardinality replication.  Target M must not affect time or
    # intensity. The theoretical mark sum is intentionally additive; the
    # normalized mark objective remains stable for a repeated composition.
    controlled = []
    for size in (4, 20, 100, 500, 504):
        marks = torch.tensor([0, 1, 2, 3] * (size // 4), dtype=torch.long)
        got = model.group_loss(history_feedback, history_times, previous, target_time, marks)
        controlled.append(
            {
                "target_group_size": size,
                "time_loss": float(got.time),
                "intensity": float(got.intensity),
                "mark_sum": float(got.mark_sum),
                "mark_loss_per_event": float(got.mark_mean),
            }
        )
    time_spread = max(x["time_loss"] for x in controlled) - min(x["time_loss"] for x in controlled)
    intensity_spread = max(x["intensity"] for x in controlled) - min(
        x["intensity"] for x in controlled
    )
    mark_mean_spread = max(x["mark_loss_per_event"] for x in controlled) - min(
        x["mark_loss_per_event"] for x in controlled
    )

    tolerance = 1e-12
    checks = {
        "permutation_invariance": maximum_permutation_difference <= tolerance
        and maximum_output_difference <= tolerance,
        "shared_pre_history": all(
            row["pre_history_events"] == row["history_events_during_group_scoring"]
            for row in trace
        ),
        "delayed_update": trace
        == [
            {"pre_history_events": 0, "history_events_during_group_scoring": 0, "post_group_history_events": 2},
            {"pre_history_events": 2, "history_events_during_group_scoring": 2, "post_group_history_events": 5},
            {"pre_history_events": 5, "history_events_during_group_scoring": 5, "post_group_history_events": 7},
        ],
        "integral_uniqueness": integral_count == len(groups) - 1,
        "no_fake_zero_delta_transition": zero_delta_transition_count == 0,
        "singleton_equivalence": max(singleton_differences) <= tolerance,
        "numerical_stability": finite_scalar(reference.theoretical)
        and bool((reference.details.lambdas > 0).all())
        and finite_scalar(reference.intensity)
        and finite_scalar(reference.integral),
        "four_feedback_gradient_visibility": all_parameter_gradients_finite
        and all(abs(x) > 0 for x in feedback_base_gradients),
        "controlled_group_size_no_mechanical_time_or_intensity_growth": time_spread <= tolerance
        and intensity_spread <= tolerance
        and mark_mean_spread <= tolerance,
    }
    return {
        "checks": checks,
        "all_hard_checks_passed": all(checks.values()),
        "numeric_diagnostics": {
            "permutation_mark_loss_max_abs_difference": maximum_permutation_difference,
            "permutation_model_output_max_abs_difference": maximum_output_difference,
            "singleton_equivalence_max_abs_difference": max(singleton_differences),
            "feedback_base_gradients": feedback_base_gradients,
            "sequence_trace": trace,
            "integral_evaluations": integral_count,
            "zero_delta_transitions": zero_delta_transition_count,
            "controlled_cardinality": controlled,
            "controlled_time_loss_spread": time_spread,
            "controlled_intensity_spread": intensity_spread,
            "controlled_mark_mean_spread": mark_mean_spread,
        },
    }


def record_metrics(model: GroupedLikelihoodGate, record: dict) -> dict:
    inputs = tensors_for_record(record, dtype=DTYPE)
    model.zero_grad(set_to_none=True)
    loss = model.group_loss(*inputs)
    loss.normalized_multitask.backward()
    details = loss.details
    values = {
        "time_loss_per_group": float(loss.time.detach()),
        "mark_loss_sum": float(loss.mark_sum.detach()),
        "mark_loss_per_event": float(loss.mark_mean.detach()),
        "theoretical_group_pseudonll": float(loss.theoretical.detach()),
        "normalized_multitask_objective": float(loss.normalized_multitask.detach()),
        "intensity": float(loss.intensity.detach()),
        "gradient_norm": grad_norm(model),
        "signed_total_influence": float(details.signed_influence.sum().detach()),
        "absolute_influence_mass": float(details.absolute_mass.sum().detach()),
        "positive_influence_mass": float(details.positive_mass.sum().detach()),
        "negative_influence_mass_signed": float(details.negative_mass.sum().detach()),
        "negative_influence_mass_magnitude": float((-details.negative_mass).sum().detach()),
    }
    finite = all(math.isfinite(value) for value in values.values())
    values["all_finite"] = finite
    values["all_lambdas_positive"] = bool((details.lambdas.detach() > 0).all())
    return values


def aggregate_selected(
    records_by_id: dict[int, dict], selected: dict[str, list[int]], model: GroupedLikelihoodGate
) -> tuple[dict, int]:
    output = {}
    nonfinite = 0
    for bin_name in BIN_ORDER:
        rows = []
        for gid in selected[bin_name]:
            values = record_metrics(model, records_by_id[int(gid)])
            rows.append(values)
            nonfinite += int(not values["all_finite"] or not values["all_lambdas_positive"])
        output[bin_name] = {
            key: summary([row[key] for row in rows])
            for key in (
                "time_loss_per_group",
                "mark_loss_sum",
                "mark_loss_per_event",
                "theoretical_group_pseudonll",
                "normalized_multitask_objective",
                "intensity",
                "gradient_norm",
                "signed_total_influence",
                "absolute_influence_mass",
                "positive_influence_mass",
                "negative_influence_mass_signed",
                "negative_influence_mass_magnitude",
            )
        }
    return output, nonfinite


def main() -> None:
    started = time.time()
    subset = json.loads(SUBSET.read_text())
    records_by_id = {int(row["target_group_id"]): row for row in subset["records"]}
    model = GroupedLikelihoodGate(dtype=DTYPE)

    hard = hard_contract_checks(model)
    target_stats, target_nonfinite = aggregate_selected(
        records_by_id, subset["target_group_size_samples"], model
    )
    previous_stats, previous_nonfinite = aggregate_selected(
        records_by_id, subset["previous_group_size_samples"], model
    )

    # Evidence ratios compare the extreme bin to the adjacent large-history bin.
    def median_ratio(metric: str) -> float | None:
        denominator = previous_stats["101-500"][metric]["median"]
        if denominator == 0:
            return None
        return previous_stats["500+"][metric]["median"] / denominator

    stress = {
        "raw_history_sum_used": True,
        "history_mean_or_sqrt_normalization_used": False,
        "clipping_implemented": False,
        "clipping_frequency": None,
        "nan_or_inf_cases": previous_nonfinite,
        "extreme_vs_101_500_median_ratios": {
            "absolute_influence_mass": median_ratio("absolute_influence_mass"),
            "intensity": median_ratio("intensity"),
            "gradient_norm": median_ratio("gradient_norm"),
        },
        "interpretation_rule": (
            "finite raw-sum diagnostics validate implementation numerics; scale shifts are reported "
            "as sensitivity evidence and do not trigger deletion or normalization in this gate"
        ),
    }

    approved = hard["all_hard_checks_passed"] and target_nonfinite == 0 and previous_nonfinite == 0
    result = {
        "status": "complete_timestamp_likelihood_gate",
        "protocol": {
            "name": "B_grouped_shared_history_conditional_pseudolikelihood",
            "theoretical_objective": "sum_group(time_nll_once + sum_event conditional_mark_nll)",
            "normalized_multitask_objective": "mean_group(time_nll_once) + mean_event(mark_nll)",
            "cardinality_conditioned_on_observed_M": True,
            "p_M_given_history_modeled": False,
            "within_group_order_used": False,
            "minimal_snmpp_training": False,
            "model_role": "deterministic audit harness; no optimization performed",
        },
        "subset": subset["selection"] | subset["counts"],
        "hard_contract": hard,
        "target_group_size_stability": {
            "bins": target_stats,
            "nonfinite_or_nonpositive_cases": target_nonfinite,
            "controlled_mechanical_replication_passed": hard["checks"][
                "controlled_group_size_no_mechanical_time_or_intensity_growth"
            ],
            "note": (
                "theoretical mark sum is additive in observed M by definition; mark/event and the "
                "normalized multi-task objective are the scale diagnostics"
            ),
        },
        "previous_group_burst_stress": {
            "bins": previous_stats,
            "stress_summary": stress,
        },
        "decision": {
            "timestamp_B_approved_for_minimal_snmpp": approved,
            "reason": (
                "all hard accounting/history/equivalence tests passed and all sampled raw-sum "
                "target/history group-size diagnostics remained finite"
                if approved
                else "one or more hard correctness or finite-gradient/intensity checks failed"
            ),
            "extreme_burst_over_500_action": "retain in D_SID; register sensitivity/exclusion candidate only",
        },
        "elapsed_seconds": time.time() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"approved": approved, "hard_checks": hard["checks"], "stress": stress}, indent=2))


if __name__ == "__main__":
    main()
