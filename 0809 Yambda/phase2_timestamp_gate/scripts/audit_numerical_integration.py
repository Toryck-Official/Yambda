#!/usr/bin/env python3
"""Compare deterministic midpoint integration Q on a train-only subset.

No parameters are trained. Every Q uses identical model parameters, history,
intervals, and target intensities. Q=32 is retained as the user-requested
temporary reference. Because the first audit showed it had not converged, Q=128
is evaluated on the full subset and Q=256 on a small representative subset.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase2_timestamp_gate"))

from grouped_likelihood import GroupedLikelihoodGate, tensors_for_record  # noqa: E402


SUBSET = ROOT / "phase2_timestamp_gate/artifacts/integration_audit_subset.json"
OUT = ROOT / "phase2_timestamp_gate/artifacts/numerical_integration_metrics.json"
Q_VALUES = (2, 4, 8, 16, 32, 64, 128)
REQUESTED_REFERENCE_Q = 32
REFERENCE_Q = 128
SANITY_Q = 256
RUNTIME_REPEATS = 3
DTYPE = torch.float64


def dist(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def integrate(model: GroupedLikelihoodGate, tensors: tuple[torch.Tensor, ...], q: int) -> float:
    history_feedback, history_times, previous, target, _ = tensors
    model.quadrature_points = q
    value = model.integrate_total_intensity(history_feedback, history_times, previous, target)
    return float(value)


def error_bundle(values: list[float], reference: list[float]) -> dict:
    absolute = [abs(x - y) for x, y in zip(values, reference, strict=True)]
    relative = [abs(x - y) / max(abs(y), 1e-12) for x, y in zip(values, reference, strict=True)]
    return {"absolute": dist(absolute), "relative": dist(relative)}


def slice_q(rows: list[dict], indices: list[int], q: int) -> dict:
    absolute_integral = [rows[i][f"q{q}_abs_integral_error"] for i in indices]
    relative_integral = [rows[i][f"q{q}_rel_integral_error"] for i in indices]
    absolute_nll = [rows[i][f"q{q}_abs_time_nll_difference"] for i in indices]
    relative_nll = [rows[i][f"q{q}_rel_time_nll_difference"] for i in indices]
    return {
        "count": len(indices),
        "integral_absolute_error": dist(absolute_integral),
        "integral_relative_error": dist(relative_integral),
        "time_nll_absolute_difference": dist(absolute_nll),
        "time_nll_relative_difference": dist(relative_nll),
    }


def main() -> None:
    started = time.time()
    torch.set_num_threads(1)
    subset = json.loads(SUBSET.read_text())
    records = subset["records"]
    model = GroupedLikelihoodGate(dtype=DTYPE)
    model.eval()
    tensors = [tensors_for_record(record, dtype=DTYPE) for record in records]

    integrals: dict[int, list[float]] = {q: [] for q in Q_VALUES}
    target_intensity = []
    influence_mass = []
    with torch.no_grad():
        for record_tensors in tensors:
            details = model.intensities(record_tensors[0], record_tensors[1], record_tensors[3])
            target_intensity.append(float(details.lambdas.sum()))
            influence_mass.append(float(details.absolute_mass.sum()))
        for q in Q_VALUES:
            integrals[q] = [integrate(model, item, q) for item in tensors]

    reference_integral = integrals[REFERENCE_Q]
    log_target_intensity = [math.log(x) for x in target_intensity]
    time_nll = {
        q: [integral - log_lam for integral, log_lam in zip(integrals[q], log_target_intensity, strict=True)]
        for q in Q_VALUES
    }
    reference_nll = time_nll[REFERENCE_Q]

    q_metrics = {}
    for q in Q_VALUES:
        q_metrics[str(q)] = {
            "integral_error_vs_q128": error_bundle(integrals[q], reference_integral),
            "time_nll_difference_vs_q128": error_bundle(time_nll[q], reference_nll),
        }
    requested_reference_integral = integrals[REQUESTED_REFERENCE_Q]
    requested_reference_nll = time_nll[REQUESTED_REFERENCE_Q]
    requested_q32_comparison = {
        str(q): {
            "integral_error_vs_q32": error_bundle(integrals[q], requested_reference_integral),
            "time_nll_difference_vs_q32": error_bundle(time_nll[q], requested_reference_nll),
        }
        for q in (2, 4, 8, 16, 32)
    }

    # Runtime benchmark: warm cache, then report median of three complete passes.
    with torch.no_grad():
        _ = integrate(model, tensors[0], REFERENCE_Q)
    runtime = {}
    for q in Q_VALUES:
        repeats = []
        for _ in range(RUNTIME_REPEATS):
            tic = time.perf_counter()
            with torch.no_grad():
                checksum = sum(integrate(model, item, q) for item in tensors)
            elapsed = time.perf_counter() - tic
            if not math.isfinite(checksum):
                raise RuntimeError(f"non-finite benchmark checksum for Q={q}")
            repeats.append(elapsed)
        median_seconds = float(statistics.median(repeats))
        runtime[str(q)] = {
            "seconds_repeats": repeats,
            "median_seconds": median_seconds,
            "throughput_intervals_per_second": len(tensors) / median_seconds,
        }
    q4_seconds = runtime["4"]["median_seconds"]
    for q in Q_VALUES:
        runtime[str(q)]["compute_cost_multiple_vs_q4"] = runtime[str(q)]["median_seconds"] / q4_seconds

    rows = []
    for index, record in enumerate(records):
        row = {
                "target_group_id": record["target_group_id"],
                "gap_bin": record["gap_bin"],
                "previous_group_size_bin": record["previous_group_size_bin"],
                "gap_seconds": record["gap_seconds"],
                "previous_group_size": record["previous_group_size"],
                "target_intensity": target_intensity[index],
                "absolute_influence_mass": influence_mass[index],
        }
        for q in (4, 32, 64):
            absolute_integral = abs(integrals[q][index] - reference_integral[index])
            row[f"q{q}_abs_integral_error"] = absolute_integral
            row[f"q{q}_rel_integral_error"] = absolute_integral / max(
                abs(reference_integral[index]), 1e-12
            )
            absolute_nll = abs(time_nll[q][index] - reference_nll[index])
            row[f"q{q}_abs_time_nll_difference"] = absolute_nll
            row[f"q{q}_rel_time_nll_difference"] = absolute_nll / max(
                abs(reference_nll[index]), 1e-12
            )
        rows.append(row)

    by_gap: dict[str, list[int]] = defaultdict(list)
    by_previous: dict[str, list[int]] = {"le_500": [], "gt_500": []}
    for index, row in enumerate(rows):
        by_gap[row["gap_bin"]].append(index)
        by_previous["gt_500" if row["previous_group_size"] > 500 else "le_500"].append(index)

    lambda_median = float(np.median(target_intensity))
    influence_median = float(np.median(influence_mass))
    by_lambda = {
        "low": [i for i, x in enumerate(target_intensity) if x <= lambda_median],
        "high": [i for i, x in enumerate(target_intensity) if x > lambda_median],
    }
    by_influence = {
        "low": [i for i, x in enumerate(influence_mass) if x <= influence_median],
        "high": [i for i, x in enumerate(influence_mass) if x > influence_median],
    }
    slices = {}
    for q in (4, 32, 64):
        slices[f"q{q}"] = {
            "gap": {name: slice_q(rows, ids, q) for name, ids in by_gap.items()},
            "previous_group_size": {
                name: slice_q(rows, ids, q) for name, ids in by_previous.items()
            },
            "target_intensity": {
                "sample_median": lambda_median,
                "bins": {name: slice_q(rows, ids, q) for name, ids in by_lambda.items()},
            },
            "absolute_influence_mass": {
                "sample_median": influence_median,
                "bins": {name: slice_q(rows, ids, q) for name, ids in by_influence.items()},
            },
        }

    # Q64 sanity: one deterministic representative from every non-empty cell.
    sanity_ids = []
    for key in sorted(subset["selected_group_ids_by_cell"]):
        values = subset["selected_group_ids_by_cell"][key]
        if values:
            sanity_ids.append(int(values[len(values) // 2]))
    record_index = {int(row["target_group_id"]): i for i, row in enumerate(records)}
    sanity_indices = [record_index[x] for x in sanity_ids]
    q128_sanity = [reference_integral[i] for i in sanity_indices]
    with torch.no_grad():
        q256_sanity = [integrate(model, tensors[i], SANITY_Q) for i in sanity_indices]
    high_precision_check = {
        "records": len(sanity_indices),
        "selection": "one middle group-id representative per non-empty stratification cell",
        "q128_error_vs_q256": error_bundle(q128_sanity, q256_sanity),
    }

    # Direct convergence diagnostics requested in addition to the common Q32 reference.
    q4_pairwise = {
        f"q4_vs_q{q}": error_bundle(integrals[4], integrals[q])
        for q in (8, 16, 32, 64, 128)
    }

    all_finite = all(
        math.isfinite(value)
        for values in integrals.values()
        for value in values
    ) and all(x > 0 and math.isfinite(x) for x in target_intensity)
    result = {
        "status": "complete_train_only_numerical_integration_accuracy_audit",
        "data_contract": subset["data_contract"]
        | {
            "records": len(records),
            "model_parameters_identical_across_Q": True,
            "parameters_trained": False,
            "integration_rule": "deterministic midpoint",
            "q_values": list(Q_VALUES),
            "requested_temporary_reference_q": REQUESTED_REFERENCE_Q,
            "final_full_sample_numerical_reference_q": REFERENCE_Q,
        },
        "sample_coverage": {
            "gap_counts": {name: len(ids) for name, ids in by_gap.items()},
            "previous_group_size_counts": {
                name: sum(row["previous_group_size_bin"] == name for row in rows)
                for name in ("1", "2_5", "6_20", "21_100", "101_500", "gt_500")
            },
            "stratification_cells_nonempty": subset["counts"]["nonempty_cells"],
            "stratification_cells_total": subset["counts"]["total_cells"],
        },
        "requested_q_comparison_vs_q32": requested_q32_comparison,
        "high_precision_q_comparison_vs_q128": q_metrics,
        "q4_pairwise_convergence": q4_pairwise,
        "runtime": runtime,
        "stress_slices_vs_q128": slices,
        "q256_reference_sanity": high_precision_check,
        "all_integrals_intensities_finite": all_finite,
        "per_sample_q4_diagnostics": rows,
        "boundaries": {
            "validation_or_test_used": False,
            "minimal_snmpp_training_started": False,
            "model_structure_modified": False,
            "burst_deleted_or_normalized": False,
        },
        "elapsed_seconds": time.time() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "records": len(records),
        "coverage": result["sample_coverage"],
        "q4_vs_q32": requested_q32_comparison["4"],
        "q4_vs_q128": q_metrics["4"],
        "q128_vs_q256": high_precision_check["q128_error_vs_q256"],
        "runtime": runtime,
        "all_finite": all_finite,
    }, indent=2))


if __name__ == "__main__":
    main()
