from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def test_train_only_stratification_and_boundaries():
    subset = load("phase2_timestamp_gate/artifacts/integration_audit_subset.json")
    metrics = load("phase2_timestamp_gate/artifacts/numerical_integration_metrics.json")
    assert subset["data_contract"]["validation_or_test_used"] is False
    assert subset["counts"]["nonempty_cells"] == 30
    assert subset["counts"]["total_cells"] == 30
    assert metrics["data_contract"]["parameters_trained"] is False
    assert metrics["all_integrals_intensities_finite"] is True
    assert metrics["boundaries"]["minimal_snmpp_training_started"] is False
    assert metrics["boundaries"]["burst_deleted_or_normalized"] is False


def test_q64_is_materially_more_accurate_than_q4_and_status_is_frozen():
    metrics = load("phase2_timestamp_gate/artifacts/numerical_integration_metrics.json")
    status = load("phase2_timestamp_gate/status.json")
    q4 = metrics["high_precision_q_comparison_vs_q128"]["4"]["integral_error_vs_q128"]
    q64 = metrics["high_precision_q_comparison_vs_q128"]["64"]["integral_error_vs_q128"]
    assert q64["relative"]["p95"] < q4["relative"]["p95"]
    assert q64["relative"]["p99"] < q4["relative"]["p99"]
    assert status["numerical_integration_validated"] is True
    assert status["recommended_integration_Q"] == 64
    assert status["minimal_snmpp_pilot_approved"] is True
    assert status["minimal_snmpp_training_started"] is False


def test_protocol_status_matches_runtime_status():
    protocol = load("snmpp_phase2_protocol/protocol_status.json")
    status = load("phase2_timestamp_gate/status.json")
    keys = (
        "gate2_materialized",
        "timestamp_protocol_selected",
        "timestamp_protocol",
        "timestamp_implementation_validated",
        "timestamp_B_approved_for_minimal_snmpp",
        "numerical_integration_validated",
        "recommended_integration_Q",
        "minimal_snmpp_pilot_approved",
        "minimal_snmpp_training_started",
    )
    for key in keys:
        assert protocol[key] == status[key]
