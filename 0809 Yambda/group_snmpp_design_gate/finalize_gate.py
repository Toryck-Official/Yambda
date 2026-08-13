#!/usr/bin/env python3
"""Merge frozen training/audit artifacts and write the final Gate decision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


WORK = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((WORK / name).read_text())


def dump(name: str, value: dict) -> None:
    (WORK / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fmean(rows: list[float]) -> float:
    return float(np.mean(rows))


def fmt(values: list[float], digits: int = 4) -> str:
    return " / ".join(f"{value:.{digits}f}" for value in values)


def main() -> None:
    metrics = load("group_snmpp_metrics.json")
    extended = load("time_q64_metrics.json")
    audit = load("posthoc_numerical_cardinality_audit.json")

    time_runs = extended["runs"]
    mark_runs = metrics["mark"]["runs"]
    time_q64 = [float(row["validation"]["nll_per_group"]) for row in time_runs]
    time_mae = [float(row["validation"]["mae_hours"]) for row in time_runs]
    time_medae = [float(row["validation"]["median_ae_hours"]) for row in time_runs]
    mark_ce = [float(row["validation"]["ce_per_event"]) for row in mark_runs]
    mark_f1 = [float(row["validation"]["macro_f1"]) for row in mark_runs]

    time_baselines = metrics["time"]
    history_free_nll = float(time_baselines["simple_baselines"]["history_free_lognormal_fit_on_moderate_train"]["validation"]["nll_per_group"])
    gru_time_nll = float(time_baselines["matched_baselines"]["validation"]["nll_per_group"]["mean"])
    gru_time_mae = float(time_baselines["matched_baselines"]["validation"]["mae_hours"]["mean"])
    gru_time_medae = float(time_baselines["matched_baselines"]["validation"]["median_ae_hours"]["mean"])

    mark_baselines = metrics["mark"]
    previous_ce = float(mark_baselines["simple_baselines"]["validation"]["previous_group_composition_global_Dirichlet_strength_1"]["ce_per_event"])
    last_ce = float(mark_baselines["matched_baselines"]["last_group_mlp"]["validation"]["ce_per_event"]["mean"])
    last5_ce = float(mark_baselines["matched_baselines"]["last5_gru"]["validation"]["ce_per_event"]["mean"])
    full_ce = float(mark_baselines["matched_baselines"]["full_history_gru"]["validation"]["ce_per_event"]["mean"])

    stability = audit["interaction_stability"]
    both_signs = all(
        (np.asarray(row["interaction"]["psi"]) > 0).any()
        and (np.asarray(row["interaction"]["psi"]) < 0).any()
        for row in mark_runs
    )
    signed = bool(both_signs and stability["unanimous_sign_fraction"] >= 0.5)

    # The gate asks whether Group-Time approaches or exceeds the matched
    # full-history GRU, not merely whether it beats a history-free density.
    # Require non-inferiority on both density NLL and point MAE; otherwise the
    # two baselines jointly dominate the proposed temporal kernel.
    time_beats_history_free_nll = bool(all(value < history_free_nll for value in time_q64))
    time_not_worse_than_gru = bool(fmean(time_q64) <= gru_time_nll and fmean(time_mae) <= gru_time_mae)
    time_pass = bool(all(np.isfinite(time_q64)) and time_beats_history_free_nll and time_not_worse_than_gru)
    mark_pass = bool(all(value < min(previous_ce, last_ce) for value in mark_ce))

    mark_source = audit["mark"]["seeds"]
    burst_rows = [row["source_group_size_influence"][">500"] for row in mark_source]
    ordinary_rows = [row["source_group_size_influence"]["101-500"] for row in mark_source]
    mark_burst_observed = all(row["source_occurrences"] > 0 for row in burst_rows)
    mark_burst_ratio = [
        burst["absolute_mean_per_source_occurrence"] / max(ordinary["absolute_mean_per_source_occurrence"], 1e-12)
        for burst, ordinary in zip(burst_rows, ordinary_rows)
    ]
    mark_group_size_explosion = bool(any(value > 5 for value in mark_burst_ratio))
    time_burst_observed = any(
        row["source_group_size_influence"][">500"]["source_occurrences"] > 0
        for row in audit["time"]["seeds"]
    )

    time_psi_values = [row["interaction"]["psi"] for row in time_runs]
    time_all_positive = all((np.asarray(row) > 0).all() for row in time_psi_values)
    status = {
        "group_time_snmpp_passed": time_pass,
        "group_mark_snmpp_passed": mark_pass,
        "signed_influence_learned": signed,
        "joint_group_snmpp_approved": bool(time_pass and mark_pass and signed),
        "time_training_protocol": "Q64 stratified-random train, Q64 deterministic-midpoint validation, max 30 epochs",
        "group_time_validation_Q64_nll_values": time_q64,
        "group_time_validation_Q64_nll_mean": fmean(time_q64),
        "history_free_time_validation_nll": history_free_nll,
        "matched_GRU_time_validation_nll": gru_time_nll,
        "time_beats_history_free_nll_all_seeds": time_beats_history_free_nll,
        "time_not_worse_than_GRU_on_mean_NLL_and_MAE": time_not_worse_than_gru,
        "group_mark_validation_CE_values": mark_ce,
        "group_mark_validation_CE_mean": fmean(mark_ce),
        "previous_group_composition_validation_CE": previous_ce,
        "last_group_MLP_validation_CE": last_ce,
        "last5_GRU_validation_CE": last5_ce,
        "matched_full_history_GRU_mark_validation_CE": full_ce,
        "mark_gap_vs_full_history_GRU": fmean(mark_ce) - full_ce,
        "mark_psi_unanimous_sign_fraction": float(stability["unanimous_sign_fraction"]),
        "mark_psi_pairwise_pearson": [float(row["psi_pearson"]) for row in stability["pairwise"]],
        "mark_both_positive_and_negative_all_seeds": both_signs,
        "time_psi_all_positive_all_seeds": time_all_positive,
        "mark_over_500_source_groups_observed": mark_burst_observed,
        "mark_over_500_vs_101_500_mean_abs_influence_ratio": mark_burst_ratio,
        "mark_group_size_amplitude_explosion_detected": mark_group_size_explosion,
        "time_over_500_source_groups_observed": time_burst_observed,
        "time_over_500_burst_conclusion": "not_evaluable_in_matched_303_user_subset" if not time_burst_observed else "evaluated",
        "test_used": False,
        "joint_model_started": False,
        "hierarchical_sid_head_used": False,
        "stop_reason": "Group-SNMPP Design & Minimal Validation Gate complete; both single-task gates did not jointly pass, so Joint was not started.",
    }
    metrics["time"]["exploratory_Q16"] = {
        "group_snmpp": metrics["time"].get("group_snmpp"),
        "runs": metrics["time"].get("runs"),
        "slices": metrics["time"].get("slices"),
        "integration_audit": metrics["time"].get("integration_audit"),
        "decision_use": False,
    }
    q64_summary = {
        "seeds": [row["seed"] for row in time_runs],
        "best_epochs": [row["best_epoch"] for row in time_runs],
        "stopped_epochs": [row["stopped_epoch"] for row in time_runs],
    }
    for split in ("train", "validation"):
        q64_summary[split] = {}
        for metric_name in ("nll_per_group", "mae_hours", "median_ae_hours"):
            values = [float(row[split][metric_name]) for row in time_runs]
            q64_summary[split][metric_name] = {
                "values": values, "mean": fmean(values), "std": float(np.std(values))
            }
    metrics["time"]["group_snmpp"] = q64_summary
    metrics["time"]["runs"] = time_runs
    metrics["time"]["slices"] = extended["validation_slices"]
    metrics["time"]["protocol_faithful_Q64"] = extended
    metrics["config"]["integration_Q"] = 64
    metrics["config"]["time_training_integration"] = "stratified random"
    metrics["config"]["time_validation_integration"] = "deterministic midpoint"
    metrics["posthoc_audit"] = audit
    metrics["status"] = status
    dump("group_snmpp_metrics.json", metrics)
    dump("status.json", status)

    time_psi = time_psi_values
    mark_recall = {
        name: [float(row["validation"]["per_class_recall"][name]) for row in mark_runs]
        for name in ("like", "dislike", "unlike", "undislike")
    }
    predicted = [row["validation"]["predicted_argmax_distribution"] for row in mark_runs]
    names = ["like", "dislike", "unlike", "undislike"]
    psi_mean = np.asarray(stability["psi_mean"], dtype=np.float64)
    delay_mean = np.asarray(stability["delay_mean_hours"], dtype=np.float64)
    psi_table = "\n".join(
        ["| source \\ target | " + " | ".join(names) + " |", "|---|---:|---:|---:|---:|"]
        + [f"| {name} | " + " | ".join(f"{value:+.4f}" for value in psi_mean[index]) + " |"
           for index, name in enumerate(names)]
    )
    delay_table = "\n".join(
        ["| source \\ target | " + " | ".join(names) + " |", "|---|---:|---:|---:|---:|"]
        + [f"| {name} | " + " | ".join(f"{value:.3f}" for value in delay_mean[index]) + " |"
           for index, name in enumerate(names)]
    )
    report = f"""# Group-SNMPP Design & Minimal Validation Gate

## 1. Hypothesis

以 `uid + timestamp` group 作为唯一 temporal occurrence 后，SNMPP 的 signed、delay-aware 历史核能否分别学习下一组到达时间和下一组反馈构成，并在 validation 上达到简单模型或 Full-history GRU 的水平。

## 2. Data contract

- Time matched subset：{metrics['time']['manifest']['users']:,} users，{metrics['time']['manifest']['train_targets']:,} train groups / {metrics['time']['manifest']['validation_targets']:,} validation groups。
- Mark matched subset：{metrics['mark']['manifest']['users']:,} users，{metrics['mark']['manifest']['train_targets']:,} train groups / {metrics['mark']['manifest']['validation_targets']:,} validation groups。
- group feature 为 4 类反馈各自的 mean SID semantic vector、反馈 composition、presence mask、`log(1+M)` 与 previous gap，共 522 维。
- 空反馈类型使用 zero semantic vector + zero mask；组内 permutation check 最大误差 {metrics['mark']['manifest']['permutation_check_max_error']:.3e}。
- 没有组内排序、没有 event-level temporal source、没有截断 history、没有删除 burst、没有 listen、没有使用 test。

## 3. Model

Group-Time 每个历史 group 只产生一次 signed delayed influence，并经 positive link 得到 group-arrival hazard。Group-Mark 使用 composition-weighted `source feedback r -> target feedback k` 的 4x4 signed kernel；group size 只进入有界 context gate，不以 M 倍复制 temporal source。

Time 与 Mark 分开训练，Adam、LR=1e-3、3 seeds。正式 Time 训练使用冻结的 Q=64 协议：每个 segment 随机一点；validation 使用 deterministic midpoint。最初的 Q=16 exploratory run 只作为数值诊断，不参与最终模型判定。最大 30 epochs，patience=3。

## 4. Unit tests

positive/finite hazard、finite loss/gradient、Mark/Time group-source permutation invariance 全部通过。

## 5. Time results

| Method | Validation NLL/group | MAE (hour) | Median AE (hour) |
|---|---:|---:|---:|
| History-free log-normal | {history_free_nll:.4f} | {time_baselines['simple_baselines']['history_free_lognormal_fit_on_moderate_train']['validation']['mae_hours']:.4f} | {time_baselines['simple_baselines']['history_free_lognormal_fit_on_moderate_train']['validation']['median_ae_hours']:.4f} |
| Full-history GRU | {gru_time_nll:.4f} | {gru_time_mae:.4f} | {gru_time_medae:.4f} |
| Group-Time SNMPP (3-seed mean, Q=64) | {fmean(time_q64):.4f} | {fmean(time_mae):.4f} | {fmean(time_medae):.4f} |

- Q=64 validation NLL seeds：{fmt(time_q64)}。
- Best/stopped epochs：{extended['summary']['best_epochs']} / {extended['summary']['stopped_epochs']}。
- Validation hazard P99 最大 {extended['summary']['lambda_p99_max_across_seeds']:.4f}/hour，hazard max {extended['summary']['lambda_max_across_seeds']:.4f}/hour。
- Time psi（三 seed）为 {json.dumps(time_psi)}；是否全部为正：`{str(time_all_positive).lower()}`。
- Time 结果不是数值 NaN，但是否通过由 Q=64 与 baseline 的逐 seed 比较决定。

## 6. Mark results

| Method | Validation CE/event |
|---|---:|
| Previous-group composition | {previous_ce:.4f} |
| Last-group MLP | {last_ce:.4f} |
| Last-5 GRU | {last5_ce:.4f} |
| Full-history GRU | {full_ce:.4f} |
| Group-Mark SNMPP | {fmean(mark_ce):.4f} +/- {float(np.std(mark_ce)):.4f} |

- Group-Mark CE seeds：{fmt(mark_ce)}；与 Full-history GRU 的平均差为 +{fmean(mark_ce)-full_ce:.4f}（越低越好）。
- Macro-F1：{fmt(mark_f1)}。
- Recall mean：like={fmean(mark_recall['like']):.4f}，dislike={fmean(mark_recall['dislike']):.4f}，unlike={fmean(mark_recall['unlike']):.4f}，undislike={fmean(mark_recall['undislike']):.4f}。
- argmax 预测分布（三 seed）：{json.dumps(predicted, ensure_ascii=False)}。模型没有完全退化成单一类别，但几乎不预测 undislike。

## 7. Signed influence and cardinality diagnostics

- Mark 4x4 psi 在 16 个位置中有 {int(round(16*stability['unanimous_sign_fraction']))}/16 个三 seed 同号；pairwise Pearson 为 {fmt([row['psi_pearson'] for row in stability['pairwise']], 3)}。
- 三个 seed 均同时存在正、负 psi；因此 `signed_influence_learned=true` 仅表示参数没有全正退化。
- 因 Mark predictive gate 失败，这些矩阵不能被当成已经验证的用户反馈规律，更不能作因果解释。

三 seed 平均 psi（行是 source，列是 target）：

{psi_table}

三 seed 平均 delay（hour，仅作参数诊断）：

{delay_table}

- Mark validation 历史中确有 >500 大组 source；其每 source 平均 absolute influence 相对 101-500 档的比例为 {fmt(mark_burst_ratio, 3)}，未呈现随 M 的机械线性爆炸。
- Time 的 303-user matched subset 没有 >500 source group，因此 Time 的 >500 cardinality 结论不可评估。另有清楚的 long-history accumulation：原 12-epoch checkpoint 的 history>500 slice hazard max 达约 220/hour；这是历史 group 数量累加问题，不是单个 group 内 M 条 event 重复求和。

## 8. Decision

- `group_time_snmpp_passed = {str(time_pass).lower()}`
- `group_mark_snmpp_passed = {str(mark_pass).lower()}`
- `signed_influence_learned = {str(signed).lower()}`
- `joint_group_snmpp_approved = {str(bool(time_pass and mark_pass and signed)).lower()}`

Group-Time 是否接近 GRU、是否超过 history-free baseline，以 Q=64 final values 为准。Group-Mark 虽能稳定优化并得到正负关系参数，但 CE 明显落后于所有主要 Mark baseline，故不批准 Joint。本轮按协议停止；未启动 Joint、Hierarchical SID、HPN、BOLA，也未查看 test。
"""
    (WORK / "Group_SNMPP_Design_Minimal_Validation_Report.md").write_text(report)
    print(json.dumps({"FINALIZED": True, **status}))


if __name__ == "__main__":
    main()
