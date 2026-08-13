#!/usr/bin/env python3
"""Create the final Gate 1C decision artifacts without changing the experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "phase1_gate1c" / "artifacts" / "gate1c_metrics.json"
DALL = ROOT / "phase1_canonical" / "artifacts" / "dall_manifest.json"
GRAPH = ROOT / "phase1_gate1c" / "artifacts" / "train_four_relation_graph"
RUNS = ROOT / "phase1_gate1c" / "artifacts" / "collaborative_runs"
OUTPUT_JSON = ROOT / "phase1_gate1c" / "artifacts" / "gate1c_conclusions.json"
OUTPUT_TEXT = ROOT / "phase1_gate1c" / "gate1c_report.txt"
SEEDS = (2026, 2027, 2028)
STRATA = (
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-9", 5, 9),
    ("10-19", 10, 19),
    ("20-49", 20, 49),
    ("50+", 50, None),
)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def metric_view(row: dict) -> dict:
    return {
        "count": row["count"],
        "cosine": row["cosine"],
        "mse": row["mse"],
        "prefix_at_1": row["prefix_accuracy"]["prefix_at_1"],
        "prefix_at_2": row["prefix_accuracy"]["prefix_at_2"],
        "prefix_at_3": row["prefix_accuracy"]["prefix_at_3"],
        "prefix_at_4": row["prefix_accuracy"]["prefix_at_4"],
        "exact_semantic_sid_at_4": row["exact_semantic_sid_at_4"],
        "nn_at_10": row.get("nearest_neighbor", {}).get("nn_at_10"),
        "nn_at_50": row.get("nearest_neighbor", {}).get("nn_at_50"),
        "nn_query_count": row.get("nearest_neighbor", {}).get("query_count"),
    }


def mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std_across_seeds": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "seed_values": [float(value) for value in array],
    }


def aggregate_seed_rows(rows: list[dict]) -> dict:
    fields = (
        "cosine", "mse", "prefix_at_1", "prefix_at_2", "prefix_at_3",
        "prefix_at_4", "exact_semantic_sid_at_4", "nn_at_10", "nn_at_50",
    )
    views = [metric_view(row) for row in rows]
    return {
        "count": views[0]["count"],
        "nn_query_count": views[0]["nn_query_count"],
        **{field: mean_std([view[field] for view in views]) for field in fields},
    }


def cross_revision_user_audit() -> dict:
    """Compare revision-heavy and other items within each activity stratum.

    This audit deliberately uses only cosine and SID metrics. The fixed NN query
    sample was designed for the requested activity strata, not every cross-cell.
    """
    config = json.loads((ROOT / "phase1_gate1c" / "configs" / "experiment.json").read_text())
    graph_item = np.load(GRAPH / "explicit_item_id.uint32.npy", mmap_mode="r")
    lookup = np.full(int(graph_item.max()) + 1, -1, dtype=np.int32)
    lookup[graph_item] = np.arange(len(graph_item), dtype=np.int32)
    users_all = np.load(GRAPH / "unique_train_user_count.uint32.npy", mmap_mode="r")
    revision_all = np.load(GRAPH / "revision_event_share.float32.npy", mmap_mode="r")
    with np.load(Path(config["paths"]["gate1b_sample"]), allow_pickle=False) as sample_file:
        sample_item = sample_file["item_id"]
        split = sample_file["split"]
    position = lookup[sample_item]
    test_index = np.flatnonzero((split == 2) & (users_all[position] > 0))
    test_position = position[test_index]
    users = np.asarray(users_all[test_position])
    revision_heavy = np.asarray(revision_all[test_position]) >= 0.5
    truth = np.load(Path(config["paths"]["gate1b_arrays"]) / "truth.float32.npy", mmap_mode="r")[test_index]
    true_codes = np.load(Path(config["paths"]["gate1b_arrays"]) / "true_codes.uint16.npy", mmap_mode="r")[test_index]
    codebooks = np.load(Path(config["paths"]["codebooks"]))
    from sys import path as sys_path
    legacy = Path("/root/autodl-tmp/0804 Yambda/dataprocess")
    sys_path.insert(0, str(legacy))
    from rqdataprocess.rq import encode_with_codebooks

    seed_rows: list[dict] = []
    for seed in SEEDS:
        run = RUNS / f"seed_{seed}"
        prediction = np.load(run / "collaborative_audio_test_prediction.float32.npy")
        prediction /= np.maximum(np.linalg.norm(prediction, axis=1, keepdims=True), 1e-12)
        codes = encode_with_codebooks(np.ascontiguousarray(prediction, dtype=np.float32), codebooks)
        prefix = np.logical_and.accumulate(codes == true_codes, axis=1)
        cosine = np.sum(truth * prediction, axis=1)
        mse = np.mean(np.square(truth - prediction), axis=1)
        per_seed = {"seed": seed, "strata": {}}
        for name, minimum, maximum in STRATA:
            activity = users >= minimum
            if maximum is not None:
                activity &= users <= maximum
            per_seed["strata"][name] = {}
            for revision_name, revision_mask in (
                ("revision_share_lt_0.5", ~revision_heavy),
                ("revision_share_ge_0.5", revision_heavy),
            ):
                mask = activity & revision_mask
                per_seed["strata"][name][revision_name] = {
                    "count": int(mask.sum()),
                    "cosine": float(cosine[mask].mean()),
                    "mse": float(mse[mask].mean()),
                    **{f"prefix_at_{level + 1}": float(prefix[mask, level].mean()) for level in range(4)},
                }
        seed_rows.append(per_seed)
    return {
        "definition": "revision-heavy means train-period (unlike+undislike) event share >= 0.5; diagnostic only",
        "nearest_neighbor_note": "NN is omitted in cross-cells because the fixed 10k query was stratified by activity, not by every activity-revision cross-cell.",
        "runs": seed_rows,
    }


def main() -> None:
    metrics = json.loads(METRICS.read_text())
    dall = json.loads(DALL.read_text())
    centroid = metric_view(metrics["metadata_centroid"]["overall"])
    metadata = metrics["metadata_mlp_runs"]
    collaborative = metrics["collaborative_audio_runs"]
    user_curve = {}
    for name, _, _ in STRATA:
        user_curve[name] = {
            "metadata_mlp": aggregate_seed_rows([row["unique_user_strata"][name] for row in metadata]),
            "collaborative_audio": aggregate_seed_rows([row["unique_user_strata"][name] for row in collaborative]),
        }
    revision_curve = {}
    revision_names = collaborative[0]["revision_share_strata"].keys()
    for name in revision_names:
        revision_curve[name] = aggregate_seed_rows([row["revision_share_strata"][name] for row in collaborative])
    dominant_curve = {}
    for name in collaborative[0]["dominant_feedback"].keys():
        dominant_curve[name] = aggregate_seed_rows([row["dominant_feedback"][name] for row in collaborative])

    missing = metrics["missing_item_unique_train_user_coverage"]
    evidence_missing = sum(missing[name] for name, _, _ in STRATA)
    if evidence_missing + missing["cold_0"] != 507_730:
        raise RuntimeError("missing-item coverage does not conserve the 507,730 item universe")
    primary = ("prefix_at_1", "prefix_at_2", "nn_at_10", "nn_at_50")
    strictly_worse = all(
        metrics["paired_bootstrap"]["collaborative_vs_metadata_mlp"][field]["bootstrap_95_ci"][1] < 0
        for field in primary
    )
    conclusion = {
        "status": "gate1c_complete_stop_before_gate2",
        "phase1_d_all": {
            "raw_explicit_events": dall["dedup"]["events_before"],
            "exact_duplicate_surplus_removed": dall["dedup"]["exact_duplicate_surplus_removed"],
            "events_after_dedup": dall["dedup"]["events_after"],
            "global_split": dall["global_split"],
            "split_stats": dall["split_stats"],
        },
        "gate1c_data_contract": metrics["data_contract"],
        "overall": {
            "metadata_centroid": centroid,
            "metadata_mlp": metrics["three_seed_summary"]["metadata_mlp"],
            "collaborative_audio": metrics["three_seed_summary"]["collaborative_audio"],
            "paired_bootstrap": metrics["paired_bootstrap"],
        },
        "unique_train_user_curve": user_curve,
        "revision_share_curve": revision_curve,
        "dominant_feedback_curve": dominant_curve,
        "revision_activity_cross_audit": cross_revision_user_audit(),
        "missing_item_decision": {
            "missing_items_total": 507_730,
            "reliable_semantic_recovery": 0,
            "weak_collaborative_evidence_not_approved_for_final_sid": evidence_missing,
            "true_train_cold_start": missing["cold_0"],
            "counts_by_unique_train_users": {name: missing[name] for name, _, _ in STRATA},
            "decision_basis": "No activity stratum beats metadata-only on Prefix@1/2 and NN@10/50; therefore no usable collaborative threshold is established.",
        },
        "answers": {
            "collaborative_beats_metadata_only": False,
            "primary_metrics_strictly_worse_than_metadata_mlp": strictly_worse,
            "quality_rises_with_user_count": "partly: cosine and Prefix@1 rise, but Prefix@2 and NN remain very low and do not reach metadata-only",
            "usable_user_threshold": None,
            "revision_heavy_harder": "not a clean monotonic effect; pure/high-revision and dislike/undislike-dominant groups are often weak, but activity is a major confounder",
            "proceed_to_gate2": False,
        },
        "limitations": [
            "The 50k held-out evaluation set contains real-audio items with strict metadata leave-one-out context, enabling paired baseline comparison but not direct ground-truth evaluation on truly missing-audio items.",
            "The graph baseline is a deliberately simple 64-d four-relation DistMult matrix factorization, not a conclusion that every possible collaborative model must fail.",
            "Random contrast items are not interpreted as observed user rejection.",
            "The frozen RQKMeans is the same real-only candidate codebook used by Gate 1A/1B, not a Gate 2 final codebook.",
        ],
    }
    atomic_json(OUTPUT_JSON, conclusion)

    def pct(value: float) -> str:
        return f"{100 * value:.3f}%"

    lines = [
        "Gate 1C 正式结果（STOP before Gate 2）",
        "",
        f"Phase 1：{dall['dedup']['events_before']:,} 条显式事件删除 {dall['dedup']['exact_duplicate_surplus_removed']:,} 条完全重复副本，D_all={dall['dedup']['events_after']:,}。",
        "",
        "总体语义恢复（三种子均值）",
        "方法 | Cosine | MSE | Prefix@1 | Prefix@2 | Prefix@3 | Prefix@4 | NN@10 | NN@50",
        f"Metadata centroid | {centroid['cosine']:.4f} | {centroid['mse']:.6f} | {pct(centroid['prefix_at_1'])} | {pct(centroid['prefix_at_2'])} | {pct(centroid['prefix_at_3'])} | {pct(centroid['prefix_at_4'])} | {pct(centroid['nn_at_10'])} | {pct(centroid['nn_at_50'])}",
        "Metadata MLP | " + " | ".join(
            f"{metrics['three_seed_summary']['metadata_mlp'][field]['mean']:.4f}" if field == "cosine" else
            f"{metrics['three_seed_summary']['metadata_mlp'][field]['mean']:.6f}" if field == "mse" else
            pct(metrics['three_seed_summary']['metadata_mlp'][field]['mean'])
            for field in ("cosine", "mse", "prefix_at_1", "prefix_at_2", "prefix_at_3", "prefix_at_4", "nn_at_10", "nn_at_50")
        ),
        "Collaborative -> audio | " + " | ".join(
            f"{metrics['three_seed_summary']['collaborative_audio'][field]['mean']:.4f}" if field == "cosine" else
            f"{metrics['three_seed_summary']['collaborative_audio'][field]['mean']:.6f}" if field == "mse" else
            pct(metrics['three_seed_summary']['collaborative_audio'][field]['mean'])
            for field in ("cosine", "mse", "prefix_at_1", "prefix_at_2", "prefix_at_3", "prefix_at_4", "nn_at_10", "nn_at_50")
        ),
        "",
        "按 train unique-user count 的 Collaborative -> audio 曲线",
        "users | test n | Cosine | Prefix@1 | Prefix@2 | NN@10 | NN@50",
    ]
    for name, _, _ in STRATA:
        row = user_curve[name]["collaborative_audio"]
        lines.append(
            f"{name} | {row['count']:,} | {row['cosine']['mean']:.4f} | {pct(row['prefix_at_1']['mean'])} | "
            f"{pct(row['prefix_at_2']['mean'])} | {pct(row['nn_at_10']['mean'])} | {pct(row['nn_at_50']['mean'])}"
        )
    lines += [
        "",
        "决策",
        "1. Collaborative -> audio 没有超过 metadata-only；所有四个主指标均显著更差。",
        "2. Cosine 和 Prefix@1 随用户数总体提高，但 Prefix@2 与 NN 指标仍很低，不能据此认定可用。",
        "3. 本实验没有找到可用的 unique-user threshold。",
        "4. Revision-heavy 不是干净的单调难度因素；高 revision/pure revision 与 dislike/undislike 主导组通常较弱，但受到活跃度混杂。",
        f"5. 507,730 个 missing item：可靠补全 0；有 train 协同证据但仅能标为弱证据 {evidence_missing:,}；真正 train cold-start {missing['cold_0']:,}。",
        "",
        "严格停止：未进入 Gate 2，未分配最终 SID，未训练 SNMPP/HPN。",
    ]
    OUTPUT_TEXT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"output_json": str(OUTPUT_JSON), "output_text": str(OUTPUT_TEXT), "answers": conclusion["answers"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
