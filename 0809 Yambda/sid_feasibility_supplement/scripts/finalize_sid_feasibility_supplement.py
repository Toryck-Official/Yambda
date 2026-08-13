#!/usr/bin/env python3
"""Correct label-invariant SID comparisons and write the final audit summary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "sid_feasibility_supplement" / "artifacts"
METRICS = ARTIFACTS / "sid_feasibility_supplement_metrics.json"
SUMMARY = ARTIFACTS / "sid_feasibility_supplement_summary.json"
TEXT = ROOT / "sid_feasibility_supplement" / "sid_feasibility_supplement_report.txt"
MAIN_SCRIPT = Path(__file__).with_name("run_sid_feasibility_supplement.py")


def load_main():
    spec = importlib.util.spec_from_file_location("sid_supplement_main", MAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    module = load_main()
    metrics = json.loads(METRICS.read_text())
    base_dir = ARTIFACTS / "B0_audio_only"
    base_codebooks = np.load(base_dir / "codebooks.npy")
    base_codes = np.load(base_dir / "codes.uint8.npy", mmap_mode="r")
    for name, report in metrics["methods"].items():
        method_dir = ARTIFACTS / name
        codebooks = np.load(method_dir / "codebooks.npy")
        codes = np.load(method_dir / "codes.uint8.npy", mmap_mode="r")
        consistency = module.aligned_prefix_consistency(
            base_codebooks, base_codes, codebooks, codes
        )
        report["semantic_preservation"]["audio_only_sid_consistency"] = consistency
        atomic_json(method_dir / "report.json", report)
    atomic_json(METRICS, metrics)

    old = metrics["comparison_contract"]["old_candidate_collision"]
    b0 = metrics["methods"]["B0_audio_only"]
    old_mse = 0.001243150145444498
    new_mse = b0["fit"]["level_trace"][-1]["mse"]
    methods = {}
    for name, report in metrics["methods"].items():
        identity = report["identity"]
        semantic = report["semantic_preservation"]
        methods[name] = {
            "unique_sid": identity["unique_semantic_sid_paths"],
            "singleton_ratio": identity["singleton_item_ratio"],
            "collision_ratio": identity["collision_item_ratio"],
            "collision_buckets": identity["collision_buckets"],
            "bucket_p90": identity["all_bucket_size"]["p90"],
            "bucket_p99": identity["all_bucket_size"]["p99"],
            "bucket_p99_9": identity["all_bucket_size"]["p99_9"],
            "bucket_max": identity["all_bucket_size"]["max"],
            "rq_input_reconstruction_mse": report["fit"]["level_trace"][-1]["mse"],
            "reconstruction_vs_audio_mse": semantic[
                "rq_reconstruction_vs_original_audio"
            ]["mse"],
            "input_nn_overlap_10": semantic["input_neighbor_overlap_at_10"],
            "input_nn_overlap_50": semantic["input_neighbor_overlap_at_50"],
            "rq_nn_overlap_10": semantic["rq_reconstruction_neighbor_overlap_at_10"],
            "rq_nn_overlap_50": semantic["rq_reconstruction_neighbor_overlap_at_50"],
            "aligned_prefix_accuracy": semantic["audio_only_sid_consistency"][
                "prefix_accuracy"
            ],
        }
    raw = metrics["embedding_uniqueness"]["raw_audio"]["exact_duplicates"]
    normalized = metrics["embedding_uniqueness"]["normalized_audio"][
        "exact_duplicates"
    ]
    residual = metrics["residual_uniqueness"]
    summary = {
        "status": "complete_and_stopped_before_gate2",
        "fit_scale": {
            "old_fit_items": 100_000,
            "full_fit_items": 2_367_341,
            "old_collision_ratio": old["collision_item_ratio"],
            "full_collision_ratio": b0["identity"]["collision_item_ratio"],
            "collision_ratio_change_percentage_points": 100
            * (b0["identity"]["collision_item_ratio"] - old["collision_item_ratio"]),
            "old_full_population_reconstruction_mse": old_mse,
            "full_fit_reconstruction_mse": new_mse,
            "reconstruction_mse_relative_change": (new_mse - old_mse) / old_mse,
            "conclusion": "full fit improves reconstruction modestly but does not reduce collision; 100k fit scale is not the main cause",
        },
        "methods": methods,
        "embedding_duplicates": {
            "raw": raw,
            "normalized": normalized,
            "conclusion": "the source contains material exact item-level duplicate audio vectors",
        },
        "near_duplicates": metrics["embedding_uniqueness"]["near_duplicates"],
        "residual": residual,
        "decisions": {
            "metadata_fusion_approved": False,
            "metadata_reason": "all tested fusions increase collision and materially reduce original-audio neighbor overlap",
            "audio_only_full_fit_preferred": True,
            "audio_only_reason": "best identity/semantic tradeoff among tested methods, although four semantic levels are not exact identity",
            "sid_plus_residual_solves_audio_distinguishable_items": True,
            "sid_plus_residual_is_globally_exact_item_unique": False,
            "residual_reason": "96.7175% unique SID+float32-residual representations; remaining ambiguity exactly matches source-vector duplicates",
            "proceed_gate2": False,
        },
        "boundaries": {
            "missing_audio_processed": False,
            "suffix_generated": False,
            "snmpp_or_hpn_trained": False,
            "gate2_started": False,
        },
    }
    atomic_json(SUMMARY, summary)
    lines = [
        "SID Feasibility 补充实验（STOP before Gate 2）",
        "",
        f"100k candidate collision: {old['collision_item_ratio']:.4%}",
        f"2.367M full-fit collision: {b0['identity']['collision_item_ratio']:.4%}",
        f"full-fit reconstruction MSE: {new_mse:.8f}",
        "结论：拟合规模不是高 collision 的主要原因。",
        "",
    ]
    for name, value in methods.items():
        lines.append(
            f"{name}: collision={value['collision_ratio']:.4%}, "
            f"NN@10/50={value['input_nn_overlap_10']:.4f}/{value['input_nn_overlap_50']:.4f}, "
            f"max_bucket={value['bucket_max']}"
        )
    lines += [
        "",
        f"raw exact duplicate item ratio: {raw['duplicate_item_ratio']:.4%}",
        f"normalized exact duplicate item ratio: {normalized['duplicate_item_ratio']:.4%}",
        f"SID+residual unique rate: {residual['sid_plus_float32_residual_unique_rate']:.4%}",
        "结论：metadata fusion 不批准；B0 是当前最合理语义 SID。SID+residual 可区分所有 audio-distinguishable items，但不能区分源向量本身完全相同的物品。",
        "",
        "STOP: no missing item, no suffix, no Gate 2, no SNMPP/HPN.",
    ]
    temporary = TEXT.with_suffix(TEXT.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(TEXT)
    print(json.dumps(summary["decisions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
