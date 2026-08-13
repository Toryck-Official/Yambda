#!/usr/bin/env python3
"""Create concise human and machine-readable Gate 1B final reports."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "phase1_gate1b"
ART = ROOT / "artifacts"


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def pct(value: float) -> str:
    return f"{100 * value:.3f}%"


def main() -> None:
    stability = json.loads((ART / "rq_perturbation_stability.json").read_text())
    metadata = json.loads((ART / "metadata_information_audit.json").read_text())
    metrics = json.loads((ART / "gate1b_metrics.json").read_text())
    base = metrics["baselines"]["conditional_centroid"]["overall"]
    learned = metrics["three_seed_summary"]["learned_embedding"]
    direct = metrics["three_seed_summary"]["direct_sid"]
    comparisons = metrics["paired_bootstrap_comparisons"]
    support = json.loads(
        Path(
            "/root/autodl-tmp/0809 Yambda/phase1_gates/artifacts/metadata_support_sample_report.json"
        ).read_text()
    )
    learned_machine = {
        "status": "complete",
        "method": "two-layer MLP metadata-context to normalized 128-d audio embedding",
        "input": metrics["data_contract"]["model_input"],
        "training_items": metrics["data_contract"]["train_validation_test"][0],
        "validation_items": metrics["data_contract"]["train_validation_test"][1],
        "test_items": metrics["data_contract"]["train_validation_test"][2],
        "seeds": metrics["data_contract"]["seeds"],
        "baselines": metrics["baselines"],
        "runs": metrics["learned_embedding_runs"],
        "three_seed_summary": learned,
        "paired_vs_conditional_centroid": comparisons[
            "learned_embedding_vs_conditional_centroid"
        ],
        "missing_item_potential_application_coverage": support[
            "missing_item_coverage"
        ],
    }
    direct_machine = {
        "status": "complete",
        "method": "four-head hierarchical SID classifier with strict autoregressive test inference",
        "input": metrics["data_contract"]["model_input"],
        "test_truth_prefix_used": False,
        "runs": metrics["direct_sid_runs"],
        "three_seed_summary": direct,
        "paired_vs_learned_embedding": comparisons["direct_sid_vs_learned_embedding"],
        "paired_vs_conditional_centroid": comparisons[
            "direct_sid_vs_conditional_centroid"
        ],
    }
    conclusions = {
        "status": "gate1b_complete_stop_before_gate2",
        "classification": "C",
        "scope": "current pinned Yambda artist/album-only metadata context and the frozen Gate-1 candidate RQKMeans; not a universal claim that metadata can never recover SID",
        "answers": {
            "1_rqkmeans_sensitive": {
                "answer": True,
                "summary": "Even random unit-vector perturbations at cosine 0.85 retain only 2.843% exact four-level SID, although level-1 prefix remains 85.607%. Greedy residual levels amplify earlier boundary changes.",
            },
            "2_learned_embedding_significantly_better": {
                "answer": False,
                "summary": "The MLP improves cosine by 0.002824 and Prefix@1 by 0.423 percentage points with bootstrap CIs excluding zero, but significantly worsens Prefix@2/3/4 by 0.339/0.620/0.585 percentage points and worsens NN@10. It is not a task-level improvement over the centroid.",
            },
            "3_direct_sid_further_better": {
                "answer": False,
                "summary": "Direct SID is statistically indistinguishable from the centroid at Prefix@1/2 and significantly worse at Prefix@3/4. Against learned embedding, only Prefix@2 improves slightly; Prefix@3/4 are indistinguishable.",
            },
        },
        "decision": {
            "gate2_started": False,
            "missing_items_assigned_final_sid": False,
            "snmpp_or_hpn_trained": False,
            "next_action_requires_user_confirmation": True,
        },
    }
    write_json(ART / "learned_embedding_metrics.json", learned_machine)
    write_json(ART / "direct_sid_metrics.json", direct_machine)
    write_json(ART / "gate1b_conclusions.json", conclusions)

    c85 = stability["curve"]["target_0.85"]
    miss = metadata["missing_items"]
    base_nn = metrics["baselines"]["conditional_centroid"][
        "nearest_neighbor_consistency"
    ]
    lines = [
        "# Gate 1B：SID 可恢复性补充验证",
        "",
        "结论分类：**C（在当前 artist/album-only 输入下，学习模型仍不足以可靠恢复多层 SID）**。这不是“任何 metadata 都无法恢复 SID”的普遍结论。Gate 2 未启动。",
        "",
        "## 固定协议",
        "",
        "- 100,000 个真实 embedding 用于 RQ 扰动校准。",
        "- 学习任务使用 300,000 / 50,000 / 50,000 个严格 LOO train/validation/test item。",
        "- 三个 seed；同一 real-only 冻结 candidate RQKMeans；测试集沿用 Gate 1A 固定 50k。",
        "- 模型输入只有 artist/album LOO centroid、可用性和上下文规模；item ID 与显式交互频率不进入模型。",
        "- Direct SID 测试严格自回归，API 不接受真实 prefix。",
        "",
        "## A. RQKMeans 稳定性",
        "",
        "| 实际 cosine | Prefix@1 | Prefix@2 | Prefix@3 | Prefix@4 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for value in [0.99, 0.95, 0.90, 0.85, 0.80, 0.70]:
        row = stability["curve"][f"target_{value:.2f}"]
        lines.append(
            f"| {row['actual_cosine']['mean']:.2f} | "
            + " | ".join(
                pct(row["prefix_accuracy"][f"prefix_at_{level}"])
                for level in range(1, 5)
            )
            + " |"
        )
    lines += [
        "",
        f"cosine≈0.85 时 Prefix@4 只剩 {pct(c85['prefix_accuracy']['prefix_at_4'])}，说明残差量化本身高度敏感；但 Prefix@1 仍有 {pct(c85['prefix_accuracy']['prefix_at_1'])}。Gate 1A 在相近 cosine 下 Prefix@1 只有 {pct(base['prefix_accuracy']['prefix_at_1'])}，所以 centroid 还存在方向结构偏差，不能把全部损失归因于 RQ 边界。低 margin 样本的 token flip 显著更多。",
        "",
        "## B. Metadata 信息能力",
        "",
        f"本地固定版本真实存在的 item 静态关系只有 artist 与 album；没有 track title、genre、release 字段。{metadata['real_items_strict_leave_one_out_context']['count']:,} / 2,367,341 个 real item 可构造严格 LOO context。",
        "",
        f"507,730 个 missing item 只有 {miss['same_context_collision']['unique_contexts']:,} 种 artist/album context；{miss['same_context_collision']['items_in_nonunique_context']:,} 个 item 的可用 context 与其他 item 完全相同，最大桶 {miss['same_context_collision']['maximum_context_collision_bucket']:,}。因此仅靠这些输入无法逐首恢复完整音乐语义。",
        "",
        "## C/D. 固定测试集结果",
        "",
        "| 方法 | Cosine | MSE | Prefix@1 | Prefix@2 | Prefix@3 | Prefix@4 | NN@10 | NN@50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Conditional centroid | {base['cosine']['mean']:.6f} | {base['mse']['mean']:.6f} | {pct(base['prefix_accuracy']['prefix_at_1'])} | {pct(base['prefix_accuracy']['prefix_at_2'])} | {pct(base['prefix_accuracy']['prefix_at_3'])} | {pct(base['prefix_accuracy']['prefix_at_4'])} | {pct(base_nn['nn_overlap_at_10_mean'])} | {pct(base_nn['nn_overlap_at_50_mean'])} |",
        f"| Learned embedding MLP（3-seed mean） | {learned['cosine']['mean']:.6f} | {learned['mse']['mean']:.6f} | {pct(learned['prefix_at_1']['mean'])} | {pct(learned['prefix_at_2']['mean'])} | {pct(learned['prefix_at_3']['mean'])} | {pct(learned['prefix_at_4']['mean'])} | {pct(learned['nn_at_10']['mean'])} | {pct(learned['nn_at_50']['mean'])} |",
        f"| Direct SID（3-seed mean） | — | — | {pct(direct['prefix_at_1']['mean'])} | {pct(direct['prefix_at_2']['mean'])} | {pct(direct['prefix_at_3']['mean'])} | {pct(direct['prefix_at_4']['mean'])} | — | — |",
        "",
        "MLP 的 cosine 与 Prefix@1 有很小但统计可检出的提升；任务优先的 Prefix@2/3/4 反而显著下降。Direct SID 没有进一步改善完整层级路径。后层独立 TokenAcc 较高不等于路径可用：只要前缀错了，后层单 token 猜对也不能恢复真实 SID。",
        "",
        "未继续上 DeepSets：当前 Gate 只要求先验证简单 learned predictor；更重要的是，同一 artist/album context 的 item 会收到相同的允许输入，DeepSets 不能凭空补回缺失的 track-specific 信息。是否引入新的 item-specific metadata/embedding 来源属于后续新方案，需要另行确认。",
        "",
        "## 最终三个回答",
        "",
        "1. **RQKMeans 对小扰动高度敏感：是。** 特别是残差后层和低 codeword margin 样本。",
        "2. **Learned embedding 明显优于 centroid：否。** 只有 cosine/第一层小幅改善，核心多层 SID 与 NN 指标没有整体改善。",
        "3. **Direct SID 进一步优于 embedding→RQKMeans：否。** Prefix@3/4 无显著提升，并且仍明显低于 centroid。",
        "",
        "**STOP：Gate 2 未启动；没有给 missing item 分配最终 SID；没有训练 SNMPP/HPN。**",
    ]
    report_path = ROOT / "reports" / "Gate1B_SID可恢复性报告.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
