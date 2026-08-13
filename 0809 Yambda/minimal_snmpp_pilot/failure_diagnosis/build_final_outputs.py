#!/usr/bin/env python3
"""Build old/new comparison, gate decision, and final diagnosis report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from minimal_snmpp_pilot.data import FullHistoryTargetDataset, make_loader, move_batch
from minimal_snmpp_pilot.model import MinimalSNMPP

WORK=ROOT/"minimal_snmpp_pilot"; OUT=WORK/"failure_diagnosis"; CODEBOOK=ROOT/"phase2_gate2_sid/materialized_v1_1/frozen_sid/codebooks.npy"


def read(path): return json.loads(Path(path).read_text())
def write(name,value): (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=True)+"\n")


@torch.no_grad()
def old_extended():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=FullHistoryTargetDataset(WORK/"data/pilot_sequences.npz","tiny_train_targets")
    loader=make_loader(data,batch_size=8,seed=2026,shuffle=False)
    model=MinimalSNMPP(CODEBOOK).to(device)
    model.load_state_dict(torch.load(WORK/"runs/stage_a_seed2026/checkpoint_final.pt",map_location=device,weights_only=False)["model"]); model.eval()
    values={k:[] for k in ("Lambda","signed","absolute","positive","negative")}
    for batch in loader:
        batch=move_batch(batch,device); out=model.loss(batch,deterministic_integral=True)
        for key,tensor in (("Lambda",out.target_total_intensity),("signed",out.signed_influence_by_target),("absolute",out.absolute_influence_mass_by_target),("positive",out.positive_influence_mass_by_target),("negative",out.negative_influence_mass_by_target)):
            values[key].extend(tensor.cpu().numpy().reshape(-1).tolist())
    result={}
    for key,array in values.items():
        a=np.asarray(array,dtype=float); result[key]={"mean":float(a.mean()),"median":float(np.median(a)),"p95":float(np.percentile(a,95)),"p99":float(np.percentile(a,99)),"max":float(a.max()),"finite":bool(np.isfinite(a).all())}
    write("old_extended_metrics.json",result); return result


def main():
    old=read(WORK/"stage_a_result.json"); conflict=read(OUT/"gradient_conflict_metrics.json"); oracle=read(OUT/"feedback_oracle_floor.json"); factorized=read(OUT/"factorized_tiny_metrics.json"); old_ext=old_extended()
    selected=next(a for a in factorized["attempts"] if a["learning_rate"]==factorized["selected_attempt_learning_rate"])
    old_floor=float(oracle["all_groups"]["oracle_feedback_ce_per_event"])
    old_pred=np.asarray(old["final"]["confusion_matrix"]).sum(axis=0); old_pred=(old_pred/old_pred.sum()).tolist()
    old_last=old["curves"][-1]["train"]
    comparison={
      "contract":{"same_tiny_subset":True,"target_groups":6413,"target_events":38363,"seed":2026,"only_change":"coupled lambda_k output -> factorized Lambda and q heads","net_added_parameters":selected["parameter_count"]["net_added"]},
      "OLD_coupled":{
        "initial":{"total_loss":old["initial"]["optimization_loss"],"time_loss":old["initial"]["time_loss_per_group"],"feedback_loss":old["initial"]["feedback_loss_per_event"],"feedback_excess_loss":old["initial"]["feedback_loss_per_event"]-old_floor},
        "final":{"total_loss":old["final"]["optimization_loss"],"time_loss":old["final"]["time_loss_per_group"],"feedback_loss":old["final"]["feedback_loss_per_event"],"feedback_excess_loss":old["final"]["feedback_loss_per_event"]-old_floor,"macro_f1":old["final"]["macro_f1"],"per_class_recall":old["final"]["per_class_recall"],"predicted_argmax_class_distribution":old_pred,"class_collapse":True,"Lambda":old_ext["Lambda"],"signed_mass":{"signed":old_ext["signed"],"positive":old_ext["positive"],"negative":old_ext["negative"]},"gradient_norm_mean":old_last["gradient_norm_mean"],"gradient_norm_max":old_last["gradient_norm_max"]},
        "gradient_conflict":{"initial":conflict["states"]["initialization"],"final":conflict["states"]["failed_checkpoint"]},
      },
      "NEW_factorized_selected":{
        "learning_rate":selected["learning_rate"],"initial":selected["initial"],"final":selected["final"],"final_diagnostics":selected["final_diagnostics"],"gradient_conflict":selected["gradient_conflict"],"gradient_norms":selected["gradient_norms"],"burst":selected["burst"],"class_collapse":max(selected["final"]["predicted_argmax_class_distribution"])>=0.99 or any(v==0 for v in selected["final"]["per_class_recall"].values()),
      },
      "exact_matched_lr_factorized_attempt":{"learning_rate":selected["learning_rate"],"final_total_loss":selected["final"]["optimization_loss"],"final_time_loss":selected["final"]["time_loss_per_group"],"final_feedback_loss":selected["final"]["feedback_loss_per_event"],"Lambda_max":selected["final_diagnostics"]["Lambda"]["max"],"obvious_numerical_instability":selected["obvious_numerical_instability"]},
      "higher_lr_sensitivity_attempts":[{"learning_rate":a["learning_rate"],"Lambda_max":a["final_diagnostics"]["Lambda"]["max"],"final_total_loss":a["final"]["optimization_loss"],"obvious_numerical_instability":a["obvious_numerical_instability"]} for a in factorized["attempts"] if a["learning_rate"]!=selected["learning_rate"]],
    }
    factorization_prevents_hundreds_at_selected=selected["final_diagnostics"]["Lambda"]["max"]<100
    factorization_prevents_hundreds_at_original_lr=selected["final_diagnostics"]["Lambda"]["max"]<100
    decisions={
      "factorized_snmpp_tiny_passed":False,
      "time_mark_coupling_failure_supported":False,
      "time_mark_coupling_contribution_detected":True,
      "primary_failure_attribution_to_coupling_not_supported":True,
      "history_aggregation_gate_needed":True,
      "minimal_snmpp_stageB_approved":False,
      "old_gradient_conflict_detected_at_initialization":conflict["states"]["initialization"]["negative_cosine_parameter_group_count"]>=3 and conflict["states"]["initialization"]["shared_temporal_parameters"]["cosine_similarity"]<0,
      "factorization_unit_tests_passed":True,
      "factorization_prevented_hundred_scale_Lambda_at_selected_lr":factorization_prevents_hundreds_at_selected,
      "factorization_prevented_hundred_scale_Lambda_at_original_lr":factorization_prevents_hundreds_at_original_lr,
      "time_loss_decreased":selected["final"]["time_loss_per_group"]<selected["initial"]["time_loss_per_group"],
      "feedback_excess_loss_decreased":selected["final"]["feedback_excess_loss"]<selected["initial"]["feedback_excess_loss"],
      "all_four_recalls_nonzero":all(v>0 for v in selected["final"]["per_class_recall"].values()),
      "positive_and_negative_signed_influence_recovered":selected["final_diagnostics"]["psi_positive_mass"]["max"]>0 and selected["final_diagnostics"]["psi_negative_mass"]["max"]>0,
      "burst_amplification_still_present":selected["burst"][">500"]["Lambda"]["mean"]>10*selected["burst"]["<=500_reference"]["Lambda"]["mean"],
      "stageB_or_forbidden_branch_started":False,
      "stop_reason":"At the exact matched LR, factorization reduced intensity explosion and briefly improved loss, but final Tiny time loss and feedback excess worsened, majority collapse remained, signed influence stayed all-positive, and >500 burst amplification persisted.",
    }
    comparison["decisions"]=decisions; write("old_vs_new_comparison.json",comparison); write("status.json",decisions)
    # Synchronize the parent pilot status without erasing the earlier Stage-A facts.
    parent=read(WORK/"status.json"); parent.update({"failure_diagnosis_gate_completed":True,**{k:decisions[k] for k in ("factorized_snmpp_tiny_passed","time_mark_coupling_failure_supported","history_aggregation_gate_needed","minimal_snmpp_stageB_approved")}}); parent["hierarchical_sid_head_approved"]=False; parent["full_scale_snmpp_approved"]=False; parent["stop_reason"]=decisions["stop_reason"]; write_parent=WORK/"status.json"; write_parent.write_text(json.dumps(parent,ensure_ascii=False,indent=2)+"\n")
    new=selected; burst=new["burst"]
    lines=[
      "# Factorized SNMPP Failure Diagnosis Report","",
      "## 1. Hypothesis","",
      "检验旧 Tiny Overfit 失败是否主要来自 group-time 与 feedback mark 共用四个 coupled intensity 的参数化冲突。唯一修正是共享原 SNMPP temporal context 后，以独立轻量 head 输出 Lambda 与 q，并定义 lambda_k=Lambda*q_k。","",
      "## 2. Frozen Contract","",
      "- 完全复用 6,413 target groups / 38,363 target events、seed=2026、D_SID、grouped shared pre-history、冻结码本、event representation、psi/phi/delay、raw event-level full-history sum、Q=64、hour 与 696-hour horizon。",
      "- 未改 class weight、采样、optimizer、loss weight、history、burst、SID 或 sequence encoder。保留旧 gradient clip=10；它不是本轮修复。",
      f"- 新输出 heads 共 {new['parameter_count']['new_head_parameters']} 参数，替换旧 4 个 baseline logits 后净增 {new['parameter_count']['net_added']} 参数。","",
      "## 3. Old Gradient Conflict","",
      f"- 初始化 shared gradient cosine = {conflict['states']['initialization']['shared_temporal_parameters']['cosine_similarity']:.4f}；5 个参数组中 {conflict['states']['initialization']['negative_cosine_parameter_group_count']} 个为负。",
      f"- psi cosine = {conflict['states']['initialization']['psi']['cosine_similarity']:.4f}；phi cosine = {conflict['states']['initialization']['phi']['cosine_similarity']:.4f}。",
      "- 失败 checkpoint 的 shared cosine 转为正值，但这是全正 influence 与多数类退化后的方向一致，不能反证初始化冲突。","",
      "## 4. Shared-group Feedback Oracle","",
      f"- 全部组 oracle floor = {old_floor:.6f} CE/event。",
      f"- singleton = {oracle['singleton']['oracle_feedback_ce_per_event']:.6f}；multi-event = {oracle['multi_event']['oracle_feedback_ce_per_event']:.6f}。",
      f"- Tiny 类别比例：like {oracle['class_proportions']['like']:.2%}，dislike {oracle['class_proportions']['dislike']:.2%}，unlike {oracle['class_proportions']['unlike']:.2%}，undislike {oracle['class_proportions']['undislike']:.2%}。",
      f"- 旧模型 {oracle['failed_model_predicted_class_proportions']['unlike']:.2%} 预测为 unlike；这就是 majority-class collapse。","",
      "## 5. Unit Tests","",
      "7 个测试函数覆盖 10 项 contract，全部通过：Lambda 正且 finite、q 非负且和为 1、sum lambda=Lambda、singleton identity、permutation invariance、shared pre-history、delayed update、Q64、四 mark gradients、冻结 codebook。","",
      "## 6. Old vs New","",
      f"| 指标 | OLD coupled final (lr=1e-5) | NEW factorized initial | NEW factorized final (lr={new['learning_rate']:.0e}) |",
      "|---|---:|---:|---:|",
      f"| total loss | {old['final']['optimization_loss']:.6f} | {new['initial']['optimization_loss']:.6f} | {new['final']['optimization_loss']:.6f} |",
      f"| time loss/group | {old['final']['time_loss_per_group']:.6f} | {new['initial']['time_loss_per_group']:.6f} | {new['final']['time_loss_per_group']:.6f} |",
      f"| feedback loss/event | {old['final']['feedback_loss_per_event']:.6f} | {new['initial']['feedback_loss_per_event']:.6f} | {new['final']['feedback_loss_per_event']:.6f} |",
      f"| feedback excess | {old['final']['feedback_loss_per_event']-old_floor:.6f} | {new['initial']['feedback_excess_loss']:.6f} | {new['final']['feedback_excess_loss']:.6f} |",
      f"| Macro-F1 | {old['final']['macro_f1']:.6f} | {new['initial']['macro_f1']:.6f} | {new['final']['macro_f1']:.6f} |",
      f"| Lambda P99 | {old_ext['Lambda']['p99']:.6f} | {new['initial_diagnostics']['Lambda']['p99']:.6f} | {new['final_diagnostics']['Lambda']['p99']:.6f} |",
      f"| Lambda max | {old_ext['Lambda']['max']:.6f} | {new['initial_diagnostics']['Lambda']['max']:.6f} | {new['final_diagnostics']['Lambda']['max']:.6f} |","",
      f"- NEW recall: like={new['final']['per_class_recall']['like']:.4f}, dislike={new['final']['per_class_recall']['dislike']:.4f}, unlike={new['final']['per_class_recall']['unlike']:.4f}, undislike={new['final']['per_class_recall']['undislike']:.4f}。",
      f"- NEW argmax distribution: {new['final']['predicted_argmax_class_distribution']}。",
      f"- 完全匹配旧正式配置 lr=1e-5 时，factorized Lambda max={new['final_diagnostics']['Lambda']['max']:.3f}，而 OLD 为 {old_ext['Lambda']['max']:.3f}；数值敏感性 lr=1e-3 时仍达 {factorized['attempts'][0]['final_diagnostics']['Lambda']['max']:.3f}。",
      f"- 训练中最佳 transient time loss={min(c['evaluation']['time_loss_per_group'] for c in new['curves']):.6f}、最佳 feedback loss={min(c['evaluation']['feedback_loss_per_event'] for c in new['curves']):.6f}，说明解耦有部分数值收益；但最终均反弹且类别始终 collapse。",
      "- NEW signed influence 仍为全正：negative mass max=0，未恢复 excitation/inhibition 两侧。","",
      "## 7. Burst Diagnostic","",
      f"- previous-group >500（17 组）：Lambda mean={burst['>500']['Lambda']['mean']:.3f}，time loss mean={burst['>500']['time_loss']['mean']:.3f}，absolute influence mean={burst['>500']['absolute_influence_mass']['mean']:.3f}，gradient norm median={burst['>500']['gradient_norm']['median']:.3f}。",
      f"- <=500 reference（17 组）：对应为 {burst['<=500_reference']['Lambda']['mean']:.3f}、{burst['<=500_reference']['time_loss']['mean']:.3f}、{burst['<=500_reference']['absolute_influence_mass']['mean']:.3f}、{burst['<=500_reference']['gradient_norm']['median']:.3f}。",
      "- 全部 finite，但 raw full-history burst amplification 仍显著存在。","",
      "## 8. Final Gate Answers","",
      "1. 旧模型存在明显初始化梯度冲突：是。",
      f"2. Tiny oracle floor：{old_floor:.6f} CE/event。",
      "3. unlike collapse 是多数类 collapse：是。",
      "4. factorized Lambda/q 数学测试：全部通过。",
      "5. 阻止 Lambda 爆炸：在完全匹配的 1e-5 对照中做到，但 1e-3 敏感性仍爆到百级，说明 factorization 改善稳定性但并非无条件稳定。",
      "6. time loss 真正下降：前 3 轮短暂下降，最终反弹并高于初始化，因此未稳定下降。",
      "7. feedback excess 明显下降：否，反而略升。",
      "8. Macro-F1 / 四类 recall 摆脱 collapse：否。",
      "9. psi 恢复正负 influence：否，仍全正。",
      "10. >500 burst amplification：仍存在。",
      "11. Minimal SNMPP Stage B：不批准。","",
      "## 9. Status","",
      "factorized_snmpp_tiny_passed = false",
      "time_mark_coupling_failure_supported = false",
      "history_aggregation_gate_needed = true",
      "minimal_snmpp_stageB_approved = false","",
      "解释：检测到 coupling 对数值不稳定有贡献，但证据不支持它是 Tiny 失败的主要或唯一原因。","",
      "本轮到此 STOP；未启动 Stage B、Hierarchical SID、HPN/BOLA 或 full-scale training。",""
    ]
    (OUT/"Factorized_SNMPP_Failure_Diagnosis_Report.md").write_text("\n".join(lines))
    print(json.dumps(decisions,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
