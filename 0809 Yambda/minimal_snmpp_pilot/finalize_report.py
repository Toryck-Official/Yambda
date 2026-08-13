#!/usr/bin/env python3
"""Materialize the requested report/status after the authorized stopping point."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "minimal_snmpp_pilot"


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True)+"\n")


def mean_std(values):
    a=np.asarray(values,dtype=float); return {"mean":float(a.mean()),"std":float(a.std(ddof=1)) if len(a)>1 else 0.0,"values":a.tolist()}


def main():
    config=read(WORK/"pilot_config.json"); manifest=read(WORK/"data/subset_manifest.json"); stage_a=read(WORK/"stage_a_result.json")
    stage_b_path=WORK/"stage_b_results.json"; scale_path=WORK/"scalability_profile.json"
    stage_b=read(stage_b_path) if stage_b_path.exists() else []
    scale=read(scale_path) if scale_path.exists() and stage_b else {"status":"not_run_because_stage_a_failed_or_stage_b_incomplete","protocol":{"integration_Q":64,"full_history":True}}
    aggregate={}
    feedback_pass=time_pass=no_collapse=False
    if stage_b:
        aggregate={
          "feedback_macro_f1":mean_std([r["metrics"]["macro_f1"] for r in stage_b]),
          "feedback_loss_per_event":mean_std([r["metrics"]["feedback_loss_per_event"] for r in stage_b]),
          "time_nll_per_group":mean_std([r["metrics"]["time_loss_per_group"] for r in stage_b]),
          "time_mae_hours":mean_std([r["metrics"]["time_prediction"]["time_mae_hours"] for r in stage_b]),
          "time_median_ae_hours":mean_std([r["metrics"]["time_prediction"]["time_median_ae_hours"] for r in stage_b]),
        }
        feedback_pass=sum(r["metrics"]["feedback_loss_per_event"] < r["metrics"]["baselines"]["global_feedback_frequency"]["feedback_loss_per_event"] and r["metrics"]["macro_f1"] > r["metrics"]["baselines"]["global_feedback_frequency"]["macro_f1"] for r in stage_b)>=2
        time_pass=sum(r["metrics"]["time_loss_per_group"] < r["metrics"]["baselines"]["global_median_gap_and_constant_exponential"]["time_nll_per_group"] and r["metrics"]["time_prediction"]["time_mae_hours"] < r["metrics"]["baselines"]["global_median_gap_and_constant_exponential"]["time_mae_hours"] for r in stage_b)>=2
        no_collapse=all(sum(v>0 for v in r["metrics"]["per_class_recall"].values())>=2 for r in stage_b)
    pilot_pass=bool(stage_a["passed"] and len(stage_b)==3 and feedback_pass and time_pass and no_collapse)
    scale_feasible=False
    if stage_b and "by_history_bin" in scale:
        rates=[v["groups_per_second"] for v in scale["by_history_bin"].values() if v["groups_per_second"]>0]
        if rates:
            conservative=min(rates); estimated_epoch_hours=77565283/conservative/3600
            scale["conservative_full_epoch_estimate_hours_single_gpu"]=estimated_epoch_hours
            scale_feasible=estimated_epoch_hours<=24
            write(scale_path,scale)
    status={
      "protocol_version":"Phase_2_v1.1","minimal_snmpp_training_started":True,
      "stage_a_tiny_overfit_passed":bool(stage_a["passed"]),
      "stage_b_pilot_completed":len(stage_b)==3,
      "baseline_comparison_completed":len(stage_b)==3,
      "scalability_profile_completed":bool(stage_b and "by_history_bin" in scale),
      "minimal_snmpp_pilot_passed":pilot_pass,
      "hierarchical_sid_head_approved":pilot_pass,
      "full_scale_snmpp_approved":bool(pilot_pass and scale_feasible),
      "forbidden_branches_started":{"hierarchical_sid":False,"residual":False,"hpn":False,"bola":False,"full_scale":False},
      "stop_reason":None if pilot_pass else ("Tiny Overfit Gate failed; Stage B and scalability were not authorized." if not stage_a["passed"] else "Pilot did not jointly beat matched feedback and time baselines across seeds."),
    }
    metrics={"data_manifest":manifest,"configuration":config,"stage_a":stage_a,"stage_b":stage_b,"three_seed_aggregate":aggregate,"decisions":status}
    write(WORK/"pilot_metrics.json",metrics); write(WORK/"status.json",status)
    if not stage_b: write(WORK/"scalability_profile.json",scale)
    commit="not_a_standalone_git_repository"
    try: commit=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: pass
    data=manifest["counts"]; final=stage_a["final"]
    burst_path=WORK/"runs/stage_a_seed2026/burst_diagnostic.json"
    burst=read(burst_path) if burst_path.exists() else None
    lines=[
      "# Minimal SNMPP Pilot Report","",
      "## 1. Hypothesis","",
      "在固定 Phase 2 v1.1 协议下，先验证全历史 signed temporal SNMPP 能否在 Tiny 同集上学习非平凡的时间与四类反馈信号；只有通过后才允许 Pilot 泛化、基线比较与扩展性分析。","",
      "## 2. Data Contract","",
      "- D_SID；仅 like/dislike/unlike/undislike；不使用 listen 或 missing-audio item。",
      "- 同 timestamp 组共享严格前序历史；整组评分后再更新历史。",
      "- 4x256 frozen audio-only RQKMeans codebook；SID 只进入历史表示，无 SID 输出头。",
      "- hour 单位、696-hour horizon、Q=64；train stratified-random，evaluation midpoint。","",
      "## 3. Data Counts","",
      f"- D_SID: 121,819,651 events / 854,649 users / 2,367,341 items / 77,565,283 groups。",
      f"- Tiny: {data['tiny_train_targets']:,} target groups，{stage_a['data']['target_events']:,} target events。",
      f"- Pilot manifest: {data['pilot_train_targets']:,} train targets / {data['pilot_validation_targets']:,} validation targets；test 未使用。","",
      "## 4. Model Input / Output","",
      "输入为完整前序事件的 feedback embedding、冻结 SID codeword 向量和连续时间；输出仅为下一 timestamp-group 时间分布和组内 feedback 分布。","",
      "## 5. Training Objective","",
      "normalized optimization objective = mean_group(time pseudo-likelihood loss) + mean_event(feedback cross-entropy)。它不是原始严格 NLL。","",
      "## 6. Tiny Overfit Results","",
      f"- PASS: {stage_a['passed']}",
      f"- Loss: {stage_a['initial']['optimization_loss']:.6f} -> {final['optimization_loss']:.6f} (relative drop {stage_a['relative_loss_drop']*100:.2f}%)",
      f"- Feedback Accuracy / Macro-F1: {final['accuracy']:.6f} / {final['macro_f1']:.6f}",
      "- Per-class recall: "+", ".join(f"{k}={v:.4f}" for k,v in final["per_class_recall"].items()),
      f"- Q64 stochastic time-loss CV: {stage_a['integration_noise_final']['cv']:.3e}","",
      "### 数值与退化诊断","",
      f"- Feedback loss/event: {stage_a['initial']['feedback_loss_per_event']:.6f} -> {final['feedback_loss_per_event']:.6f}（略降，但主要来自多数类 unlike）。",
      f"- Time loss/group: {stage_a['initial']['time_loss_per_group']:.6f} -> {final['time_loss_per_group']:.6f}（恶化）。",
      f"- total intensity 最大值: {stage_a['initial']['total_lambda']['max']:.6f} -> {final['total_lambda']['max']:.6f} / hour。",
      "- 四类目标的 baseline、feedback embedding 与 delay 均有非零有限梯度；失败不是断图。",
      "- psi、phi、delay 均偏离初始化，但 signed mass 退化成全正累积，并未形成可信的 excitation/inhibition 结构。",
      "- 已保留 lr=1e-3、1e-4 的摆动日志；最终 lr=1e-5 仍失败。","",
      "### Previous-group >500 诊断","",
      (f"17 个 burst 目标的 time loss/group 均值 {burst['>500']['time_loss']['mean']:.3f}、total intensity 均值 {burst['>500']['total_lambda']['mean']:.3f}、gradient norm 中位数 {burst['>500']['gradient_norm']['median']:.1f}；对照分别为 {burst['<=500_matched']['time_loss']['mean']:.3f}、{burst['<=500_matched']['total_lambda']['mean']:.3f}、{burst['<=500_matched']['gradient_norm']['median']:.1f}。均保持 finite，但 burst 明显放大 raw sum；样本仅 17 组，不能把全局失败只归因于 burst。" if burst else "未生成 burst 诊断。"),"",
      "## 7. Pilot / Baseline / Scalability","",
      ("完成 3 seeds；详细结果见 pilot_metrics.json 与 scalability_profile.json。" if stage_b else "未执行：Tiny Overfit Gate 未通过时必须 STOP。"),"",
      "## 8. Failure / Anomaly","",
      status["stop_reason"] or "未发现阻止进入下一阶段的 Gate 失败。","",
      "## 9. Conclusion","",
      f"minimal_snmpp_pilot_passed = {str(status['minimal_snmpp_pilot_passed']).lower()}",
      f"hierarchical_sid_head_approved = {str(status['hierarchical_sid_head_approved']).lower()}",
      f"full_scale_snmpp_approved = {str(status['full_scale_snmpp_approved']).lower()}","",
      "## 10. Gate Questions","",
      "1. Tiny overfit：失败。",
      "2. 非平凡 signal：没有形成；参数虽更新，但输出退化。",
      "3. Pilot validation vs feedback baseline：未执行，Stage A 失败后无授权。",
      "4. Time NLL/MAE vs baseline：未执行泛化比较；Tiny time loss 已恶化。",
      "5. Collapse：发生 unlike collapse，不是 like collapse，但同样属于单类退化。",
      "6. psi/phi/delay：均有更新，但方向退化，不能视为有效学习。",
      "7. >500 burst：保持 finite，但强度、损失与梯度显著放大。",
      "8. Q64 noise：可接受，CV 约 5e-7，不是本次失败原因。",
      "9. Full-history 扩展性：未进入正式 profiling；当前 raw sum 已暴露数值风险，full scale 不批准。",
      "10. Hierarchical SID：不批准。","",
      "## 11. Reproducibility","",
      f"- code revision: {commit}",
      f"- subset SHA256: {manifest['artifacts']['npz_sha256']}",
      "- 已保存 config、checkpoint、曲线、所有被拒绝数值尝试日志。",""
    ]
    (WORK/"Minimal_SNMPP_Pilot_Report.md").write_text("\n".join(lines))
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
