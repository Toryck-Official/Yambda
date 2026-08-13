#!/usr/bin/env python3
"""Run the frozen Tiny overfit comparison with only Lambda/q factorized."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from minimal_snmpp_pilot.data import FullHistoryTargetDataset, make_loader, move_batch, collate
from minimal_snmpp_pilot.model import MinimalSNMPP
from minimal_snmpp_pilot.training_utils import (
    evaluate_feedback_and_loss, parameter_change, parameter_snapshot,
    save_json, seed_everything, train_epoch,
)
from minimal_snmpp_pilot.failure_diagnosis.factorized_model import FactorizedMinimalSNMPP

WORK = ROOT / "minimal_snmpp_pilot"
OUT = WORK / "failure_diagnosis"
RUNS = OUT / "runs"
DATA = WORK / "data/pilot_sequences.npz"
CODEBOOK = ROOT / "phase2_gate2_sid/materialized_v1_1/frozen_sid/codebooks.npy"


def compact_evaluation(model, loader, device, oracle_floor: float) -> tuple[dict, list[dict]]:
    result = evaluate_feedback_and_loss(model, loader, device)
    rows = result.pop("rows")
    result["feedback_oracle_loss_per_event"] = oracle_floor
    result["feedback_excess_loss"] = result["feedback_loss_per_event"] - oracle_floor
    confusion = np.asarray(result["confusion_matrix"], dtype=np.int64)
    predicted = confusion.sum(axis=0)
    result["predicted_argmax_class_distribution"] = (predicted / predicted.sum()).tolist()
    return result, rows


@torch.no_grad()
def output_diagnostics(model, loader, device) -> dict:
    totals=[]; entropies=[]; phi=[]; signed=[]; absolute=[]; positive=[]; negative=[]
    for batch in loader:
        batch=move_batch(batch,device)
        total,q,_,diagnostics=model.conditional_outputs(
            batch["target_time"][:,None],batch["history_times"],batch["history_feedback"],
            batch["history_sid"],batch["history_mask"],return_influence=True,
        )
        totals.extend(total[:,0].cpu().numpy().tolist())
        entropies.extend((-(q[:,0,:]*torch.log(q[:,0,:].clamp_min(1e-12))).sum(-1)).cpu().numpy().tolist())
        assert diagnostics is not None
        phi.extend(diagnostics["phi_mean"][:,0,:].cpu().numpy().reshape(-1).tolist())
        for target,key in ((signed,"signed"),(absolute,"absolute"),(positive,"positive"),(negative,"negative_magnitude")):
            target.extend(diagnostics[key][:,0,:].cpu().numpy().reshape(-1).tolist())
    def summary(values):
        array=np.asarray(values,dtype=float)
        return {"mean":float(array.mean()),"median":float(np.median(array)),"p95":float(np.percentile(array,95)),"p99":float(np.percentile(array,99)),"max":float(array.max()),"finite":bool(np.isfinite(array).all())}
    return {"Lambda":summary(totals),"q_entropy":summary(entropies),"phi":summary(phi),"psi_signed_mass":summary(signed),"psi_absolute_mass":summary(absolute),"psi_positive_mass":summary(positive),"psi_negative_mass":summary(negative)}


PARAMETER_GROUPS={
 "feedback_embedding":("feedback_embedding.",),"P_SID":("sid_projection.",),
 "psi":("interaction_network.",),"phi":("temporal_network.",),"delay":("raw_delays",),
}


def gradient_vector(model,prefixes):
    chunks=[]
    for name,p in model.named_parameters():
        if any(name==prefix or name.startswith(prefix) for prefix in prefixes):
            chunks.append((torch.zeros_like(p) if p.grad is None else p.grad).detach().reshape(-1))
    return torch.cat(chunks)


def conflict(model,batch,device):
    batch=move_batch(batch,device); vectors={}
    for objective in ("time","feedback"):
        model.zero_grad(set_to_none=True); out=model.loss(batch,deterministic_integral=True)
        (out.time_loss if objective=="time" else out.feedback_loss).backward()
        vectors[objective]={k:gradient_vector(model,v) for k,v in PARAMETER_GROUPS.items()}
        vectors[objective]["all_shared"]=torch.cat(list(vectors[objective].values()))
    result={}; negative=0
    for group in (*PARAMETER_GROUPS,"all_shared"):
        a=vectors["time"][group]; b=vectors["feedback"][group]
        na=float(torch.linalg.vector_norm(a)); nb=float(torch.linalg.vector_norm(b)); dot=float(torch.dot(a,b)); cosine=dot/(na*nb) if na and nb else float("nan")
        result[group]={"gradient_norm_time":na,"gradient_norm_feedback":nb,"cosine_similarity":cosine,"combined_gradient_norm":float(torch.linalg.vector_norm(a+b)),"finite":bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
        if group!="all_shared" and math.isfinite(cosine) and cosine<0: negative+=1
    result["negative_cosine_parameter_group_count"]=negative
    return result


def fixed_conflict_batch(dataset):
    counts=dataset.group_feedback_counts[dataset.targets]
    mask=(dataset.history_event_counts>=100)&(dataset.history_event_counts<=500)
    positions=[]
    for feedback in range(4): positions.extend(np.flatnonzero(mask&(counts[:,feedback]>0))[:2].tolist())
    return collate([dataset[i] for i in positions])


def burst_diagnostic(model,dataset,device):
    previous=dataset.group_feedback_counts[dataset.targets-1].sum(axis=1)
    positions={">500":np.flatnonzero(previous>500),"<=500_reference":np.flatnonzero(previous<=500)[:max(17,int(np.sum(previous>500)))]}
    result={}
    for label,indices in positions.items():
        values={k:[] for k in ("Lambda","time_loss","absolute_influence_mass","gradient_norm")}
        for start in range(0,len(indices),4):
            batch=move_batch(collate([dataset[int(i)] for i in indices[start:start+4]]),device)
            model.zero_grad(set_to_none=True); out=model.loss(batch,deterministic_integral=True)
            values["Lambda"].extend(out.target_total_intensity.detach().cpu().numpy().tolist())
            values["time_loss"].extend(out.time_loss_by_group.detach().cpu().numpy().tolist())
            values["absolute_influence_mass"].extend(out.absolute_influence_mass_by_target.detach().cpu().numpy().reshape(-1).tolist())
            out.optimization_loss.backward(); values["gradient_norm"].append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),float("inf"))))
        result[label]={"groups":len(indices),**{k:{"mean":float(np.mean(v)),"median":float(np.median(v)),"p99":float(np.percentile(v,99)),"max":float(np.max(v)),"finite":bool(np.isfinite(v).all())} for k,v in values.items()}}
    return result


def train_attempt(learning_rate,dataset,oracle_floor,device,manifest):
    seed_everything(2026)
    model=FactorizedMinimalSNMPP(CODEBOOK).to(device)
    model.initialize_total_rate(float(manifest["baselines"]["constant_exponential_rate_per_hour"]))
    initial_parameters=parameter_snapshot(model)
    train_loader=make_loader(dataset,batch_size=8,seed=2026,shuffle=True)
    eval_loader=make_loader(dataset,batch_size=8,seed=2026,shuffle=False)
    initial, _=compact_evaluation(model,eval_loader,device,oracle_floor)
    initial_diagnostics=output_diagnostics(model,eval_loader,device)
    conflict_batch=fixed_conflict_batch(dataset)
    initial_conflict=conflict(model,conflict_batch,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    curves=[]
    for epoch in range(1,6):
        train=train_epoch(model,train_loader,optimizer,device,10.0)
        evaluation,_=compact_evaluation(model,eval_loader,device,oracle_floor)
        curves.append({"epoch":epoch,"train":train,"evaluation":evaluation})
        print(json.dumps({"lr":learning_rate,"epoch":epoch,"train_loss":train["optimization_loss"],"eval_loss":evaluation["optimization_loss"],"time":evaluation["time_loss_per_group"],"feedback":evaluation["feedback_loss_per_event"],"macro_f1":evaluation["macro_f1"],"recall":evaluation["per_class_recall"],"Lambda_max":evaluation["total_lambda"]["max"]},ensure_ascii=False),flush=True)
        if train["nonfinite_steps"] or not math.isfinite(evaluation["optimization_loss"]): break
    final,rows=compact_evaluation(model,eval_loader,device,oracle_floor)
    final_diagnostics=output_diagnostics(model,eval_loader,device)
    final_conflict=conflict(model,conflict_batch,device)
    burst=burst_diagnostic(model,dataset,device)
    nonfinite=any(epoch["train"]["nonfinite_steps"] for epoch in curves) or not final_diagnostics["Lambda"]["finite"]
    obvious_instability=bool(nonfinite or final_diagnostics["Lambda"]["max"]>=100 or final["time_loss_per_group"]>initial["time_loss_per_group"]*2)
    recalls=list(final["per_class_recall"].values())
    signed_recovered=final_diagnostics["psi_positive_mass"]["max"]>0 and final_diagnostics["psi_negative_mass"]["max"]>0
    passed=bool(
      final["optimization_loss"]<initial["optimization_loss"] and
      final["time_loss_per_group"]<initial["time_loss_per_group"] and
      final["feedback_excess_loss"]<initial["feedback_excess_loss"] and
      all(value>0 for value in recalls) and
      max(final["predicted_argmax_class_distribution"])<0.99 and
      final_diagnostics["Lambda"]["max"]<100 and signed_recovered and not nonfinite
    )
    old=MinimalSNMPP(CODEBOOK)
    result={
      "learning_rate":learning_rate,"epochs":len(curves),"passed":passed,"obvious_numerical_instability":obvious_instability,
      "initial":initial,"final":final,"initial_diagnostics":initial_diagnostics,"final_diagnostics":final_diagnostics,
      "gradient_conflict":{"initial":initial_conflict,"final":final_conflict},"burst":burst,
      "parameter_change_l2":parameter_change(initial_parameters,model),"delay_matrix_hours":model.delays().detach().cpu().tolist(),
      "gradient_norms":{"mean_by_epoch":[c["train"]["gradient_norm_mean"] for c in curves],"max_by_epoch":[c["train"]["gradient_norm_max"] for c in curves],"clip_frequency_by_epoch":[c["train"]["gradient_clip_frequency"] for c in curves]},
      "NaN_or_Inf":nonfinite,"curves":curves,
      "parameter_count":{"old_trainable":sum(p.numel() for p in old.parameters()),"new_trainable":sum(p.numel() for p in model.parameters()),"new_head_parameters":model.head_parameter_count,"net_added":sum(p.numel() for p in model.parameters())-sum(p.numel() for p in old.parameters())},
      "pass_criteria":{"total_loss_down":final["optimization_loss"]<initial["optimization_loss"],"time_loss_down":final["time_loss_per_group"]<initial["time_loss_per_group"],"feedback_excess_down":final["feedback_excess_loss"]<initial["feedback_excess_loss"],"all_four_recalls_nonzero":all(v>0 for v in recalls),"no_99pct_single_class":max(final["predicted_argmax_class_distribution"])<0.99,"Lambda_below_100":final_diagnostics["Lambda"]["max"]<100,"positive_and_negative_influence":signed_recovered,"finite":not nonfinite},
    }
    run_dir=RUNS/f"lr_{learning_rate:.0e}"; run_dir.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"learning_rate":learning_rate,"seed":2026,"factorized":True},run_dir/"checkpoint_final.pt")
    save_json(run_dir/"training_curve.json",curves); save_json(run_dir/"metrics.json",result)
    return result


def main():
    dataset=FullHistoryTargetDataset(DATA,"tiny_train_targets")
    if len(dataset)!=6413 or int(dataset.group_feedback_counts[dataset.targets].sum())!=38363: raise RuntimeError("frozen Tiny contract mismatch")
    oracle=json.loads((OUT/"feedback_oracle_floor.json").read_text()); floor=float(oracle["all_groups"]["oracle_feedback_ce_per_event"])
    manifest=json.loads((WORK/"data/subset_manifest.json").read_text()); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attempts=[]
    for lr in (1e-3,1e-4,1e-5):
        result=train_attempt(lr,dataset,floor,device,manifest); attempts.append(result)
        if result["passed"] or not result["obvious_numerical_instability"]: break
    chosen=next((r for r in attempts if r["passed"]),attempts[-1])
    final={"frozen_contract":{"target_groups":6413,"target_events":38363,"seed":2026,"same_subset_sha256":manifest["artifacts"]["npz_sha256"],"batch_size":8,"optimizer":"Adam unchanged","gradient_clip_norm":10.0,"gradient_clipping_used_as_repair":False,"Q":64,"time_unit":"hour","horizon_hours":696,"history_truncated":False,"burst_normalized":False},"attempt_order":[r["learning_rate"] for r in attempts],"attempts":attempts,"selected_attempt_learning_rate":chosen["learning_rate"],"factorized_snmpp_tiny_passed":chosen["passed"]}
    save_json(OUT/"factorized_tiny_metrics.json",final)
    print(json.dumps({"FACTORIZED_TINY_COMPLETE":True,"attempts":final["attempt_order"],"selected_lr":chosen["learning_rate"],"passed":chosen["passed"]},ensure_ascii=False),flush=True)


if __name__=="__main__": main()
