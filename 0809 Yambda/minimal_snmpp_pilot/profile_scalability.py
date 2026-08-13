#!/usr/bin/env python3
"""Full-history Q=64 latency/memory profiling without any history approximation."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimal_snmpp_pilot.data import FullHistoryTargetDataset, collate, move_batch
from minimal_snmpp_pilot.model import MinimalSNMPP
from minimal_snmpp_pilot.training_utils import save_json


BINS = (("<=50", 0, 50), ("51-100", 51, 100), ("101-500", 101, 500),
        ("501-1000", 501, 1000), ("1001-5000", 1001, 5000), ("5000+", 5001, 10**12))


def checkpoint_path(work: Path) -> Path:
    stage_b = work / "runs/stage_b_seed2026/checkpoint_best.pt"
    if stage_b.exists(): return stage_b
    stage_a = work / "runs/stage_a_seed2026/checkpoint_final.pt"
    if stage_a.exists(): return stage_a
    raise FileNotFoundError("no trained Minimal SNMPP checkpoint")


def main():
    work = ROOT / "minimal_snmpp_pilot"; device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=MinimalSNMPP(ROOT/"phase2_gate2_sid/materialized_v1_1/frozen_sid/codebooks.npy").to(device)
    saved=torch.load(checkpoint_path(work),map_location=device,weights_only=False); model.load_state_dict(saved["model"]); model.train()
    pilot=FullHistoryTargetDataset(work/"data/pilot_sequences.npz","pilot_validation_targets")
    tiny=FullHistoryTargetDataset(work/"data/pilot_sequences.npz","tiny_train_targets")
    records=[]
    for label,low,high in BINS:
        candidates=[]
        for dataset_name,dataset in (("pilot_validation",pilot),("tiny_train",tiny)):
            positions=np.flatnonzero((dataset.history_event_counts>=low)&(dataset.history_event_counts<=high))
            for index in positions[:8]: candidates.append((dataset_name,dataset,int(index)))
            if len(candidates)>=8: break
        for dataset_name,dataset,index in candidates[:8]:
            batch=move_batch(collate([dataset[index]]),device)
            # Warm-up uses the exact same Q=64 full-history objective.
            model.zero_grad(set_to_none=True); warm=model.loss(batch,deterministic_integral=False); warm.optimization_loss.backward(); torch.cuda.synchronize() if device.type=="cuda" else None
            forward=[]; backward=[]; target_only=[]; peaks=[]
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                if device.type=="cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                start=time.perf_counter(); output=model.loss(batch,deterministic_integral=False)
                if device.type=="cuda": torch.cuda.synchronize()
                middle=time.perf_counter(); output.optimization_loss.backward()
                if device.type=="cuda": torch.cuda.synchronize()
                end=time.perf_counter(); forward.append(middle-start); backward.append(end-middle)
                peaks.append(torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else float("nan"))
                with torch.no_grad():
                    start=time.perf_counter(); model.conditional_intensity(batch["target_time"][:,None],batch["history_times"],batch["history_feedback"],batch["history_sid"],batch["history_mask"])
                    if device.type=="cuda": torch.cuda.synchronize()
                    target_only.append(time.perf_counter()-start)
            h=int(batch["history_event_count"][0]); target_events=int(batch["target_group_size"][0]); f=float(np.median(forward)); b=float(np.median(backward))
            records.append({"bin":label,"source":dataset_name,"history_events":h,"target_events":target_events,"forward_seconds":f,"backward_seconds":b,"total_seconds":f+b,"groups_per_second":1/(f+b),"history_events_per_second":h/(f+b),"peak_gpu_memory_mb":float(max(peaks)),"target_intensity_only_seconds":float(np.median(target_only)),"q64_integration_increment_seconds":max(0.0,f-float(np.median(target_only)))})
    summaries={}
    for label,_,_ in BINS:
        rows=[r for r in records if r["bin"]==label]
        if rows:
            summaries[label]={"samples":len(rows),**{key:float(np.median([r[key] for r in rows])) for key in ("history_events","forward_seconds","backward_seconds","total_seconds","groups_per_second","history_events_per_second","peak_gpu_memory_mb","q64_integration_increment_seconds")}}
    x=np.log(np.asarray([r["history_events"] for r in records],dtype=float)); y=np.log(np.asarray([r["total_seconds"] for r in records],dtype=float))
    slope=float(np.polyfit(x,y,1)[0]) if len(records)>2 else float("nan")
    result={"protocol":{"integration_Q":64,"full_history":True,"history_truncation":False,"history_sampling":False,"batch_size":1,"repeats":3},"checkpoint":str(checkpoint_path(work)),"records":records,"by_history_bin":summaries,"log_runtime_vs_log_history_slope":slope,"interpretation":"Per-target work sums all previous events (approximately linear in H); evaluating every target in a user sequence therefore accumulates toward quadratic sequence cost.","full_D_SID_feasibility":None}
    save_json(work/"scalability_profile.json",result); print(json.dumps({"PROFILE_COMPLETE":True,"slope":slope,"bins":summaries},ensure_ascii=False),flush=True)


if __name__=="__main__": main()
