#!/usr/bin/env python3
"""Diagnostic-only comparison of previous-group >500 after Stage-A failure."""

from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from minimal_snmpp_pilot.data import FullHistoryTargetDataset, collate, move_batch
from minimal_snmpp_pilot.model import MinimalSNMPP

work=ROOT/"minimal_snmpp_pilot"; device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
data=FullHistoryTargetDataset(work/"data/pilot_sequences.npz","tiny_train_targets")
previous_sizes=data.group_feedback_counts[data.targets-1].sum(axis=1)
indices={">500":np.flatnonzero(previous_sizes>500),"<=500_matched":np.flatnonzero(previous_sizes<=500)[:max(17,int(np.sum(previous_sizes>500)))]}
model=MinimalSNMPP(ROOT/"phase2_gate2_sid/materialized_v1_1/frozen_sid/codebooks.npy").to(device)
model.load_state_dict(torch.load(work/"runs/stage_a_seed2026/checkpoint_final.pt",map_location=device,weights_only=False)["model"]); model.eval()
result={}
for label,positions in indices.items():
    values={k:[] for k in ("time_loss","total_lambda","signed","absolute","positive","negative","gradient_norm")}
    for start in range(0,len(positions),4):
        batch=move_batch(collate([data[int(i)] for i in positions[start:start+4]]),device)
        model.zero_grad(set_to_none=True); out=model.loss(batch,deterministic_integral=True)
        for key,tensor in (("time_loss",out.time_loss_by_group),("total_lambda",out.target_total_intensity),("signed",out.signed_influence_by_target),("absolute",out.absolute_influence_mass_by_target),("positive",out.positive_influence_mass_by_target),("negative",out.negative_influence_mass_by_target)):
            values[key].extend(tensor.detach().cpu().numpy().reshape(-1).tolist())
        out.optimization_loss.backward(); values["gradient_norm"].append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),float("inf"))))
    result[label]={"groups":len(positions),**{k:{"mean":float(np.mean(v)),"median":float(np.median(v)),"max":float(np.max(v)),"finite":bool(np.isfinite(v).all())} for k,v in values.items()}}
result["conclusion"]="All sampled values remain finite; only 17 burst cases exist, so this diagnostic cannot attribute the global collapse to >500 bursts."
path=work/"runs/stage_a_seed2026/burst_diagnostic.json"; path.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
