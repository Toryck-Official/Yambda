#!/usr/bin/env python3
"""Post-hoc four-feedback gradient visibility audit on the final Stage-A checkpoint."""

from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from minimal_snmpp_pilot.data import FullHistoryTargetDataset, collate, move_batch
from minimal_snmpp_pilot.model import MinimalSNMPP

work=ROOT/"minimal_snmpp_pilot"; device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset=FullHistoryTargetDataset(work/"data/pilot_sequences.npz","tiny_train_targets")
model=MinimalSNMPP(ROOT/"phase2_gate2_sid/materialized_v1_1/frozen_sid/codebooks.npy").to(device)
saved=torch.load(work/"runs/stage_a_seed2026/checkpoint_final.pt",map_location=device,weights_only=False); model.load_state_dict(saved["model"]); model.train()
rows={}
for feedback in range(4):
    local=np.flatnonzero(dataset.group_feedback_counts[dataset.targets,feedback]>0)
    samples=[dataset[int(i)] for i in local[:min(8,len(local))]]
    batch=move_batch(collate(samples),device); model.zero_grad(set_to_none=True)
    output=model.loss(batch,deterministic_integral=True); output.optimization_loss.backward()
    rows[str(feedback)]={
      "samples":len(samples),"loss":float(output.optimization_loss),
      "baseline_logit_gradient":float(model.baseline_logits.grad[feedback]),
      "feedback_embedding_gradient_norm":float(torch.linalg.vector_norm(model.feedback_embedding.weight.grad[feedback])),
      "target_delay_column_gradient_norm":float(torch.linalg.vector_norm(model.raw_delays.grad[:,feedback])),
      "all_finite":bool(torch.isfinite(model.baseline_logits.grad[feedback]) and torch.isfinite(model.feedback_embedding.weight.grad[feedback]).all() and torch.isfinite(model.raw_delays.grad[:,feedback]).all()),
    }
path=work/"runs/stage_a_seed2026/four_feedback_gradient_audit.json"; path.write_text(json.dumps(rows,indent=2)+"\n")
metrics=json.loads((work/"stage_a_result.json").read_text()); metrics["four_feedback_gradient_visibility"]=rows; metrics["all_four_feedback_nonzero_finite_gradient"]=all(r["all_finite"] and abs(r["baseline_logit_gradient"])>0 and r["feedback_embedding_gradient_norm"]>0 for r in rows.values()); (work/"stage_a_result.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(rows,indent=2))
