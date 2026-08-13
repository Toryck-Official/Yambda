#!/usr/bin/env python3
"""Add the exact LR=1e-5 factorized run matching the frozen failed Stage A."""

from __future__ import annotations
import json, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from minimal_snmpp_pilot.data import FullHistoryTargetDataset
from minimal_snmpp_pilot.failure_diagnosis.run_factorized_tiny import train_attempt, WORK, OUT, DATA
from minimal_snmpp_pilot.training_utils import save_json

factorized_path=OUT/"factorized_tiny_metrics.json"
existing=json.loads(factorized_path.read_text())
if any(float(a["learning_rate"])==1e-5 for a in existing["attempts"]):
    print("exact matched LR already present",flush=True); raise SystemExit(0)
dataset=FullHistoryTargetDataset(DATA,"tiny_train_targets")
oracle=json.loads((OUT/"feedback_oracle_floor.json").read_text())
manifest=json.loads((WORK/"data/subset_manifest.json").read_text())
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
result=train_attempt(1e-5,dataset,float(oracle["all_groups"]["oracle_feedback_ce_per_event"]),device,manifest)
existing["attempts"].append(result); existing["attempt_order"].append(1e-5)
existing["selected_attempt_learning_rate"]=1e-5
existing["selection_reason"]="exactly matches the frozen failed Stage-A final learning rate; higher-LR attempts are sensitivity diagnostics only"
existing["factorized_snmpp_tiny_passed"]=result["passed"]
save_json(factorized_path,existing)
print(json.dumps({"EXACT_MATCH_COMPLETE":True,"lr":1e-5,"passed":result["passed"]},ensure_ascii=False),flush=True)
