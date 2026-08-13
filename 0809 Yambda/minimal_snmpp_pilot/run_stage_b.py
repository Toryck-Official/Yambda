#!/usr/bin/env python3
"""Three-seed Pilot generalization, matched baselines, and validation slices."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimal_snmpp_pilot.data import FullHistoryTargetDataset, make_loader
from minimal_snmpp_pilot.model import MinimalSNMPP
from minimal_snmpp_pilot.training_utils import (
    FEEDBACK_NAMES,
    evaluate_feedback_and_loss,
    initialize_constant_rate,
    macro_metrics,
    parameter_change,
    parameter_snapshot,
    save_json,
    seed_everything,
    train_epoch,
)


def feedback_baseline(rows, probabilities):
    prediction = int(np.argmax(probabilities))
    confusion = np.zeros((4, 4), dtype=np.int64)
    loss = events = 0.0
    for row in rows:
        counts = np.asarray(row["counts"], dtype=np.int64)
        confusion[:, prediction] += counts
        loss -= float(np.dot(counts, np.log(probabilities)))
        events += counts.sum()
    return {"feedback_loss_per_event": loss / events, **macro_metrics(confusion)}


def time_baseline(rows, median_gap_hours, rate):
    actual = np.asarray([row["actual_gap_hours"] for row in rows])
    nll = -np.log(rate) + rate * actual
    error = np.abs(actual - median_gap_hours)
    return {"time_nll_per_group": float(np.mean(nll)), "time_mae_hours": float(np.mean(error)), "time_median_ae_hours": float(np.median(error))}


def neural_time(rows):
    actual = np.asarray([row["actual_gap_hours"] for row in rows])
    predicted = np.asarray([row["predicted_gap_hours"] for row in rows])
    error = np.abs(actual - predicted)
    return {"time_mae_hours": float(np.mean(error)), "time_median_ae_hours": float(np.median(error)), "horizon_event_mass_mean": float(np.mean([row["horizon_event_mass"] for row in rows]))}


def slice_rows(rows, key, bins):
    output = {}
    for label, low, high in bins:
        selected = [row for row in rows if low <= row[key] <= high]
        if not selected:
            continue
        confusion = np.zeros((4, 4), dtype=np.int64)
        feedback_sum = events = 0
        for row in selected:
            counts = np.asarray(row["counts"], dtype=np.int64)
            confusion[:, row["prediction"]] += counts
            feedback_sum += row["feedback_loss_sum"]
            events += row["target_events"]
        output[label] = {
            "groups": len(selected), "events": events,
            "time_loss_per_group": float(np.mean([r["time_loss"] for r in selected])),
            "feedback_loss_per_event": feedback_sum / events,
            **macro_metrics(confusion), **neural_time(selected),
        }
    return output


def evaluate_and_compare(model, loader, device, manifest):
    evaluation = evaluate_feedback_and_loss(model, loader, device, compute_time_predictions=True)
    rows = evaluation.pop("rows")
    probabilities = np.asarray(manifest["baselines"]["global_train_feedback_probabilities"])
    median_gap = float(manifest["baselines"]["global_train_gap_median_seconds"]) / 3600.0
    rate = float(manifest["baselines"]["constant_exponential_rate_per_hour"])
    evaluation["time_prediction"] = neural_time(rows)
    evaluation["baselines"] = {"global_feedback_frequency": feedback_baseline(rows, probabilities), "global_median_gap_and_constant_exponential": time_baseline(rows, median_gap, rate)}
    evaluation["slices"] = {
        "target_group": slice_rows(rows, "target_group_size", [("singleton",1,1),("multi_event",2,10**9)]),
        "previous_group_size": slice_rows(rows, "previous_group_size", [("1",1,1),("2-5",2,5),("6-20",6,20),("21-100",21,100),("101-500",101,500),(">500",501,10**9)]),
    }
    return evaluation


def main():
    work = ROOT / "minimal_snmpp_pilot"
    config = json.loads((work / "pilot_config.json").read_text())
    stage_a = json.loads((work / "stage_a_result.json").read_text())
    if not stage_a["passed"]:
        raise SystemExit("Stage A did not pass; Stage B is forbidden")
    manifest = json.loads((work / "data/subset_manifest.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = FullHistoryTargetDataset(work / "data/pilot_sequences.npz", "pilot_train_targets")
    validation_data = FullHistoryTargetDataset(work / "data/pilot_sequences.npz", "pilot_validation_targets")
    all_results = []
    for seed in config["stage_b"]["seeds"]:
        seed_everything(int(seed))
        model = MinimalSNMPP(ROOT / "phase2_gate2_sid/materialized_v1_1/frozen_sid/codebooks.npy", feedback_embedding_dim=int(config["model"]["feedback_embedding_dim"]), hidden_dims=tuple(config["model"]["interaction_hidden_dims"]), integration_q=64, initial_delay_hours=float(config["model"]["initial_delay_hours"])).to(device)
        initialize_constant_rate(model, float(manifest["baselines"]["constant_exponential_rate_per_hour"]))
        initial = parameter_snapshot(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["stage_b"]["learning_rate"]))
        train_loader = make_loader(train_data, batch_size=int(config["stage_b"]["batch_size"]), seed=int(seed), shuffle=True)
        validation_loader = make_loader(validation_data, batch_size=int(config["stage_b"]["batch_size"]), seed=int(seed), shuffle=False)
        output_dir = work / f"runs/stage_b_seed{seed}"; output_dir.mkdir(parents=True, exist_ok=True)
        curves=[]; best=float("inf"); best_epoch=0
        for epoch in range(1, int(config["stage_b"]["max_epochs"])+1):
            train=train_epoch(model,train_loader,optimizer,device,float(config["stage_b"]["gradient_clip_norm"]))
            val=evaluate_feedback_and_loss(model,validation_loader,device)
            val_small={k:v for k,v in val.items() if k!="rows"}
            curves.append({"epoch":epoch,"train":train,"validation":val_small})
            if val["optimization_loss"] < best:
                best=val["optimization_loss"]; best_epoch=epoch
                torch.save({"model":model.state_dict(),"seed":seed,"epoch":epoch,"config":config},output_dir/"checkpoint_best.pt")
            print(json.dumps({"seed":seed,"epoch":epoch,"train_loss":train["optimization_loss"],"val_loss":val["optimization_loss"],"val_macro_f1":val["macro_f1"],"recall":val["per_class_recall"],"sec":train["seconds"]},ensure_ascii=False),flush=True)
        saved=torch.load(output_dir/"checkpoint_best.pt",map_location=device,weights_only=False); model.load_state_dict(saved["model"])
        final=evaluate_and_compare(model,validation_loader,device,manifest)
        result={"seed":seed,"best_epoch":best_epoch,"train_targets":len(train_data),"validation_targets":len(validation_data),"metrics":final,"parameter_change_l2":parameter_change(initial,model),"delay_matrix_hours":model.delays().detach().cpu().tolist(),"curves":curves}
        save_json(output_dir/"training_curve.json",curves); save_json(output_dir/"pilot_seed_metrics.json",result)
        all_results.append(result)
        print(json.dumps({"SEED_COMPLETE":seed,"best_epoch":best_epoch,"feedback_macro_f1":final["macro_f1"],"time_nll":final["time_loss_per_group"],"time_mae":final["time_prediction"]["time_mae_hours"]},ensure_ascii=False),flush=True)
    save_json(work/"stage_b_results.json",all_results)


if __name__ == "__main__": main()
