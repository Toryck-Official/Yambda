#!/usr/bin/env python3
"""Trace RAPI item changes and simulator counterfactual scores.

This is a diagnostic script. It does not train models or alter rollout logic.
For each RAPI-branch state, it compares the base argmax item against the
RAPI+candidate-rerank argmax item on the same history, then asks the simulator
for expected feedback components for both candidate items.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = PROJECT_ROOT / "scripts/09_eval_simulator_rollout.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_eval_module() -> Any:
    spec = importlib.util.spec_from_file_location("eval_simulator_rollout", EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {EVAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_mod = load_eval_module()


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in str(text).replace(",", " ").split() if x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace RAPI changed actions with simulator counterfactuals")
    parser.add_argument("--transition_root", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "replay_val", "replay_test"])
    parser.add_argument("--item_features_npy", required=True)
    parser.add_argument("--dense_item2sid_npy", required=True)
    parser.add_argument("--actor_checkpoint", required=True)
    parser.add_argument("--simulator_checkpoint", required=True)
    parser.add_argument("--save_meta", required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--num_episodes", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--read_batch_size", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--decode_top_k", type=int, default=8)
    parser.add_argument("--trace_action_mode", default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--action_temperature", type=float, default=1.0)
    parser.add_argument("--sasrec_n_layer", type=int, default=2)
    parser.add_argument("--sasrec_d_model", type=int, default=64)
    parser.add_argument("--sasrec_d_forward", type=int, default=128)
    parser.add_argument("--sasrec_n_head", type=int, default=4)
    parser.add_argument("--sasrec_dropout", type=float, default=0.1)
    parser.add_argument("--sid_temp", type=float, default=1.0)
    parser.add_argument("--use_history_feedback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_history_event_type", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--event_type_vocab_size", type=int, default=8)
    parser.add_argument("--sara_eta", type=float, default=0.10)
    parser.add_argument("--sara_layer_weights", default="0.05,0.25,0.70")
    parser.add_argument("--rapi_candidate_eta", type=float, default=1.0)
    parser.add_argument("--regret_pool_size", type=int, default=20)
    parser.add_argument("--regret_gamma", type=float, default=0.9)
    parser.add_argument("--regret_phi_scale", type=float, default=1.0)
    parser.add_argument("--regret_phi_clip", type=float, default=2.0)
    parser.add_argument("--regret_reward_threshold", type=float, default=0.0)
    parser.add_argument("--use_precomputed_regret_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--history_memory_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--memory_signal_scope",
        default="explicit_negative",
        choices=["all_failed", "explicit_negative", "revision_only", "rrca", "none"],
    )
    parser.add_argument("--gate_revision_by_history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--structured_simulator_response", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--failure_signal_scope", default="explicit_negative", choices=["all_failed", "explicit_negative"])
    parser.add_argument("--simulator_negative_prob_scale", type=float, default=1.0)
    parser.add_argument("--negative_patience", type=int, default=5)
    parser.add_argument("--max_case_examples", type=int, default=30)
    return parser.parse_args()


def make_loader(args: argparse.Namespace) -> DataLoader:
    dataset = eval_mod.TransitionIterableDataset(
        Path(args.transition_root) / args.split,
        args.item_features_npy,
        max_seq_len=args.max_seq_len,
        max_rows=args.num_episodes,
        batch_size=args.read_batch_size,
        shuffle_files=False,
        shuffle_buffer_size=0,
        seed=args.seed,
        sample_across_files=False,
    )
    print(f"[data] split={args.split} files={len(dataset.files)} episodes={args.num_episodes}")
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=0)


def new_totals() -> dict[str, float]:
    return {
        "states": 0.0,
        "active_memory_states": 0.0,
        "changed_states": 0.0,
        "changed_active_states": 0.0,
        "memory_entries_sum": 0.0,
        "base_expected_reward_sum": 0.0,
        "rapi_expected_reward_sum": 0.0,
        "base_expected_play_sum": 0.0,
        "rapi_expected_play_sum": 0.0,
        "base_like_prob_sum": 0.0,
        "rapi_like_prob_sum": 0.0,
        "base_dislike_prob_sum": 0.0,
        "rapi_dislike_prob_sum": 0.0,
        "base_unlike_valid_prob_sum": 0.0,
        "rapi_unlike_valid_prob_sum": 0.0,
        "changed_base_expected_reward_sum": 0.0,
        "changed_rapi_expected_reward_sum": 0.0,
        "changed_base_expected_play_sum": 0.0,
        "changed_rapi_expected_play_sum": 0.0,
        "changed_base_like_prob_sum": 0.0,
        "changed_rapi_like_prob_sum": 0.0,
        "changed_base_dislike_prob_sum": 0.0,
        "changed_rapi_dislike_prob_sum": 0.0,
        "changed_base_unlike_valid_prob_sum": 0.0,
        "changed_rapi_unlike_valid_prob_sum": 0.0,
        "candidate_penalty_sum": 0.0,
        "changed_candidate_penalty_sum": 0.0,
    }


def simulator_expectation(env: Any, obs: dict[str, torch.Tensor], item_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    action_np = item_ids.detach().cpu().numpy().astype(np.int64)
    action_features = torch.tensor(env.features[action_np].astype(np.float32), device=env.device)
    batch = {
        "history_features": obs["history_features"],
        "history_feedbacks": obs["history_feedbacks"],
        "history_event_type_ids": obs["history_event_type_ids"],
        "history_mask": obs["history_mask"],
        "action_features": action_features,
        "prior_stats": torch.zeros(item_ids.shape[0], env.model.prior_dim, device=env.device),
    }
    out = env.model(batch)
    listen_prob = out["listen_prob"].detach()
    play_prob = out["play_prob"].detach()
    feedback = out["reward_feedback_probs"].detach()
    expected_play = listen_prob * play_prob
    prior_like = eval_mod.history_has_same_item_event(obs, item_ids, eval_mod.EVENT_TYPE_TO_ID["like"])
    unlike_valid_prob = feedback[:, 2] * prior_like.float()
    return {
        "expected_play": expected_play,
        "like_prob": feedback[:, 0],
        "dislike_prob": feedback[:, 1],
        "unlike_valid_prob": unlike_valid_prob,
        "expected_reward": expected_play + feedback[:, 0] - feedback[:, 1] - unlike_valid_prob,
        "listen_prob": listen_prob,
        "play_prob": play_prob,
        "negative_prob": out["negative_prob"].detach(),
    }


def add_metric(totals: dict[str, float], key: str, values: torch.Tensor, mask: torch.Tensor | None = None) -> None:
    if mask is not None:
        values = values[mask]
    if values.numel() <= 0:
        return
    totals[key] += float(values.detach().sum().cpu())


def memory_examples(pool_tokens: torch.Tensor, pool_phis: torch.Tensor, row_idx: int, max_items: int = 5) -> list[dict[str, Any]]:
    active_idx = torch.nonzero(pool_phis[row_idx] > 0.0, as_tuple=False).view(-1).detach().cpu().tolist()
    out = []
    for mem_idx in active_idx[:max_items]:
        out.append(
            {
                "sid_path": [int(x) for x in pool_tokens[row_idx, mem_idx].detach().cpu().tolist()],
                "phi": float(pool_phis[row_idx, mem_idx].detach().cpu()),
            }
        )
    return out


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = eval_mod.resolve_device(args.device)
    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
    sid_levels, sid_vocab_size = eval_mod.infer_sid_spec(dense_item2sid)
    feature_shape = np.load(args.item_features_npy, mmap_mode="r").shape
    print(f"[sid] levels={sid_levels} vocab={sid_vocab_size}; building decoder...")
    decoder = eval_mod.SIDDecoder(dense_item2sid, sid_vocab_size)
    print(f"[sid] decoder valid_items={decoder.dense_ids.shape[0]}")
    actor = eval_mod.build_actor(args, dense_item2sid, int(feature_shape[1]), device)
    env = eval_mod.RegretUserResponseEnv(
        checkpoint_path=args.simulator_checkpoint,
        transition_path=Path(args.transition_root) / args.split,
        dense_item_features_npy=args.item_features_npy,
        max_seq_len=args.max_seq_len,
        device=str(device),
        max_step_per_episode=args.max_steps,
        sample_response=True,
        structured_response=args.structured_simulator_response,
        gate_revision_by_history=args.gate_revision_by_history,
        failure_signal_scope=args.failure_signal_scope,
        negative_prob_scale=args.simulator_negative_prob_scale,
        negative_patience=args.negative_patience,
        seed=args.seed,
    )
    base_rng = np.random.default_rng(int(args.seed))
    rapi_rng = np.random.default_rng(int(args.seed))
    totals = new_totals()
    cases: list[dict[str, Any]] = []
    loader = make_loader(args)
    seen = 0
    pbar = tqdm(total=int(args.num_episodes) * int(args.max_steps), desc="[trace]", ncols=120)
    for batch in loader:
        batch_n = int(batch["history_ids"].shape[0])
        if seen >= int(args.num_episodes):
            break
        if seen + batch_n > int(args.num_episodes):
            keep = int(args.num_episodes) - seen
            batch = {key: value[:keep] if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            batch_n = keep
        obs = env.reset_from_batch(batch)
        memory = eval_mod.RegretMemoryPool(
            pool_size=args.regret_pool_size,
            sid_levels=sid_levels,
            gamma=args.regret_gamma,
            phi_scale=args.regret_phi_scale,
            phi_clip=args.regret_phi_clip,
            reward_threshold=args.regret_reward_threshold,
            negative_type_ids=(1, 2, 3),
        )
        loaded_snapshot = 0
        loaded_history = 0
        if args.use_precomputed_regret_memory:
            loaded_snapshot = eval_mod.init_memory_from_snapshot(
                memory,
                batch,
                dense_item2sid,
                device,
                args.memory_signal_scope,
            )
        if loaded_snapshot <= 0 and args.history_memory_fallback:
            loaded_history = eval_mod.init_memory_from_history(
                memory,
                obs,
                dense_item2sid,
                device,
                args.memory_signal_scope,
            )
        active = torch.ones(batch_n, dtype=torch.bool, device=device)
        for step_idx in range(int(args.max_steps)):
            if not bool(active.any()):
                break
            state_obs = {
                key: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                for key, value in env.current_observation.items()
            }
            feed = eval_mod.actor_obs(state_obs)
            base_out = actor(feed)
            pool_tokens, pool_phis = memory.get(state_obs["user_id"].long(), device)
            rapi_out = actor.get_sara_logits(feed, pool_tokens=pool_tokens, pool_phis=pool_phis)
            fallback = eval_mod.fallback_items_from_history(state_obs)
            base_items, base_sids = decoder.decode_logits(
                base_out["sid_logits"],
                fallback_items=fallback,
                top_k=args.decode_top_k,
                device=device,
                mode=args.trace_action_mode,
                rng=base_rng,
                temperature=args.action_temperature,
            )
            rapi_items, rapi_sids, details = decoder.decode_logits(
                rapi_out["sid_logits"],
                fallback_items=fallback,
                top_k=args.decode_top_k,
                device=device,
                mode=args.trace_action_mode,
                rng=rapi_rng,
                temperature=args.action_temperature,
                pool_tokens=pool_tokens,
                pool_phis=pool_phis,
                candidate_rerank=True,
                candidate_eta=args.rapi_candidate_eta,
                candidate_layer_weights=parse_float_list(args.sara_layer_weights),
                return_details=True,
            )
            active_memory = (pool_phis > 0.0).any(dim=1)
            changed = (base_items != rapi_items) & active
            active_mask = active
            active_memory_mask = active & active_memory
            totals["states"] += float(active_mask.float().sum().detach().cpu())
            totals["active_memory_states"] += float(active_memory_mask.float().sum().detach().cpu())
            totals["changed_states"] += float(changed.float().sum().detach().cpu())
            totals["changed_active_states"] += float((changed & active_memory).float().sum().detach().cpu())
            totals["memory_entries_sum"] += float(((pool_phis > 0.0).float().sum(dim=1)[active_mask]).sum().detach().cpu())
            penalty = torch.tensor([float(d["candidate_penalty"]) for d in details], dtype=torch.float32, device=device)
            add_metric(totals, "candidate_penalty_sum", penalty, active_memory_mask)
            add_metric(totals, "changed_candidate_penalty_sum", penalty, changed)

            base_exp = simulator_expectation(env, state_obs, base_items)
            rapi_exp = simulator_expectation(env, state_obs, rapi_items)
            for source, exp in (("base", base_exp), ("rapi", rapi_exp)):
                add_metric(totals, f"{source}_expected_reward_sum", exp["expected_reward"], active_mask)
                add_metric(totals, f"{source}_expected_play_sum", exp["expected_play"], active_mask)
                add_metric(totals, f"{source}_like_prob_sum", exp["like_prob"], active_mask)
                add_metric(totals, f"{source}_dislike_prob_sum", exp["dislike_prob"], active_mask)
                add_metric(totals, f"{source}_unlike_valid_prob_sum", exp["unlike_valid_prob"], active_mask)
                add_metric(totals, f"changed_{source}_expected_reward_sum", exp["expected_reward"], changed)
                add_metric(totals, f"changed_{source}_expected_play_sum", exp["expected_play"], changed)
                add_metric(totals, f"changed_{source}_like_prob_sum", exp["like_prob"], changed)
                add_metric(totals, f"changed_{source}_dislike_prob_sum", exp["dislike_prob"], changed)
                add_metric(totals, f"changed_{source}_unlike_valid_prob_sum", exp["unlike_valid_prob"], changed)

            if len(cases) < int(args.max_case_examples) and bool(changed.any()):
                user_ids = state_obs["user_id"].detach().cpu().tolist()
                changed_rows = torch.nonzero(changed.detach().cpu(), as_tuple=False).view(-1).tolist()
                for row_idx in changed_rows:
                    if len(cases) >= int(args.max_case_examples):
                        break
                    cases.append(
                        {
                            "global_episode_offset": int(seen + row_idx),
                            "step": int(step_idx),
                            "user_id": int(user_ids[row_idx]),
                            "base_item": int(base_items[row_idx].detach().cpu()),
                            "rapi_item": int(rapi_items[row_idx].detach().cpu()),
                            "base_sid": [int(x) for x in base_sids[row_idx].detach().cpu().tolist()],
                            "rapi_sid": [int(x) for x in rapi_sids[row_idx].detach().cpu().tolist()],
                            "candidate_detail": details[row_idx],
                            "base_expected": {k: float(v[row_idx].detach().cpu()) for k, v in base_exp.items()},
                            "rapi_expected": {k: float(v[row_idx].detach().cpu()) for k, v in rapi_exp.items()},
                            "memory_entries": memory_examples(pool_tokens, pool_phis, row_idx),
                        }
                    )

            _, raw_reward, done, info = env.step(rapi_items)
            eval_mod.update_memory_from_simulated_step(
                memory,
                state_obs,
                env.current_observation["user_id"].long(),
                rapi_sids,
                rapi_items,
                raw_reward,
                info,
                active,
                bool(args.gate_revision_by_history),
                args.memory_signal_scope,
            )
            active = active & ~done.bool()
            pbar.update(batch_n)
        seen += batch_n
    pbar.close()

    states = max(float(totals["states"]), 1.0)
    active_memory_states = max(float(totals["active_memory_states"]), 1.0)
    changed_states = max(float(totals["changed_states"]), 1.0)
    summary = {
        **totals,
        "active_memory_state_rate": totals["active_memory_states"] / states,
        "changed_state_rate": totals["changed_states"] / states,
        "changed_active_state_rate": totals["changed_active_states"] / active_memory_states,
        "mean_memory_entries_per_state": totals["memory_entries_sum"] / states,
        "mean_memory_entries_per_active_memory_state": totals["memory_entries_sum"] / active_memory_states,
        "mean_candidate_penalty_active": totals["candidate_penalty_sum"] / active_memory_states,
        "mean_candidate_penalty_changed": totals["changed_candidate_penalty_sum"] / changed_states,
        "base_expected_reward": totals["base_expected_reward_sum"] / states,
        "rapi_expected_reward": totals["rapi_expected_reward_sum"] / states,
        "delta_expected_reward": (totals["rapi_expected_reward_sum"] - totals["base_expected_reward_sum"]) / states,
        "changed_base_expected_reward": totals["changed_base_expected_reward_sum"] / changed_states,
        "changed_rapi_expected_reward": totals["changed_rapi_expected_reward_sum"] / changed_states,
        "changed_delta_expected_reward": (
            totals["changed_rapi_expected_reward_sum"] - totals["changed_base_expected_reward_sum"]
        )
        / changed_states,
    }
    for metric in ["expected_play", "like_prob", "dislike_prob", "unlike_valid_prob"]:
        summary[f"base_{metric}"] = totals[f"base_{metric}_sum"] / states
        summary[f"rapi_{metric}"] = totals[f"rapi_{metric}_sum"] / states
        summary[f"delta_{metric}"] = (totals[f"rapi_{metric}_sum"] - totals[f"base_{metric}_sum"]) / states
        summary[f"changed_base_{metric}"] = totals[f"changed_base_{metric}_sum"] / changed_states
        summary[f"changed_rapi_{metric}"] = totals[f"changed_rapi_{metric}_sum"] / changed_states
        summary[f"changed_delta_{metric}"] = (
            totals[f"changed_rapi_{metric}_sum"] - totals[f"changed_base_{metric}_sum"]
        ) / changed_states

    meta = {
        "args": vars(args),
        "summary": summary,
        "cases": cases,
    }
    out_path = Path(args.save_meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[done] meta saved to {out_path}")


if __name__ == "__main__":
    main()
