#!/usr/bin/env python3
"""Diagnose RAPI intervention strength and small eta reward sensitivity."""

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
import torch.nn.functional as F
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
    parser = argparse.ArgumentParser(description="RAPI diagnostic statistics")
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
    parser.add_argument("--num_episodes", type=int, default=1024)
    parser.add_argument("--reward_sweep_episodes", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--read_batch_size", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--sample_response", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--structured_simulator_response", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--simulator_negative_prob_scale", type=float, default=1.0)
    parser.add_argument("--failure_signal_scope", default="explicit_negative", choices=["all_failed", "explicit_negative"])
    parser.add_argument("--negative_patience", type=int, default=5)
    parser.add_argument("--reward_done_threshold", type=float, default=None)
    parser.add_argument(
        "--eval_reward_mode",
        default="paper_effective",
        choices=["simulator", "paper", "paper_effective"],
    )
    parser.add_argument("--decode_top_k", type=int, default=8)
    parser.add_argument("--action_mode", default="sample", choices=["sample", "argmax"])
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
    parser.add_argument("--eta_list", default="0,0.02,0.05,0.10,0.20")
    parser.add_argument("--sara_layer_weights", type=str, default="0.05,0.25,0.70")
    parser.add_argument(
        "--rapi_candidate_eta",
        type=float,
        default=1.0,
        help="Candidate-level rerank penalty used to compare old token-only RAPI against the new decoder rerank.",
    )
    parser.add_argument("--max_case_examples", type=int, default=12)
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
    return parser.parse_args()


def make_loader(args: argparse.Namespace, num_episodes: int) -> DataLoader:
    dataset = eval_mod.TransitionIterableDataset(
        Path(args.transition_root) / args.split,
        args.item_features_npy,
        max_seq_len=args.max_seq_len,
        max_rows=int(num_episodes),
        batch_size=args.read_batch_size,
        shuffle_files=False,
        shuffle_buffer_size=0,
        seed=args.seed,
        sample_across_files=False,
    )
    print(f"[data] split={args.split} files={len(dataset.files)} episodes={num_episodes}")
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=0)


def new_level_stat() -> dict[str, float]:
    return {
        "rows": 0.0,
        "active_memory_rows": 0.0,
        "mean_abs_logit_delta_sum": 0.0,
        "max_abs_logit_delta_sum": 0.0,
        "base_top1_delta_sum": 0.0,
        "base_top1_abs_delta_sum": 0.0,
        "base_top1_prob_delta_sum": 0.0,
        "top1_changed": 0.0,
        "base_top1_memory_hit": 0.0,
        "rapi_top1_memory_hit": 0.0,
        "base_top5_memory_hit": 0.0,
        "prob_l1_sum": 0.0,
        "kl_base_to_rapi_sum": 0.0,
        "base_entropy_sum": 0.0,
        "rapi_entropy_sum": 0.0,
        "distinct_memory_tokens_sum": 0.0,
    }


def new_diag_stat(sid_levels: int) -> dict[str, Any]:
    return {
        "rows": 0.0,
        "memory_entries": 0.0,
        "active_memory_rows": 0.0,
        "memory_phi_sum": 0.0,
        "memory_phi_max": 0.0,
        "memory_type_1_low_play": 0.0,
        "memory_type_2_dislike": 0.0,
        "memory_type_3_unlike": 0.0,
        "memory_type_other": 0.0,
        "base_rapi_argmax_same_item": 0.0,
        "base_candidate_argmax_same_item": 0.0,
        "token_candidate_argmax_same_item": 0.0,
        "candidate_changed": 0.0,
        "candidate_changed_active": 0.0,
        "base_greedy_path_in_memory": 0.0,
        "rapi_greedy_path_in_memory": 0.0,
        "case_examples": [],
        "levels": [new_level_stat() for _ in range(sid_levels)],
    }


def update_memory_summary(stats: dict[str, Any], pool_tokens: torch.Tensor, pool_phis: torch.Tensor, memory: Any) -> None:
    active = pool_phis > 0.0
    rows = float(pool_phis.shape[0])
    stats["rows"] += rows
    stats["memory_entries"] += float(active.float().sum().detach().cpu())
    stats["active_memory_rows"] += float(active.any(dim=1).float().sum().detach().cpu())
    stats["memory_phi_sum"] += float(pool_phis[active].sum().detach().cpu()) if bool(active.any()) else 0.0
    if bool(active.any()):
        stats["memory_phi_max"] = max(float(stats["memory_phi_max"]), float(pool_phis[active].max().detach().cpu()))

    for entries in memory._pool.values():
        for entry in entries:
            if int(entry.regret_type_id) == 1:
                stats["memory_type_1_low_play"] += 1.0
            elif int(entry.regret_type_id) == 2:
                stats["memory_type_2_dislike"] += 1.0
            elif int(entry.regret_type_id) == 3:
                stats["memory_type_3_unlike"] += 1.0
            else:
                stats["memory_type_other"] += 1.0

    for level, level_stats in enumerate(stats["levels"]):
        tokens_l = pool_tokens[:, :, level]
        distinct_sum = 0.0
        for row_idx in range(tokens_l.shape[0]):
            row_active = active[row_idx]
            if bool(row_active.any()):
                distinct_sum += float(torch.unique(tokens_l[row_idx][row_active]).numel())
        level_stats["distinct_memory_tokens_sum"] += distinct_sum


def path_in_memory(path_tokens: torch.Tensor, pool_tokens: torch.Tensor, pool_phis: torch.Tensor) -> torch.Tensor:
    active = pool_phis > 0.0
    same = (pool_tokens == path_tokens.unsqueeze(1)).all(dim=-1)
    return (same & active).any(dim=1)


def token_hit(token: torch.Tensor, pool_tokens: torch.Tensor, pool_phis: torch.Tensor, level: int) -> torch.Tensor:
    active = pool_phis > 0.0
    return ((pool_tokens[:, :, level] == token.view(-1, 1)) & active).any(dim=1)


def topk_hit(tokens: torch.Tensor, pool_tokens: torch.Tensor, pool_phis: torch.Tensor, level: int) -> torch.Tensor:
    active = pool_phis > 0.0
    hits = pool_tokens[:, :, level].unsqueeze(1) == tokens.unsqueeze(-1)
    return (hits & active.unsqueeze(1)).any(dim=(1, 2))


def update_logit_stats(
    stats: dict[str, Any],
    base_logits: list[torch.Tensor],
    rapi_logits: list[torch.Tensor],
    pool_tokens: torch.Tensor,
    pool_phis: torch.Tensor,
    decoder: Any,
    obs: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> None:
    active_memory_rows = (pool_phis > 0.0).any(dim=1)
    base_path_tokens = []
    rapi_path_tokens = []
    for level, (base_l, rapi_l) in enumerate(zip(base_logits, rapi_logits)):
        level_stats = stats["levels"][level]
        base_probs = F.softmax(base_l, dim=-1)
        rapi_probs = F.softmax(rapi_l, dim=-1)
        base_log_probs = F.log_softmax(base_l, dim=-1)
        rapi_log_probs = F.log_softmax(rapi_l, dim=-1)
        delta = rapi_l - base_l
        base_top1 = base_l.argmax(dim=-1)
        rapi_top1 = rapi_l.argmax(dim=-1)
        base_top5 = torch.topk(base_l, k=min(5, base_l.shape[-1]), dim=-1).indices
        top1_delta = delta.gather(1, base_top1.view(-1, 1)).squeeze(1)
        top1_prob_delta = rapi_probs.gather(1, base_top1.view(-1, 1)).squeeze(1) - base_probs.gather(
            1, base_top1.view(-1, 1)
        ).squeeze(1)
        prob_l1 = torch.abs(rapi_probs - base_probs).sum(dim=-1)
        kl = (base_probs * (base_log_probs - rapi_log_probs)).sum(dim=-1)
        base_entropy = -(base_probs * base_log_probs).sum(dim=-1)
        rapi_entropy = -(rapi_probs * rapi_log_probs).sum(dim=-1)

        n = float(base_l.shape[0])
        level_stats["rows"] += n
        level_stats["active_memory_rows"] += float(active_memory_rows.float().sum().detach().cpu())
        level_stats["mean_abs_logit_delta_sum"] += float(delta.abs().mean(dim=-1).sum().detach().cpu())
        level_stats["max_abs_logit_delta_sum"] += float(delta.abs().max(dim=-1).values.sum().detach().cpu())
        level_stats["base_top1_delta_sum"] += float(top1_delta.sum().detach().cpu())
        level_stats["base_top1_abs_delta_sum"] += float(top1_delta.abs().sum().detach().cpu())
        level_stats["base_top1_prob_delta_sum"] += float(top1_prob_delta.sum().detach().cpu())
        level_stats["top1_changed"] += float((base_top1 != rapi_top1).float().sum().detach().cpu())
        level_stats["base_top1_memory_hit"] += float(token_hit(base_top1, pool_tokens, pool_phis, level).float().sum().detach().cpu())
        level_stats["rapi_top1_memory_hit"] += float(token_hit(rapi_top1, pool_tokens, pool_phis, level).float().sum().detach().cpu())
        level_stats["base_top5_memory_hit"] += float(topk_hit(base_top5, pool_tokens, pool_phis, level).float().sum().detach().cpu())
        level_stats["prob_l1_sum"] += float(prob_l1.sum().detach().cpu())
        level_stats["kl_base_to_rapi_sum"] += float(kl.sum().detach().cpu())
        level_stats["base_entropy_sum"] += float(base_entropy.sum().detach().cpu())
        level_stats["rapi_entropy_sum"] += float(rapi_entropy.sum().detach().cpu())
        base_path_tokens.append(base_top1)
        rapi_path_tokens.append(rapi_top1)

    base_path = torch.stack(base_path_tokens, dim=1)
    rapi_path = torch.stack(rapi_path_tokens, dim=1)
    stats["base_greedy_path_in_memory"] += float(path_in_memory(base_path, pool_tokens, pool_phis).float().sum().detach().cpu())
    stats["rapi_greedy_path_in_memory"] += float(path_in_memory(rapi_path, pool_tokens, pool_phis).float().sum().detach().cpu())

    rng = np.random.default_rng(int(args.seed))
    base_items, _ = decoder.decode_logits(
        base_logits,
        fallback_items=eval_mod.fallback_items_from_history(obs),
        top_k=args.decode_top_k,
        device=obs["history_features"].device,
        mode="argmax",
        rng=rng,
        temperature=args.action_temperature,
    )
    rapi_items, rapi_sids = decoder.decode_logits(
        rapi_logits,
        fallback_items=eval_mod.fallback_items_from_history(obs),
        top_k=args.decode_top_k,
        device=obs["history_features"].device,
        mode="argmax",
        rng=rng,
        temperature=args.action_temperature,
    )
    candidate_items, candidate_sids, candidate_details = decoder.decode_logits(
        rapi_logits,
        fallback_items=eval_mod.fallback_items_from_history(obs),
        top_k=args.decode_top_k,
        device=obs["history_features"].device,
        mode="argmax",
        rng=rng,
        temperature=args.action_temperature,
        pool_tokens=pool_tokens,
        pool_phis=pool_phis,
        candidate_rerank=True,
        candidate_eta=float(args.rapi_candidate_eta),
        candidate_layer_weights=parse_float_list(args.sara_layer_weights),
        return_details=True,
    )
    candidate_changed = base_items != candidate_items
    stats["base_rapi_argmax_same_item"] += float((base_items == rapi_items).float().sum().detach().cpu())
    stats["base_candidate_argmax_same_item"] += float((base_items == candidate_items).float().sum().detach().cpu())
    stats["token_candidate_argmax_same_item"] += float((rapi_items == candidate_items).float().sum().detach().cpu())
    stats["candidate_changed"] += float(candidate_changed.float().sum().detach().cpu())
    stats["candidate_changed_active"] += float((candidate_changed & active_memory_rows).float().sum().detach().cpu())

    examples = stats.setdefault("case_examples", [])
    max_examples = max(0, int(getattr(args, "max_case_examples", 0)))
    if len(examples) < max_examples and bool(candidate_changed.any()):
        user_ids = obs["user_id"].detach().cpu().view(-1).tolist()
        pool_tokens_cpu = pool_tokens.detach().cpu()
        pool_phis_cpu = pool_phis.detach().cpu()
        base_path_cpu = base_path.detach().cpu()
        rapi_path_cpu = rapi_path.detach().cpu()
        rapi_sids_cpu = rapi_sids.detach().cpu()
        candidate_sids_cpu = candidate_sids.detach().cpu()
        changed_rows = torch.nonzero(candidate_changed.detach().cpu(), as_tuple=False).view(-1).tolist()
        for row_idx in changed_rows:
            if len(examples) >= max_examples:
                break
            active_idx = torch.nonzero(pool_phis_cpu[row_idx] > 0.0, as_tuple=False).view(-1).tolist()
            memory_entries = []
            for mem_idx in active_idx[:5]:
                memory_entries.append(
                    {
                        "sid_path": [int(x) for x in pool_tokens_cpu[row_idx, mem_idx].tolist()],
                        "phi": float(pool_phis_cpu[row_idx, mem_idx]),
                    }
                )
            examples.append(
                {
                    "row_idx": int(row_idx),
                    "user_id": int(user_ids[row_idx]),
                    "base_item": int(base_items[row_idx].detach().cpu()),
                    "old_token_rapi_item": int(rapi_items[row_idx].detach().cpu()),
                    "new_candidate_rapi_item": int(candidate_items[row_idx].detach().cpu()),
                    "base_greedy_sid_path": [int(x) for x in base_path_cpu[row_idx].tolist()],
                    "old_token_rapi_greedy_sid_path": [int(x) for x in rapi_path_cpu[row_idx].tolist()],
                    "old_token_rapi_decoded_sid_path": [int(x) for x in rapi_sids_cpu[row_idx].tolist()],
                    "new_candidate_rapi_decoded_sid_path": [int(x) for x in candidate_sids_cpu[row_idx].tolist()],
                    "new_candidate_detail": candidate_details[row_idx],
                    "memory_entries": memory_entries,
                }
            )


def finalize_diag(stats: dict[str, Any], sid_vocab_size: int) -> dict[str, Any]:
    rows = max(float(stats["rows"]), 1.0)
    entries = max(float(stats["memory_entries"]), 1.0)
    out = {
        key: value
        for key, value in stats.items()
        if key != "levels"
    }
    out.update(
        {
            "active_memory_row_rate": stats["active_memory_rows"] / rows,
            "mean_memory_entries_per_row": stats["memory_entries"] / rows,
            "mean_phi_per_entry": stats["memory_phi_sum"] / entries,
            "base_rapi_argmax_same_item_rate": stats["base_rapi_argmax_same_item"] / rows,
            "base_candidate_argmax_same_item_rate": stats["base_candidate_argmax_same_item"] / rows,
            "token_candidate_argmax_same_item_rate": stats["token_candidate_argmax_same_item"] / rows,
            "candidate_changed_rate": stats["candidate_changed"] / rows,
            "candidate_changed_active_rate": stats["candidate_changed_active"] / max(float(stats["active_memory_rows"]), 1.0),
            "base_greedy_path_in_memory_rate": stats["base_greedy_path_in_memory"] / rows,
            "rapi_greedy_path_in_memory_rate": stats["rapi_greedy_path_in_memory"] / rows,
        }
    )
    out["levels"] = []
    for level, level_stats in enumerate(stats["levels"]):
        n = max(float(level_stats["rows"]), 1.0)
        active_rows = max(float(level_stats["active_memory_rows"]), 1.0)
        out["levels"].append(
            {
                "level": level,
                **level_stats,
                "active_memory_row_rate": level_stats["active_memory_rows"] / n,
                "mean_abs_logit_delta": level_stats["mean_abs_logit_delta_sum"] / n,
                "mean_max_abs_logit_delta": level_stats["max_abs_logit_delta_sum"] / n,
                "mean_base_top1_delta": level_stats["base_top1_delta_sum"] / n,
                "mean_abs_base_top1_delta": level_stats["base_top1_abs_delta_sum"] / n,
                "mean_base_top1_prob_delta": level_stats["base_top1_prob_delta_sum"] / n,
                "top1_changed_rate": level_stats["top1_changed"] / n,
                "base_top1_memory_hit_rate": level_stats["base_top1_memory_hit"] / n,
                "rapi_top1_memory_hit_rate": level_stats["rapi_top1_memory_hit"] / n,
                "base_top5_memory_hit_rate": level_stats["base_top5_memory_hit"] / n,
                "prob_l1": level_stats["prob_l1_sum"] / n,
                "kl_base_to_rapi": level_stats["kl_base_to_rapi_sum"] / n,
                "base_entropy": level_stats["base_entropy_sum"] / n,
                "rapi_entropy": level_stats["rapi_entropy_sum"] / n,
                "mean_distinct_memory_tokens_per_active_row": level_stats["distinct_memory_tokens_sum"] / active_rows,
                "mean_distinct_memory_token_vocab_coverage": (
                    level_stats["distinct_memory_tokens_sum"] / active_rows / max(float(sid_vocab_size), 1.0)
                ),
            }
        )
    return out


@torch.no_grad()
def run_static_diagnostic(args: argparse.Namespace, actor: Any, decoder: Any, dense_item2sid: np.ndarray, device: torch.device) -> dict[str, Any]:
    sid_levels, sid_vocab_size = eval_mod.infer_sid_spec(dense_item2sid)
    stats_by_eta = {str(eta): new_diag_stat(sid_levels) for eta in parse_float_list(args.eta_list) if eta > 0.0}
    loader = make_loader(args, int(args.num_episodes))
    seen = 0
    for batch in tqdm(loader, desc="[static]", ncols=120):
        batch_n = int(batch["history_ids"].shape[0])
        if seen >= int(args.num_episodes):
            break
        if seen + batch_n > int(args.num_episodes):
            keep = int(args.num_episodes) - seen
            batch = {key: value[:keep] if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            batch_n = keep
        obs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        feed = eval_mod.actor_obs(obs)
        base_out = actor(feed)

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
        pool_tokens, pool_phis = memory.get(obs["user_id"].long(), device)

        for eta_text, stats in stats_by_eta.items():
            actor.sara_eta = float(eta_text)
            rapi_out = actor.get_sara_logits(feed, pool_tokens=pool_tokens, pool_phis=pool_phis)
            update_memory_summary(stats, pool_tokens, pool_phis, memory)
            stats["loaded_snapshot_entries"] = float(stats.get("loaded_snapshot_entries", 0.0)) + float(loaded_snapshot)
            stats["loaded_history_entries"] = float(stats.get("loaded_history_entries", 0.0)) + float(loaded_history)
            update_logit_stats(
                stats,
                base_out["sid_logits"],
                rapi_out["sid_logits"],
                pool_tokens,
                pool_phis,
                decoder,
                obs,
                args,
            )
        seen += batch_n
    return {eta: finalize_diag(stats, sid_vocab_size) for eta, stats in stats_by_eta.items()}


def build_env(args: argparse.Namespace, device: torch.device) -> Any:
    return eval_mod.RegretUserResponseEnv(
        checkpoint_path=args.simulator_checkpoint,
        transition_path=Path(args.transition_root) / args.split,
        dense_item_features_npy=args.item_features_npy,
        max_seq_len=args.max_seq_len,
        device=str(device),
        max_step_per_episode=args.max_steps,
        sample_response=args.sample_response,
        structured_response=args.structured_simulator_response,
        gate_revision_by_history=args.gate_revision_by_history,
        failure_signal_scope=args.failure_signal_scope,
        negative_prob_scale=args.simulator_negative_prob_scale,
        negative_patience=args.negative_patience,
        reward_done_threshold=args.reward_done_threshold,
        seed=args.seed,
    )


@torch.no_grad()
def run_reward_sweep(args: argparse.Namespace, actor: Any, decoder: Any, dense_item2sid: np.ndarray, device: torch.device) -> dict[str, Any]:
    if int(args.reward_sweep_episodes) <= 0:
        return {}
    results: dict[str, Any] = {}
    for eta in parse_float_list(args.eta_list):
        sweep_args = copy.copy(args)
        sweep_args.num_episodes = int(args.reward_sweep_episodes)
        actor.sara_eta = float(eta)
        loader = make_loader(sweep_args, int(sweep_args.num_episodes))
        base_env = build_env(sweep_args, device)
        rapi_env = build_env(sweep_args, device)
        base_stats = eval_mod.new_stats()
        rapi_stats = eval_mod.new_stats()
        base_rng = np.random.default_rng(int(args.seed))
        rapi_rng = np.random.default_rng(int(args.seed))
        total = int(sweep_args.num_episodes) * int(args.max_steps) * 2
        pbar = tqdm(total=total, desc=f"[eta {eta:g}]", ncols=120)
        seen = 0
        for batch in loader:
            batch_n = int(batch["history_ids"].shape[0])
            if seen >= int(sweep_args.num_episodes):
                break
            if seen + batch_n > int(sweep_args.num_episodes):
                keep = int(sweep_args.num_episodes) - seen
                batch = {key: value[:keep] if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                batch_n = keep
            eval_mod.rollout_one_policy_batch(
                actor,
                base_env,
                batch,
                decoder,
                sweep_args,
                dense_item2sid,
                False,
                base_stats,
                pbar,
                base_rng,
            )
            eval_mod.rollout_one_policy_batch(
                actor,
                rapi_env,
                batch,
                decoder,
                sweep_args,
                dense_item2sid,
                True,
                rapi_stats,
                pbar,
                rapi_rng,
            )
            seen += batch_n
            base_now = eval_mod.finalize_stats(base_stats)
            rapi_now = eval_mod.finalize_stats(rapi_stats)
            pbar.set_postfix(
                base_r=base_now["avg_cum_reward"],
                rapi_r=rapi_now["avg_cum_reward"],
                delta=rapi_now["avg_cum_reward"] - base_now["avg_cum_reward"],
            )
        pbar.close()
        base = eval_mod.finalize_stats(base_stats)
        rapi = eval_mod.finalize_stats(rapi_stats)
        results[str(float(eta))] = {
            "base": base,
            "rapi": rapi,
            "delta_rapi_minus_base": {
                key: float(rapi.get(key, 0.0) - base.get(key, 0.0))
                for key in [
                    "avg_cum_reward",
                    "avg_step",
                    "reward_per_step",
                    "negative_rate",
                    "positive_rate",
                    "early_stop_rate",
                    "failure_rate",
                    "failure_dislike_rate",
                    "failure_unlike_rate",
                    "invalid_revision_mass_per_step",
                ]
            },
        }
    return results


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

    static = run_static_diagnostic(args, actor, decoder, dense_item2sid, device)
    reward_sweep = run_reward_sweep(args, actor, decoder, dense_item2sid, device)
    meta = {
        "args": vars(args),
        "static_logit_diagnostic": static,
        "reward_sweep": reward_sweep,
    }
    out_path = Path(args.save_meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] meta saved to {out_path}")
    for eta, result in reward_sweep.items():
        delta = result["delta_rapi_minus_base"]["avg_cum_reward"]
        print(
            f"[eta {eta}] base={result['base']['avg_cum_reward']:.4f} "
            f"rapi={result['rapi']['avg_cum_reward']:.4f} delta={delta:.4f}"
        )


if __name__ == "__main__":
    main()
