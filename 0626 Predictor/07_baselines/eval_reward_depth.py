#!/usr/bin/env python3
"""Evaluate offline SID baselines with one shared reward/depth rollout protocol."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
sys.path.extend([str(ROOT / "02_model"), str(BASELINE_DIR), str(WORKSPACE_ROOT)])

from baseline_models import SASRecSIDPolicy  # noqa: E402
from hpn import FutureHPNPolicy  # noqa: E402


def load_rollout_module() -> ModuleType:
    path = ROOT / "06_rl" / "scripts" / "09_eval_simulator_rollout.py"
    spec = importlib.util.spec_from_file_location("baseline_rollout_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load rollout helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROLLOUT = load_rollout_module()


class HSRLEnvSpec:
    def __init__(self, n_item: int, item_dim: int, max_seq_len: int) -> None:
        self.action_space = {
            "item_id": ("nominal", n_item),
            "item_feature": ("continuous", item_dim, "normal"),
        }
        self.observation_space = {
            "history": ("sequence", max_seq_len, ("continuous", item_dim)),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SID baselines by reward/depth in one learned simulator.")
    parser.add_argument("--model_type", choices=["sasrec_sid", "hpn_sid", "hsrl_offline", "hsrl_sid"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--transition_root", default=str(ROOT / "01_data" / "processed" / "regret_current_data"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "replay_val", "replay_test"])
    parser.add_argument("--item_features_npy", default=str(ROOT / "01_data" / "processed" / "raw_rqkmeans" / "dense_item_features.npy"))
    parser.add_argument("--dense_item2sid_npy", default=str(ROOT / "01_data" / "processed" / "raw_rqkmeans" / "dense_item2sid.npy"))
    parser.add_argument("--simulator_checkpoint", default=str(ROOT / "06_rl" / "artifacts" / "user_response" / "base_v1_sim" / "regret_user_response.pt"))
    parser.add_argument("--save_meta", default=str(BASELINE_DIR / "artifacts" / "evals" / "reward_depth.meta.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--read_batch_size", type=int, default=2048)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--decode_top_k", type=int, default=8)
    parser.add_argument("--action_mode", default="sample", choices=["sample", "argmax"])
    parser.add_argument("--action_temperature", type=float, default=1.0)
    parser.add_argument("--sample_response", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--structured_simulator_response", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--simulator_negative_prob_scale", type=float, default=1.0)
    parser.add_argument("--failure_signal_scope", default="all_failed", choices=["all_failed", "explicit_negative"])
    parser.add_argument("--negative_patience", type=int, default=3)
    parser.add_argument("--reward_done_threshold", type=float, default=None)
    parser.add_argument(
        "--eval_reward_mode",
        default="paper_effective",
        choices=["simulator", "paper", "paper_effective"],
    )
    parser.add_argument("--reward_w_listen", type=float, default=1.0)
    parser.add_argument("--reward_w_like", type=float, default=1.0)
    parser.add_argument("--reward_w_dislike", type=float, default=1.0)
    parser.add_argument("--rrca_unlike_weight", type=float, default=1.0)
    parser.add_argument("--rrca_undislike_weight", type=float, default=1.0)
    parser.add_argument("--hsrl_project_root", default=str(WORKSPACE_ROOT / "HSRL"))
    parser.add_argument("--hsrl_bootstrap_root", default=str(WORKSPACE_ROOT))
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--d_forward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sid_temp", type=float, default=1.0)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def canonical_model_type(model_type: str) -> str:
    if model_type == "hsrl_sid":
        return "hsrl_offline"
    return model_type


def load_state_dict_compatible(model: torch.nn.Module, checkpoint: str, device: torch.device) -> dict[str, int]:
    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    else:
        state = ckpt
    own = model.state_dict()
    compatible = {}
    skipped = 0
    for key, value in state.items():
        if key in own and tuple(own[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped += 1
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {"loaded": len(compatible), "missing": len(missing), "skipped_or_unexpected": skipped + len(unexpected)}


def load_checkpoint_config(checkpoint: str) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, dict):
        return dict(ckpt.get("config", {}))
    return {}


def build_model(
    args: argparse.Namespace,
    item_dim: int,
    sid_levels: int,
    sid_vocab_size: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, int]]:
    model_type = canonical_model_type(args.model_type)
    cfg = load_checkpoint_config(args.checkpoint)
    if model_type == "sasrec_sid":
        model = SASRecSIDPolicy(
            item_dim=int(cfg.get("item_dim", item_dim)),
            d_model=int(cfg.get("d_model", args.d_model)),
            max_seq_len=int(cfg.get("max_seq_len", args.max_seq_len)),
            n_layer=int(cfg.get("n_layer", args.n_layer)),
            n_head=int(cfg.get("n_head", args.n_head)),
            dropout=float(cfg.get("dropout", args.dropout)),
            sid_levels=int(cfg.get("sid_levels", sid_levels)),
            sid_vocab_size=int(cfg.get("sid_vocab_size", sid_vocab_size)),
            state_pooling=str(cfg.get("state_pooling", "last_mean")),
            use_history_feedback=bool(cfg.get("use_history_feedback", False)),
            use_history_event_type=bool(cfg.get("use_history_event_type", False)),
        ).to(device)
    elif model_type == "hpn_sid":
        model = FutureHPNPolicy(
            item_dim=int(cfg.get("item_dim", item_dim)),
            d_model=int(cfg.get("d_model", args.d_model)),
            max_seq_len=int(cfg.get("max_seq_len", args.max_seq_len)),
            n_layer=int(cfg.get("n_layer", args.n_layer)),
            n_head=int(cfg.get("n_head", args.n_head)),
            dropout=float(cfg.get("dropout", args.dropout)),
            sid_levels=int(cfg.get("sid_levels", sid_levels)),
            sid_vocab_size=int(cfg.get("sid_vocab_size", sid_vocab_size)),
            sid_temp=float(cfg.get("sid_temp", args.sid_temp)),
            state_pooling=str(cfg.get("state_pooling", "last_mean")),
        ).to(device)
    else:
        sys.path.insert(0, str(Path(args.hsrl_bootstrap_root)))
        from adapter.bootstrap import install_hsrl_adapter  # type: ignore

        install_hsrl_adapter(str(Path(args.hsrl_project_root) / "hsrl_core"))
        from model.policy.SIDPolicy_credit import SIDPolicy_credit  # type: ignore

        hsrl_args = SimpleNamespace(
            sasrec_n_layer=int(args.n_layer),
            sasrec_d_model=int(args.d_model),
            sasrec_d_forward=int(args.d_forward),
            sasrec_n_head=int(args.n_head),
            sasrec_dropout=float(args.dropout),
            sid_levels=int(sid_levels),
            sid_vocab_sizes=int(sid_vocab_size),
            sid_temp=float(args.sid_temp),
            sara_eta=0.0,
            sara_layer_weights="1,1,1,1",
        )
        env = HSRLEnvSpec(n_item=int(np.load(args.dense_item2sid_npy, mmap_mode="r").shape[0] - 1), item_dim=item_dim, max_seq_len=args.max_seq_len)
        model = SIDPolicy_credit(hsrl_args, env).to(device)
    load_info = load_state_dict_compatible(model, args.checkpoint, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, load_info


def actor_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "history_features": obs["history_features"],
        "history_feedbacks": obs["history_feedbacks"].float(),
        "history_event_type_ids": obs["history_event_type_ids"].long(),
        "history_mask": obs["history_mask"].float(),
    }


def fallback_items_from_history(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    hist = obs["history_ids"].long()
    mask = hist > 0
    counts = mask.long().sum(dim=1).clamp_min(1) - 1
    return hist.gather(1, counts.view(-1, 1)).squeeze(1).clamp_min(1)


def select_actions(
    model: torch.nn.Module,
    obs: dict[str, torch.Tensor],
    decoder: Any,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(actor_obs(obs))
    logits = out["sid_logits"]
    if not isinstance(logits, list):
        raise TypeError("Baseline model must return sid_logits as a list of tensors.")
    return decoder.decode_logits(
        logits,
        fallback_items=fallback_items_from_history(obs),
        top_k=int(args.decode_top_k),
        device=obs["history_features"].device,
        mode=str(args.action_mode),
        rng=rng,
        temperature=float(args.action_temperature),
    )


def rollout_batch(
    model: torch.nn.Module,
    env: Any,
    batch: dict[str, torch.Tensor],
    decoder: Any,
    args: argparse.Namespace,
    stats: dict[str, float],
    rng: np.random.Generator,
) -> int:
    obs = env.reset_from_batch(batch)
    batch_size = int(obs["history_ids"].shape[0])
    active = torch.ones(batch_size, dtype=torch.bool, device=env.device)
    episode_rewards = torch.zeros(batch_size, dtype=torch.float32, device=env.device)
    episode_steps = torch.zeros(batch_size, dtype=torch.float32, device=env.device)
    for _ in range(int(args.max_steps)):
        if not bool(active.any()):
            break
        state_obs = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in env.current_observation.items()
        }
        action_items, _sid_paths = select_actions(model, state_obs, decoder, args, rng)
        _, raw_reward, done, info = env.step(action_items)
        eval_reward = ROLLOUT.compute_eval_reward(raw_reward, info, args)
        ROLLOUT.update_step_stats(stats, eval_reward, info, active, raw_reward=raw_reward)
        episode_rewards = episode_rewards + torch.where(active, eval_reward, torch.zeros_like(eval_reward))
        episode_steps = episode_steps + active.float()
        active = active & ~done.bool()
    ROLLOUT.finish_episode_stats(stats, episode_rewards, episode_steps, int(args.max_steps))
    return batch_size


def make_env(args: argparse.Namespace, device: torch.device) -> Any:
    return ROLLOUT.RegretUserResponseEnv(
        checkpoint_path=args.simulator_checkpoint,
        transition_path=Path(args.transition_root) / args.split,
        dense_item_features_npy=args.item_features_npy,
        max_seq_len=args.max_seq_len,
        device=str(device),
        max_step_per_episode=args.max_steps,
        sample_response=args.sample_response,
        structured_response=args.structured_simulator_response,
        gate_revision_by_history=True,
        failure_signal_scope=args.failure_signal_scope,
        negative_prob_scale=args.simulator_negative_prob_scale,
        negative_patience=args.negative_patience,
        reward_done_threshold=args.reward_done_threshold,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    args.model_type = canonical_model_type(args.model_type)
    set_seed(args.seed)
    device = choose_device(args.device)
    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
    item_features = np.load(args.item_features_npy, mmap_mode="r")
    sid_levels, sid_vocab_size = ROLLOUT.infer_sid_spec(dense_item2sid)
    print(f"[device] using {device}")
    print(f"[sid] levels={sid_levels} vocab={sid_vocab_size}; building decoder...")
    decoder = ROLLOUT.SIDDecoder(dense_item2sid, sid_vocab_size)
    model, load_info = build_model(args, int(item_features.shape[1]), sid_levels, sid_vocab_size, device)
    print(
        f"[model] type={args.model_type} checkpoint={args.checkpoint} "
        f"loaded={load_info['loaded']} missing={load_info['missing']} "
        f"skipped_or_unexpected={load_info['skipped_or_unexpected']}"
    )
    loader = ROLLOUT.make_loader(args)
    env = make_env(args, device)
    stats = ROLLOUT.new_stats()
    rng = np.random.default_rng(int(args.seed))
    seen = 0
    pbar = tqdm(total=int(args.num_episodes), desc=f"[reward_depth {args.model_type}]", dynamic_ncols=True)
    with torch.no_grad():
        for batch in loader:
            if seen >= int(args.num_episodes):
                break
            batch_n = rollout_batch(model, env, batch, decoder, args, stats, rng)
            seen += batch_n
            metrics = ROLLOUT.finalize_stats(stats)
            pbar.update(batch_n)
            pbar.set_postfix(
                reward=f"{metrics['avg_cum_reward']:.3f}",
                depth=f"{metrics['avg_step']:.2f}",
                fail=f"{metrics['failure_rate']:.4f}",
                refresh=False,
            )
    pbar.close()
    metrics = ROLLOUT.finalize_stats(stats)
    meta = {
        "args": vars(args),
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "simulator_checkpoint": args.simulator_checkpoint,
        "sid_levels": sid_levels,
        "sid_vocab_size": sid_vocab_size,
        "load_info": load_info,
        "metrics": metrics,
    }
    save_path = Path(args.save_meta)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[done] "
        f"reward={metrics['avg_cum_reward']:.4f} depth={metrics['avg_step']:.3f} "
        f"reward_per_step={metrics['reward_per_step']:.4f} "
        f"neg={metrics['negative_rate']:.4f} fail={metrics['failure_rate']:.4f}"
    )
    print(f"[done] meta saved to {save_path}")


if __name__ == "__main__":
    main()
