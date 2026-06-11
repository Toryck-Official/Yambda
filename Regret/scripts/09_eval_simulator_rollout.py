#!/usr/bin/env python3
"""主线评估：在 learned simulator 里 rollout，对比 base 策略和 RAPI 策略的 reward/depth/negative rate。"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regret_core.data.schema import EVENT_TYPE_TO_ID  # noqa: E402
from regret_core.data.transition_dataset import TransitionIterableDataset  # noqa: E402
from regret_core.env.user_response_env import RegretUserResponseEnv  # noqa: E402
from regret_core.model.regret_memory import RegretMemoryPool  # noqa: E402
from regret_core.model.sid_policy import RegretAwareSIDPolicy  # noqa: E402


class OfflineTransitionEnvSpec:
    def __init__(self, n_item: int, item_dim: int, max_seq_len: int) -> None:
        self.action_space = {
            "item_id": ("nominal", n_item),
            "item_feature": ("continuous", item_dim, "normal"),
        }
        self.observation_space = {
            "history": ("sequence", max_seq_len, ("continuous", item_dim)),
        }


class SIDDecoder:
    """Decode generated SID paths to valid dense item ids."""

    def __init__(self, dense_item2sid: np.ndarray, sid_vocab_size: int) -> None:
        sid = np.asarray(dense_item2sid)
        self.sid_levels = int(sid.shape[1])
        self.sid_vocab_size = int(sid_vocab_size)
        self.multipliers = (self.sid_vocab_size ** np.arange(self.sid_levels - 1, -1, -1)).astype(np.int64)
        valid = (sid >= 0).all(axis=1) & (sid < self.sid_vocab_size).all(axis=1)
        # Keep dense id 0 as invalid padding even if its SID accidentally looks valid.
        valid[0] = False
        dense_ids = np.nonzero(valid)[0].astype(np.int64)
        codes = self.pack_np(sid[dense_ids].astype(np.int64))
        order = np.argsort(codes, kind="mergesort")
        self.codes = codes[order]
        self.dense_ids = dense_ids[order]
        self.sid = sid

    def pack_np(self, sid_paths: np.ndarray) -> np.ndarray:
        return (sid_paths.astype(np.int64) * self.multipliers.reshape(1, -1)).sum(axis=1)

    def lookup_code(self, code: int) -> int | None:
        idx = int(np.searchsorted(self.codes, np.int64(code), side="left"))
        if idx < self.codes.shape[0] and int(self.codes[idx]) == int(code):
            return int(self.dense_ids[idx])
        return None

    def sid_for_item(self, dense_item_id: int) -> np.ndarray:
        item_id = int(dense_item_id)
        if item_id <= 0 or item_id >= self.sid.shape[0] or (self.sid[item_id] < 0).any():
            item_id = int(self.dense_ids[0])
        return np.asarray(self.sid[item_id], dtype=np.int64)

    def _candidate_penalty(
        self,
        sid_tuple: np.ndarray,
        row_idx: int,
        pool_tokens_np: np.ndarray | None,
        pool_phis_np: np.ndarray | None,
        layer_weights_np: np.ndarray,
    ) -> float:
        if pool_tokens_np is None or pool_phis_np is None:
            return 0.0
        phis = pool_phis_np[row_idx]
        active = phis > 0.0
        if not bool(active.any()):
            return 0.0
        tokens = pool_tokens_np[row_idx, active, : self.sid_levels]
        weights = layer_weights_np[: self.sid_levels].reshape(1, -1)
        matches = (tokens == sid_tuple.reshape(1, -1)).astype(np.float32)
        similarity = (matches * weights).sum(axis=1)
        return float((similarity * phis[active]).sum())

    def decode_logits(
        self,
        logits_list: list[torch.Tensor],
        fallback_items: torch.Tensor,
        top_k: int,
        device: torch.device,
        mode: str,
        rng: np.random.Generator,
        temperature: float,
        pool_tokens: torch.Tensor | None = None,
        pool_phis: torch.Tensor | None = None,
        candidate_rerank: bool = False,
        candidate_eta: float = 1.0,
        candidate_layer_weights: list[float] | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        log_probs = [F.log_softmax(logits.detach(), dim=-1).cpu() for logits in logits_list]
        k = int(max(1, min(top_k, log_probs[0].shape[-1])))
        top_values: list[torch.Tensor] = []
        top_indices: list[torch.Tensor] = []
        for level_log_probs in log_probs:
            values, indices = torch.topk(level_log_probs, k=k, dim=-1)
            top_values.append(values)
            top_indices.append(indices)

        batch_size = int(log_probs[0].shape[0])
        out_items: list[int] = []
        out_sids: list[np.ndarray] = []
        details: list[dict[str, Any]] = []
        fallback_np = fallback_items.detach().cpu().numpy().astype(np.int64)
        use_candidate_rerank = bool(candidate_rerank) and pool_tokens is not None and pool_phis is not None
        pool_tokens_np = None
        pool_phis_np = None
        if use_candidate_rerank:
            pool_tokens_np = pool_tokens.detach().cpu().numpy().astype(np.int64)
            pool_phis_np = pool_phis.detach().cpu().numpy().astype(np.float32)
        weights = candidate_layer_weights or [1.0] * self.sid_levels
        if len(weights) < self.sid_levels:
            weights = weights + [weights[-1] if weights else 1.0] * (self.sid_levels - len(weights))
        layer_weights_np = np.asarray(weights[: self.sid_levels], dtype=np.float32)
        eta = max(0.0, float(candidate_eta))
        for row_idx in range(batch_size):
            candidates: list[tuple[int, np.ndarray, float, float, float]] = []
            level_ranges = [range(k) for _ in range(self.sid_levels)]
            for combo in itertools.product(*level_ranges):
                sid_tuple = np.asarray(
                    [int(top_indices[level][row_idx, combo[level]]) for level in range(self.sid_levels)],
                    dtype=np.int64,
                )
                code = int((sid_tuple * self.multipliers).sum())
                dense_item_id = self.lookup_code(code)
                if dense_item_id is None:
                    continue
                score = float(sum(float(top_values[level][row_idx, combo[level]]) for level in range(self.sid_levels)))
                penalty = self._candidate_penalty(
                    sid_tuple,
                    row_idx,
                    pool_tokens_np,
                    pool_phis_np,
                    layer_weights_np,
                )
                adjusted_score = score - eta * penalty
                candidates.append((dense_item_id, sid_tuple, score, penalty, adjusted_score))
            if not candidates:
                best_item = int(fallback_np[row_idx])
                best_sid = self.sid_for_item(best_item)
                best_score = 0.0
                best_penalty = 0.0
                best_adjusted = 0.0
            elif mode == "sample":
                scores = np.asarray([item[4] for item in candidates], dtype=np.float64)
                temp = max(float(temperature), 1e-6)
                probs = np.exp((scores - scores.max()) / temp)
                probs = probs / probs.sum()
                chosen = int(rng.choice(len(candidates), p=probs))
                best_item = int(candidates[chosen][0])
                best_sid = candidates[chosen][1]
                best_score = float(candidates[chosen][2])
                best_penalty = float(candidates[chosen][3])
                best_adjusted = float(candidates[chosen][4])
            else:
                best_item, best_sid, best_score, best_penalty, best_adjusted = max(candidates, key=lambda item: item[4])
            out_items.append(int(best_item))
            out_sids.append(best_sid.astype(np.int64))
            if return_details:
                details.append(
                    {
                        "candidate_count": len(candidates),
                        "score": best_score,
                        "candidate_penalty": best_penalty,
                        "adjusted_score": best_adjusted,
                    }
                )
        result = (
            torch.tensor(out_items, dtype=torch.long, device=device),
            torch.tensor(np.stack(out_sids, axis=0), dtype=torch.long, device=device),
        )
        if return_details:
            return result[0], result[1], details
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base vs RAPI actor rollout in a learned simulator")
    parser.add_argument("--transition_root", default=str(PROJECT_ROOT / "artifacts/transitions/raw_rqkmeans_v2_session_run_1h_span6h_run100_replay50"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "replay_val", "replay_test"])
    parser.add_argument("--item_features_npy", default=str(PROJECT_ROOT / "artifacts/mappings/raw_rqkmeans/dense_item_features.npy"))
    parser.add_argument("--dense_item2sid_npy", default=str(PROJECT_ROOT / "artifacts/mappings/raw_rqkmeans/dense_item2sid.npy"))
    parser.add_argument("--actor_checkpoint", default=str(PROJECT_ROOT / "artifacts/models/regret_sid_session_run_v2_actor"))
    parser.add_argument("--simulator_checkpoint", default=str(PROJECT_ROOT / "artifacts/user_response/raw_rqkmeans_v2_session_run_simulator_decoupled_1m_e3/regret_user_response.pt"))
    parser.add_argument("--save_meta", default=str(PROJECT_ROOT / "artifacts/models/simulator_rollout_session_run.meta.json"))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--read_batch_size", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--sample_response", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--structured_simulator_response",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample mutually exclusive simulator outcomes and mask invalid revision events before writing history.",
    )
    parser.add_argument("--simulator_negative_prob_scale", type=float, default=1.0)
    parser.add_argument(
        "--failure_signal_scope",
        default="all_failed",
        choices=["all_failed", "explicit_negative"],
        help=(
            "Controls what counts as env failure and negative_patience. "
            "explicit_negative ignores low_play for failure/done and only counts dislike/valid unlike."
        ),
    )
    parser.add_argument("--negative_patience", type=int, default=5)
    parser.add_argument("--reward_done_threshold", type=float, default=None)
    parser.add_argument(
        "--eval_reward_mode",
        default="paper_effective",
        choices=["simulator", "paper", "paper_effective"],
        help="Reward used for reported metrics. simulator uses the learned env reward; paper matches policy training base reward; paper_effective adds valid revision psi.",
    )
    parser.add_argument("--reward_w_listen", type=float, default=1.0)
    parser.add_argument("--reward_w_like", type=float, default=1.0)
    parser.add_argument("--reward_w_dislike", type=float, default=1.0)
    parser.add_argument("--rrca_unlike_weight", type=float, default=1.0)
    parser.add_argument("--rrca_undislike_weight", type=float, default=1.0)
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
    parser.add_argument("--sara_eta", type=float, default=0.5)
    parser.add_argument("--sara_layer_weights", type=str, default="0.05,0.25,0.70")
    parser.add_argument(
        "--rapi_candidate_rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply RAPI again at SIDDecoder candidate level so memory can change the final item ranking.",
    )
    parser.add_argument(
        "--rapi_candidate_eta",
        type=float,
        default=1.0,
        help="Penalty weight for candidate-level RAPI reranking after valid SID candidates are generated.",
    )
    parser.add_argument("--regret_pool_size", type=int, default=20)
    parser.add_argument("--regret_gamma", type=float, default=0.9)
    parser.add_argument("--regret_phi_scale", type=float, default=1.0)
    parser.add_argument("--regret_phi_clip", type=float, default=2.0)
    parser.add_argument("--regret_reward_threshold", type=float, default=0.0)
    parser.add_argument("--use_precomputed_regret_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--history_memory_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--memory_signal_scope",
        default="all_failed",
        choices=["all_failed", "explicit_negative", "revision_only", "rrca", "none"],
        help=(
            "Controls which feedback enters the RAPI B_rev pool during evaluation. "
            "explicit_negative stores dislike and valid unlike only."
        ),
    )
    parser.add_argument(
        "--gate_revision_by_history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only let revision events affect RAPI memory if the same item has the corresponding prior like/dislike in history.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


def infer_sid_spec(dense_item2sid: np.ndarray) -> tuple[int, int]:
    valid = dense_item2sid[1:]
    sid_levels = int(valid.shape[1])
    vocab_sizes = [int(valid[:, i].max()) + 1 for i in range(sid_levels)]
    if len(set(vocab_sizes)) != 1:
        raise ValueError(f"Inconsistent per-level SID vocab sizes: {vocab_sizes}")
    return sid_levels, int(vocab_sizes[0])


def build_actor(args: argparse.Namespace, dense_item2sid: np.ndarray, item_dim: int, device: torch.device) -> RegretAwareSIDPolicy:
    sid_levels, sid_vocab_size = infer_sid_spec(dense_item2sid)
    env = OfflineTransitionEnvSpec(
        n_item=int(dense_item2sid.shape[0] - 1),
        item_dim=int(item_dim),
        max_seq_len=args.max_seq_len,
    )
    model_args = SimpleNamespace(
        sasrec_n_layer=args.sasrec_n_layer,
        sasrec_d_model=args.sasrec_d_model,
        sasrec_d_forward=args.sasrec_d_forward,
        sasrec_n_head=args.sasrec_n_head,
        sasrec_dropout=args.sasrec_dropout,
        sid_levels=sid_levels,
        sid_vocab_sizes=sid_vocab_size,
        sid_temp=args.sid_temp,
        use_history_feedback=args.use_history_feedback,
        use_history_event_type=args.use_history_event_type,
        event_type_vocab_size=args.event_type_vocab_size,
        sara_eta=args.sara_eta,
        sara_layer_weights=args.sara_layer_weights,
    )
    actor = RegretAwareSIDPolicy(model_args, env).to(device)
    state_dict = torch.load(args.actor_checkpoint, map_location=device, weights_only=False)
    missing, unexpected = actor.load_state_dict(state_dict, strict=False)
    print(f"[actor] loaded={args.actor_checkpoint} missing={len(missing)} unexpected={len(unexpected)}")
    actor.eval()
    return actor


def make_loader(args: argparse.Namespace) -> DataLoader:
    dataset = TransitionIterableDataset(
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
    fallback = hist.gather(1, counts.view(-1, 1)).squeeze(1)
    return fallback.clamp_min(1)


def allowed_memory_type_ids(memory_signal_scope: str) -> set[int]:
    scope = str(memory_signal_scope)
    if scope == "none":
        return set()
    if scope in {"revision_only", "rrca"}:
        return {3}
    if scope == "explicit_negative":
        return {2, 3}
    return {1, 2, 3}


def init_memory_from_history(
    memory: RegretMemoryPool,
    obs: dict[str, torch.Tensor],
    dense_item2sid: np.ndarray,
    device: torch.device,
    memory_signal_scope: str,
) -> int:
    allowed_types = allowed_memory_type_ids(memory_signal_scope)
    if not allowed_types:
        return 0
    user_ids = obs["user_id"].detach().cpu().numpy().astype(np.int64)
    hist_ids = obs["history_ids"].detach().cpu().numpy().astype(np.int64)
    feedbacks = obs["history_feedbacks"].detach().cpu().numpy().astype(np.float32)
    event_ids = obs["history_event_type_ids"].detach().cpu().numpy().astype(np.int64)
    masks = obs["history_mask"].detach().cpu().numpy() > 0
    inserted = 0
    for row_idx, user_id in enumerate(user_ids):
        for col_idx in range(hist_ids.shape[1]):
            if not masks[row_idx, col_idx]:
                continue
            item_id = int(hist_ids[row_idx, col_idx])
            if item_id <= 0 or item_id >= dense_item2sid.shape[0]:
                continue
            event_id = int(event_ids[row_idx, col_idx])
            reward = float(feedbacks[row_idx, col_idx])
            if event_id == EVENT_TYPE_TO_ID["dislike"]:
                regret_type_id, strength = 2, 1.2
            elif event_id == EVENT_TYPE_TO_ID["unlike"]:
                regret_type_id, strength = 3, 0.6
            elif reward < 0.0:
                regret_type_id, strength = 1, max(1e-6, -reward)
            else:
                continue
            if regret_type_id not in allowed_types:
                continue
            sid_path = torch.tensor(np.asarray(dense_item2sid[item_id], dtype=np.int64), dtype=torch.long, device=device)
            if bool((sid_path < 0).any()):
                continue
            inserted += int(
                memory.update(
                    user_id=int(user_id),
                    sid_path=sid_path,
                    regret_type_id=regret_type_id,
                    regret_strength=float(strength),
                    reward=reward,
                    item_id=item_id,
                )
            )
    return int(inserted)


def init_memory_from_snapshot(
    memory: RegretMemoryPool,
    batch: dict[str, torch.Tensor],
    dense_item2sid: np.ndarray,
    device: torch.device,
    memory_signal_scope: str,
) -> int:
    item_ids = batch.get("regret_memory_item_ids")
    phis = batch.get("regret_memory_phis")
    type_ids = batch.get("regret_memory_type_ids")
    if item_ids is None or phis is None or type_ids is None:
        return 0
    allowed_types = allowed_memory_type_ids(memory_signal_scope)
    if not allowed_types:
        return 0
    item_ids = item_ids.to(device=device).long()
    phis = phis.to(device=device).float()
    type_ids = type_ids.to(device=device).long()
    if not bool((item_ids > 0).any() and (phis > 0.0).any()):
        return 0
    user_ids = batch["user_id"].detach().cpu().numpy().astype(np.int64)
    sid_levels = memory.sid_levels
    loaded = 0
    for row_idx, user_id in enumerate(user_ids):
        row_item_ids = item_ids[row_idx]
        row_phis = phis[row_idx]
        row_type_ids = type_ids[row_idx]
        allowed_mask = torch.zeros_like(row_type_ids, dtype=torch.bool)
        for type_id in allowed_types:
            allowed_mask = allowed_mask | (row_type_ids == int(type_id))
        row_phis = torch.where(allowed_mask, row_phis, torch.zeros_like(row_phis))
        row_sid_paths = torch.zeros(
            row_item_ids.shape[0],
            sid_levels,
            dtype=torch.long,
            device=device,
        )
        valid_items = row_item_ids.detach().cpu().numpy().astype(np.int64)
        for col_idx, item_id in enumerate(valid_items):
            if item_id <= 0 or item_id >= dense_item2sid.shape[0]:
                continue
            sid_path = np.asarray(dense_item2sid[int(item_id)], dtype=np.int64)
            if (sid_path < 0).any():
                continue
            row_sid_paths[col_idx] = torch.tensor(sid_path[:sid_levels], dtype=torch.long, device=device)
        loaded += memory.load_snapshot(
            user_id=int(user_id),
            sid_paths=row_sid_paths,
            phis=row_phis,
            regret_type_ids=row_type_ids,
            item_ids=row_item_ids,
        )
    return int(loaded)


def update_memory_from_simulated_step(
    memory: RegretMemoryPool,
    state_obs: dict[str, torch.Tensor],
    user_ids: torch.Tensor,
    sid_paths: torch.Tensor,
    action_items: torch.Tensor,
    reward: torch.Tensor,
    info: dict[str, Any],
    active_mask: torch.Tensor,
    gate_revision_by_history: bool,
    memory_signal_scope: str,
) -> None:
    allowed_types = allowed_memory_type_ids(memory_signal_scope)
    if not allowed_types:
        return
    event_ids = info["event_type_id"].detach().cpu().numpy().astype(np.int64)
    rewards = reward.detach().cpu().numpy().astype(np.float32)
    failure_type_ids = info.get("failure_type_id")
    failure_type_np = (
        failure_type_ids.detach().cpu().numpy().astype(np.int64)
        if failure_type_ids is not None
        else None
    )
    users = user_ids.detach().cpu().numpy().astype(np.int64)
    actions = action_items.detach().cpu().numpy().astype(np.int64)
    prior_like_mask = history_has_same_item_event(state_obs, action_items, EVENT_TYPE_TO_ID["like"])
    for row_idx in torch.nonzero(active_mask.detach().cpu(), as_tuple=False).view(-1).tolist():
        event_id = int(event_ids[row_idx])
        r = float(rewards[row_idx])
        failure_type = int(failure_type_np[row_idx]) if failure_type_np is not None else 0
        if failure_type == 1:
            regret_type_id, strength = 1, max(1e-6, -r) if r < 0.0 else 1.0
        elif failure_type == 2 or event_id == EVENT_TYPE_TO_ID["dislike"]:
            regret_type_id, strength = 2, 1.2
        elif failure_type == 3 or event_id == EVENT_TYPE_TO_ID["unlike"]:
            if gate_revision_by_history and not bool(prior_like_mask[row_idx].detach().cpu()):
                continue
            regret_type_id, strength = 3, 0.6
        elif r < 0.0:
            regret_type_id, strength = 1, max(1e-6, -r)
        else:
            continue
        if regret_type_id not in allowed_types:
            continue
        memory.update(
            user_id=int(users[row_idx]),
            sid_path=sid_paths[row_idx],
            regret_type_id=regret_type_id,
            regret_strength=float(strength),
            reward=r,
            item_id=int(actions[row_idx]),
        )


def history_has_same_item_event(
    obs: dict[str, torch.Tensor],
    action_items: torch.Tensor,
    event_type_id: int,
) -> torch.Tensor:
    hist_ids = obs["history_ids"].long()
    hist_event_ids = obs["history_event_type_ids"].long()
    hist_mask = obs["history_mask"].float() > 0.0
    same_item = hist_ids == action_items.long().view(-1, 1)
    same_event = hist_event_ids == int(event_type_id)
    return (same_item & same_event & hist_mask).any(dim=1)


def compute_eval_reward(raw_reward: torch.Tensor, info: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    if args.eval_reward_mode == "simulator":
        return raw_reward.detach()
    feedback = info["feedback_samples"].float()
    reward = (
        float(getattr(args, "reward_w_listen", 1.0)) * info["play_ratio"].float()
        + float(getattr(args, "reward_w_like", 1.0)) * feedback[:, 0]
        - float(getattr(args, "reward_w_dislike", 1.0)) * feedback[:, 1]
    )
    if args.eval_reward_mode == "paper_effective":
        psi = (
            -float(getattr(args, "rrca_unlike_weight", 1.0)) * feedback[:, 2]
            + float(getattr(args, "rrca_undislike_weight", 1.0)) * feedback[:, 3]
        )
        valid_revision = info.get("valid_revision")
        if valid_revision is not None:
            psi = psi * valid_revision.float()
        reward = reward + psi
    return reward.detach()


def new_stats() -> dict[str, float]:
    return {
        "episodes": 0.0,
        "episode_reward_sum": 0.0,
        "episode_step_sum": 0.0,
        "early_stop": 0.0,
        "step_n": 0.0,
        "reward_sum": 0.0,
        "negative_steps": 0.0,
        "positive_steps": 0.0,
        "listen_steps": 0.0,
        "play_sum": 0.0,
        "like_steps": 0.0,
        "dislike_steps": 0.0,
        "unlike_steps": 0.0,
        "undislike_steps": 0.0,
        "failure_steps": 0.0,
        "failure_low_play_steps": 0.0,
        "failure_dislike_steps": 0.0,
        "failure_unlike_steps": 0.0,
        "invalid_revision_mass": 0.0,
        "raw_reward_sum": 0.0,
        "raw_negative_steps": 0.0,
        "memory_init_batches": 0.0,
        "memory_init_snapshot_entries": 0.0,
        "memory_init_history_entries": 0.0,
        "memory_init_active_users": 0.0,
        "memory_init_entries": 0.0,
    }


def update_step_stats(
    stats: dict[str, float],
    reward: torch.Tensor,
    info: dict[str, Any],
    active: torch.Tensor,
    raw_reward: torch.Tensor | None = None,
) -> None:
    n = float(active.float().sum().item())
    if n <= 0:
        return
    reward_active = reward[active]
    raw_reward_active = reward_active if raw_reward is None else raw_reward[active]
    feedback = info["feedback_samples"][active]
    stats["step_n"] += n
    stats["reward_sum"] += float(reward_active.sum().detach().cpu())
    stats["raw_reward_sum"] += float(raw_reward_active.sum().detach().cpu())
    stats["negative_steps"] += float((reward_active < 0.0).float().sum().detach().cpu())
    stats["raw_negative_steps"] += float((raw_reward_active < 0.0).float().sum().detach().cpu())
    stats["positive_steps"] += float((reward_active > 0.0).float().sum().detach().cpu())
    stats["listen_steps"] += float(info["listen"][active].sum().detach().cpu())
    stats["play_sum"] += float(info["play_ratio"][active].sum().detach().cpu())
    stats["like_steps"] += float(feedback[:, 0].sum().detach().cpu())
    stats["dislike_steps"] += float(feedback[:, 1].sum().detach().cpu())
    stats["unlike_steps"] += float(feedback[:, 2].sum().detach().cpu())
    stats["undislike_steps"] += float(feedback[:, 3].sum().detach().cpu())
    failure_type = info.get("failure_type_id")
    if failure_type is not None:
        failure_type = failure_type[active].detach().long()
        stats["failure_steps"] += float((failure_type > 0).float().sum().detach().cpu())
        stats["failure_low_play_steps"] += float((failure_type == 1).float().sum().detach().cpu())
        stats["failure_dislike_steps"] += float((failure_type == 2).float().sum().detach().cpu())
        stats["failure_unlike_steps"] += float((failure_type == 3).float().sum().detach().cpu())
    invalid_revision_mass = info.get("invalid_revision_mass")
    if invalid_revision_mass is not None:
        stats["invalid_revision_mass"] += float(invalid_revision_mass[active].sum().detach().cpu())


def finish_episode_stats(stats: dict[str, float], episode_rewards: torch.Tensor, episode_steps: torch.Tensor, max_steps: int) -> None:
    steps = episode_steps.detach().cpu().float()
    rewards = episode_rewards.detach().cpu().float()
    stats["episodes"] += float(steps.numel())
    stats["episode_reward_sum"] += float(rewards.sum())
    stats["episode_step_sum"] += float(steps.sum())
    stats["early_stop"] += float((steps < int(max_steps)).float().sum())


def finalize_stats(stats: dict[str, float]) -> dict[str, float]:
    episodes = max(stats["episodes"], 1.0)
    step_n = max(stats["step_n"], 1.0)
    return {
        **stats,
        "avg_cum_reward": stats["episode_reward_sum"] / episodes,
        "avg_step": stats["episode_step_sum"] / episodes,
        "early_stop_rate": stats["early_stop"] / episodes,
        "reward_per_step": stats["reward_sum"] / step_n,
        "negative_rate": stats["negative_steps"] / step_n,
        "positive_rate": stats["positive_steps"] / step_n,
        "listen_rate": stats["listen_steps"] / step_n,
        "mean_play_ratio": stats["play_sum"] / step_n,
        "like_rate": stats["like_steps"] / step_n,
        "dislike_rate": stats["dislike_steps"] / step_n,
        "unlike_rate": stats["unlike_steps"] / step_n,
        "undislike_rate": stats["undislike_steps"] / step_n,
        "failure_rate": stats["failure_steps"] / step_n,
        "failure_low_play_rate": stats["failure_low_play_steps"] / step_n,
        "failure_dislike_rate": stats["failure_dislike_steps"] / step_n,
        "failure_unlike_rate": stats["failure_unlike_steps"] / step_n,
        "invalid_revision_mass_per_step": stats["invalid_revision_mass"] / step_n,
        "raw_reward_per_step": stats["raw_reward_sum"] / step_n,
        "raw_negative_rate": stats["raw_negative_steps"] / step_n,
    }


def select_actions(
    actor: RegretAwareSIDPolicy,
    obs: dict[str, torch.Tensor],
    decoder: SIDDecoder,
    top_k: int,
    action_mode: str,
    action_temperature: float,
    rng: np.random.Generator,
    sara_layer_weights: str,
    rapi_candidate_rerank: bool,
    rapi_candidate_eta: float,
    memory: RegretMemoryPool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    feed = actor_obs(obs)
    pool_tokens = None
    pool_phis = None
    if memory is None:
        out = actor(feed)
    else:
        pool_tokens, pool_phis = memory.get(obs["user_id"].long(), obs["history_features"].device)
        out = actor.get_sara_logits(feed, pool_tokens=pool_tokens, pool_phis=pool_phis)
    return decoder.decode_logits(
        out["sid_logits"],
        fallback_items=fallback_items_from_history(obs),
        top_k=top_k,
        device=obs["history_features"].device,
        mode=action_mode,
        rng=rng,
        temperature=action_temperature,
        pool_tokens=pool_tokens,
        pool_phis=pool_phis,
        candidate_rerank=memory is not None and bool(rapi_candidate_rerank),
        candidate_eta=float(rapi_candidate_eta),
        candidate_layer_weights=[float(x) for x in str(sara_layer_weights).split(",") if x],
    )


def rollout_one_policy_batch(
    actor: RegretAwareSIDPolicy,
    env: RegretUserResponseEnv,
    batch: dict[str, torch.Tensor],
    decoder: SIDDecoder,
    args: argparse.Namespace,
    dense_item2sid: np.ndarray,
    use_rapi: bool,
    stats: dict[str, float],
    pbar: tqdm,
    rng: np.random.Generator,
) -> None:
    obs = env.reset_from_batch(batch)
    batch_size = int(obs["history_ids"].shape[0])
    active = torch.ones(batch_size, dtype=torch.bool, device=env.device)
    episode_rewards = torch.zeros(batch_size, dtype=torch.float32, device=env.device)
    episode_steps = torch.zeros(batch_size, dtype=torch.float32, device=env.device)
    memory: RegretMemoryPool | None = None
    if use_rapi:
        sid_levels, _ = infer_sid_spec(dense_item2sid)
        memory = RegretMemoryPool(
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
            loaded_snapshot = init_memory_from_snapshot(
                memory,
                batch,
                dense_item2sid,
                env.device,
                args.memory_signal_scope,
            )
        if loaded_snapshot <= 0 and args.history_memory_fallback:
            loaded_history = init_memory_from_history(
                memory,
                obs,
                dense_item2sid,
                env.device,
                args.memory_signal_scope,
            )
        summary = memory.summary()
        stats["memory_init_batches"] += 1.0
        stats["memory_init_snapshot_entries"] += float(loaded_snapshot)
        stats["memory_init_history_entries"] += float(loaded_history)
        stats["memory_init_active_users"] += float(summary["active_users"])
        stats["memory_init_entries"] += float(summary["entries"])

    for _ in range(int(args.max_steps)):
        if not bool(active.any()):
            break
        state_obs = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in env.current_observation.items()
        }
        action_items, sid_paths = select_actions(
            actor,
            state_obs,
            decoder,
            args.decode_top_k,
            args.action_mode,
            args.action_temperature,
            rng,
            args.sara_layer_weights,
            args.rapi_candidate_rerank,
            args.rapi_candidate_eta,
            memory,
        )
        _, raw_reward, done, info = env.step(action_items)
        eval_reward = compute_eval_reward(raw_reward, info, args)
        update_step_stats(stats, eval_reward, info, active, raw_reward=raw_reward)
        episode_rewards = episode_rewards + torch.where(active, eval_reward, torch.zeros_like(eval_reward))
        episode_steps = episode_steps + active.float()
        if use_rapi and memory is not None:
            update_memory_from_simulated_step(
                memory,
                state_obs,
                env.current_observation["user_id"].long(),
                sid_paths,
                action_items,
                raw_reward,
                info,
                active,
                bool(args.gate_revision_by_history),
                args.memory_signal_scope,
            )
        active = active & ~done.bool()
        pbar.update(int(batch_size))
    finish_episode_stats(stats, episode_rewards, episode_steps, int(args.max_steps))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
    sid_levels, sid_vocab_size = infer_sid_spec(dense_item2sid)
    feature_shape = np.load(args.item_features_npy, mmap_mode="r").shape
    print(f"[sid] levels={sid_levels} vocab={sid_vocab_size}; building decoder...")
    decoder = SIDDecoder(dense_item2sid, sid_vocab_size)
    print(f"[sid] decoder valid_items={decoder.dense_ids.shape[0]}")
    actor = build_actor(args, dense_item2sid, int(feature_shape[1]), device)
    loader = make_loader(args)
    transition_path = Path(args.transition_root) / args.split
    base_env = RegretUserResponseEnv(
        checkpoint_path=args.simulator_checkpoint,
        transition_path=transition_path,
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
    rapi_env = RegretUserResponseEnv(
        checkpoint_path=args.simulator_checkpoint,
        transition_path=transition_path,
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

    base_stats = new_stats()
    rapi_stats = new_stats()
    base_rng = np.random.default_rng(int(args.seed))
    rapi_rng = np.random.default_rng(int(args.seed))
    total_updates = int(args.num_episodes) * int(args.max_steps) * 2
    pbar = tqdm(total=total_updates, desc="[rollout]", ncols=120)
    seen = 0
    for batch in loader:
        batch_n = int(batch["history_ids"].shape[0])
        if seen >= int(args.num_episodes):
            break
        if seen + batch_n > int(args.num_episodes):
            keep = int(args.num_episodes) - seen
            batch = {key: value[:keep] if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            batch_n = keep
        rollout_one_policy_batch(actor, base_env, batch, decoder, args, dense_item2sid, False, base_stats, pbar, base_rng)
        rollout_one_policy_batch(actor, rapi_env, batch, decoder, args, dense_item2sid, True, rapi_stats, pbar, rapi_rng)
        seen += batch_n
        pbar.set_postfix(
            base_r=finalize_stats(base_stats)["avg_cum_reward"],
            rapi_r=finalize_stats(rapi_stats)["avg_cum_reward"],
            base_step=finalize_stats(base_stats)["avg_step"],
            rapi_step=finalize_stats(rapi_stats)["avg_step"],
        )
    pbar.close()

    base = finalize_stats(base_stats)
    rapi = finalize_stats(rapi_stats)
    delta = {
        key: float(rapi.get(key, 0.0) - base.get(key, 0.0))
        for key in [
            "avg_cum_reward",
            "avg_step",
            "reward_per_step",
            "negative_rate",
            "positive_rate",
            "early_stop_rate",
            "listen_rate",
            "mean_play_ratio",
            "failure_rate",
            "failure_low_play_rate",
            "failure_dislike_rate",
            "failure_unlike_rate",
            "invalid_revision_mass_per_step",
        ]
    }
    meta = {
        "args": vars(args),
        "base": base,
        "rapi": rapi,
        "delta_rapi_minus_base": delta,
    }
    Path(args.save_meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        "[done] "
        f"base_reward={base['avg_cum_reward']:.4f} rapi_reward={rapi['avg_cum_reward']:.4f} "
        f"delta_reward={delta['avg_cum_reward']:.4f} "
        f"base_step={base['avg_step']:.3f} rapi_step={rapi['avg_step']:.3f} "
        f"delta_step={delta['avg_step']:.3f} "
        f"base_neg={base['negative_rate']:.4f} rapi_neg={rapi['negative_rate']:.4f} "
        f"base_fail={base['failure_rate']:.4f} rapi_fail={rapi['failure_rate']:.4f}"
    )
    print(f"[done] meta saved to {args.save_meta}")


if __name__ == "__main__":
    main()
