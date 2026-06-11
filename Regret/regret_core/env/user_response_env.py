from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from regret_core.data.schema import EVENT_TYPE_TO_ID, ID_TO_EVENT_TYPE
from regret_core.data.transition_dataset import TransitionIterableDataset
from regret_core.model.user_response import RegretUserResponse


class RegretUserResponseEnv:
    """Yambda simulator environment backed by a trained RegretUserResponse.

    The wrapper follows the session-run transition semantics: each policy action
    is treated as one recommended item step, the simulator predicts behavior
    components, derives reward via the model's rule-based reward path, and then
    appends a synthetic step-level feedback signal to the history.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        transition_path: str | Path,
        dense_item_features_npy: str | Path | None = None,
        max_seq_len: int | None = None,
        device: str = "cpu",
        max_step_per_episode: int = 10,
        sample_response: bool = True,
        negative_patience: int = 5,
        reward_done_threshold: float | None = None,
        structured_response: bool = True,
        gate_revision_by_history: bool = True,
        failure_signal_scope: str = "all_failed",
        negative_prob_scale: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model_args = dict(ckpt["model_args"])
        # A simulator should predict behavior components and let the known reward
        # rule compose the scalar reward, not directly hallucinate reward.
        model_args.setdefault("decouple_reward_model", True)
        self.max_seq_len = int(max_seq_len or ckpt["data_args"]["max_seq_len"])
        self.feature_path = str(dense_item_features_npy or ckpt["data_args"]["dense_item_features_npy"])
        self.features = np.load(self.feature_path, mmap_mode="r")
        self.model = RegretUserResponse(**model_args).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.transition_path = transition_path
        self.max_step_per_episode = int(max_step_per_episode)
        self.sample_response = bool(sample_response)
        self.negative_patience = int(negative_patience)
        self.reward_done_threshold = reward_done_threshold
        self.structured_response = bool(structured_response)
        self.gate_revision_by_history = bool(gate_revision_by_history)
        self.failure_signal_scope = str(failure_signal_scope)
        if self.failure_signal_scope not in {"all_failed", "explicit_negative"}:
            raise ValueError(
                "failure_signal_scope must be one of {'all_failed', 'explicit_negative'}"
            )
        self.negative_prob_scale = float(negative_prob_scale)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed))
        self.iter = None
        self.current_observation = None

    def reset_from_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = int(batch["history_ids"].shape[0])
        self.current_observation = {
            "user_id": batch["user_id"].to(self.device),
            "history_ids": batch["history_ids"].to(self.device),
            "history_features": batch["history_features"].to(self.device),
            "history_feedbacks": batch["history_feedbacks"].to(self.device),
            "history_event_type_ids": batch["history_event_type_ids"].to(self.device),
            "history_mask": batch["history_mask"].to(self.device),
            "cummulative_reward": torch.zeros(batch_size, device=self.device),
            "step": torch.zeros(batch_size, device=self.device, dtype=torch.long),
            "consecutive_negative": torch.zeros(batch_size, device=self.device, dtype=torch.long),
        }
        return deepcopy(self.current_observation)

    def reset(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        dataset = TransitionIterableDataset(
            self.transition_path,
            self.feature_path,
            max_seq_len=self.max_seq_len,
            max_rows=0,
        )
        self.iter = iter(DataLoader(dataset, batch_size=batch_size, num_workers=0))
        return self.reset_from_batch(next(self.iter))

    def _bernoulli(self, probs: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp(0.0, 1.0)
        if self.sample_response:
            return torch.bernoulli(probs, generator=self.rng)
        return (probs >= 0.5).float()

    def _sample_play_bucket(self, play_bucket_probs: torch.Tensor) -> torch.Tensor:
        probs = play_bucket_probs.clamp_min(0.0)
        denom = probs.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(probs, 1.0 / max(int(probs.shape[-1]), 1))
        probs = torch.where(denom > 1e-8, probs / denom.clamp_min(1e-8), uniform)
        if self.sample_response:
            return torch.multinomial(probs, num_samples=1, replacement=True, generator=self.rng).squeeze(1)
        return probs.argmax(dim=-1)

    def _sample_categorical(self, probs: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp_min(0.0)
        denom = probs.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(probs, 1.0 / max(int(probs.shape[-1]), 1))
        probs = torch.where(denom > 1e-8, probs / denom.clamp_min(1e-8), uniform)
        if self.sample_response:
            return torch.multinomial(probs, num_samples=1, replacement=True, generator=self.rng).squeeze(1)
        return probs.argmax(dim=-1)

    def _history_has_same_item_event(self, action: torch.Tensor, event_type_id: int) -> torch.Tensor:
        hist_ids = self.current_observation["history_ids"].long()
        hist_event_ids = self.current_observation["history_event_type_ids"].long()
        hist_mask = self.current_observation["history_mask"].float() > 0.0
        same_item = hist_ids == action.long().view(-1, 1)
        same_event = hist_event_ids == int(event_type_id)
        return (same_item & same_event & hist_mask).any(dim=1)

    def _sample_structured_response(
        self,
        action: torch.Tensor,
        listen_prob: torch.Tensor,
        play_bucket_probs: torch.Tensor,
        reward_feedback_probs: torch.Tensor,
        negative_prob: torch.Tensor,
        negative_type_probs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = int(action.shape[0])
        device = action.device
        prior_like = self._history_has_same_item_event(action, EVENT_TYPE_TO_ID["like"])
        prior_dislike = self._history_has_same_item_event(action, EVENT_TYPE_TO_ID["dislike"])

        scaled_negative_prob = (negative_prob * self.negative_prob_scale).clamp(0.0, 1.0)
        explicit_dislike_prob = reward_feedback_probs[:, 1].clamp(0.0, 1.0)
        explicit_unlike_prob = reward_feedback_probs[:, 2].clamp(0.0, 1.0)
        raw_invalid_unlike_mass = explicit_unlike_prob * (~prior_like).float()
        if self.gate_revision_by_history:
            explicit_unlike_prob = torch.where(
                prior_like,
                explicit_unlike_prob,
                torch.zeros_like(explicit_unlike_prob),
            )

        explicit_none_prob = (1.0 - explicit_dislike_prob - explicit_unlike_prob).clamp_min(0.0)
        explicit_event_probs = torch.stack(
            [explicit_none_prob, explicit_dislike_prob, explicit_unlike_prob],
            dim=-1,
        )
        explicit_event_id = self._sample_categorical(explicit_event_probs)
        dislike_mask = explicit_event_id == 1
        unlike_mask = explicit_event_id == 2

        negative_type_probs = negative_type_probs.clone().clamp_min(0.0)
        low_play_prob = (scaled_negative_prob * negative_type_probs[:, 0]).clamp(0.0, 1.0)
        low_play_mask = (self._bernoulli(low_play_prob) > 0.5) & ~(dislike_mask | unlike_mask)

        nonnegative_mask = ~(low_play_mask | dislike_mask | unlike_mask)
        nonnegative_play_probs = play_bucket_probs.clone().clamp_min(0.0)
        nonnegative_play_probs[:, 0:2] = 0.0
        empty_nonnegative = nonnegative_play_probs.sum(dim=-1, keepdim=True) <= 1e-8
        nonnegative_play_probs = torch.where(
            empty_nonnegative,
            torch.tensor([0.0, 0.0, 0.5, 0.5], dtype=nonnegative_play_probs.dtype, device=device).view(1, -1),
            nonnegative_play_probs,
        )
        nonnegative_play_probs = nonnegative_play_probs / nonnegative_play_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        nonnegative_play_bucket = self._sample_play_bucket(nonnegative_play_probs)

        low_play_probs = play_bucket_probs.clone().clamp_min(0.0)
        low_play_probs[:, 2:] = 0.0
        empty_low = low_play_probs.sum(dim=-1, keepdim=True) <= 1e-8
        low_play_probs = torch.where(
            empty_low,
            torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=low_play_probs.dtype, device=device).view(1, -1),
            low_play_probs,
        )
        low_play_bucket = self._sample_play_bucket(low_play_probs)

        listen_sample = torch.zeros(batch_size, dtype=torch.float32, device=device)
        nonnegative_listen = self._bernoulli(listen_prob)
        listen_sample = torch.where(nonnegative_mask, nonnegative_listen, listen_sample)
        listen_sample = torch.where(low_play_mask, torch.ones_like(listen_sample), listen_sample)

        play_bucket_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        play_bucket_id = torch.where(nonnegative_mask, nonnegative_play_bucket, play_bucket_id)
        play_bucket_id = torch.where(low_play_mask, low_play_bucket, play_bucket_id)
        play_values = self.model.play_bucket_values.to(device)[play_bucket_id]
        play_value = play_values * listen_sample

        feedback_samples = torch.zeros(batch_size, 4, dtype=torch.float32, device=device)
        like_context = nonnegative_mask & (listen_sample > 0.5) & (play_bucket_id >= 2)
        like_sample = (self._bernoulli(reward_feedback_probs[:, 0]) > 0.5) & like_context
        undislike_prob = reward_feedback_probs[:, 3]
        raw_invalid_undislike_mass = undislike_prob * (~prior_dislike).float()
        if self.gate_revision_by_history:
            undislike_prob = torch.where(prior_dislike, undislike_prob, torch.zeros_like(undislike_prob))
        undislike_sample = (self._bernoulli(undislike_prob) > 0.5) & nonnegative_mask
        like_sample = like_sample & ~undislike_sample

        feedback_samples[:, 0] = like_sample.float()
        feedback_samples[:, 1] = dislike_mask.float()
        feedback_samples[:, 2] = unlike_mask.float()
        feedback_samples[:, 3] = undislike_sample.float()

        event_type = self._dominant_event_type(listen_sample, feedback_samples)
        failure_type_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        if self.failure_signal_scope == "all_failed":
            failure_type_id = torch.where(low_play_mask, torch.full_like(failure_type_id, 1), failure_type_id)
        failure_type_id = torch.where(dislike_mask, torch.full_like(failure_type_id, 2), failure_type_id)
        failure_type_id = torch.where(unlike_mask, torch.full_like(failure_type_id, 3), failure_type_id)
        invalid_revision_mass = raw_invalid_unlike_mass + raw_invalid_undislike_mass
        valid_revision = torch.ones(batch_size, dtype=torch.bool, device=device)
        valid_revision = torch.where(unlike_mask, prior_like, valid_revision)
        valid_revision = torch.where(undislike_sample, prior_dislike, valid_revision)
        return {
            "listen_sample": listen_sample,
            "play_bucket_id": play_bucket_id,
            "play_value": play_value,
            "feedback_samples": feedback_samples,
            "event_type": event_type,
            "failure_type_id": failure_type_id,
            "valid_revision": valid_revision,
            "invalid_revision_mass": invalid_revision_mass,
            "prior_like": prior_like,
            "prior_dislike": prior_dislike,
        }

    def _dominant_event_type(
        self,
        listen: torch.Tensor,
        feedback_samples: torch.Tensor,
    ) -> torch.Tensor:
        event_type = torch.full_like(listen.long(), EVENT_TYPE_TO_ID["recommend"])
        event_type = torch.where(listen > 0.5, torch.full_like(event_type, EVENT_TYPE_TO_ID["listen"]), event_type)
        # Match session-run split priority: dislike > unlike > like > undislike > listen.
        event_type = torch.where(
            feedback_samples[:, 3] > 0.5,
            torch.full_like(event_type, EVENT_TYPE_TO_ID["undislike"]),
            event_type,
        )
        event_type = torch.where(
            feedback_samples[:, 0] > 0.5,
            torch.full_like(event_type, EVENT_TYPE_TO_ID["like"]),
            event_type,
        )
        event_type = torch.where(
            feedback_samples[:, 2] > 0.5,
            torch.full_like(event_type, EVENT_TYPE_TO_ID["unlike"]),
            event_type,
        )
        event_type = torch.where(
            feedback_samples[:, 1] > 0.5,
            torch.full_like(event_type, EVENT_TYPE_TO_ID["dislike"]),
            event_type,
        )
        return event_type

    def _append_history(
        self,
        action: torch.Tensor,
        action_features: torch.Tensor,
        reward: torch.Tensor,
        event_type: torch.Tensor,
    ) -> None:
        self.current_observation["history_ids"] = torch.cat(
            [self.current_observation["history_ids"], action.unsqueeze(1)],
            dim=1,
        )[:, -self.max_seq_len:]
        self.current_observation["history_features"] = torch.cat(
            [self.current_observation["history_features"], action_features.unsqueeze(1)],
            dim=1,
        )[:, -self.max_seq_len:, :]
        # session-run split stores the step-level reward as history signal.
        self.current_observation["history_feedbacks"] = torch.cat(
            [self.current_observation["history_feedbacks"], reward.unsqueeze(1)],
            dim=1,
        )[:, -self.max_seq_len:]
        self.current_observation["history_event_type_ids"] = torch.cat(
            [self.current_observation["history_event_type_ids"], event_type.unsqueeze(1)],
            dim=1,
        )[:, -self.max_seq_len:]
        self.current_observation["history_mask"] = (self.current_observation["history_ids"] > 0).float()

    def step(self, action: torch.Tensor, action_features: torch.Tensor | None = None):
        if self.current_observation is None:
            raise RuntimeError("Call reset() before step().")
        action = action.to(self.device).long()
        if action.dim() == 2:
            action = action[:, 0]
        if action_features is None:
            action_np = action.detach().cpu().numpy()
            action_features = torch.tensor(self.features[action_np].astype(np.float32), device=self.device)
        else:
            action_features = action_features.to(self.device)
            if action_features.dim() == 3:
                action_features = action_features[:, 0, :]

        batch = {
            "history_features": self.current_observation["history_features"],
            "history_feedbacks": self.current_observation["history_feedbacks"],
            "history_event_type_ids": self.current_observation["history_event_type_ids"],
            "history_mask": self.current_observation["history_mask"],
            "action_features": action_features,
            "prior_stats": torch.zeros(action.shape[0], self.model.prior_dim, device=self.device),
        }
        with torch.no_grad():
            out = self.model(batch)
            listen_prob = out["listen_prob"].detach()
            play_bucket_probs = out["play_bucket_probs"].detach()
            feedback_probs = out["feedback_probs"].detach()
            reward_feedback_probs = out["reward_feedback_probs"].detach()
            positive_gate = out["positive_gate"].detach()
            negative_prob = out["negative_prob"].detach()
            negative_type_probs = out["negative_type_probs"].detach()

        if self.structured_response:
            sampled = self._sample_structured_response(
                action,
                listen_prob,
                play_bucket_probs,
                reward_feedback_probs,
                negative_prob,
                negative_type_probs,
            )
            listen_sample = sampled["listen_sample"]
            play_bucket_id = sampled["play_bucket_id"]
            play_value = sampled["play_value"]
            feedback_samples = sampled["feedback_samples"]
            event_type = sampled["event_type"]
            failure_type_id = sampled["failure_type_id"]
            valid_revision = sampled["valid_revision"]
            invalid_revision_mass = sampled["invalid_revision_mass"]
        else:
            listen_sample = self._bernoulli(listen_prob)
            play_bucket_id = self._sample_play_bucket(play_bucket_probs)
            play_values = self.model.play_bucket_values.to(self.device)[play_bucket_id]
            play_value = play_values * listen_sample
            feedback_samples = self._bernoulli(reward_feedback_probs)
            event_type = self._dominant_event_type(listen_sample, feedback_samples)
            failure_type_id = torch.zeros(action.shape[0], dtype=torch.long, device=self.device)
            valid_revision = torch.ones(action.shape[0], dtype=torch.bool, device=self.device)
            invalid_revision_mass = torch.zeros(action.shape[0], dtype=torch.float32, device=self.device)
        reward = self.model.compose_reward(listen_sample, play_value, feedback_samples).detach()
        if not self.structured_response:
            if self.failure_signal_scope == "explicit_negative":
                failure_type_id = torch.where(
                    feedback_samples[:, 1] > 0.5,
                    torch.full_like(failure_type_id, 2),
                    failure_type_id,
                )
                failure_type_id = torch.where(
                    feedback_samples[:, 2] > 0.5,
                    torch.full_like(failure_type_id, 3),
                    failure_type_id,
                )
            else:
                failure_type_id = torch.where(reward < 0.0, torch.full_like(failure_type_id, 1), failure_type_id)
        expected_reward = out["preds"].detach()
        feedback_signal = self.model.compose_feedback_signal(listen_sample, play_value, feedback_samples).detach()
        self._append_history(action, action_features, reward, event_type)
        self.current_observation["cummulative_reward"] += reward
        self.current_observation["step"] += 1
        if self.failure_signal_scope == "explicit_negative":
            is_negative = (failure_type_id == 2) | (failure_type_id == 3)
        else:
            is_negative = reward < 0.0
        self.current_observation["consecutive_negative"] = torch.where(
            is_negative,
            self.current_observation["consecutive_negative"] + 1,
            torch.zeros_like(self.current_observation["consecutive_negative"]),
        )
        done = self.current_observation["step"] >= self.max_step_per_episode
        if self.negative_patience > 0:
            done = done | (self.current_observation["consecutive_negative"] >= self.negative_patience)
        if self.reward_done_threshold is not None:
            done = done | (self.current_observation["cummulative_reward"] <= float(self.reward_done_threshold))
        return deepcopy(self.current_observation), reward, done, {
            "response": (reward > 0).float(),
            "expected_reward": expected_reward,
            "listen_prob": listen_prob,
            "listen": listen_sample,
            "play_bucket_probs": play_bucket_probs,
            "play_bucket_id": play_bucket_id,
            "play_ratio": play_value,
            "feedback_probs": feedback_probs,
            "reward_feedback_probs": reward_feedback_probs,
            "feedback_samples": feedback_samples,
            "positive_gate": positive_gate,
            "negative_prob": negative_prob,
            "negative_type_probs": negative_type_probs,
            "feedback_signal": feedback_signal,
            "event_type_id": event_type,
            "event_type": [ID_TO_EVENT_TYPE.get(int(x), "unknown") for x in event_type.detach().cpu().tolist()],
            "failure_type_id": failure_type_id,
            "is_failure": (failure_type_id > 0).float(),
            "is_low_play": (failure_type_id == 1).float(),
            "is_dislike": (failure_type_id == 2).float(),
            "is_unlike": (failure_type_id == 3).float(),
            "valid_revision": valid_revision.float(),
            "invalid_revision_mass": invalid_revision_mass,
            "cummulative_reward": self.current_observation["cummulative_reward"].detach().clone(),
            "step": self.current_observation["step"].detach().clone(),
        }

    @staticmethod
    def write_env_meta(checkpoint_path: str | Path, out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"checkpoint_path": str(checkpoint_path)}, indent=2), encoding="utf-8")
