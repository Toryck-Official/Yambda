from __future__ import annotations

import torch
import torch.nn as nn


class SoftStateBuilder(nn.Module):
    def __init__(
        self,
        item_dim: int = 128,
        d_model: int = 128,
        response_classes: int = 5,
        regret_classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.item_map = nn.Linear(int(item_dim), int(d_model))
        self.response_emb = nn.Embedding(int(response_classes), int(d_model))
        self.regret_map = nn.Linear(int(regret_classes), int(d_model), bias=False)
        self.reward_map = nn.Linear(1, int(d_model), bias=False)
        self.play_map = nn.Linear(1, int(d_model), bias=False)
        self.source_emb = nn.Parameter(torch.zeros(int(d_model)))
        self.token_norm = nn.LayerNorm(int(d_model))
        self.state_norm = nn.LayerNorm(int(d_model))
        self.drop = nn.Dropout(float(dropout))
        self.update = nn.Sequential(
            nn.Linear(int(d_model) * 2, int(d_model) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model) * 2, int(d_model)),
        )

    def _flat_forward(
        self,
        state_emb: torch.Tensor,
        action_features: torch.Tensor,
        response_probs: torch.Tensor,
        predicted_play_ratio: torch.Tensor,
        predicted_reward: torch.Tensor,
        regret_probs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        response_token = response_probs @ self.response_emb.weight
        token = (
            self.item_map(action_features)
            + response_token
            + self.regret_map(regret_probs)
            + self.reward_map(predicted_reward.unsqueeze(-1))
            + self.play_map(predicted_play_ratio.unsqueeze(-1))
            + self.source_emb.view(1, -1)
        )
        soft_token = self.token_norm(self.drop(token))
        delta = self.update(torch.cat([state_emb, soft_token], dim=-1))
        next_state = self.state_norm(state_emb + delta)
        return {"soft_next_token": soft_token, "next_state_emb": next_state}

    def forward(
        self,
        state_emb: torch.Tensor,
        action_features: torch.Tensor,
        pred: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if action_features.dim() == 2:
            return self._flat_forward(
                state_emb,
                action_features,
                pred["response_probs"],
                pred["predicted_play_ratio"],
                pred["predicted_reward"],
                pred["regret_probs"],
            )

        batch_size, candidate_k, item_dim = action_features.shape
        state_flat = state_emb.unsqueeze(1).expand(batch_size, candidate_k, -1).reshape(batch_size * candidate_k, -1)
        action_flat = action_features.reshape(batch_size * candidate_k, item_dim)
        pred_flat = {
            "response_probs": pred["response_probs"].reshape(batch_size * candidate_k, -1),
            "predicted_play_ratio": pred["predicted_play_ratio"].reshape(batch_size * candidate_k),
            "predicted_reward": pred["predicted_reward"].reshape(batch_size * candidate_k),
            "regret_probs": pred["regret_probs"].reshape(batch_size * candidate_k, -1),
        }
        out = self._flat_forward(
            state_flat,
            action_flat,
            pred_flat["response_probs"],
            pred_flat["predicted_play_ratio"],
            pred_flat["predicted_reward"],
            pred_flat["regret_probs"],
        )
        return {
            "soft_next_token": out["soft_next_token"].reshape(batch_size, candidate_k, -1),
            "next_state_emb": out["next_state_emb"].reshape(batch_size, candidate_k, -1),
        }
