from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    def __init__(self, d_model: int = 128, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(d_model), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.LayerNorm(int(hidden_dim) // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) // 2, 1),
        )

    def forward(self, state_emb: torch.Tensor) -> torch.Tensor:
        return self.net(state_emb).squeeze(-1)


class CandidateScorer(nn.Module):
    def __init__(self, alpha: float = 1.0, eta: float = 0.2, gamma: float = 0.9) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.eta = nn.Parameter(torch.tensor(float(eta)))
        self.gamma = float(gamma)

    @staticmethod
    def normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def forward(
        self,
        hpn_logit: torch.Tensor,
        predicted_reward: torch.Tensor,
        next_value: torch.Tensor | None = None,
        regret_probs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        future_score = predicted_reward
        if next_value is not None:
            future_score = future_score + self.gamma * next_value
        if regret_probs is not None:
            regret_risk = regret_probs[..., 1:].sum(dim=-1)
            future_score = future_score - torch.relu(self.eta) * regret_risk
        final_logit = self.normalize(hpn_logit) + torch.relu(self.alpha) * self.normalize(future_score)
        return {"future_score": future_score, "final_logit": final_logit}


def bellman_value_loss(
    state_value: torch.Tensor,
    predicted_reward: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.9,
    candidate_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    target_q = predicted_reward + float(gamma) * next_value
    if candidate_mask is not None:
        target_q = target_q.masked_fill(~candidate_mask.bool(), -1e9)
    target = target_q.max(dim=1).values.detach()
    loss = F.mse_loss(state_value, target)
    return {"loss": loss, "target_value": target}
