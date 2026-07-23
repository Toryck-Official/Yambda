from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from state_encoder import StateEncoder


class FuturePredictor(nn.Module):
    def __init__(
        self,
        item_dim: int = 128,
        d_model: int = 128,
        max_seq_len: int = 50,
        n_layer: int = 2,
        n_head: int = 4,
        dropout: float = 0.1,
        response_classes: int = 5,
        regret_classes: int = 4,
    ) -> None:
        super().__init__()
        self.item_dim = int(item_dim)
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)
        self.response_classes = int(response_classes)
        self.regret_classes = int(regret_classes)

        self.state_encoder = StateEncoder(
            item_dim=self.item_dim,
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            n_layer=int(n_layer),
            n_head=int(n_head),
            dropout=float(dropout),
            response_dim=self.response_classes,
        )
        self.action_proj = nn.Linear(self.item_dim, self.d_model)

        joint_dim = self.d_model * 4
        self.joint = nn.Sequential(
            nn.Linear(joint_dim, self.d_model * 2),
            nn.LayerNorm(self.d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )
        self.response_head = nn.Linear(self.d_model, self.response_classes)
        self.play_head = nn.Linear(self.d_model, 1)
        self.reward_head = nn.Linear(self.d_model, 1)
        self.regret_head = nn.Linear(self.d_model, self.regret_classes)

    def encode_history(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.state_encoder(batch)

    def _predict_flat(self, state_emb: torch.Tensor, action_features: torch.Tensor) -> dict[str, torch.Tensor]:
        action_emb = self.action_proj(action_features)
        joint = torch.cat(
            [
                state_emb,
                action_emb,
                state_emb * action_emb,
                torch.abs(state_emb - action_emb),
            ],
            dim=-1,
        )
        hidden = self.joint(joint)
        response_logits = self.response_head(hidden)
        regret_logits = self.regret_head(hidden)
        play_ratio = F.softplus(self.play_head(hidden).squeeze(-1))
        response_probs = torch.sigmoid(response_logits)
        predicted_reward = 2.0 * torch.tanh(self.reward_head(hidden).squeeze(-1))
        return {
            "hidden": hidden,
            "response_logits": response_logits,
            "response_probs": response_probs,
            "predicted_play_ratio": play_ratio,
            "predicted_reward": predicted_reward,
            "regret_logits": regret_logits,
            "regret_probs": torch.softmax(regret_logits, dim=-1),
        }

    def forward(self, batch: dict[str, torch.Tensor], action_features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        enc = self.encode_history(batch)
        state_emb = enc["state_emb"]
        actions = batch["action_features"] if action_features is None else action_features

        if actions.dim() == 2:
            out = self._predict_flat(state_emb, actions)
            out.update(enc)
            return out

        if actions.dim() != 3:
            raise ValueError("action_features must have shape [B,D] or [B,K,D].")
        batch_size, candidate_k, item_dim = actions.shape
        state_flat = state_emb.unsqueeze(1).expand(batch_size, candidate_k, -1).reshape(batch_size * candidate_k, -1)
        action_flat = actions.reshape(batch_size * candidate_k, item_dim)
        out_flat = self._predict_flat(state_flat, action_flat)
        out = {
            key: value.reshape(batch_size, candidate_k, *value.shape[1:])
            for key, value in out_flat.items()
            if key != "hidden"
        }
        out["hidden"] = out_flat["hidden"].reshape(batch_size, candidate_k, -1)
        out.update(enc)
        return out


def predictor_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    response_weight: float = 1.0,
    play_weight: float = 0.1,
    reward_weight: float = 0.1,
    regret_weight: float = 0.1,
    response_pos_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    targets = batch["response_targets"].float()
    response_loss = F.binary_cross_entropy_with_logits(
        out["response_logits"],
        targets,
        pos_weight=response_pos_weight,
    )
    listen_mask = targets[:, 0] > 0.5
    if listen_mask.any():
        play_loss = F.smooth_l1_loss(
            out["predicted_play_ratio"][listen_mask],
            batch["played_ratio"][listen_mask].float(),
        )
    else:
        play_loss = out["predicted_play_ratio"].sum() * 0.0
    reward_loss = F.mse_loss(out["predicted_reward"], batch["reward"].float())
    regret_loss = F.cross_entropy(out["regret_logits"], batch["regret_type_id"].long())
    total = (
        float(response_weight) * response_loss
        + float(play_weight) * play_loss
        + float(reward_weight) * reward_loss
        + float(regret_weight) * regret_loss
    )
    return {
        "loss": total,
        "response_loss": response_loss.detach(),
        "play_loss": play_loss.detach(),
        "reward_loss": reward_loss.detach(),
        "regret_loss": regret_loss.detach(),
    }
