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
        event_vocab_size: int = 6,
        response_classes: int = 5,
        regret_classes: int = 4,
        state_pooling: str = "last",
        sid_levels: int = 4,
        sid_vocab_size: int = 256,
    ) -> None:
        super().__init__()
        self.item_dim = int(item_dim)
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)
        self.response_classes = int(response_classes)
        self.regret_classes = int(regret_classes)
        self.state_pooling = str(state_pooling)
        if self.state_pooling not in {"last", "mean", "last_mean"}:
            raise ValueError("state_pooling must be one of: last, mean, last_mean")
        self.sid_levels = int(sid_levels)
        self.sid_vocab_size = int(sid_vocab_size)

        self.action_proj = nn.Linear(self.item_dim, self.d_model)
        self.state_action_norm = nn.LayerNorm(self.d_model)
        self.action_norm = nn.LayerNorm(self.d_model)
        self.state_encoder = StateEncoder(
            item_dim=self.item_dim,
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            n_layer=int(n_layer),
            n_head=int(n_head),
            dropout=float(dropout),
            response_dim=self.response_classes,
        )

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
        self.action_residual = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.response_head = nn.Linear(self.d_model, self.response_classes)
        self.play_head = nn.Linear(self.d_model, 1)
        self.reward_head = nn.Linear(self.d_model, 1)
        self.regret_head = nn.Linear(self.d_model, self.regret_classes)
        self.future_return_head = nn.Linear(self.d_model, 1)
        self.future_regret_head = nn.Linear(self.d_model, 1)
        self.soft_token_head = nn.Linear(self.d_model, self.d_model)
        self.candidate_head = nn.Linear(self.d_model, 1)
        self.sid_heads = nn.ModuleList([nn.Linear(self.d_model, self.sid_vocab_size) for _ in range(self.sid_levels)])

    def _semantic_logits(self, state_emb: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(state_emb) for head in self.sid_heads], dim=1)

    def encode_history(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        enc = self.state_encoder(batch)
        enc["sid_logits"] = self._semantic_logits(enc["state_emb"])
        return enc

    def _predict_flat(self, state_emb: torch.Tensor, action_features: torch.Tensor) -> dict[str, torch.Tensor]:
        state_cond = self.state_action_norm(state_emb)
        action_emb = self.action_norm(self.action_proj(action_features))
        joint = torch.cat(
            [
                state_cond,
                action_emb,
                state_cond * action_emb,
                torch.abs(state_cond - action_emb),
            ],
            dim=-1,
        )
        hidden = self.joint(joint) + self.action_residual(torch.cat([state_cond, action_emb], dim=-1))
        response_logits = self.response_head(hidden)
        regret_logits = self.regret_head(hidden)
        play_ratio = F.softplus(self.play_head(hidden).squeeze(-1))
        predicted_reward = 2.0 * torch.tanh(self.reward_head(hidden).squeeze(-1))
        predicted_future_return = self.future_return_head(hidden).squeeze(-1)
        future_regret_logit = self.future_regret_head(hidden).squeeze(-1)
        soft_next_token = self.soft_token_head(hidden)
        candidate_logit = self.candidate_head(hidden).squeeze(-1)
        return {
            "hidden": hidden,
            "response_logits": response_logits,
            "response_probs": torch.sigmoid(response_logits),
            "predicted_play_ratio": play_ratio,
            "predicted_reward": predicted_reward,
            "predicted_future_return": predicted_future_return,
            "future_regret_logit": future_regret_logit,
            "future_regret_prob": torch.sigmoid(future_regret_logit),
            "regret_logits": regret_logits,
            "regret_probs": torch.softmax(regret_logits, dim=-1),
            "soft_next_token": soft_next_token,
            "candidate_logit": candidate_logit,
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
    play_weight: float = 1.0,
    reward_weight: float = 1.0,
    regret_weight: float = 1.0,
    future_return_weight: float = 0.0,
    future_regret_weight: float = 0.0,
    future_regret_pos_weight: torch.Tensor | None = None,
    response_pos_weight: torch.Tensor | None = None,
    response_class_weight: torch.Tensor | None = None,
    regret_class_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if "response_targets" in batch:
        response_loss_raw = F.binary_cross_entropy_with_logits(
            out["response_logits"],
            batch["response_targets"].float(),
            pos_weight=response_pos_weight,
            reduction="none",
        )
        if response_class_weight is not None:
            response_loss_raw = response_loss_raw * response_class_weight.view(1, -1)
        response_loss = response_loss_raw.mean()
    else:
        response_loss = F.cross_entropy(out["response_logits"], batch["response_target"].long())
    listen_mask = batch.get("response_targets")
    if listen_mask is None or not (listen_mask[:, 0] > 0.5).any():
        play_loss = out["predicted_play_ratio"].sum() * 0.0
    else:
        listen_mask = listen_mask[:, 0] > 0.5
        play_loss = F.smooth_l1_loss(
            out["predicted_play_ratio"][listen_mask],
            batch["played_ratio"][listen_mask].float(),
        )
    reward_loss = F.mse_loss(out["predicted_reward"], batch["reward"].float())
    regret_loss = F.cross_entropy(out["regret_logits"], batch["regret_type_id"].long(), weight=regret_class_weight)
    if "future_return" in batch and "predicted_future_return" in out:
        future_return_loss = F.smooth_l1_loss(out["predicted_future_return"], batch["future_return"].float())
    else:
        future_return_loss = reward_loss.new_tensor(0.0)
    if "future_regret_any" in batch and "future_regret_logit" in out:
        future_regret_loss = F.binary_cross_entropy_with_logits(
            out["future_regret_logit"],
            batch["future_regret_any"].float(),
            pos_weight=future_regret_pos_weight,
        )
    else:
        future_regret_loss = reward_loss.new_tensor(0.0)
    total = (
        float(response_weight) * response_loss
        + float(play_weight) * play_loss
        + float(reward_weight) * reward_loss
        + float(regret_weight) * regret_loss
        + float(future_return_weight) * future_return_loss
        + float(future_regret_weight) * future_regret_loss
    )
    return {
        "loss": total,
        "response_loss": response_loss.detach(),
        "play_loss": play_loss.detach(),
        "reward_loss": reward_loss.detach(),
        "regret_loss": regret_loss.detach(),
        "future_return_loss": future_return_loss.detach(),
        "future_regret_loss": future_regret_loss.detach(),
    }
