from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.item_proj = nn.Linear(self.item_dim, self.d_model)
        self.action_proj = nn.Linear(self.item_dim, self.d_model)
        self.state_action_norm = nn.LayerNorm(self.d_model)
        self.action_norm = nn.LayerNorm(self.d_model)
        self.feedback_proj = nn.Linear(1, self.d_model, bias=False)
        self.event_emb = nn.Embedding(int(event_vocab_size), self.d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_head),
            dim_feedforward=self.d_model * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layer))
        self.register_buffer("pos_idx", torch.arange(self.max_seq_len, dtype=torch.long), persistent=False)
        causal = torch.tril(torch.ones((self.max_seq_len, self.max_seq_len), dtype=torch.bool))
        self.register_buffer("attn_mask_full", ~causal, persistent=False)

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
        self.state_fuse = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
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

    def _masked_mean_state(self, seq: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch.get("history_mask")
        if mask is None:
            valid = torch.ones(seq.shape[:2], dtype=seq.dtype, device=seq.device)
        else:
            valid = mask.to(seq.device, dtype=seq.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (seq * valid.unsqueeze(-1)).sum(dim=1) / denom

    def _pool_state(self, seq: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        last = seq[:, -1, :]
        if self.state_pooling == "last":
            return last
        mean = self._masked_mean_state(seq, batch)
        if self.state_pooling == "mean":
            return mean
        return self.state_fuse(torch.cat([last, mean], dim=-1))

    def _semantic_logits(self, state_emb: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(state_emb) for head in self.sid_heads], dim=1)

    def encode_history(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hist = batch["history_features"]
        batch_size, hist_len, _ = hist.shape
        pos = self.pos_emb(self.pos_idx[:hist_len]).unsqueeze(0).expand(batch_size, hist_len, -1)
        x = self.item_proj(hist) + pos

        feedback = batch.get("history_feedbacks")
        if feedback is not None:
            x = x + self.feedback_proj(feedback.to(hist.device, dtype=hist.dtype).unsqueeze(-1))

        event_ids = batch.get("history_event_type_ids")
        if event_ids is not None:
            event_ids = event_ids.to(hist.device).long().clamp(min=0, max=self.event_emb.num_embeddings - 1)
            x = x + self.event_emb(event_ids)

        x = self.input_norm(self.drop(x))
        attn_mask = self.attn_mask_full[:hist_len, :hist_len]
        seq = self.encoder(x, mask=attn_mask)
        state_emb = self._pool_state(seq, batch)
        return {"seq_emb": seq, "state_emb": state_emb, "sid_logits": self._semantic_logits(state_emb)}

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
        predicted_reward = self.reward_head(hidden).squeeze(-1)
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
    play_loss = F.smooth_l1_loss(out["predicted_play_ratio"], batch["played_ratio"].float())
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
