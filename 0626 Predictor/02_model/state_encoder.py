from __future__ import annotations

import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """Encode observable history into short-term, long-term and fused states."""

    def __init__(
        self,
        item_dim: int = 128,
        d_model: int = 128,
        max_seq_len: int = 50,
        n_layer: int = 2,
        n_head: int = 4,
        dropout: float = 0.1,
        response_dim: int = 5,
    ) -> None:
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.item_map = nn.Linear(int(item_dim), int(d_model))
        self.response_map = nn.Linear(int(response_dim), int(d_model), bias=False)
        self.play_map = nn.Linear(2, int(d_model), bias=False)
        self.time_map = nn.Linear(1, int(d_model), bias=False)
        self.source_emb = nn.Embedding(2, int(d_model))
        self.session_emb = nn.Embedding(2, int(d_model))
        self.pos_emb = nn.Embedding(self.max_seq_len + 1, int(d_model))
        self.start_token = nn.Parameter(torch.zeros(int(d_model)))
        self.input_norm = nn.LayerNorm(int(d_model))
        self.drop = nn.Dropout(float(dropout))

        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_head),
            dim_feedforward=int(d_model) * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layer))
        self.fusion = nn.Sequential(
            nn.Linear(int(d_model) * 3, int(d_model) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model) * 2, int(d_model)),
            nn.LayerNorm(int(d_model)),
        )
        causal = torch.triu(torch.ones(self.max_seq_len + 1, self.max_seq_len + 1, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)

    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(values.dtype).unsqueeze(-1)
        return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def forward(self, batch: dict[str, torch.Tensor], prefix: str = "history") -> dict[str, torch.Tensor]:
        features = batch[f"{prefix}_features"]
        mask = batch[f"{prefix}_mask"].bool()
        batch_size, seq_len, _ = features.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"history length {seq_len} exceeds max_seq_len={self.max_seq_len}")

        responses = batch.get(f"{prefix}_response_targets")
        if responses is None:
            responses = features.new_zeros(batch_size, seq_len, self.response_map.in_features)
        play_ratio = batch.get(f"{prefix}_play_ratios", features.new_zeros(batch_size, seq_len))
        play_excess = batch.get(f"{prefix}_play_excesses", features.new_zeros(batch_size, seq_len))
        source = batch.get(f"{prefix}_is_organic", torch.zeros_like(mask, dtype=torch.long)).long().clamp(0, 1)
        same_session = batch.get(f"{prefix}_same_session", mask.long()).long().clamp(0, 1)
        gaps = batch.get(f"{prefix}_time_gap_seconds", features.new_zeros(batch_size, seq_len))

        positions = torch.arange(1, seq_len + 1, device=features.device).unsqueeze(0)
        x = self.item_map(features)
        x = x + self.response_map(responses.to(features.dtype))
        x = x + self.play_map(torch.stack([play_ratio, play_excess], dim=-1).to(features.dtype))
        x = x + self.time_map(torch.log1p(gaps.clamp_min(0.0)).unsqueeze(-1).to(features.dtype))
        x = x + self.source_emb(source) + self.session_emb(same_session) + self.pos_emb(positions)
        x = self.input_norm(self.drop(x))

        start = self.start_token.view(1, 1, -1).expand(batch_size, 1, -1)
        sequence = torch.cat([start, x], dim=1)
        full_mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        encoded = self.encoder(
            sequence,
            mask=self.causal_mask[: seq_len + 1, : seq_len + 1],
            src_key_padding_mask=~full_mask,
        )
        history_encoded = encoded[:, 1:, :]

        index = torch.arange(1, seq_len + 1, device=mask.device).unsqueeze(0).expand(batch_size, -1)
        last_index = (index * mask.long()).max(dim=1).values
        full_state = encoded[torch.arange(batch_size, device=features.device), last_index]
        short_mask = mask & same_session.bool()
        long_mask = mask & ~same_session.bool()
        short_state = self.masked_mean(history_encoded, short_mask)
        long_state = self.masked_mean(history_encoded, long_mask)
        state = self.fusion(torch.cat([full_state, short_state, long_state], dim=-1))
        return {
            "seq_emb": history_encoded,
            "state_emb": state,
            "short_state_emb": short_state,
            "long_state_emb": long_state,
        }
