"""Explicit-feedback HPN without predictor, simulator, play, organic, or sessions."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ExplicitFeedbackHistoryEncoder(nn.Module):
    """Encode item, explicit-mark, and continuous-recency history."""

    def __init__(
        self,
        *,
        item_dim: int,
        d_model: int,
        max_history: int,
        num_event_types: int = 4,
        include_source: bool = False,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        time_scale_seconds: float = 3600.0,
        state_pooling: str = "last_mean",
    ) -> None:
        super().__init__()
        if state_pooling not in {"last", "mean", "last_mean"}:
            raise ValueError("state_pooling must be last, mean, or last_mean")
        self.max_history = int(max_history)
        self.time_scale_seconds = float(time_scale_seconds)
        self.state_pooling = state_pooling
        self.include_source = bool(include_source)
        self.item_projection = nn.Linear(int(item_dim), int(d_model))
        self.event_embedding = nn.Embedding(
            int(num_event_types) + 1,
            int(d_model),
            padding_idx=0,
        )
        if self.include_source:
            self.source_embedding = nn.Embedding(3, int(d_model), padding_idx=0)
        self.time_projection = nn.Sequential(
            nn.Linear(1, int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), int(d_model)),
        )
        self.position_embedding = nn.Embedding(self.max_history + 1, int(d_model))
        self.start_token = nn.Parameter(torch.zeros(int(d_model)))
        self.input_norm = nn.LayerNorm(int(d_model))
        self.dropout = nn.Dropout(float(dropout))
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(num_heads),
            dim_feedforward=int(d_model) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.last_mean_fusion = nn.Sequential(
            nn.Linear(int(d_model) * 2, int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(int(d_model)),
        )
        causal = torch.triu(
            torch.ones(
                self.max_history + 1,
                self.max_history + 1,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = batch["history_item_features"]
        marks = batch["history_event_type_ids"].long()
        ages = batch["history_age_seconds"]
        mask = batch["history_mask"].bool()
        batch_size, history_len, _ = features.shape
        if history_len > self.max_history:
            raise ValueError(
                f"history length {history_len} exceeds configured maximum {self.max_history}"
            )
        if marks.shape != mask.shape or ages.shape != mask.shape:
            raise ValueError("mark, age, and mask tensors must share [batch, history] shape")
        if torch.any(marks < 0) or torch.any(marks >= self.event_embedding.num_embeddings):
            raise ValueError("history contains an invalid explicit mark id")
        if self.include_source:
            sources = batch["history_source_ids"].long()
            if sources.shape != marks.shape:
                raise ValueError("history source ids must share the mark shape")

        positions = torch.arange(1, history_len + 1, device=features.device).view(1, -1)
        time_value = torch.log1p(
            ages.clamp_min(0.0) / max(self.time_scale_seconds, 1.0e-6)
        ).unsqueeze(-1)
        encoded_input = (
            self.item_projection(features)
            + self.event_embedding(marks)
            + self.time_projection(time_value.to(features.dtype))
            + self.position_embedding(positions)
        )
        if self.include_source:
            encoded_input = encoded_input + self.source_embedding(sources)
        encoded_input = self.input_norm(self.dropout(encoded_input))

        start = self.start_token.view(1, 1, -1).expand(batch_size, 1, -1)
        sequence = torch.cat([start, encoded_input], dim=1)
        full_mask = torch.cat(
            [
                torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device),
                mask,
            ],
            dim=1,
        )
        encoded = self.encoder(
            sequence,
            mask=self.causal_mask[: history_len + 1, : history_len + 1],
            src_key_padding_mask=~full_mask,
        )
        history_encoded = encoded[:, 1:, :]
        indices = torch.arange(1, history_len + 1, device=mask.device).view(1, -1)
        last_indices = (indices * mask.long()).max(dim=1).values
        last_state = encoded[
            torch.arange(batch_size, device=features.device),
            last_indices,
        ]
        weights = mask.to(features.dtype).unsqueeze(-1)
        mean_state = (history_encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        if self.state_pooling == "last":
            state = last_state
        elif self.state_pooling == "mean":
            state = mean_state
        else:
            state = self.last_mean_fusion(torch.cat([last_state, mean_state], dim=-1))
        return {
            "state_embedding": state,
            "sequence_embeddings": history_encoded,
            "last_state": last_state,
            "mean_state": mean_state,
        }


class ExplicitFeedbackHPN(nn.Module):
    """Hierarchical policy network trained only by logged next-item supervision."""

    def __init__(
        self,
        *,
        item_dim: int,
        d_model: int = 128,
        max_history: int = 50,
        num_event_types: int = 4,
        include_source: bool = False,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        sid_levels: int = 4,
        sid_vocab_size: int = 256,
        sid_temperature: float = 1.0,
        path_conditioning: str = "expected_residual",
        time_scale_seconds: float = 3600.0,
        state_pooling: str = "last_mean",
    ) -> None:
        super().__init__()
        self.sid_levels = int(sid_levels)
        self.sid_vocab_size = int(sid_vocab_size)
        self.sid_temperature = float(sid_temperature)
        if path_conditioning not in {
            "expected_residual",
            "prefix_autoregressive",
        }:
            raise ValueError("path_conditioning must be expected_residual or prefix_autoregressive")
        self.path_conditioning = str(path_conditioning)
        self.history_encoder = ExplicitFeedbackHistoryEncoder(
            item_dim=item_dim,
            d_model=d_model,
            max_history=max_history,
            include_source=include_source,
            num_event_types=num_event_types,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            time_scale_seconds=time_scale_seconds,
            state_pooling=state_pooling,
        )
        self.sid_heads = nn.ModuleList(
            [nn.Linear(int(d_model), self.sid_vocab_size) for _ in range(self.sid_levels)]
        )
        self.sid_token_embeddings = nn.ModuleList(
            [nn.Embedding(self.sid_vocab_size, int(d_model)) for _ in range(self.sid_levels)]
        )
        self.residual_norms = nn.ModuleList(
            [nn.LayerNorm(int(d_model)) for _ in range(self.sid_levels)]
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        history = self.history_encoder(batch)
        context = history["state_embedding"]
        logits: list[torch.Tensor] = []
        temperature = max(self.sid_temperature, 1.0e-6)
        for level in range(self.sid_levels):
            level_logits = self.sid_heads[level](context)
            logits.append(level_logits)
            probabilities = torch.softmax(level_logits / temperature, dim=-1)
            expected_token = probabilities @ self.sid_token_embeddings[level].weight
            context = self.residual_norms[level](context - expected_token)
        return {
            "sid_logits": logits,
            "state_embedding": history["state_embedding"],
            "sequence_embeddings": history["sequence_embeddings"],
        }

    def path_level_logits(
        self,
        state_embedding: torch.Tensor,
        candidate_sids: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Return per-level logits conditioned on each candidate prefix.

        The legacy expected-residual mode reproduces the original independent
        path factorization.  In prefix-autoregressive mode, level ``l`` is
        conditioned on the concrete candidate tokens at levels ``< l``.
        """

        if state_embedding.ndim != 2:
            raise ValueError("state_embedding must have [batch, dimension] shape")
        if candidate_sids.ndim != 3:
            raise ValueError("candidate_sids must have [batch, candidate, level] shape")
        if candidate_sids.shape[0] != state_embedding.shape[0]:
            raise ValueError("state and candidate batch dimensions differ")
        if candidate_sids.shape[-1] < self.sid_levels:
            raise ValueError("candidate SID path has fewer levels than the model")

        candidate_count = int(candidate_sids.shape[1])
        if self.path_conditioning == "expected_residual":
            context = state_embedding
            expanded: list[torch.Tensor] = []
            temperature = max(self.sid_temperature, 1.0e-6)
            for level in range(self.sid_levels):
                logits = self.sid_heads[level](context)
                expanded.append(logits.unsqueeze(1).expand(-1, candidate_count, -1))
                probabilities = torch.softmax(logits / temperature, dim=-1)
                expected_token = probabilities @ self.sid_token_embeddings[level].weight
                context = self.residual_norms[level](context - expected_token)
            return expanded

        context = state_embedding.unsqueeze(1).expand(-1, candidate_count, -1)
        conditional: list[torch.Tensor] = []
        for level in range(self.sid_levels):
            logits = self.sid_heads[level](context)
            conditional.append(logits)
            if level + 1 < self.sid_levels:
                token = candidate_sids[:, :, level].long()
                if torch.any(token < 0) or torch.any(token >= self.sid_vocab_size):
                    raise ValueError("candidate SID token is outside the model vocabulary")
                prefix_embedding = self.sid_token_embeddings[level](token)
                context = self.residual_norms[level](context + prefix_embedding)
        return conditional

    def score_paths(
        self,
        output: dict[str, torch.Tensor | list[torch.Tensor]],
        candidate_sids: torch.Tensor,
    ) -> torch.Tensor:
        """Score complete candidate prefixes by their conditional joint probability."""

        state = output["state_embedding"]
        if not isinstance(state, torch.Tensor):
            raise TypeError("state_embedding must be a tensor")
        logits_by_level = self.path_level_logits(state, candidate_sids)
        scores = state.new_zeros(candidate_sids.shape[:2])
        temperature = max(self.sid_temperature, 1.0e-6)
        for level, logits in enumerate(logits_by_level):
            token = candidate_sids[:, :, level].long()
            if torch.any(token < 0) or torch.any(token >= logits.shape[-1]):
                raise ValueError("candidate SID token is outside the model vocabulary")
            log_probabilities = F.log_softmax(logits / temperature, dim=-1)
            scores = scores + log_probabilities.gather(2, token.unsqueeze(-1)).squeeze(-1)
        return scores / max(len(logits_by_level), 1)


def _require_sid_logits(
    output: dict[str, torch.Tensor | list[torch.Tensor]],
) -> list[torch.Tensor]:
    logits = output["sid_logits"]
    if not isinstance(logits, list):
        raise TypeError("sid_logits must be a list")
    return logits


def score_sid_candidates(
    sid_logits: list[torch.Tensor],
    candidate_sids: torch.Tensor,
) -> torch.Tensor:
    """Average log probabilities over exactly the levels predicted by the HPN.

    Candidate paths may contain a later identity suffix.  It is intentionally
    ignored when the model predicts only the semantic RQ prefix.
    """

    if candidate_sids.ndim != 3:
        raise ValueError("candidate_sids must have [batch, candidate, level] shape")
    if candidate_sids.shape[-1] < len(sid_logits):
        raise ValueError("candidate SID path has fewer levels than the model")
    scores = sid_logits[0].new_zeros(candidate_sids.shape[:2])
    for level, logits in enumerate(sid_logits):
        token = candidate_sids[:, :, level].long()
        if torch.any(token < 0) or torch.any(token >= logits.shape[-1]):
            raise ValueError("candidate SID token is outside the model vocabulary")
        log_probabilities = F.log_softmax(logits, dim=-1)
        scores = scores + log_probabilities.gather(1, token)
    return scores / max(len(sid_logits), 1)


def unique_positive_path_mask(
    positive_sids: torch.Tensor,
    positive_mask: torch.Tensor,
    levels: int | None = None,
) -> torch.Tensor:
    """Remove duplicate paths at the resolution actually predicted by the model."""

    unique = positive_mask.bool().clone()
    compared = positive_sids[:, :, :levels] if levels is not None else positive_sids
    positive_count = positive_sids.shape[1]
    for position in range(1, positive_count):
        duplicate = torch.zeros(
            positive_sids.shape[0],
            dtype=torch.bool,
            device=positive_sids.device,
        )
        for earlier in range(position):
            duplicate |= unique[:, earlier] & (compared[:, position] == compared[:, earlier]).all(
                dim=-1
            )
        unique[:, position] &= ~duplicate
    return unique


def multi_positive_hpn_loss(
    model: ExplicitFeedbackHPN,
    output: dict[str, torch.Tensor | list[torch.Tensor]],
    positive_sids: torch.Tensor,
    positive_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Supervise every positive item without imposing an order within a tied group."""

    if positive_sids.ndim != 3 or positive_mask.ndim != 2:
        raise ValueError("positive targets must have [batch, positive, level] plus mask")
    if positive_sids.shape[:2] != positive_mask.shape:
        raise ValueError("positive SID and mask shapes disagree")
    levels = min(model.sid_levels, positive_sids.shape[-1])
    path_mask = unique_positive_path_mask(positive_sids, positive_mask, levels)
    if not torch.all(path_mask.any(dim=1)):
        raise ValueError("every recommendation row must contain a mapped positive path")

    state = output["state_embedding"]
    if not isinstance(state, torch.Tensor):
        raise TypeError("state_embedding must be a tensor")
    logits_by_level = model.path_level_logits(state, positive_sids)
    nll_sum = state.new_tensor(0.0)
    denominator = state.new_tensor(0.0)
    token_hits: list[torch.Tensor] = []
    predicted_path: list[torch.Tensor] = []
    temperature = max(model.sid_temperature, 1.0e-6)
    for level, logits in enumerate(logits_by_level[:levels]):
        target = positive_sids[:, :, level].long()
        log_probabilities = F.log_softmax(logits / temperature, dim=-1)
        selected = log_probabilities.gather(2, target.unsqueeze(-1)).squeeze(-1)
        nll_sum = nll_sum - (selected * path_mask.to(selected.dtype)).sum()
        denominator = denominator + path_mask.sum()
        prediction = logits.argmax(dim=-1)
        predicted_path.append(prediction)
        token_hits.append(((prediction == target) & path_mask).any(dim=1).float().mean())
    loss = nll_sum / denominator.clamp_min(1.0)
    predicted = torch.stack(predicted_path, dim=2)
    full_path_hit = ((predicted == positive_sids[:, :, :levels]).all(dim=-1) & path_mask).any(dim=1)
    metrics: dict[str, torch.Tensor] = {
        "loss": loss,
        "predicted_path_hit": full_path_hit.float().mean().detach(),
        "unique_positive_paths": path_mask.sum(dim=1).float().mean().detach(),
    }
    for level, value in enumerate(token_hits, start=1):
        metrics[f"token_hit_l{level}"] = value.detach()
    return metrics


def multi_positive_candidate_loss(
    model: ExplicitFeedbackHPN,
    output: dict[str, torch.Tensor | list[torch.Tensor]],
    positive_sids: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_sids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Set likelihood over all positive paths versus sampled negative paths."""

    predicted_levels = model.sid_levels
    positive_path_mask = unique_positive_path_mask(
        positive_sids,
        positive_mask,
        predicted_levels,
    )
    candidates = torch.cat([positive_sids, negative_sids], dim=1)
    negative_mask = torch.ones(
        negative_sids.shape[:2],
        dtype=torch.bool,
        device=negative_sids.device,
    )
    candidate_mask = torch.cat([positive_path_mask, negative_mask], dim=1)
    positive_candidate_mask = torch.cat(
        [
            positive_path_mask,
            torch.zeros_like(negative_mask),
        ],
        dim=1,
    )
    scores = model.score_paths(output, candidates)
    masked_all = scores.masked_fill(~candidate_mask, float("-inf"))
    masked_positive = scores.masked_fill(~positive_candidate_mask, float("-inf"))
    loss = (torch.logsumexp(masked_all, dim=1) - torch.logsumexp(masked_positive, dim=1)).mean()

    order = masked_all.argsort(dim=1, descending=True)
    ranked_positive = positive_candidate_mask.gather(1, order)
    rank_positions = torch.arange(
        1,
        ranked_positive.shape[1] + 1,
        device=scores.device,
    ).view(1, -1)
    first_rank = (
        torch.where(
            ranked_positive,
            rank_positions,
            torch.full_like(rank_positions, ranked_positive.shape[1] + 1),
        )
        .min(dim=1)
        .values
    )
    return {
        "loss": loss,
        "sampled_hit_at_1": (first_rank == 1).float().mean().detach(),
        "sampled_mrr": (1.0 / first_rank.float()).mean().detach(),
    }
