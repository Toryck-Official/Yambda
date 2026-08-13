"""Protocol-v1.1 item-aware Minimal SNMPP.

Outputs are limited to next timestamp-group time and feedback composition.
There is intentionally no SID prediction head, recurrent encoder, history
truncation, burst normalization, or history-contribution clipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def inverse_softplus(value: float) -> float:
    x = torch.tensor(value, dtype=torch.float64)
    return float(torch.log(torch.expm1(x)))


class PositiveLinear(nn.Module):
    """Linear layer with nonnegative effective weights."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        target = 1.0 / max(1, input_dim)
        self.raw_weight = nn.Parameter(
            torch.empty(output_dim, input_dim).normal_(inverse_softplus(target), 0.02)
        )
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, value: Tensor) -> Tensor:
        return F.linear(value, F.softplus(self.raw_weight), self.bias)


class SignedInteractionNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        dims = (input_dim, *hidden_dims)
        layers: list[nn.Module] = []
        for left, right in zip(dims[:-1], dims[1:], strict=True):
            layers.extend([nn.Linear(left, right), nn.GELU()])
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(dims[-1], 1)
        # Full-history summation makes a conventional 1e-2 output scale grow
        # with thousands of sources before training.  A near-zero signed-link
        # initialization preserves the exact sum model while avoiding a false
        # numerical failure caused solely by initialization.
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: Tensor) -> Tensor:
        return self.output(self.hidden(value)).squeeze(-1)


class MonotoneTemporalNetwork(nn.Module):
    """Contextual temporal response monotone decreasing in distance."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        dims = (input_dim, *hidden_dims, 1)
        self.layers = nn.ModuleList(
            PositiveLinear(left, right) for left, right in zip(dims[:-1], dims[1:], strict=True)
        )
        nn.init.constant_(self.layers[-1].bias, -0.5)

    def forward(self, value: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index < len(self.layers) - 1:
                value = F.softplus(value)
        return torch.sigmoid(value).squeeze(-1)


@dataclass
class MinimalLossOutput:
    optimization_loss: Tensor
    time_loss: Tensor
    feedback_loss: Tensor
    time_loss_by_group: Tensor
    feedback_loss_sum: Tensor
    num_groups: Tensor
    num_events: Tensor
    target_lambdas: Tensor
    target_total_intensity: Tensor
    target_feedback_probabilities: Tensor
    integral_by_group: Tensor
    signed_influence_by_target: Tensor
    absolute_influence_mass_by_target: Tensor
    positive_influence_mass_by_target: Tensor
    negative_influence_mass_by_target: Tensor


class MinimalSNMPP(nn.Module):
    """Full-history item-aware SNMPP under grouped protocol B."""

    def __init__(
        self,
        codebook_path: str | Path,
        *,
        feedback_embedding_dim: int = 4,
        hidden_dims: tuple[int, ...] = (16, 16),
        integration_q: int = 64,
        initial_delay_hours: float = 0.1,
        query_chunk_size: int = 4,
        minimum_intensity: float = 1e-8,
    ) -> None:
        super().__init__()
        codebook = np.load(Path(codebook_path), allow_pickle=False)
        if codebook.shape != (4, 256, 128):
            raise ValueError(f"expected codebook [4,256,128], got {codebook.shape}")
        self.register_buffer("codebook", torch.from_numpy(codebook.astype(np.float32, copy=True)))
        self.feedback_embedding = nn.Embedding(4, feedback_embedding_dim)
        nn.init.normal_(self.feedback_embedding.weight, mean=0.0, std=0.1)
        self.sid_projection = nn.Linear(128, feedback_embedding_dim, bias=False)
        nn.init.normal_(self.sid_projection.weight, mean=0.0, std=0.02)

        pair_dim = 2 * feedback_embedding_dim
        self.interaction_network = SignedInteractionNetwork(pair_dim, hidden_dims)
        self.temporal_network = MonotoneTemporalNetwork(pair_dim + 1, hidden_dims)
        self.baseline_logits = nn.Parameter(torch.zeros(4))
        self.raw_delays = nn.Parameter(
            torch.full((4, 4), inverse_softplus(initial_delay_hours), dtype=torch.float32)
        )
        self.integration_q = int(integration_q)
        self.query_chunk_size = int(query_chunk_size)
        self.minimum_intensity = float(minimum_intensity)
        if self.integration_q != 64:
            raise ValueError("Phase 2 v1.1 freezes integration_q=64")
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")

    def delays(self) -> Tensor:
        return F.softplus(self.raw_delays)

    def item_representation(self, sid: Tensor) -> Tensor:
        if sid.shape[-1] != 4:
            raise ValueError("SID must contain four semantic tokens")
        pieces = [self.codebook[level][sid[..., level].long()] for level in range(4)]
        return torch.stack(pieces, dim=0).sum(dim=0)

    def event_representation(self, feedback: Tensor, sid: Tensor) -> Tensor:
        return self.feedback_embedding(feedback.long()) + self.sid_projection(
            self.item_representation(sid)
        )

    def _source_target_features(self, source_z: Tensor) -> Tensor:
        batch, history, dim = source_z.shape
        source = source_z[:, :, None, :].expand(batch, history, 4, dim)
        target = self.feedback_embedding.weight[None, None, :, :].expand(
            batch, history, 4, dim
        )
        return torch.cat([source, target], dim=-1)

    def _positive_link(self, score: Tensor) -> Tensor:
        return F.softplus(score).clamp_min(self.minimum_intensity)

    def conditional_intensity(
        self,
        query_times: Tensor,
        history_times: Tensor,
        history_feedback: Tensor,
        history_sid: Tensor,
        history_mask: Tensor,
        *,
        return_influence: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        """Evaluate λ_k with strict history_time < query_time.

        Shapes: query [B,Q], history time/feedback/mask [B,H], SID [B,H,4].
        Every source event is summed; no count normalization or truncation occurs.
        """
        source_z = self.event_representation(history_feedback, history_sid)
        pair_features = self._source_target_features(source_z)
        psi = self.interaction_network(pair_features)  # [B,H,4], signed
        source_delays = self.delays()[history_feedback.long()]  # [B,H,4]
        outputs = []
        signed_parts = []
        absolute_parts = []
        positive_parts = []
        negative_parts = []
        for start in range(0, query_times.shape[1], self.query_chunk_size):
            query = query_times[:, start : start + self.query_chunk_size]
            lags = query[:, :, None] - history_times[:, None, :]
            valid = (lags > 0) & history_mask[:, None, :]
            distance = torch.abs(lags[:, :, :, None] - source_delays[:, None, :, :])
            pair = pair_features[:, None, :, :, :].expand(
                -1, query.shape[1], -1, -1, -1
            )
            temporal_input = torch.cat([pair, -distance[..., None]], dim=-1)
            phi = self.temporal_network(temporal_input)
            contribution = psi[:, None, :, :] * phi
            contribution = contribution * valid[..., None]
            signed = contribution.sum(dim=2)
            outputs.append(self._positive_link(self.baseline_logits + signed))
            if return_influence:
                signed_parts.append(signed)
                absolute_parts.append(contribution.abs().sum(dim=2))
                positive_parts.append(contribution.clamp_min(0).sum(dim=2))
                negative_parts.append((-contribution.clamp_max(0)).sum(dim=2))
        intensities = torch.cat(outputs, dim=1)
        if not return_influence:
            return intensities, None
        diagnostics = {
            "signed": torch.cat(signed_parts, dim=1),
            "absolute": torch.cat(absolute_parts, dim=1),
            "positive": torch.cat(positive_parts, dim=1),
            "negative_magnitude": torch.cat(negative_parts, dim=1),
        }
        return intensities, diagnostics

    def stratified_ratios(self, batch_size: int, *, deterministic: bool) -> Tensor:
        strata = torch.arange(
            self.integration_q,
            device=self.baseline_logits.device,
            dtype=self.baseline_logits.dtype,
        )
        if deterministic:
            within = torch.full(
                (batch_size, self.integration_q),
                0.5,
                device=strata.device,
                dtype=strata.dtype,
            )
        else:
            within = torch.rand(
                batch_size,
                self.integration_q,
                device=strata.device,
                dtype=strata.dtype,
            )
        return (strata[None, :] + within) / self.integration_q

    def loss(self, batch: dict[str, Tensor], *, deterministic_integral: bool) -> MinimalLossOutput:
        target_time = batch["target_time"]
        previous_time = batch["previous_time"]
        interval = target_time - previous_time
        if torch.any(interval <= 0):
            raise ValueError("all target intervals must be strictly positive")
        query_target = target_time[:, None]
        target_lambdas, influence = self.conditional_intensity(
            query_target,
            batch["history_times"],
            batch["history_feedback"],
            batch["history_sid"],
            batch["history_mask"],
            return_influence=True,
        )
        target_lambdas = target_lambdas[:, 0, :]
        total = target_lambdas.sum(dim=-1)
        q = target_lambdas / total[:, None]

        ratios = self.stratified_ratios(len(target_time), deterministic=deterministic_integral)
        sample_times = previous_time[:, None] + interval[:, None] * ratios
        sampled, _ = self.conditional_intensity(
            sample_times,
            batch["history_times"],
            batch["history_feedback"],
            batch["history_sid"],
            batch["history_mask"],
        )
        integral = interval * sampled.sum(dim=-1).mean(dim=-1)
        time_by_group = -torch.log(total) + integral
        target_counts = batch["target_feedback_counts"]
        feedback_sum = -(target_counts * torch.log(q)).sum()
        num_events = target_counts.sum()
        time_loss = time_by_group.mean()
        feedback_loss = feedback_sum / num_events.clamp_min(1.0)
        objective = time_loss + feedback_loss
        assert influence is not None
        return MinimalLossOutput(
            optimization_loss=objective,
            time_loss=time_loss,
            feedback_loss=feedback_loss,
            time_loss_by_group=time_by_group,
            feedback_loss_sum=feedback_sum,
            num_groups=target_time.new_tensor(float(len(target_time))),
            num_events=num_events,
            target_lambdas=target_lambdas,
            target_total_intensity=total,
            target_feedback_probabilities=q,
            integral_by_group=integral,
            signed_influence_by_target=influence["signed"][:, 0, :],
            absolute_influence_mass_by_target=influence["absolute"][:, 0, :],
            positive_influence_mass_by_target=influence["positive"][:, 0, :],
            negative_influence_mass_by_target=influence["negative_magnitude"][:, 0, :],
        )

    @torch.no_grad()
    def time_distribution(
        self,
        batch: dict[str, Tensor],
        *,
        horizon_hours: float = 696.0,
        grid_size: int = 256,
    ) -> dict[str, Tensor]:
        if grid_size < 8:
            raise ValueError("grid_size must be at least 8")
        minimum = 5.0 / 3600.0
        positive_grid = torch.logspace(
            torch.log10(torch.tensor(minimum, device=self.baseline_logits.device)),
            torch.log10(torch.tensor(horizon_hours, device=self.baseline_logits.device)),
            grid_size - 1,
            device=self.baseline_logits.device,
            dtype=self.baseline_logits.dtype,
        )
        delta = torch.cat([positive_grid.new_zeros(1), positive_grid])
        reference = batch["previous_time"]
        query = reference[:, None] + delta[None, :]
        query[:, 0] += torch.finfo(query.dtype).eps
        lambdas, _ = self.conditional_intensity(
            query,
            batch["history_times"],
            batch["history_feedback"],
            batch["history_sid"],
            batch["history_mask"],
        )
        total = lambdas.sum(dim=-1)
        widths = delta[1:] - delta[:-1]
        increments = 0.5 * (total[:, 1:] + total[:, :-1]) * widths[None, :]
        cumulative = torch.cat(
            [torch.zeros(len(query), 1, device=query.device), torch.cumsum(increments, dim=1)],
            dim=1,
        )
        survival = torch.exp(-cumulative.clamp(max=80.0))
        event_mass = 1.0 - survival[:, -1]
        expected_capped = torch.trapezoid(survival, delta, dim=1)
        return {
            "expected_delta_hours": expected_capped,
            "event_mass_within_horizon": event_mass,
            "survival_at_horizon": survival[:, -1],
        }
