"""Minimal SNMPP with the sole experimental change: factorized Lambda and q heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from minimal_snmpp_pilot.model import MinimalLossOutput, MinimalSNMPP, inverse_softplus


class FactorizedMinimalSNMPP(MinimalSNMPP):
    """Shared signed-temporal context with separate group-time and mark heads.

    Everything before the output parameterization is inherited unchanged from
    MinimalSNMPP.  The former four coupled positive intensities are replaced by
    Lambda=softplus(time_head(context)) and q=softmax(mark_head(context)); the
    diagnostic per-mark intensity is exactly lambda_k=Lambda*q_k.
    """

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
        super().__init__(
            codebook_path,
            feedback_embedding_dim=feedback_embedding_dim,
            hidden_dims=hidden_dims,
            integration_q=integration_q,
            initial_delay_hours=initial_delay_hours,
            query_chunk_size=query_chunk_size,
            minimum_intensity=minimum_intensity,
        )
        del self.baseline_logits
        self.time_head = nn.Linear(4, 1)
        self.mark_head = nn.Linear(4, 4)
        nn.init.normal_(self.time_head.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.time_head.bias)
        nn.init.normal_(self.mark_head.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.mark_head.bias)

    @property
    def head_parameter_count(self) -> int:
        return sum(parameter.numel() for module in (self.time_head, self.mark_head) for parameter in module.parameters())

    def initialize_total_rate(self, total_rate_per_hour: float) -> None:
        with torch.no_grad():
            self.time_head.bias.fill_(inverse_softplus(total_rate_per_hour))

    def stratified_ratios(self, batch_size: int, *, deterministic: bool) -> Tensor:
        device = self.time_head.weight.device
        dtype = self.time_head.weight.dtype
        strata = torch.arange(self.integration_q, device=device, dtype=dtype)
        if deterministic:
            within = torch.full((batch_size, self.integration_q), 0.5, device=device, dtype=dtype)
        else:
            within = torch.rand(batch_size, self.integration_q, device=device, dtype=dtype)
        return (strata[None, :] + within) / self.integration_q

    def shared_temporal_context(
        self,
        query_times: Tensor,
        history_times: Tensor,
        history_feedback: Tensor,
        history_sid: Tensor,
        history_mask: Tensor,
        *,
        return_influence: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        source_z = self.event_representation(history_feedback, history_sid)
        pair_features = self._source_target_features(source_z)
        psi = self.interaction_network(pair_features)
        source_delays = self.delays()[history_feedback.long()]
        contexts = []
        signed_parts = []
        absolute_parts = []
        positive_parts = []
        negative_parts = []
        phi_parts = []
        for start in range(0, query_times.shape[1], self.query_chunk_size):
            query = query_times[:, start : start + self.query_chunk_size]
            lags = query[:, :, None] - history_times[:, None, :]
            valid = (lags > 0) & history_mask[:, None, :]
            distance = torch.abs(lags[:, :, :, None] - source_delays[:, None, :, :])
            pair = pair_features[:, None, :, :, :].expand(-1, query.shape[1], -1, -1, -1)
            temporal_input = torch.cat([pair, -distance[..., None]], dim=-1)
            phi = self.temporal_network(temporal_input)
            contribution = psi[:, None, :, :] * phi * valid[..., None]
            signed = contribution.sum(dim=2)
            contexts.append(signed)
            if return_influence:
                signed_parts.append(signed)
                absolute_parts.append(contribution.abs().sum(dim=2))
                positive_parts.append(contribution.clamp_min(0).sum(dim=2))
                negative_parts.append((-contribution.clamp_max(0)).sum(dim=2))
                valid_phi = phi * valid[..., None]
                denominator = valid[..., None].sum(dim=2).clamp_min(1)
                phi_parts.append(valid_phi.sum(dim=2) / denominator)
        context = torch.cat(contexts, dim=1)
        if not return_influence:
            return context, None
        return context, {
            "signed": torch.cat(signed_parts, dim=1),
            "absolute": torch.cat(absolute_parts, dim=1),
            "positive": torch.cat(positive_parts, dim=1),
            "negative_magnitude": torch.cat(negative_parts, dim=1),
            "phi_mean": torch.cat(phi_parts, dim=1),
        }

    def conditional_outputs(
        self,
        query_times: Tensor,
        history_times: Tensor,
        history_feedback: Tensor,
        history_sid: Tensor,
        history_mask: Tensor,
        *,
        return_influence: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor] | None]:
        context, diagnostics = self.shared_temporal_context(
            query_times, history_times, history_feedback, history_sid, history_mask,
            return_influence=return_influence,
        )
        total = F.softplus(self.time_head(context).squeeze(-1)).clamp_min(self.minimum_intensity)
        mark_logits = self.mark_head(context)
        probabilities = torch.softmax(mark_logits, dim=-1)
        lambdas = total[..., None] * probabilities
        return total, probabilities, lambdas, diagnostics

    def loss(self, batch: dict[str, Tensor], *, deterministic_integral: bool) -> MinimalLossOutput:
        target_time = batch["target_time"]
        previous_time = batch["previous_time"]
        interval = target_time - previous_time
        if torch.any(interval <= 0):
            raise ValueError("all target intervals must be strictly positive")
        total, probabilities, lambdas, influence = self.conditional_outputs(
            target_time[:, None], batch["history_times"], batch["history_feedback"],
            batch["history_sid"], batch["history_mask"], return_influence=True,
        )
        total = total[:, 0]
        probabilities = probabilities[:, 0, :]
        lambdas = lambdas[:, 0, :]
        ratios = self.stratified_ratios(len(target_time), deterministic=deterministic_integral)
        sample_times = previous_time[:, None] + interval[:, None] * ratios
        sampled_total, _, _, _ = self.conditional_outputs(
            sample_times, batch["history_times"], batch["history_feedback"],
            batch["history_sid"], batch["history_mask"], return_influence=False,
        )
        integral = interval * sampled_total.mean(dim=-1)
        time_by_group = -torch.log(total) + integral
        target_counts = batch["target_feedback_counts"]
        feedback_sum = -(target_counts * torch.log(probabilities.clamp_min(1e-12))).sum()
        num_events = target_counts.sum()
        time_loss = time_by_group.mean()
        feedback_loss = feedback_sum / num_events.clamp_min(1.0)
        assert influence is not None
        return MinimalLossOutput(
            optimization_loss=time_loss + feedback_loss,
            time_loss=time_loss,
            feedback_loss=feedback_loss,
            time_loss_by_group=time_by_group,
            feedback_loss_sum=feedback_sum,
            num_groups=target_time.new_tensor(float(len(target_time))),
            num_events=num_events,
            target_lambdas=lambdas,
            target_total_intensity=total,
            target_feedback_probabilities=probabilities,
            integral_by_group=integral,
            signed_influence_by_target=influence["signed"][:, 0, :],
            absolute_influence_mass_by_target=influence["absolute"][:, 0, :],
            positive_influence_mass_by_target=influence["positive"][:, 0, :],
            negative_influence_mass_by_target=influence["negative_magnitude"][:, 0, :],
        )

    @torch.no_grad()
    def time_distribution(
        self, batch: dict[str, Tensor], *, horizon_hours: float = 696.0, grid_size: int = 256
    ) -> dict[str, Tensor]:
        minimum = 5.0 / 3600.0
        positive_grid = torch.logspace(
            torch.log10(torch.tensor(minimum, device=self.time_head.weight.device)),
            torch.log10(torch.tensor(horizon_hours, device=self.time_head.weight.device)),
            grid_size - 1, device=self.time_head.weight.device, dtype=self.time_head.weight.dtype,
        )
        delta = torch.cat([positive_grid.new_zeros(1), positive_grid])
        query = batch["previous_time"][:, None] + delta[None, :]
        query[:, 0] += torch.finfo(query.dtype).eps
        total, _, _, _ = self.conditional_outputs(
            query, batch["history_times"], batch["history_feedback"],
            batch["history_sid"], batch["history_mask"], return_influence=False,
        )
        widths = delta[1:] - delta[:-1]
        increments = 0.5 * (total[:, 1:] + total[:, :-1]) * widths[None, :]
        cumulative = torch.cat([torch.zeros(len(query), 1, device=query.device), torch.cumsum(increments, dim=1)], dim=1)
        survival = torch.exp(-cumulative.clamp(max=80.0))
        return {
            "expected_delta_hours": torch.trapezoid(survival, delta, dim=1),
            "event_mass_within_horizon": 1.0 - survival[:, -1],
            "survival_at_horizon": survival[:, -1],
        }
