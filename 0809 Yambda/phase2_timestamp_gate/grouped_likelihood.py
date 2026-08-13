"""Audit-only implementation of protocol B grouped pseudo-likelihood.

This is a mathematical/engineering gate harness, not Minimal SNMPP training.
It deliberately uses only four feedback marks and a transparent signed temporal
kernel so that history timing and likelihood accounting are directly testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class IntensityDetails:
    lambdas: Tensor
    signed_influence: Tensor
    absolute_mass: Tensor
    positive_mass: Tensor
    negative_mass: Tensor


@dataclass
class GroupLoss:
    time: Tensor
    mark_sum: Tensor
    mark_mean: Tensor
    theoretical: Tensor
    normalized_multitask: Tensor
    intensity: Tensor
    q: Tensor
    integral: Tensor
    details: IntensityDetails


class GroupedLikelihoodGate(nn.Module):
    """Small signed temporal model used only to validate protocol mechanics."""

    def __init__(self, *, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        # Fixed deterministic, non-degenerate initialization. No optimization is run.
        self.base_score = nn.Parameter(torch.tensor([-0.10, -0.35, -0.20, -0.55], dtype=dtype))
        amp = torch.tensor(
            [
                [0.030, -0.022, 0.045, -0.014],
                [-0.018, 0.028, -0.012, 0.040],
                [-0.035, 0.016, 0.025, -0.009],
                [0.012, -0.030, 0.010, 0.022],
            ],
            dtype=dtype,
        )
        self.signed_amplitude = nn.Parameter(amp)
        self.raw_decay = nn.Parameter(torch.full((4, 4), -1.50, dtype=dtype))
        self.raw_delay = nn.Parameter(torch.full((4, 4), -4.00, dtype=dtype))
        self.eps = 1e-12
        self.quadrature_points = 8

    @property
    def dtype(self) -> torch.dtype:
        return self.base_score.dtype

    def intensities(
        self, history_feedback: Tensor, history_times_hours: Tensor, query_time_hours: Tensor
    ) -> IntensityDetails:
        query_time_hours = query_time_hours.to(dtype=self.dtype)
        if history_feedback.numel() == 0:
            influence_by_event = self.base_score.new_zeros((0, 4))
        else:
            history_feedback = history_feedback.to(dtype=torch.long)
            history_times_hours = history_times_hours.to(dtype=self.dtype)
            delta = (query_time_hours - history_times_hours).clamp_min(0.0)[:, None]
            decay = F.softplus(self.raw_decay[history_feedback]) + 1e-8
            delay = F.softplus(self.raw_delay[history_feedback])
            temporal = torch.exp(-decay * torch.abs(delta - delay))
            influence_by_event = self.signed_amplitude[history_feedback] * temporal

        signed = influence_by_event.sum(dim=0)
        absolute = influence_by_event.abs().sum(dim=0)
        positive = influence_by_event.clamp_min(0.0).sum(dim=0)
        negative = influence_by_event.clamp_max(0.0).sum(dim=0)
        lambdas = F.softplus(self.base_score + signed) + self.eps
        return IntensityDetails(lambdas, signed, absolute, positive, negative)

    def integrate_total_intensity(
        self,
        history_feedback: Tensor,
        history_times_hours: Tensor,
        interval_start_hours: Tensor,
        interval_end_hours: Tensor,
    ) -> Tensor:
        width = interval_end_hours - interval_start_hours
        if not bool(width > 0):
            raise ValueError("group-level interval must be strictly positive")
        fractions = (
            torch.arange(self.quadrature_points, dtype=self.dtype, device=width.device) + 0.5
        ) / self.quadrature_points
        query = interval_start_hours + width * fractions
        totals = []
        for t in query:
            totals.append(self.intensities(history_feedback, history_times_hours, t).lambdas.sum())
        return width * torch.stack(totals).mean()

    def group_loss(
        self,
        history_feedback: Tensor,
        history_times_hours: Tensor,
        previous_time_hours: Tensor,
        target_time_hours: Tensor,
        target_feedback: Tensor,
    ) -> GroupLoss:
        if target_feedback.numel() == 0:
            raise ValueError("a timestamp group must contain at least one mark")
        details = self.intensities(history_feedback, history_times_hours, target_time_hours)
        total = details.lambdas.sum()
        q = details.lambdas / total
        integral = self.integrate_total_intensity(
            history_feedback, history_times_hours, previous_time_hours, target_time_hours
        )
        time_loss = -torch.log(total) + integral
        per_mark = -torch.log(q[target_feedback.to(dtype=torch.long)])
        mark_sum = per_mark.sum()
        mark_mean = per_mark.mean()
        return GroupLoss(
            time=time_loss,
            mark_sum=mark_sum,
            mark_mean=mark_mean,
            theoretical=time_loss + mark_sum,
            normalized_multitask=time_loss + mark_mean,
            intensity=total,
            q=q,
            integral=integral,
            details=details,
        )


def tensors_for_record(record: dict, *, dtype: torch.dtype = torch.float64) -> tuple[Tensor, ...]:
    """Use the immediately preceding complete group as strict pre-history."""
    previous_time = torch.tensor(float(record["previous_timestamp"]) / 3600.0, dtype=dtype)
    target_time = torch.tensor(float(record["target_timestamp"]) / 3600.0, dtype=dtype)
    history_feedback = torch.tensor(record["previous_feedback_multiset"], dtype=torch.long)
    history_times = torch.full((history_feedback.numel(),), previous_time.item(), dtype=dtype)
    target_feedback = torch.tensor(record["target_feedback_multiset"], dtype=torch.long)
    return history_feedback, history_times, previous_time, target_time, target_feedback
