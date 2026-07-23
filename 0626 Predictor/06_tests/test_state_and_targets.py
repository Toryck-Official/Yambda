from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_model"))

from predictor import predictor_loss
from soft_state import SoftStateBuilder
from state_encoder import StateEncoder


def state_batch(padded_value: float = 0.0, same_session: list[int] | None = None) -> dict[str, torch.Tensor]:
    features = torch.zeros(1, 4, 4)
    features[:, :2] = padded_value
    features[0, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    features[0, 3] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    return {
        "history_features": features,
        "history_mask": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        "history_response_targets": torch.zeros(1, 4, 5),
        "history_play_ratios": torch.zeros(1, 4),
        "history_play_excesses": torch.zeros(1, 4),
        "history_is_organic": torch.zeros(1, 4, dtype=torch.long),
        "history_time_gap_seconds": torch.zeros(1, 4),
        "history_same_session": torch.tensor([same_session or [0, 0, 1, 1]]),
    }


class StateEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.encoder = StateEncoder(
            item_dim=4,
            d_model=8,
            max_seq_len=4,
            n_layer=1,
            n_head=2,
            dropout=0.0,
        ).eval()

    def test_padding_values_do_not_change_state(self) -> None:
        base = self.encoder(state_batch(0.0))["state_emb"]
        changed = self.encoder(state_batch(1000.0))["state_emb"]
        torch.testing.assert_close(base, changed, atol=1e-6, rtol=1e-6)

    def test_session_membership_changes_short_and_long_state(self) -> None:
        current_only = self.encoder(state_batch(same_session=[0, 0, 0, 1]))
        both_current = self.encoder(state_batch(same_session=[0, 0, 1, 1]))
        self.assertFalse(torch.allclose(current_only["short_state_emb"], both_current["short_state_emb"]))
        self.assertFalse(torch.allclose(current_only["long_state_emb"], both_current["long_state_emb"]))


class ObjectiveTest(unittest.TestCase):
    def test_response_target_is_multi_label(self) -> None:
        out = {
            "response_logits": torch.tensor([[8.0, 8.0, -8.0, -8.0, -8.0]]),
            "predicted_play_ratio": torch.tensor([0.9]),
            "predicted_reward": torch.tensor([1.0]),
            "regret_logits": torch.tensor([[8.0, -8.0, -8.0, -8.0]]),
        }
        batch = {
            "response_targets": torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]]),
            "played_ratio": torch.tensor([0.9]),
            "reward": torch.tensor([1.0]),
            "regret_type_id": torch.tensor([0]),
        }
        losses = predictor_loss(
            out,
            batch,
            play_weight=0.0,
            reward_weight=0.0,
            regret_weight=0.0,
        )
        self.assertLess(float(losses["response_loss"]), 0.001)

    def test_soft_state_receives_state_supervision_gradient(self) -> None:
        builder = SoftStateBuilder(item_dim=4, d_model=8, dropout=0.0)
        pred = {
            "response_probs": torch.full((2, 5), 0.2),
            "predicted_play_ratio": torch.tensor([0.5, 0.8]),
            "predicted_reward": torch.tensor([0.1, 0.3]),
            "regret_probs": torch.full((2, 4), 0.25),
        }
        output = builder(torch.zeros(2, 8), torch.ones(2, 4), pred)
        output["next_state_emb"].square().mean().backward()
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in builder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
