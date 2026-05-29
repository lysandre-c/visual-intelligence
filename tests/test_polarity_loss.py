"""Unit tests for the Symmetric Polarity Preference Loss module."""

from __future__ import annotations

import pytest
import torch

from src.rl.loss import SymmetricPolarityPreferenceLoss


# ────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────

@pytest.fixture
def loss_fn() -> SymmetricPolarityPreferenceLoss:
    return SymmetricPolarityPreferenceLoss(
        beta=0.1, gamma=1.0, label_lambda=0.5, eta=0.1,
    )


@pytest.fixture
def batch_size() -> int:
    return 4


def _random_logps(batch_size: int) -> torch.Tensor:
    """Generate random negative log-probs (typical range -10 to -1)."""
    return -torch.rand(batch_size) * 9 - 1


# ────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────


class TestSymmetricPolarityPreferenceLoss:
    """Tests for the composite DPO loss."""

    def test_output_shapes(self, loss_fn, batch_size):
        """All five outputs should be scalar tensors."""
        total, l_dpo, l_sym, l_margin, l_ancpo = loss_fn(
            _random_logps(batch_size),
            _random_logps(batch_size),
            _random_logps(batch_size),
            _random_logps(batch_size),
            _random_logps(batch_size),
            _random_logps(batch_size),
            _random_logps(batch_size),
            _random_logps(batch_size),
        )
        for t in (total, l_dpo, l_sym, l_margin, l_ancpo):
            assert t.shape == (), f"Expected scalar, got shape {t.shape}"

    def test_total_is_sum_of_components(self, loss_fn, batch_size):
        """total_loss should equal the weighted sum of the four terms."""
        total, l_dpo, l_sym, l_margin, l_ancpo = loss_fn(
            _random_logps(batch_size),
            _random_logps(batch_size),
            _random_logps(batch_size),
            _random_logps(batch_size),
            _random_logps(batch_size),
            _random_logps(batch_size),
            _random_logps(batch_size),
            _random_logps(batch_size),
        )
        expected = (
            l_dpo
            + loss_fn.gamma * l_sym
            + loss_fn.label_lambda * l_margin
            + loss_fn.eta * l_ancpo
        )
        assert torch.allclose(total, expected, atol=1e-6), (
            f"Total {total.item():.6f} != expected {expected.item():.6f}"
        )

    def test_margin_zero_when_logits_equal(self, loss_fn, batch_size):
        """L_margin should be 0 when original and inverted margins match.

        If we feed identical log-probs for both the original and inverted
        directions, the margin difference is zero.
        """
        chosen_logps = _random_logps(batch_size)
        rejected_logps = _random_logps(batch_size)
        ref_chosen = _random_logps(batch_size)
        ref_rejected = _random_logps(batch_size)

        _, _, _, l_margin, _ = loss_fn(
            chosen_logps,
            rejected_logps,
            ref_chosen,
            ref_rejected,
            # Inverted = same values → identical logits
            chosen_logps,
            rejected_logps,
            ref_chosen,
            ref_rejected,
        )
        assert torch.allclose(l_margin, torch.tensor(0.0), atol=1e-6), (
            f"L_margin should be 0 when polarities are identical, got {l_margin.item()}"
        )

    def test_dpo_loss_decreases_with_higher_chosen_ratio(self, loss_fn):
        """L_dpo should decrease as policy favours the chosen response more."""
        batch_size = 8
        ref_chosen = _random_logps(batch_size)
        ref_rejected = _random_logps(batch_size)
        inv_c = _random_logps(batch_size)
        inv_r = _random_logps(batch_size)
        ref_inv_c = _random_logps(batch_size)
        ref_inv_r = _random_logps(batch_size)

        # Scenario 1: policy slightly prefers chosen
        losses_low = loss_fn(
            torch.full((batch_size,), -2.0),   # chosen
            torch.full((batch_size,), -3.0),   # rejected
            ref_chosen, ref_rejected,
            inv_c, inv_r, ref_inv_c, ref_inv_r,
        )

        # Scenario 2: policy strongly prefers chosen
        losses_high = loss_fn(
            torch.full((batch_size,), -1.0),   # chosen (higher)
            torch.full((batch_size,), -5.0),   # rejected (lower)
            ref_chosen, ref_rejected,
            inv_c, inv_r, ref_inv_c, ref_inv_r,
        )

        assert losses_high[1] < losses_low[1], (
            "L_dpo should be lower when policy more strongly prefers chosen"
        )

    def test_gradient_flows(self, loss_fn, batch_size):
        """Gradients should flow through all inputs."""
        inputs = [
            _random_logps(batch_size).requires_grad_(True)
            for _ in range(8)
        ]
        total, *_ = loss_fn(*inputs)
        total.backward()

        for i, inp in enumerate(inputs):
            assert inp.grad is not None, f"Input {i} has no gradient"
            assert not torch.all(inp.grad == 0), f"Input {i} gradient is all zeros"

    def test_ancpo_encourages_high_chosen_logps(self, loss_fn):
        """L_ancpo should be lower when chosen log-probs are higher."""
        batch_size = 4
        ref_c = _random_logps(batch_size)
        ref_r = _random_logps(batch_size)
        ref_inv_c = _random_logps(batch_size)
        ref_inv_r = _random_logps(batch_size)
        rej = _random_logps(batch_size)
        rej_inv = _random_logps(batch_size)

        # Low chosen log-probs
        _, _, _, _, ancpo_low = loss_fn(
            torch.full((batch_size,), -8.0),
            rej, ref_c, ref_r,
            torch.full((batch_size,), -8.0),
            rej_inv, ref_inv_c, ref_inv_r,
        )

        # High chosen log-probs
        _, _, _, _, ancpo_high = loss_fn(
            torch.full((batch_size,), -1.0),
            rej, ref_c, ref_r,
            torch.full((batch_size,), -1.0),
            rej_inv, ref_inv_c, ref_inv_r,
        )

        assert ancpo_high < ancpo_low, (
            "L_ancpo should be lower when chosen log-probs are higher"
        )

    def test_batch_size_one(self, loss_fn):
        """Loss should work with batch_size=1."""
        total, l_dpo, l_sym, l_margin, l_ancpo = loss_fn(
            _random_logps(1), _random_logps(1),
            _random_logps(1), _random_logps(1),
            _random_logps(1), _random_logps(1),
            _random_logps(1), _random_logps(1),
        )
        assert torch.isfinite(total), "Loss should be finite for batch_size=1"

    def test_custom_hyperparameters(self):
        """Loss should respect non-default hyperparameters."""
        fn = SymmetricPolarityPreferenceLoss(
            beta=0.5, gamma=2.0, label_lambda=1.0, eta=0.5,
        )
        assert fn.beta == 0.5
        assert fn.gamma == 2.0
        assert fn.label_lambda == 1.0
        assert fn.eta == 0.5

        bs = 4
        total, l_dpo, l_sym, l_margin, l_ancpo = fn(
            _random_logps(bs), _random_logps(bs),
            _random_logps(bs), _random_logps(bs),
            _random_logps(bs), _random_logps(bs),
            _random_logps(bs), _random_logps(bs),
        )
        expected = l_dpo + 2.0 * l_sym + 1.0 * l_margin + 0.5 * l_ancpo
        assert torch.allclose(total, expected, atol=1e-6)
