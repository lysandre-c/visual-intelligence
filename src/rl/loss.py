"""Symmetric Polarity-Inverted Preference Loss for multimodal DPO.

This module implements the four-term composite loss that jointly optimises
the original and polarity-inverted preference signals on the *same* static
illusion image, with margin consistency and anchored-preference
regularisation.

Joint loss:
    L(θ) = L_DPO_m(θ) + γ·L_Symmetric(θ) + λ·L_Margin(θ) + η·L_AncPO(θ)

Because the image I is held constant within each term, the standard DPO
partition functions cancel out perfectly, guaranteeing theoretical
consistency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricPolarityPreferenceLoss(nn.Module):
    """Four-term Symmetric Polarity DPO loss.

    Parameters
    ----------
    beta : float
        Temperature scaling for DPO log-sigmoid.  Controls how sharply
        the loss penalises deviations from the reference model.
    gamma : float
        Weight for the symmetric (polarity-inverted) DPO term.
    label_lambda : float
        Weight for the preference-margin consistency term.
    eta : float
        Weight for the anchored preference (AncPO) stabilisation term.
    """

    def __init__(
        self,
        beta: float = 0.1,
        gamma: float = 1.0,
        label_lambda: float = 0.5,
        eta: float = 0.1,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.label_lambda = label_lambda
        self.eta = eta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        policy_chosen_logps_inverted: torch.Tensor,
        policy_rejected_logps_inverted: torch.Tensor,
        reference_chosen_logps_inverted: torch.Tensor,
        reference_rejected_logps_inverted: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the joint symmetric polarity DPO loss.

        All inputs are 1-D tensors of shape ``(batch_size,)`` containing
        sequence-level log-probabilities (sum of per-token log-probs over
        the completion tokens).

        Parameters
        ----------
        policy_chosen_logps : log π_θ(y_w | x, I)
        policy_rejected_logps : log π_θ(y_l | x, I)
        reference_chosen_logps : log π_ref(y_w | x, I)
        reference_rejected_logps : log π_ref(y_l | x, I)
        policy_chosen_logps_inverted : log π_θ(y'_w | x', I)
        policy_rejected_logps_inverted : log π_θ(y'_l | x', I)
        reference_chosen_logps_inverted : log π_ref(y'_w | x', I)
        reference_rejected_logps_inverted : log π_ref(y'_l | x', I)

        Returns
        -------
        total_loss, loss_dpo_m, loss_symmetric, loss_margin, loss_ancpo
            Each a scalar tensor (shape ``()``).
        """
        # ── 1. Standard Multimodal DPO Loss ─────────────────────────────
        # logits = β · [(log π_θ(y_w) - log π_θ(y_l))
        #             - (log π_ref(y_w) - log π_ref(y_l))]
        policy_log_ratio_orig = policy_chosen_logps - policy_rejected_logps
        reference_log_ratio_orig = reference_chosen_logps - reference_rejected_logps
        logits_orig = policy_log_ratio_orig - reference_log_ratio_orig
        loss_dpo_m = -F.logsigmoid(self.beta * logits_orig).mean()

        # ── 2. Symmetric Polarity Loss ──────────────────────────────────
        # Same structure as above but on the polarity-inverted prompt x'.
        policy_log_ratio_inv = (
            policy_chosen_logps_inverted - policy_rejected_logps_inverted
        )
        reference_log_ratio_inv = (
            reference_chosen_logps_inverted - reference_rejected_logps_inverted
        )
        logits_inv = policy_log_ratio_inv - reference_log_ratio_inv
        loss_symmetric = -F.logsigmoid(self.beta * logits_inv).mean()

        # ── 3. Preference Margin Consistency Loss ───────────────────────
        # Penalises variance between the original and inverted preference
        # gaps on the same image.
        margin_difference = logits_orig - logits_inv
        loss_margin = torch.square(margin_difference).mean()

        # ── 4. Anchored Preference Loss (AncPO) ────────────────────────
        # Stabilises absolute log-likelihoods of chosen responses against
        # displacement.
        loss_ancpo = -(
            policy_chosen_logps + policy_chosen_logps_inverted
        ).mean()

        # ── Joint loss ──────────────────────────────────────────────────
        total_loss = (
            loss_dpo_m
            + self.gamma * loss_symmetric
            + self.label_lambda * loss_margin
            + self.eta * loss_ancpo
        )

        return total_loss, loss_dpo_m, loss_symmetric, loss_margin, loss_ancpo
