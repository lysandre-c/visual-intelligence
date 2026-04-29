"""Linear probe training and evaluation for CNN / ViT backbones.

The probe is a 3-way logistic regression head trained on held-out non-illusion
stimuli (e.g. ImageNet validation images) and evaluated on illusion pairs.

Design contract
---------------
- The backbone is always kept frozen; only the linear head is optimised.
- A sanity check verifies that the control-image accuracy exceeds
  ``control_ceiling_threshold`` (default 0.80) before the probe is used for
  illusion evaluation.  Failing this check means the task is not discriminable
  for the model and HEAS results would be meaningless.
- Labels: 0 → correct, 1 → illusory, 2 → other.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class LinearProbeProtocol:
    """Train and evaluate a 3-way linear probe on top of a frozen backbone.

    Parameters
    ----------
    prober :
        A ``_CNNProber`` or ``_ViTProber`` instance (anything with an
        ``extract_features`` method and a ``feature_dim`` attribute).
    lr : Learning rate for the Adam optimiser.
    epochs : Number of training epochs.
    batch_size : Training batch size.
    weight_decay : L2 regularisation.
    control_ceiling_threshold :
        Minimum fraction of control stimuli that must be correctly classified
        before probe results are accepted.
    device : Torch device string (falls back to ``prober.device``).
    """

    def __init__(
        self,
        prober: Any,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
        control_ceiling_threshold: float = 0.80,
    ) -> None:
        self.prober = prober
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.control_ceiling_threshold = control_ceiling_threshold
        self.device = prober.device

    # ------------------------------------------------------------------
    # Feature caching
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_feature_dataset(
        self,
        images: list,  # list[PIL.Image.Image]
        labels: list[int],
        desc: str = "Extracting features",
    ) -> TensorDataset:
        """Extract backbone features for a list of PIL images + integer labels."""
        from tqdm import tqdm

        feats = []
        for img in tqdm(images, desc=desc, leave=False):
            feats.append(self.prober.extract_features(img).cpu())
        X = torch.cat(feats, dim=0)
        y = torch.tensor(labels, dtype=torch.long)
        return TensorDataset(X, y)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_images: list,
        train_labels: list[int],
        val_images: list | None = None,
        val_labels: list[int] | None = None,
    ) -> nn.Linear:
        """Train the linear probe and attach it to ``self.prober``.

        Parameters
        ----------
        train_images : Non-illusion training images (PIL).
        train_labels : Integer class labels (0 = correct, 1 = illusory, 2 = other).
        val_images, val_labels : Optional validation set for early stopping.

        Returns
        -------
        nn.Linear
            The trained probe (also stored at ``self.prober.probe``).
        """
        train_ds = self.extract_feature_dataset(train_images, train_labels, "Train features")
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        probe = nn.Linear(self.prober.feature_dim, 3).to(self.device)
        optimiser = torch.optim.Adam(probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Pre-extract validation features once (not every epoch)
        val_ds: TensorDataset | None = None
        if val_images is not None and val_labels is not None:
            val_ds = self.extract_feature_dataset(val_images, val_labels, "Val features")

        best_val_loss = float("inf")
        epochs_no_improve = 0
        patience = max(5, self.epochs // 5)  # early-stop patience

        for epoch in range(self.epochs):
            probe.train()
            total_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimiser.zero_grad()
                loss = criterion(probe(X_batch), y_batch)
                loss.backward()
                optimiser.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            if val_ds is not None:
                val_loss, val_acc = self._evaluate_loss_and_accuracy(probe, val_ds, criterion)
                overfit_gap = avg_loss - val_loss  # negative = val worse than train
                logger.info(
                    "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f%s",
                    epoch + 1, self.epochs, avg_loss, val_loss, val_acc,
                    "  ⚠ overfit" if avg_loss < val_loss * 0.5 else "",
                )
                # Early stopping on validation loss
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        logger.info("Early stopping at epoch %d (no val improvement for %d epochs).", epoch + 1, patience)
                        break
            else:
                logger.info("Epoch %d/%d  loss=%.4f", epoch + 1, self.epochs, avg_loss)

        self.prober.attach_probe(probe)
        return probe

    # ------------------------------------------------------------------
    # Sanity check: control-image ceiling
    # ------------------------------------------------------------------

    def check_control_ceiling(
        self,
        control_images: list,
        control_labels: list[int],
    ) -> float:
        """Verify the probe achieves acceptable accuracy on control stimuli.

        Returns
        -------
        float
            Fraction of controls correctly classified.

        Raises
        ------
        RuntimeError
            If accuracy falls below ``self.control_ceiling_threshold``.
        """
        if self.prober.probe is None:
            raise RuntimeError("Probe not trained yet.")
        ctrl_ds = self.extract_feature_dataset(control_images, control_labels, "Control features")
        acc = self._evaluate_accuracy(self.prober.probe, ctrl_ds)
        logger.info("Control ceiling accuracy: %.3f (threshold: %.2f)", acc, self.control_ceiling_threshold)
        if acc < self.control_ceiling_threshold:
            raise RuntimeError(
                f"Control ceiling {acc:.3f} < threshold {self.control_ceiling_threshold:.2f}. "
                "HEAS comparison would be meaningless for this model."
            )
        return acc

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        self.prober.save_probe(path)

    def load(self, path: Path) -> None:
        self.prober.load_probe(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _evaluate_accuracy(probe: nn.Linear, dataset: TensorDataset) -> float:
        device = next(probe.parameters()).device
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        correct = total = 0
        probe.eval()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = probe(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
        return correct / total if total > 0 else 0.0

    @staticmethod
    @torch.no_grad()
    def _evaluate_loss_and_accuracy(
        probe: nn.Linear, dataset: TensorDataset, criterion: nn.Module
    ) -> tuple[float, float]:
        device = next(probe.parameters()).device
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        total_loss = correct = total = 0
        probe.eval()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = probe(X_batch)
            total_loss += criterion(logits, y_batch).item() * len(y_batch)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += len(y_batch)
        return (total_loss / total if total > 0 else 0.0,
                correct / total if total > 0 else 0.0)
