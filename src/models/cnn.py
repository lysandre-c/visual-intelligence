"""Supervised CNN model probers (ResNet-50, ConvNeXt-Base).

Both models use a trained linear probe on top of frozen backbone features.
The linear probe is a 3-way classifier: (correct, illusory, other).
It must be trained on held-out non-illusion stimuli before use; the training
is handled by ``src.probing.linear_probe.LinearProbeProtocol``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

from .base import ModelProber, ResponseDistribution


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_TRANSFORM = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
)


class _CNNProber(ModelProber):
    """Shared implementation for CNN-based probers."""

    model_name: str = ""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        device: str | None = None,
        probe_path: Path | None = None,
    ) -> None:
        super().__init__(device)
        self.backbone = backbone.to(self.device).eval()
        self.feature_dim = feature_dim
        self.probe: nn.Linear | None = None
        if probe_path is not None:
            self.load_probe(probe_path)

    # ------------------------------------------------------------------
    # Probe management
    # ------------------------------------------------------------------

    def attach_probe(self, probe: nn.Linear) -> None:
        self.probe = probe.to(self.device)

    def load_probe(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        probe = nn.Linear(self.feature_dim, 3)
        probe.load_state_dict(state)
        self.attach_probe(probe)

    def save_probe(self, path: Path) -> None:
        if self.probe is None:
            raise RuntimeError("No probe attached; train one first.")
        torch.save(self.probe.state_dict(), path)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """Return the backbone feature vector for a single PIL image."""
        x = _TRANSFORM(image).unsqueeze(0).to(self.device)
        return self.backbone(x)

    # ------------------------------------------------------------------
    # ModelProber interface
    # ------------------------------------------------------------------

    def probe_pair(
        self,
        illusion: Image.Image,
        control: Image.Image,
        correct_answer: str,
        illusory_answer: str,
        category: str,
        illusion_type: str,
        extra: dict[str, Any] | None = None,
    ) -> ResponseDistribution:
        if self.probe is None:
            raise RuntimeError(
                f"{self.model_name}: linear probe not attached. "
                "Run LinearProbeProtocol.train() first."
            )
        self.probe.eval()

        feat_ill = self.extract_features(illusion)
        feat_ctrl = self.extract_features(control)

        logits_ill = self.probe(feat_ill).squeeze(0)
        logits_ctrl = self.probe(feat_ctrl).squeeze(0)

        probs_ill = torch.softmax(logits_ill, dim=0).cpu().tolist()
        probs_ctrl = torch.softmax(logits_ctrl, dim=0).cpu().tolist()

        return ResponseDistribution(
            correct=probs_ill[0],
            illusory=probs_ill[1],
            other=probs_ill[2],
            raw={
                "logits_illusion": logits_ill.cpu().tolist(),
                "logits_control": logits_ctrl.cpu().tolist(),
                "probs_control": probs_ctrl,
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# ResNet-50
# ──────────────────────────────────────────────────────────────────────────────

class ResNetProber(_CNNProber):
    """ResNet-50 backbone with a 3-way linear probe head."""

    model_name = "resnet50"

    def __init__(self, device: str | None = None, probe_path: Path | None = None) -> None:
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2
        base = tvm.resnet50(weights=weights)
        # Remove the classification head; keep the global average pool
        feature_dim = base.fc.in_features
        base.fc = nn.Identity()
        super().__init__(base, feature_dim, device, probe_path)


# ──────────────────────────────────────────────────────────────────────────────
# ConvNeXt-Base
# ──────────────────────────────────────────────────────────────────────────────

class ConvNeXtProber(_CNNProber):
    """ConvNeXt-Base backbone with a 3-way linear probe head."""

    model_name = "convnext_base"

    def __init__(self, device: str | None = None, probe_path: Path | None = None) -> None:
        weights = tvm.ConvNeXt_Base_Weights.IMAGENET1K_V1
        base = tvm.convnext_base(weights=weights)
        # ConvNeXt classifier: Sequential(LayerNorm, Flatten, Linear)
        feature_dim = base.classifier[2].in_features
        base.classifier = nn.Sequential(
            base.classifier[0],  # LayerNorm
            base.classifier[1],  # Flatten
            nn.Identity(),
        )
        super().__init__(base, feature_dim, device, probe_path)
