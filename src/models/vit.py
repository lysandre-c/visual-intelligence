"""Supervised Vision Transformer model probers (ViT-B/16, ViT-L/16) via timm.

Same linear-probe paradigm as the CNN probers: the backbone is frozen and a
3-way head is trained on held-out non-illusion data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from .base import ModelProber, ResponseDistribution


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


class _ViTProber(ModelProber):
    """Shared implementation for timm-based ViT probers."""

    model_name: str = ""
    _timm_name: str = ""
    _input_size: int = 224

    def __init__(self, device: str | None = None, probe_path: Path | None = None) -> None:
        super().__init__(device)
        # Load backbone without the classification head
        self.backbone = timm.create_model(
            self._timm_name, pretrained=True, num_classes=0
        ).to(self.device).eval()
        self.feature_dim: int = self.backbone.num_features
        self.transform = _build_transform(self._input_size)
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
        x = self.transform(image).unsqueeze(0).to(self.device)
        return self.backbone(x)  # (1, feature_dim)

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
# ViT-B/16
# ──────────────────────────────────────────────────────────────────────────────

class ViTBProber(_ViTProber):
    """ViT-B/16 (ImageNet-21k → ImageNet-1k fine-tune) + linear probe."""

    model_name = "vit_b_16"
    _timm_name = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
    _input_size = 224


# ──────────────────────────────────────────────────────────────────────────────
# ViT-L/16
# ──────────────────────────────────────────────────────────────────────────────

class ViTLProber(_ViTProber):
    """ViT-L/16 (ImageNet-21k → ImageNet-1k fine-tune) + linear probe."""

    model_name = "vit_l_16"
    _timm_name = "vit_large_patch16_224.augreg_in21k_ft_in1k"
    _input_size = 224
