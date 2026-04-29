"""GradCAM saliency maps for CNN and ViT models.

Uses ``pytorch-grad-cam`` (``pip install grad-cam``) for the heavy lifting.
We expose a single high-level function ``compute_gradcam`` that works with
any model stored inside a ``_CNNProber`` or ``_ViTProber``.

The function returns a normalised heatmap (numpy float32 array in [0, 1]) at
the same spatial resolution as the input image.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


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


def _get_target_layer(prober: Any) -> torch.nn.Module:
    """Heuristically identify the last convolutional / attention layer."""
    backbone = prober.backbone
    # ResNet
    if hasattr(backbone, "layer4"):
        return backbone.layer4[-1]
    # ConvNeXt — last block
    if hasattr(backbone, "features"):
        return backbone.features[-1]
    # ViT via timm — last transformer block
    if hasattr(backbone, "blocks"):
        return backbone.blocks[-1].norm1
    raise ValueError(
        f"Cannot automatically determine target layer for {type(backbone).__name__}. "
        "Pass target_layer explicitly."
    )


def compute_gradcam(
    prober: Any,
    image: Image.Image,
    target_class: int | None = None,
    target_layer: torch.nn.Module | None = None,
    use_eigen_cam: bool = False,
) -> np.ndarray:
    """Compute a GradCAM saliency map.

    Parameters
    ----------
    prober :
        A ``_CNNProber`` or ``_ViTProber`` with an attached linear probe.
    image :
        The input PIL image.
    target_class :
        Index of the class to attribute (0=correct, 1=illusory, 2=other).
        If None, uses the argmax of the probe output.
    target_layer :
        The layer to hook.  Auto-detected if None.
    use_eigen_cam :
        Use EigenCAM instead of GradCAM (no gradients needed; faster but
        less precise).

    Returns
    -------
    np.ndarray of shape (H, W), dtype float32, values in [0, 1].
    """
    from pytorch_grad_cam import GradCAM, EigenCAM  # type: ignore
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # type: ignore

    if target_layer is None:
        target_layer = _get_target_layer(prober)

    # Build a wrapper model that composes backbone + linear probe
    class _ProbeModel(torch.nn.Module):
        def __init__(self, backbone, probe):
            super().__init__()
            self.backbone = backbone
            self.probe = probe

        def forward(self, x):
            return self.probe(self.backbone(x))

    if prober.probe is None:
        raise RuntimeError("No probe attached to prober.")

    model = _ProbeModel(prober.backbone, prober.probe).to(prober.device).eval()

    # Prepare input
    input_tensor = _TRANSFORM(image).unsqueeze(0).to(prober.device)

    # Determine target class
    if target_class is None:
        with torch.no_grad():
            logits = model(input_tensor)
        target_class = int(logits.argmax(dim=1).item())

    targets = [ClassifierOutputTarget(target_class)]

    CamClass = EigenCAM if use_eigen_cam else GradCAM

    # Wrap target layer in a list as required by pytorch-grad-cam
    with CamClass(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    return grayscale_cam[0]  # shape (H, W), float32
