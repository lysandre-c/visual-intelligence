"""Attention rollout for Vision Transformer models.

Implements the attention rollout algorithm from Abnar & Zuidema (2020),
which recursively multiplies attention matrices through the transformer
layers to obtain a single per-token relevance map.

Reference: "Quantifying Attention Flow in Transformers"
           https://arxiv.org/abs/2005.00928
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


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


def compute_attention_rollout(
    prober: Any,
    image: Image.Image,
    discard_ratio: float = 0.9,
    head_fusion: str = "mean",
    input_size: int = 224,
    patch_size: int = 16,
) -> np.ndarray:
    """Compute an attention-rollout saliency map for a ViT model.

    Parameters
    ----------
    prober :
        A ``_ViTProber`` instance (must have ``prober.backbone`` with
        ``blocks`` attribute — timm ViT).
    image :
        The input PIL image.
    discard_ratio :
        Fraction of the lowest-attention tokens to zero out per layer (noise
        reduction; Abnar & Zuidema set this to 0.9).
    head_fusion :
        How to aggregate across attention heads: ``"mean"`` (default),
        ``"min"``, or ``"max"``.
    input_size :
        Spatial resolution fed to the model.
    patch_size :
        ViT patch size in pixels (16 for ViT-B/16 and ViT-L/16).

    Returns
    -------
    np.ndarray of shape (input_size, input_size), dtype float32.
    """
    backbone = prober.backbone
    device = prober.device

    # ---- hook collection ----
    attention_matrices: list[torch.Tensor] = []

    def _hook(module, input, output):
        # timm ViT block stores attention weights in the Attention sub-module.
        # We re-run the attention manually here; alternatively hooks can be
        # registered on the Attention module directly.
        pass

    hooks = []
    attn_modules = []
    for block in backbone.blocks:
        attn_modules.append(block.attn)

    # Forward pass with stored attention
    transform = _build_transform(input_size)
    x = transform(image).unsqueeze(0).to(device)

    # Patch timm's attention to capture weights
    original_forwards = []

    def make_attn_hook(attn_module):
        original_fwd = attn_module.forward

        def forward_with_attn(x_in):
            B, N, C = x_in.shape
            qkv = attn_module.qkv(x_in).reshape(
                B, N, 3, attn_module.num_heads, C // attn_module.num_heads
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            scale = (C // attn_module.num_heads) ** -0.5
            attn_w = (q @ k.transpose(-2, -1)) * scale
            attn_w = attn_w.softmax(dim=-1)
            attention_matrices.append(attn_w.detach().cpu())
            x_out = (attn_w @ v).transpose(1, 2).reshape(B, N, C)
            x_out = attn_module.proj(x_out)
            x_out = attn_module.proj_drop(x_out)
            return x_out

        attn_module.forward = forward_with_attn
        return original_fwd

    for attn in attn_modules:
        original_forwards.append(make_attn_hook(attn))

    with torch.no_grad():
        backbone(x)

    # Restore original forward methods
    for attn, orig in zip(attn_modules, original_forwards):
        attn.forward = orig

    # ---- rollout ----
    n_patches = (input_size // patch_size) ** 2
    n_tokens = n_patches + 1  # +1 for CLS token

    result = torch.eye(n_tokens)
    for attn_w in attention_matrices:
        # attn_w shape: (1, heads, tokens, tokens)
        attn_w = attn_w.squeeze(0)  # (heads, tokens, tokens)
        if head_fusion == "mean":
            attn_fused = attn_w.mean(dim=0)
        elif head_fusion == "min":
            attn_fused = attn_w.min(dim=0).values
        elif head_fusion == "max":
            attn_fused = attn_w.max(dim=0).values
        else:
            raise ValueError(f"Unknown head_fusion: {head_fusion!r}")

        # Discard lowest-attention tokens
        flat = attn_fused.flatten()
        threshold = flat.kthvalue(int(discard_ratio * flat.numel())).values
        attn_fused[attn_fused < threshold] = 0.0

        # Add residual connection
        I = torch.eye(n_tokens)
        a = (attn_fused + I) / 2
        a = a / a.sum(dim=-1, keepdim=True)

        result = a @ result

    # Extract CLS → patch attention
    mask = result[0, 1:]  # shape (n_patches,)
    grid_size = input_size // patch_size
    mask = mask.reshape(grid_size, grid_size).numpy()

    # Upsample to image resolution
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    from PIL import Image as PImage
    mask_img = PImage.fromarray((mask * 255).astype(np.uint8)).resize(
        (input_size, input_size), resample=PImage.BICUBIC
    )
    return np.array(mask_img, dtype=np.float32) / 255.0


# Type annotation alias used in the function signature above
from typing import Any  # noqa: E402 (avoid circular at top-level)
