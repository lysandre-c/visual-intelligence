"""GradCAM for Multimodal LLMs (LLaVA) over the vision tower.

This module computes GradCAM by tracing the gradients of a target output token
back to the vision encoder's last feature layer.

Supports both raw ``LlavaForConditionalGeneration`` and PEFT-wrapped models
(e.g. after DPO fine-tuning with LoRA).
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_vision_tower(model: torch.nn.Module) -> torch.nn.Module:
    """Walk through PEFT / Transformers wrapper layers to find the CLIP vision tower.

    Handles multiple layouts:
      - Transformers v4: model.vision_tower
      - Transformers v5: model.model.vision_tower
      - PeftModel:       model.base_model.model.model.vision_tower  (v5)
                         model.base_model.model.vision_tower        (v4)
    """
    # Unpack PeftModel → base_model.model
    inner = model
    if hasattr(inner, "base_model"):
        inner = inner.base_model
    if hasattr(inner, "model"):
        inner = inner.model

    # Transformers v5: LlavaForConditionalGeneration.model.vision_tower
    if hasattr(inner, "model") and hasattr(inner.model, "vision_tower"):
        return inner.model.vision_tower
    # Transformers v4: LlavaForConditionalGeneration.vision_tower
    if hasattr(inner, "vision_tower"):
        return inner.vision_tower

    raise AttributeError(
        f"Cannot locate vision_tower on {type(model).__name__}. "
        "Expected LlavaForConditionalGeneration (raw or PEFT-wrapped)."
    )


def _resolve_device(model: torch.nn.Module) -> torch.device:
    """Get the device of the first parameter in the model."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def compute_vlm_gradcam(
    model: torch.nn.Module,
    processor: AutoProcessor,
    image: Image.Image,
    prompt: str,
    target_token: str,
) -> np.ndarray:
    """Compute GradCAM for a VLM focusing on the vision tower features.

    Parameters
    ----------
    model :
        The VLM model (e.g. LlavaForConditionalGeneration or PeftModel).
    processor :
        The corresponding processor.
    image :
        The input PIL image.
    prompt :
        The text prompt.
    target_token :
        The output token string to attribute (e.g., ' A', ' B', 'A').
        It will be encoded to its token ID.

    Returns
    -------
    np.ndarray of shape (H, W), dtype float32, values in [0, 1].
    """
    device = _resolve_device(model)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    if "pixel_values" in inputs:
        inputs["pixel_values"].requires_grad_(True)

    # Locate the vision tower through any wrapper layers
    vision_tower = _resolve_vision_tower(model)
    target_layer = vision_tower.vision_model.encoder.layers[-2]

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output[0])

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    outputs = model(**inputs, output_hidden_states=True)
    logits = outputs.logits
    # Next token prediction is at the last sequence position
    next_token_logits = logits[0, -1, :]

    target_token_id = processor.tokenizer.encode(target_token, add_special_tokens=False)[-1]
    score = next_token_logits[target_token_id]

    model.zero_grad()
    score.backward(retain_graph=True)

    h1.remove()
    h2.remove()

    if not gradients or not activations:
        raise RuntimeError("Hooks did not capture gradients or activations.")

    grads = gradients[0][0]  # (seq_len, dim)
    acts = activations[0][0] # (seq_len, dim)

    # Aggregate gradients over spatial dimensions
    weights = torch.mean(grads, dim=0)
    cam = torch.matmul(acts, weights)

    # Remove CLS token
    cam = cam[1:]

    grid_size = int(np.sqrt(cam.shape[0]))
    cam = cam.reshape(grid_size, grid_size).detach().cpu().numpy()

    cam = np.maximum(cam, 0)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)

    # Resize to original image size
    img_size = image.size
    w, h = img_size
    
    # LLaVA-1.5 pads the image to a square before resizing to 336x336. 
    # Therefore, the 24x24 CAM includes the padding. We must crop it out.
    if w > h:
        valid_h = max(1, int(grid_size * (h / w)))
        pad_h = (grid_size - valid_h) // 2
        cam = cam[pad_h : pad_h + valid_h, :]
    elif h > w:
        valid_w = max(1, int(grid_size * (w / h)))
        pad_w = (grid_size - valid_w) // 2
        cam = cam[:, pad_w : pad_w + valid_w]

    from PIL import Image as PImage
    mask_img = PImage.fromarray((cam * 255).astype(np.uint8)).resize(img_size, resample=PImage.BICUBIC)
    return np.array(mask_img, dtype=np.float32) / 255.0
