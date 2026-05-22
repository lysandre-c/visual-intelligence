"""GradCAM for Multimodal LLMs (LLaVA) over the vision tower.

This module computes GradCAM by tracing the gradients of a target output token
back to the vision encoder's last feature layer.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

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
        The VLM model (e.g. LlavaForConditionalGeneration).
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
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    if "pixel_values" in inputs:
        inputs["pixel_values"].requires_grad_(True)

    # Heuristic for LLaVA vision tower target layer
    target_layer = model.vision_tower.vision_model.encoder.layers[-2]

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
        valid_h = max(1, int(24 * (h / w)))
        pad_h = (24 - valid_h) // 2
        cam = cam[pad_h : pad_h + valid_h, :]
    elif h > w:
        valid_w = max(1, int(24 * (w / h)))
        pad_w = (24 - valid_w) // 2
        cam = cam[:, pad_w : pad_w + valid_w]

    from PIL import Image as PImage
    mask_img = PImage.fromarray((cam * 255).astype(np.uint8)).resize(img_size, resample=PImage.BICUBIC)
    return np.array(mask_img, dtype=np.float32) / 255.0
