"""Occlusion Sensitivity Analysis for VLM models.

This script sweeps a gray block over the illusion image and measures
how the probability of the illusory answer drops when specific regions
(like the arrows) are hidden.
"""

import sys
import logging
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import gc

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.vlm import LLaVAProber
from src.analysis.plots import plot_attention_overlay, save_figure
from src.stimuli.geometric import MullerLyerGenerator
from transformers import AutoProcessor
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_illusory_prob(model, processor, img: Image.Image, prompt: str, target_token: str) -> float:
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    
    # Get probability of the specific token (e.g. "B")
    target_token_id = processor.tokenizer.encode(target_token, add_special_tokens=False)[-1]
    return probs[target_token_id].item()

def generate_occlusion_heatmap(model, processor, base_img: Image.Image, prompt: str, target_token: str, box_size: int = 32, step: int = 16) -> np.ndarray:
    w, h = base_img.size
    grid_w = len(range(0, w - box_size + 1, step))
    grid_h = len(range(0, h - box_size + 1, step))
    heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)

    logger.info(f"Running occlusion grid: {grid_w}x{grid_h} ({grid_w*grid_h} total inferences)")

    y_idx = 0
    for y in range(0, h - box_size + 1, step):
        x_idx = 0
        for x in range(0, w - box_size + 1, step):
            occ_img = base_img.copy()
            draw = ImageDraw.Draw(occ_img)
            draw.rectangle([x, y, x + box_size, y + box_size], fill=(128, 128, 128))

            prob = get_illusory_prob(model, processor, occ_img, prompt, target_token)
            heatmap[y_idx, x_idx] = prob
            x_idx += 1
        y_idx += 1

    return heatmap

def run_occlusion_analysis():
    out_dir = PROJECT_ROOT / "results" / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    gen = MullerLyerGenerator()
    config = {"fin_length": 60, "fin_angle_deg": 20}
    img = gen.generate(config).illusion

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    question = "Which of the two horizontal lines looks longer?"
    options = [("A", "They are equal in length."), ("B", "The top line looks longer."), ("C", "The bottom line looks longer.")]
    
    # We borrow prompt building logic
    prompt_text = "USER: <image>\nWhich of the two horizontal lines looks longer?\nA. They are equal in length.\nB. The top line looks longer.\nC. The bottom line looks longer.\nAnswer with the option's letter from the given choices directly.\nASSISTANT:"
    
    # 1. Base Model
    logger.info("Loading Base LLaVA-1.5...")
    base_prober = LLaVAProber(load_in_4bit=True)
    base_prober._load_model()
    model = base_prober._pipe.model

    base_baseline = get_illusory_prob(model, processor, img, prompt_text, "B")
    logger.info(f"Base baseline prob: {base_baseline:.4f}")
    
    base_heatmap = generate_occlusion_heatmap(model, processor, img, prompt_text, "B", box_size=32, step=16)
    
    base_saliency = np.maximum(0, base_baseline - base_heatmap)
    if np.max(base_saliency) > 0:
        base_saliency = base_saliency / np.max(base_saliency)
        
    save_figure(
        plot_attention_overlay(img, base_saliency, title="Base LLaVA Occlusion Saliency (Muller-Lyer)"), 
        out_dir / "muller_lyer_strong_llava_base_occlusion.png"
    )

    del base_prober
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 2. DPO Model
    logger.info("Loading DPO LLaVA-1.5...")
    dpo_prober = LLaVAProber(load_in_4bit=True, adapter_path=str(PROJECT_ROOT / "results" / "rl_alignment" / "checkpoint-1000"))
    dpo_prober._load_model()
    model = dpo_prober._pipe.model

    dpo_baseline = get_illusory_prob(model, processor, img, prompt_text, "B")
    logger.info(f"DPO baseline prob: {dpo_baseline:.4f}")
    
    dpo_heatmap = generate_occlusion_heatmap(model, processor, img, prompt_text, "B", box_size=32, step=16)
    
    dpo_saliency = np.maximum(0, dpo_baseline - dpo_heatmap)
    if np.max(dpo_saliency) > 0:
        dpo_saliency = dpo_saliency / np.max(dpo_saliency)
        
    save_figure(
        plot_attention_overlay(img, dpo_saliency, title="DPO LLaVA Occlusion Saliency (Muller-Lyer)"), 
        out_dir / "muller_lyer_strong_llava_dpo_occlusion.png"
    )

    logger.info("Occlusion heatmaps saved successfully.")

if __name__ == "__main__":
    run_occlusion_analysis()
