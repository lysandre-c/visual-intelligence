"""Generate GradCAM saliency heatmaps for LLaVA base vs DPO.

This script selects a few representative illusions and generates
saliency overlays for both the base LLaVA 1.5 and the DPO-aligned version.
"""

import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.vlm import LLaVAProber
from src.analysis.vlm_saliency import compute_vlm_gradcam
from src.analysis.plots import plot_attention_overlay, save_figure
from src.stimuli.geometric import MullerLyerGenerator
from src.stimuli.color import SimultaneousContrastGenerator
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_visualization():
    out_dir = PROJECT_ROOT / "results" / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    illusions = {
        "muller_lyer": (MullerLyerGenerator(), "geometric"),
        "simultaneous_contrast": (SimultaneousContrastGenerator(), "color"),
    }
    
    # 1. Base Model
    logger.info("Loading Base LLaVA-1.5...")
    base_prober = LLaVAProber(load_in_4bit=True)
    base_prober._load_model()
    
    for name, (gen, cat) in illusions.items():
        logger.info("=== Processing %s (Base) ===", name)
        configs = [({"fin_length": 60, "fin_angle_deg": 20}, "strong")] if name == "muller_lyer" else [({"dark_lum": 60, "bright_lum": 200}, "default")]
        for config, label in configs:
            img = gen.generate(config).illusion
            question = "Which of the two horizontal lines looks longer?" if cat == "geometric" else "Which of the two grey patches looks brighter?"
            options = [("A", "They are equal in length."), ("B", "The top line looks longer."), ("C", "The bottom line looks longer.")] if cat == "geometric" else [("A", "They are the same brightness."), ("B", "The left patch looks brighter."), ("C", "The right patch looks brighter.")]
            prompt = base_prober._build_prompt(question, options, framing="neutral")
            processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            formatted_prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}], add_generation_prompt=True)
            target_token = "B"
            base_cam = compute_vlm_gradcam(base_prober._pipe.model, processor, img, formatted_prompt, target_token)
            save_figure(plot_attention_overlay(img, base_cam, title=f"Base LLaVA GradCAM: {name} ({label})"), out_dir / f"{name}_{label}_llava_base_gradcam.png")
            
    # Clear memory
    import torch, gc
    del base_prober
    gc.collect()
    torch.cuda.empty_cache()

    # 2. DPO Model
    logger.info("Loading DPO LLaVA-1.5...")
    dpo_prober = LLaVAProber(load_in_4bit=True, adapter_path=str(PROJECT_ROOT / "results" / "rl_alignment" / "checkpoint-1000"))
    dpo_prober._load_model()

    for name, (gen, cat) in illusions.items():
        logger.info("=== Processing %s (DPO) ===", name)
        configs = [({"fin_length": 60, "fin_angle_deg": 20}, "strong")] if name == "muller_lyer" else [({"dark_lum": 60, "bright_lum": 200}, "default")]
        for config, label in configs:
            img = gen.generate(config).illusion
            question = "Which of the two horizontal lines looks longer?" if cat == "geometric" else "Which of the two grey patches looks brighter?"
            options = [("A", "They are equal in length."), ("B", "The top line looks longer."), ("C", "The bottom line looks longer.")] if cat == "geometric" else [("A", "They are the same brightness."), ("B", "The left patch looks brighter."), ("C", "The right patch looks brighter.")]
            prompt = dpo_prober._build_prompt(question, options, framing="neutral")
            processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            formatted_prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}], add_generation_prompt=True)
            target_token = "B"
            dpo_cam = compute_vlm_gradcam(dpo_prober._pipe.model, processor, img, formatted_prompt, target_token)
            save_figure(plot_attention_overlay(img, dpo_cam, title=f"DPO LLaVA GradCAM: {name} ({label})"), out_dir / f"{name}_{label}_llava_dpo_gradcam.png")

    logger.info("Visualizations saved to %s", out_dir)

if __name__ == "__main__":
    run_visualization()
