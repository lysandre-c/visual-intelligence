"""Extra DPO Experiments for VLM models.

Phase 1: Psychometric Curve (Intensity Scaling)
Phase 2: Overfitting Sanity Check (Control Images)
Phase 3: The Skeptic Persona (Prompt Robustness)
"""

import sys
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gc
from transformers import AutoProcessor

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.vlm import LLaVAProber
from src.stimuli.geometric import MullerLyerGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_illusory_prob(model, processor, img, prompt, target_token):
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    
    target_token_id = processor.tokenizer.encode(target_token, add_special_tokens=False)[-1]
    return probs[target_token_id].item()

def run_experiments():
    gen = MullerLyerGenerator()
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    
    std_prompt_mc = (
        "USER: <image>\nWhich of the two horizontal lines looks longer?\n"
        "A. They are equal in length.\nB. The top line looks longer.\nC. The bottom line looks longer.\n"
        "Answer with the option's letter from the given choices directly.\nASSISTANT:"
    )
    
    std_prompt_text = (
        "Which of the two horizontal lines looks longer?\n"
        "A. They are equal in length.\nB. The top line looks longer.\nC. The bottom line looks longer.\n"
        "Think step-by-step and output exactly one letter: Option A, Option B, or Option C."
    )

    skeptic_prompt = (
        "You are an expert geometrician armed with a pixel-perfect ruler. "
        "Measure carefully and objectively. Which of the two horizontal lines is physically longer?\n"
        "A. They are equal in length.\nB. The top line is longer.\nC. The bottom line is longer.\n"
        "Think step-by-step and explain your measurement, then output exactly one letter: Option A, Option B, or Option C."
    )

    # ---------------------------------------------------------
    # PHASE 1 & 2: Base Model
    # ---------------------------------------------------------
    logger.info("Loading Base LLaVA-1.5...")
    base_prober = LLaVAProber(load_in_4bit=True)
    base_prober._load_model()
    # Temporarily increase max tokens for text phases
    base_prober._pipe.model.config.max_new_tokens = 150
    
    fin_lengths = list(range(0, 100, 10))
    base_probs = []
    
    logger.info("--- PHASE 1: Base Psychometric Curve ---")
    for fl in fin_lengths:
        config = {"fin_length": fl, "fin_angle_deg": 20}
        img = gen.generate(config).illusion
        prob = get_illusory_prob(base_prober._pipe.model, processor, img, std_prompt_mc, "B")
        base_probs.append(prob)
        logger.info(f"Fin Length {fl}: P(B) = {prob:.4f}")

    logger.info("\n--- PHASE 2: Base Control Image Sanity Check ---")
    control_img = gen.generate({"fin_length": 60, "fin_angle_deg": 20}).control
    # Note: query with the text prober to see reasoning
    base_control_resp = base_prober._query(control_img, std_prompt_text)
    print(f"[BASE CONTROL RESPONSE]:\n{base_control_resp}\n")
    
    del base_prober
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # PHASE 1, 2, & 3: DPO Model
    # ---------------------------------------------------------
    logger.info("\nLoading DPO LLaVA-1.5...")
    dpo_prober = LLaVAProber(load_in_4bit=True, adapter_path=str(PROJECT_ROOT / "results" / "rl_alignment" / "checkpoint-1000"))
    dpo_prober._load_model()
    dpo_prober._pipe.model.config.max_new_tokens = 150

    dpo_probs = []
    
    logger.info("--- PHASE 1: DPO Psychometric Curve ---")
    for fl in fin_lengths:
        config = {"fin_length": fl, "fin_angle_deg": 20}
        img = gen.generate(config).illusion
        prob = get_illusory_prob(dpo_prober._pipe.model, processor, img, std_prompt_mc, "B")
        dpo_probs.append(prob)
        logger.info(f"Fin Length {fl}: P(B) = {prob:.4f}")

    logger.info("\n--- PHASE 2: DPO Control Image Sanity Check ---")
    dpo_control_resp = dpo_prober._query(control_img, std_prompt_text)
    print(f"[DPO CONTROL RESPONSE]:\n{dpo_control_resp}\n")

    logger.info("\n--- PHASE 3: DPO Skeptic Persona ---")
    strong_illusion = gen.generate({"fin_length": 60, "fin_angle_deg": 20}).illusion
    dpo_skeptic_resp = dpo_prober._query(strong_illusion, skeptic_prompt)
    print(f"[DPO SKEPTIC RESPONSE]:\n{dpo_skeptic_resp}\n")

    # Plot Phase 1
    plt.figure(figsize=(8, 6))
    plt.plot(fin_lengths, base_probs, marker='o', label='Base LLaVA-1.5', linestyle='--')
    plt.plot(fin_lengths, dpo_probs, marker='s', label='DPO LLaVA-1.5', linewidth=2)
    plt.xlabel('Fin Length (pixels) [Illusion Intensity]', fontsize=12)
    plt.ylabel('Probability of Answering "Top Line Looks Longer" (P(B))', fontsize=12)
    plt.title('Psychometric Curve: Muller-Lyer Illusion', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "psychometric_curve.png")
    logger.info(f"Saved psychometric curve plot to {OUT_DIR / 'psychometric_curve.png'}")

if __name__ == "__main__":
    run_experiments()
