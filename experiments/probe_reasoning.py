"""Semantic Probing for VLM models.

This script uses a Chain-of-Thought multiple-choice prompt to highlight
how DPO changes the reasoning and final decision of the model.
"""

import sys
import logging
from pathlib import Path
import torch
import gc

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.vlm import LLaVAProber
from src.stimuli.geometric import MullerLyerGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_probing():
    gen = MullerLyerGenerator()
    config = {"fin_length": 60, "fin_angle_deg": 20}
    img = gen.generate(config).illusion
    
    prompt = (
        "Look carefully at the image. Answer the following question based only on what you see.\n\n"
        "Question: Which of the two horizontal lines looks longer?\n\n"
        "Options:\n"
        "  A. They are equal in length.\n"
        "  B. The top line looks longer.\n"
        "  C. The bottom line looks longer.\n\n"
        "First, think step-by-step and describe the visual features of both lines in detail. "
        "Then, output your final answer as exactly one letter: Option A, Option B, or Option C."
    )

    # 1. Base Model
    logger.info("Loading Base LLaVA-1.5...")
    base_prober = LLaVAProber(load_in_4bit=True)
    base_prober._load_model()
    
    logger.info("Querying Base Model...")
    base_response = base_prober._query(img, prompt)
    print("\n" + "="*50)
    print("BASE MODEL RESPONSE:")
    print(base_response)
    print("="*50 + "\n")

    del base_prober
    gc.collect()
    torch.cuda.empty_cache()

    # 2. DPO Model
    logger.info("Loading DPO LLaVA-1.5...")
    dpo_prober = LLaVAProber(load_in_4bit=True, adapter_path=str(PROJECT_ROOT / "results" / "rl_alignment" / "checkpoint-1000"))
    dpo_prober._load_model()
    
    logger.info("Querying DPO Model...")
    dpo_response = dpo_prober._query(img, prompt)
    print("\n" + "="*50)
    print("DPO MODEL RESPONSE:")
    print(dpo_response)
    print("="*50 + "\n")

if __name__ == "__main__":
    run_probing()
