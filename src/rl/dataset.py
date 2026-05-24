"""Generate a multimodal DPO dataset for aligning VLMs with human visual biases.

This script leverages existing stimulus generators to create triplets of:
  - image: The visual stimulus (e.g. Müller-Lyer illusion)
  - prompt: "What do you see? [Options]"
  - chosen: The human-illusory response
  - rejected: The physically correct response

The output is a directory of images and a metadata JSONL file compatible with
HuggingFace's `datasets` library.
"""

import os
import json
import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stimuli.geometric import MullerLyerGenerator, PonzoGenerator, EbbinghausGenerator
from src.stimuli.color import SimultaneousContrastGenerator, WhiteIllusionGenerator
from src.stimuli.angle import ZollnerGenerator, PoggendorffGenerator
from src.stimuli.motion import ScintillatingGridGenerator, RotatingSnakesGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Configuration for dataset generation
GEN_CONFIG = {
    "muller_lyer": {"cls": MullerLyerGenerator, "n": 200},
    "ponzo": {"cls": PonzoGenerator, "n": 100},
    "ebbinghaus": {"cls": EbbinghausGenerator, "n": 100},
    "simultaneous_contrast": {"cls": SimultaneousContrastGenerator, "n": 100},
    "whites_illusion": {"cls": WhiteIllusionGenerator, "n": 100},
    "zollner": {"cls": ZollnerGenerator, "n": 100},
    "poggendorff": {"cls": PoggendorffGenerator, "n": 100},
    "scintillating_grid": {"cls": ScintillatingGridGenerator, "n": 100},
    "rotating_snakes": {"cls": RotatingSnakesGenerator, "n": 100},
}

# Prompt and answer templates (matching src/models/vlm.py logic)
_QUESTIONS = {
    "geometric": "Which of the two horizontal lines looks longer?",
    "color": "Which of the two grey patches looks brighter?",
    "angle": "Do the main long lines appear parallel?",
    "motion": "Does the pattern appear to move, rotate, or flicker?",
}

_ANSWERS = {
    "geometric": {
        "correct": "They are equal in length.",
        "illusory": "The top line looks longer.",
    },
    "color": {
        "correct": "They are the same brightness.",
        "illusory": "The left patch looks brighter.",
    },
    "angle": {
        "correct": "Yes, they are parallel.",
        "illusory": "No, they appear to converge or diverge.",
    },
    "motion": {
        "correct": "No, the pattern appears still.",
        "illusory": "Yes, the pattern appears to move, rotate, or flicker.",
    },
}

def get_category(illusion_type):
    if illusion_type in ["muller_lyer", "ponzo", "ebbinghaus"]:
        return "geometric"
    if illusion_type in ["simultaneous_contrast", "whites_illusion"]:
        return "color"
    if illusion_type in ["zollner", "poggendorff"]:
        return "angle"
    if illusion_type in ["scintillating_grid", "rotating_snakes"]:
        return "motion"
    return "other"

def generate_dpo_dataset(output_dir="data/rl"):
    output_path = PROJECT_ROOT / output_dir
    img_dir = output_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    
    logger.info("Starting DPO dataset generation...")
    
    import random
    from src.models.vlm import VLMProber, _get_answer_descriptions

    for illusion_type, cfg in GEN_CONFIG.items():
        logger.info(f"Generating samples for {illusion_type}...")
        gen = cfg["cls"]()
        cat = get_category(illusion_type)
        question = _QUESTIONS.get(cat, "What do you see?")
        
        # Get descriptions including correct, illusory, and other
        descriptions = _get_answer_descriptions(None, cat)
        
        # Get parameter grid
        params_list = gen.param_grid()
        if len(params_list) > cfg["n"]:
            params_list = random.sample(params_list, cfg["n"])
            
        for i, params in enumerate(tqdm(params_list, desc=illusion_type)):
            pair = gen.generate(params)
            
            # Save illusion image
            img_filename = f"{illusion_type}_{i}.png"
            pair.illusion.save(img_dir / img_filename)
            
            # Construct multiple-choice DPO triplet
            letters = ["A", "B", "C"]
            label_order = ["correct", "illusory", "other"]
            shuffled = list(zip(letters, label_order))
            random.shuffle(shuffled)
            options = [(l, descriptions[lbl]) for l, lbl in shuffled]
            
            # Find which letter maps to illusory and correct
            illusory_letter = next(l for l, lbl in shuffled if lbl == "illusory")
            correct_letter = next(l for l, lbl in shuffled if lbl == "correct")
            
            framing = random.choice(["neutral", "name_blind"])
            base_prompt = VLMProber._build_prompt(question, options, framing)
            
            # Match LLaVA template syntax used in DPO training
            prompt = f"USER: <image>\n{base_prompt}\nASSISTANT:"
            
            metadata.append({
                "image": str(Path(output_dir) / "images" / img_filename),
                "prompt": prompt,
                "chosen": illusory_letter,
                "rejected": correct_letter,
                "category": cat,
                "illusion_type": illusion_type
            })

    # Save metadata as JSONL
    metadata_path = output_path / "dataset.jsonl"
    with open(metadata_path, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")
            
    logger.info(f"Dataset generated with {len(metadata)} samples.")
    logger.info(f"Metadata saved to {metadata_path}")
    logger.info(f"Images saved to {img_dir}")

if __name__ == "__main__":
    generate_dpo_dataset()
