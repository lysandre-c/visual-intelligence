#!/usr/bin/env python3
"""Quick GradCAM for the symDPO model ONLY (base is skipped — already have those).

Saves one saliency overlay per Müller-Lyer stimulus:
    results/full/post_symMPO/figures/gradcam/<sid>_dpo.png

    python experiments/quick_gradcam.py --n 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _select_samples(manifest: list[dict], n: int) -> list[dict]:
    fin_lengths = sorted(set(e["params"]["fin_length"] for e in manifest))
    indices = np.linspace(0, len(fin_lengths) - 1, n, dtype=int)
    target_fins = [fin_lengths[i] for i in indices]
    samples = []
    for fl in target_fins:
        for entry in manifest:
            if (abs(entry["params"]["fin_length"] - fl) < 0.1
                    and entry["params"].get("fin_angle_deg", 30.0) == 30.0):
                samples.append(entry)
                break
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="Number of Müller-Lyer stimuli.")
    ap.add_argument("--adapter-path", default="results/rl_alignment/final")
    args = ap.parse_args()

    import torch
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    from PIL import Image
    from src.analysis.vlm_saliency import compute_vlm_gradcam
    from src.analysis.plots import plot_attention_overlay, save_figure
    from src.models.vlm import VLMProber

    with open(ROOT / "configs" / "experiments.yaml") as fh:
        cfg = yaml.safe_load(fh)
    stimuli_dir = ROOT / cfg["paths"]["stimuli_dir"]
    out_dir = ROOT / cfg["full_eval"]["output_dir"] / "post_symMPO"
    gradcam_dir = out_dir / "figures" / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = stimuli_dir / "geometric" / "muller_lyer" / "manifest.json"
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    samples = _select_samples(manifest, args.n)
    if not samples:
        logger.error("No matching Müller-Lyer stimuli found.")
        sys.exit(1)

    hf_model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(hf_model_id)

    prompt = VLMProber._build_prompt(
        question="Which of the two horizontal lines looks longer?",
        options=[
            ("A", "They are equal in length."),
            ("B", "The top line looks longer."),
            ("C", "The bottom line looks longer."),
        ],
        framing="neutral",
    )
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    logger.info("Loading symDPO model from %s ...", args.adapter_path)
    dpo_base = LlavaForConditionalGeneration.from_pretrained(
        hf_model_id, torch_dtype=torch.float16, device_map="auto"
    )
    projector_path = Path(args.adapter_path) / "multi_modal_projector.pt"
    if projector_path.exists():
        logger.info("Loading trained projector weights from %s ...", projector_path)
        state = torch.load(str(projector_path), map_location="cpu", weights_only=True)
        dpo_base.model.multi_modal_projector.load_state_dict(state)
    dpo_model = PeftModel.from_pretrained(dpo_base, args.adapter_path).eval()

    for entry in samples:
        sid = entry["stimulus_id"]
        fin = entry["params"]["fin_length"]
        image = Image.open(entry["illusion_path"]).convert("RGB")
        logger.info("  GradCAM [symDPO] on %s (fin=%.1f) ...", sid, fin)
        try:
            cam = compute_vlm_gradcam(dpo_model, processor, image, formatted_prompt, target_token="A")
            fig = plot_attention_overlay(image, cam, title=f"SymDPO — {sid} (fin={fin})")
            save_figure(fig, gradcam_dir / f"{sid}_dpo.png")
        except Exception as e:
            logger.warning("  GradCAM [symDPO] failed for %s: %s", sid, e)

    logger.info("symDPO GradCAM images in: %s", gradcam_dir)


if __name__ == "__main__":
    main()
