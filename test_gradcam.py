"""Validate compute_vlm_gradcam on BOTH the base and the PEFT/DPO model.

Reproduces the production call exactly (real Müller-Lyer stimulus, the neutral
A/B/C prompt, target_token="A") so a green run here means the post_dpo_eval
GradCAM step will work. Prints the CAM shape per model, or the precise failure.
"""

import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.analysis.vlm_saliency import compute_vlm_gradcam
from src.models.vlm import VLMProber

HF_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
ADAPTER_PATH = "results/rl_alignment/final"  # production adapter


def _pick_stimulus():
    with open(ROOT / "configs" / "experiments.yaml") as fh:
        cfg = yaml.safe_load(fh)
    stimuli_dir = ROOT / cfg["paths"]["stimuli_dir"]
    manifest_path = stimuli_dir / "geometric" / "muller_lyer" / "manifest.json"
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    entry = manifest[0]
    return entry["stimulus_id"], Image.open(entry["illusion_path"]).convert("RGB")


def _build_prompt(processor):
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
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def _try(name, model, processor, image, formatted_prompt):
    print(f"\n=== {name} ===", flush=True)
    try:
        cam = compute_vlm_gradcam(model, processor, image, formatted_prompt, target_token="A")
        print(f"  OK  {name}: cam shape={cam.shape}, min={cam.min():.3f}, max={cam.max():.3f}", flush=True)
        return True
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}", flush=True)
        return False


def main():
    processor = AutoProcessor.from_pretrained(HF_MODEL_ID)
    sid, image = _pick_stimulus()
    formatted_prompt = _build_prompt(processor)
    print(f"Stimulus: {sid}  image size={image.size}")

    print("Loading base model ...", flush=True)
    base = LlavaForConditionalGeneration.from_pretrained(
        HF_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    ok_base = _try("base", base, processor, image, formatted_prompt)
    del base
    torch.cuda.empty_cache()

    print("\nLoading DPO/PEFT model ...", flush=True)
    dpo_base = LlavaForConditionalGeneration.from_pretrained(
        HF_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    proj = Path(ADAPTER_PATH) / "multi_modal_projector.pt"
    if proj.exists():
        state = torch.load(str(proj), map_location="cpu", weights_only=True)
        dpo_base.model.multi_modal_projector.load_state_dict(state)
    dpo = PeftModel.from_pretrained(dpo_base, ADAPTER_PATH).eval()
    ok_dpo = _try("dpo", dpo, processor, image, formatted_prompt)

    print(f"\nRESULT  base={'OK' if ok_base else 'FAIL'}  dpo={'OK' if ok_dpo else 'FAIL'}")
    sys.exit(0 if (ok_base and ok_dpo) else 1)


if __name__ == "__main__":
    main()
