#!/usr/bin/env python3
"""Decisive eval-side test: does the trained adapter change the model's forward pass?

Replicates the EXACT load sequence used in src/models/vlm.py::LLaVAProber._load_model
(pipeline -> load projector -> PeftModel.from_pretrained -> merge_and_unload) and
compares base vs merged on a REAL illusion stimulus at the LOGIT level — bypassing
generation sampling, the A/B/C bottleneck, and results caching.

Run on the cluster (needs the model + 1 GPU):
    python experiments/diagnose_eval_merge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import pipeline, AutoProcessor
from peft import PeftModel

from src.models.vlm import VLMProber

HF_ID = "llava-hf/llava-1.5-7b-hf"
ADAPTER = "results/rl_alignment/final"


def _get_stimulus_image():
    """Load a real Müller-Lyer illusion image if available, else a gradient."""
    import json
    from PIL import Image
    mani = PROJECT_ROOT / "data/stimuli/geometric/muller_lyer/manifest.json"
    if mani.exists():
        entry = json.loads(mani.read_text())[0]
        p = entry["illusion_path"]
        p = p if Path(p).is_absolute() else PROJECT_ROOT / p
        print(f"Using real stimulus: {p}")
        return Image.open(str(p)).convert("RGB")
    print("No manifest found; using a non-trivial gradient image.")
    import numpy as np
    arr = (np.linspace(0, 255, 224 * 224 * 3).reshape(224, 224, 3)).astype("uint8")
    return Image.fromarray(arr)


def _next_token_logits(model, processor, image, formatted_prompt):
    """Single forward pass; return logits for the next token (the letter)."""
    inputs = processor(images=image, text=formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0, -1, :].float().cpu()


def main() -> None:
    processor = AutoProcessor.from_pretrained(HF_ID)
    image = _get_stimulus_image()

    prompt = VLMProber._build_prompt(
        question="Which of the two horizontal lines looks longer?",
        options=[("A", "They are equal in length."),
                 ("B", "The top line looks longer."),
                 ("C", "The bottom line looks longer.")],
        framing="neutral",
    )
    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    formatted = processor.apply_chat_template(conv, add_generation_prompt=True)

    # ── BASE ────────────────────────────────────────────────────────────
    print("\n[1/2] Loading base model ...")
    pipe_base = pipeline("image-text-to-text", model=HF_ID, device_map="auto",
                         model_kwargs={"torch_dtype": torch.float16})
    base_logits = _next_token_logits(pipe_base.model, processor, image, formatted)
    # snapshot a weight to compare after merge
    w_base = pipe_base.model.model.language_model.layers[0].self_attn.q_proj.weight.detach().float().cpu().clone()
    del pipe_base
    torch.cuda.empty_cache()

    # ── MERGED (exact vlm.py sequence) ──────────────────────────────────
    print("[2/2] Loading + merging adapter (exact vlm.py sequence) ...")
    pipe_dpo = pipeline("image-text-to-text", model=HF_ID, device_map="auto",
                        model_kwargs={"torch_dtype": torch.float16})

    proj_path = Path(ADAPTER) / "multi_modal_projector.pt"
    if proj_path.exists():
        proj_state = torch.load(str(proj_path), map_location="cpu", weights_only=True)
        pipe_dpo.model.model.multi_modal_projector.load_state_dict(proj_state)
        print("  projector loaded.")
    else:
        print("  WARNING: projector file not found.")

    peft_model = PeftModel.from_pretrained(pipe_dpo.model, ADAPTER)
    # How many LoRA modules did the regex actually match at LOAD time?
    n_lora = sum(1 for n, _ in peft_model.named_modules() if n.endswith("lora_A.default"))
    print(f"  LoRA modules injected at load: {n_lora}")
    pipe_dpo.model = peft_model.merge_and_unload()

    w_merged = pipe_dpo.model.model.language_model.layers[0].self_attn.q_proj.weight.detach().float().cpu().clone()
    merged_logits = _next_token_logits(pipe_dpo.model, processor, image, formatted)

    # ── REPORT ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    w_delta = (w_merged - w_base).norm().item()
    print(f"q_proj.layer0 weight delta (base vs merged): {w_delta:.6e}")
    print("  -> 0 means merge_and_unload did NOT change LM weights (BUG).")

    logit_delta = (merged_logits - base_logits).abs().max().item()
    print(f"\nmax |logit| difference (base vs merged): {logit_delta:.6e}")
    print("  -> 0 means the adapter has NO effect on the forward pass (BUG).")

    tok = processor.tokenizer
    for name, logits in [("BASE", base_logits), ("MERGED", merged_logits)]:
        top = torch.topk(logits, 5)
        toks = [repr(tok.decode([i])) for i in top.indices.tolist()]
        print(f"\n{name} top-5 next tokens: {list(zip(toks, [round(v,2) for v in top.values.tolist()]))}")

    print("\n" + "=" * 60)
    if logit_delta < 1e-4:
        print("VERDICT: adapter is NOT affecting inference -> load/merge bug in vlm.py.")
    else:
        print("VERDICT: adapter DOES change logits. Identical HEAS is likely")
        print("         stale caching (results/full/llava_symDPO_*.json) or a weak effect.")
    print("=" * 60)


if __name__ == "__main__":
    main()
