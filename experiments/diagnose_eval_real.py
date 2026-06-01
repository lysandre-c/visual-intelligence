#!/usr/bin/env python3
"""Reproduce the REAL eval path: instantiate the actual LLaVAProber class
(base vs symDPO) exactly as full_eval.py does, and compare.

Unlike diagnose_eval_merge.py, this uses NO forced torch_dtype — it calls the
real LLaVAProber._load_model, so the pipeline loads exactly as in production
(float32, device_map="auto"). This is what determines whether the adapter
actually takes effect during the real evaluation.

Run on the cluster with the SAME allocation as the real eval:
    sbatch sbatch/run_diagnose_merge.sbatch   (after pointing it at this script)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.models.vlm import LLaVAProber, VLMProber

ADAPTER = "results/rl_alignment/final"


def _stimuli(n: int = 5):
    mani = PROJECT_ROOT / "data/stimuli/geometric/muller_lyer/manifest.json"
    entries = json.loads(mani.read_text())[:n]
    imgs = []
    for e in entries:
        p = e["illusion_path"]
        p = p if Path(p).is_absolute() else PROJECT_ROOT / p
        imgs.append((e["stimulus_id"], Image.open(str(p)).convert("RGB")))
    return imgs


def _prompt():
    p = VLMProber._build_prompt(
        question="Which of the two horizontal lines looks longer?",
        options=[("A", "They are equal in length."),
                 ("B", "The top line looks longer."),
                 ("C", "The bottom line looks longer.")],
        framing="neutral",
    )
    return p


def _describe_model(prober, tag):
    m = prober._pipe.model
    dtype = next(m.parameters()).dtype
    # device map of the language model layers (detect sharding)
    devs = {str(p.device) for p in m.parameters()}
    print(f"  [{tag}] dtype={dtype}  param devices={sorted(devs)}")
    w = m.model.language_model.layers[0].self_attn.q_proj.weight.detach().float().cpu().clone()
    return w


def main() -> None:
    imgs = _stimuli()
    prompt = _prompt()

    # ── BASE (no adapter) ───────────────────────────────────────────────
    print("\n[1/2] Real LLaVAProber (base) ...")
    base = LLaVAProber(device="cuda")
    base._load_model()
    w_base = _describe_model(base, "base")
    base_out = [base._query(img, prompt) for _, img in imgs]
    del base
    torch.cuda.empty_cache()

    # ── symDPO (adapter, exactly as full_eval registry) ─────────────────
    print("\n[2/2] Real LLaVAProber (symDPO, adapter=%s) ..." % ADAPTER)
    dpo = LLaVAProber(device="cuda", adapter_path=ADAPTER, model_name="llava_symDPO")
    dpo._load_model()
    w_dpo = _describe_model(dpo, "symDPO")
    dpo_out = [dpo._query(img, prompt) for _, img in imgs]

    # ── REPORT ──────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"q_proj.layer0 weight delta (base vs symDPO): {(w_dpo - w_base).norm().item():.6e}")
    print("  -> 0 means the adapter did NOT change the model in the REAL path.\n")
    n_diff = 0
    for (sid, _), b, d in zip(imgs, base_out, dpo_out):
        same = b.strip() == d.strip()
        n_diff += (not same)
        print(f"[{sid}] {'SAME' if same else 'DIFF'}")
        print(f"   base  : {b.strip()[:80]!r}")
        print(f"   symDPO: {d.strip()[:80]!r}")
    print("=" * 64)
    print(f"{n_diff}/{len(imgs)} outputs differ.")
    if n_diff == 0:
        print("BUG REPRODUCED: real eval path shows the adapter has no effect.")
    else:
        print("Adapter takes effect in the real path — look elsewhere for the HEAS issue.")
    print("=" * 64)


if __name__ == "__main__":
    main()
