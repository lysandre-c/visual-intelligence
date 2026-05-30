#!/usr/bin/env python3
"""Diagnose whether a saved SymMPO adapter actually contains trained weights.

No GPU, no model load — this only reads the saved files, so it runs in
seconds on a login node.

It answers the central question: did training move the LoRA + projector,
or did the adapter end up empty / untrained?

Usage
-----
    python experiments/diagnose_adapter.py results/rl_alignment/final
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(adapter_dir: str) -> None:
    adapter_dir = Path(adapter_dir)
    print(f"\n=== Inspecting adapter dir: {adapter_dir.resolve()} ===")
    print("Exists:", adapter_dir.exists())
    if adapter_dir.exists():
        print("Contents:", sorted(p.name for p in adapter_dir.iterdir()))

    # ── Library versions (train vs eval skew matters) ───────────────────
    try:
        import transformers, peft
        print(f"\ntransformers version: {transformers.__version__}")
        print(f"peft version:         {peft.__version__}")
    except Exception as e:  # noqa: BLE001
        print("Could not import transformers/peft:", e)

    # ── adapter_config.json ─────────────────────────────────────────────
    cfg_path = adapter_dir / "adapter_config.json"
    print("\n--- adapter_config.json ---")
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        print("base_model_name_or_path:", cfg.get("base_model_name_or_path"))
        print("peft_type:", cfg.get("peft_type"))
        print("r:", cfg.get("r"), " lora_alpha:", cfg.get("lora_alpha"))
        print("target_modules:", cfg.get("target_modules"))
    else:
        print("MISSING — no adapter_config.json here.")

    # ── adapter_model.safetensors: keys + lora_B norms ──────────────────
    print("\n--- adapter weights ---")
    st_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    state = None
    if st_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(st_path))
    elif bin_path.exists():
        import torch
        state = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    else:
        print("MISSING — no adapter_model.safetensors / .bin")

    if state is not None:
        keys = list(state.keys())
        print(f"total tensors: {len(keys)}")
        # Distinct module suffixes that got an adapter (e.g. self_attn.q_proj)
        suffixes = sorted({
            k.split(".lora_")[0].split("layers.")[-1].split(".", 1)[-1]
            for k in keys if ".lora_" in k
        })
        print("adapted module suffixes:", suffixes)
        print("sample keys:")
        for k in keys[:6]:
            print("   ", k)

        # The decisive check: lora_B initializes to exactly 0.
        # Nonzero norm => training moved the adapter.
        b_norms = []
        a_norms = []
        for k, v in state.items():
            n = float(v.float().norm().item())
            if "lora_B" in k:
                b_norms.append(n)
            elif "lora_A" in k:
                a_norms.append(n)
        if b_norms:
            import statistics
            print(f"\nlora_B tensors: {len(b_norms)}")
            print(f"  max  ||lora_B|| = {max(b_norms):.6e}")
            print(f"  mean ||lora_B|| = {statistics.mean(b_norms):.6e}")
            print(f"  min  ||lora_B|| = {min(b_norms):.6e}")
            n_nonzero = sum(1 for n in b_norms if n > 1e-8)
            print(f"  nonzero lora_B: {n_nonzero}/{len(b_norms)}")
            print("\n>>> VERDICT:",
                  "LoRA WAS trained (nonzero lora_B) -> bug is EVAL-side"
                  if n_nonzero > 0 else
                  "lora_B all ~0 -> training never moved LoRA -> bug is TRAIN-side")
        else:
            print("\nNo lora_B tensors found — adapter may be empty or "
                  "targeted no modules (TRAIN-side problem).")
        if a_norms:
            print(f"(lora_A tensors: {len(a_norms)}, max norm {max(a_norms):.4e} "
                  f"— nonzero by init, not diagnostic)")

    # ── multi_modal_projector.pt ────────────────────────────────────────
    print("\n--- multi_modal_projector.pt ---")
    proj_path = adapter_dir / "multi_modal_projector.pt"
    if proj_path.exists():
        import torch
        proj = torch.load(str(proj_path), map_location="cpu", weights_only=True)
        print("keys:", list(proj.keys()))
        for k, v in proj.items():
            print(f"   {k}: shape={tuple(v.shape)} ||.||={float(v.float().norm()):.4e}")
    else:
        print("MISSING — projector was not saved here "
              "(eval will skip projector loading).")

    print("\n=== done ===\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/rl_alignment/final")
