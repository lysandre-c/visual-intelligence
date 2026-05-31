#!/usr/bin/env python3
"""Tiny timing benchmark for the symDPO eval — measure real per-stimulus cost.

Loads the merged symDPO model exactly as full_eval does, times N probe_pair
calls on real stimuli, and projects the wall-clock for the full scan so we can
size the scope to fit a fixed GPU-time budget BEFORE committing the long run.

Run on ONE GPU (allocate gpu:1) so the projection matches the intended config.

Usage:
    python experiments/bench_symdpo.py --n 10
    python experiments/bench_symdpo.py --n 10 --category geometric --illusion muller_lyer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.models.vlm import LLaVAProber


def _manifest_sizes() -> dict[str, int]:
    """Count stimuli per illusion type from the generated manifests."""
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "experiments.yaml").read_text())
    stimuli_dir = PROJECT_ROOT / cfg["paths"]["stimuli_dir"]
    sizes: dict[str, int] = {}
    for spec in cfg["full_eval"]["stimuli"]:
        cat, itype = spec["category"], spec["illusion_type"]
        mpath = stimuli_dir / cat / itype / "manifest.json"
        if mpath.exists():
            sizes[f"{cat}/{itype}"] = len(json.loads(mpath.read_text()))
    return sizes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="stimuli to time")
    ap.add_argument("--category", default="geometric")
    ap.add_argument("--illusion", default="muller_lyer")
    ap.add_argument("--adapter-path", default="results/rl_alignment/final")
    ap.add_argument("--budget-hours", type=float, default=3.5,
                    help="Wall-clock budget for the eval phase (excludes compare + GradCAM).")
    ap.add_argument("--gpus", type=int, default=1,
                    help="GPUs the real run will use (data-parallel). Sets which cap is written.")
    ap.add_argument("--cap-out", default="results/tmp/symdpo_maxpertype.txt",
                    help="File to write the recommended --max-per-type value to.")
    args = ap.parse_args()

    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "experiments.yaml").read_text())
    stimuli_dir = PROJECT_ROOT / cfg["paths"]["stimuli_dir"]

    # Load stimuli (reuse the same generator/manifest path as full_eval).
    from experiments.full_eval import GENERATOR_REGISTRY
    gen = GENERATOR_REGISTRY[args.illusion]()
    manifest = stimuli_dir / args.category / args.illusion / "manifest.json"
    pairs = gen.load_manifest(manifest)[: args.n]
    print(f"Loaded {len(pairs)} stimuli from {manifest}")

    print("Loading merged symDPO model (this also forces the model load) ...")
    t0 = time.time()
    prober = LLaVAProber(
        device="cuda",
        adapter_path=args.adapter_path,
        model_name="llava_symDPO",
    )
    raw_samples: list[str] = []

    def _probe(p):
        d = prober.probe_pair(
            illusion=p.illusion, control=p.control,
            correct_answer=p.correct_answer, illusory_answer=p.illusory_answer,
            category=p.category, illusion_type=p.illusion_type,
        )
        rr = (d.raw or {}).get("raw_responses") or []
        if rr:
            raw_samples.append(rr[0])
        return d

    # Force model load + first generation (warm-up, excluded from timing).
    _probe(pairs[0])
    print(f"Model loaded + warm-up in {time.time() - t0:.1f}s")

    # Timed loop over the remaining pairs.
    timed = pairs[1:]
    t1 = time.time()
    for p in timed:
        _probe(p)
    elapsed = time.time() - t1
    per_it = elapsed / max(1, len(timed))
    print("\n" + "=" * 60)
    print(f"Timed {len(timed)} stimuli in {elapsed:.1f}s  ->  {per_it:.2f} s/it")
    print("=" * 60)

    # ── Sanity: does truncating to 8 tokens still yield a parseable letter? ──
    # The eval parses raw.strip().upper()[:1]; verify that is a real option.
    valid = sum(1 for s in raw_samples if s.strip()[:1].upper() in {"A", "B", "C"})
    print("\n" + "=" * 60)
    print("OUTPUT SANITY CHECK (max_new_tokens=8 — exactly what eval parses):")
    print(f"  {valid}/{len(raw_samples)} replies start with a valid letter A/B/C")
    for s in raw_samples[:8]:
        letter = s.strip()[:1].upper()
        flag = "ok" if letter in {"A", "B", "C"} else "!! NOT A/B/C"
        print(f"    parsed={letter!r:5} <- {s[:70]!r}   {flag}")
    if valid < len(raw_samples):
        print("  WARNING: some replies do NOT begin with A/B/C — first-char parse "
              "marks those 'other'. Raise max_new_tokens in vlm.py if so.")
    else:
        print("  All replies begin with a valid letter — 8-token truncation is safe.")
    print("=" * 60)

    sizes = _manifest_sizes()
    total = sum(sizes.values())
    n_types = max(1, len(sizes))
    max_type = max(sizes.values()) if sizes else 0
    print("\nManifest sizes (per illusion type):")
    for k, v in sizes.items():
        print(f"  {k:35s} {v:5d}   (~{v * per_it / 60:.0f} min)")

    budget_s = args.budget_hours * 3600

    def _cap_for(gpus: int) -> tuple[int, float, float]:
        """Return (per_type_cap, full_scan_h, projected_h) for `gpus` workers.
        Data-parallel: each GPU runs ~n_types/gpus types, full model per GPU."""
        types_per_gpu = max(1, -(-n_types // gpus))  # ceil
        full_h = (total / gpus) * per_it / 3600
        if full_h <= args.budget_hours:
            return max_type, full_h, full_h  # everything fits, no cap
        cap = max(1, min(int(budget_s / per_it / types_per_gpu), max_type))
        proj = cap * types_per_gpu * per_it / 3600
        return cap, full_h, proj

    print(f"\nFULL scan = {total} stimuli across {n_types} types.")
    print(f"{'config':<22}{'full-scan h':>13}{'cap/type':>11}{'proj h @cap':>13}")
    for g in (1, 2):
        cap_g, full_h, proj_g = _cap_for(g)
        tag = f"{g} GPU" + (" (data-parallel)" if g > 1 else "")
        capshow = "none(full)" if cap_g >= max_type else str(cap_g)
        print(f"{tag:<22}{full_h:>11.2f} h{capshow:>11}{proj_g:>11.2f} h")
    print(f"\n(Budget assumed = {args.budget_hours} h eval phase. 'full-scan h' is the "
          "uncapped time; if it exceeds budget you either cap, add hours, or add the 2nd GPU.)")

    # Write the cap for the requested --gpus config (consumed by the run sbatch).
    cap, _, proj = _cap_for(args.gpus)
    note = "full scan fits — no cap needed" if cap >= max_type else \
        f"capped to fit ~{args.budget_hours}h on {args.gpus} GPU(s)"
    print(f"\nChosen config = {args.gpus} GPU(s): --max-per-type = {cap}  ({note}); "
          f"projected ~{proj:.2f} h")
    cap_path = PROJECT_ROOT / args.cap_out
    cap_path.parent.mkdir(parents=True, exist_ok=True)
    cap_path.write_text(str(cap))
    print(f"  wrote cap to {cap_path}")


if __name__ == "__main__":
    main()
