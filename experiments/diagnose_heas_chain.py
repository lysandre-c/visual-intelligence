#!/usr/bin/env python3
"""End-to-end HEAS-chain test: base vs symDPO on a small stimulus subset.

This runs the EXACT production path that produces HEAS:
    VLMProtocol.probe_dataset  (option shuffling)
      -> _aggregate_responses  (first-char letter parse)
        -> compute_heas        (control-ceiling gating)

It answers the real question the user asked: does HEAS actually differ between
base and the symDPO model? (We've already proven the raw model outputs differ;
this checks whether that survives the shuffling/parsing/gating chain.)

Base uses CACHED llava_1.5 results (already on disk); symDPO is run fresh with
the final adapter.

Run on the cluster:
    python experiments/diagnose_heas_chain.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vlm import LLaVAProber
from src.probing.vlm_protocol import VLMProtocol
from src.metrics.heas import compute_heas
from src.stimuli.geometric import MullerLyerGenerator
from src.stimuli.color import SimultaneousContrastGenerator

ADAPTER = "results/rl_alignment/final"
N_PER_TYPE = 8

# (illusion_type, category, generator, results-file stem)
SUBSETS = [
    ("muller_lyer", "geometric", MullerLyerGenerator),
    ("simultaneous_contrast", "color", SimultaneousContrastGenerator),
]


def _load_pairs(gen_cls, illusion_type, category):
    mani = PROJECT_ROOT / f"data/stimuli/{category}/{illusion_type}/manifest.json"
    pairs = gen_cls().load_manifest(mani)
    return pairs[:N_PER_TYPE]


def _cached_base(illusion_type):
    p = PROJECT_ROOT / f"results/full/llava_1.5_{illusion_type}_results.json"
    if not p.exists():
        print(f"  WARNING: no cached base results at {p}")
        return {}
    return {r["stimulus_id"]: r for r in json.loads(p.read_text())}


def main() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "experiments.yaml").read_text())
    human = cfg["human_baselines"]
    ctrl_thr = cfg["control_ceiling_threshold"]
    print(f"control_ceiling_threshold = {ctrl_thr}")

    print("\nLoading symDPO prober (final adapter) ...")
    prober = LLaVAProber(device="cuda", adapter_path=ADAPTER, model_name="llava_symDPO")
    protocol = VLMProtocol(prober)

    for illusion_type, category, gen_cls in SUBSETS:
        print("\n" + "=" * 64)
        print(f"{category} / {illusion_type}")
        print("=" * 64)
        pairs = _load_pairs(gen_cls, illusion_type, category)
        base_cache = _cached_base(illusion_type)

        dpo_results = protocol.probe_dataset(pairs, verbose=False)

        # Per-stimulus label comparison
        base_subset = []
        for r in dpo_results:
            sid = r["stimulus_id"]
            b = base_cache.get(sid)
            base_label = b["predicted_label"] if b else "??"
            print(f"[{sid}] base={base_label:9s}  symDPO={r['predicted_label']:9s}"
                  f"  {'SAME' if base_label == r['predicted_label'] else 'DIFF'}")
            if b:
                base_subset.append(b)

        # HEAS on the matched subset
        hr = human.get(category, 0.5)
        base_heas = compute_heas(base_subset, hr, ctrl_thr, category=category) if base_subset else None
        dpo_heas = compute_heas(dpo_results, hr, ctrl_thr, category=category)

        print("-" * 64)
        if base_heas:
            print(f"  BASE  : HEAS={base_heas['heas']!s:8}  p_illusory={base_heas['p_model_illusory']!s:8}"
                  f"  n={base_heas['n_stimuli']}  excluded={base_heas['n_excluded']}")
        print(f"  symDPO: HEAS={dpo_heas['heas']!s:8}  p_illusory={dpo_heas['p_model_illusory']!s:8}"
              f"  n={dpo_heas['n_stimuli']}  excluded={dpo_heas['n_excluded']}")
        if base_heas:
            same = (str(base_heas["heas"]) == str(dpo_heas["heas"]))
            print(f"  -> HEAS {'IDENTICAL (investigate gating/parsing)' if same else 'DIFFERS (adapter effect is real)'}")


if __name__ == "__main__":
    main()
