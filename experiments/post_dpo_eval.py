#!/usr/bin/env python3
"""Post-DPO evaluation: HEAS comparison, GradCAM visualizations, and diagnostics.

This script is designed to run **after** DPO training completes. It:

1. Loads cached ``llava_1.5`` results from a previous ``full_eval.py`` run.
2. Loads (or computes) ``llava_symDPO`` results from the current run.
3. Computes a side-by-side HEAS comparison table and delta chart.
4. Generates GradCAM saliency visualizations on Müller-Lyer stimuli for
   both the base and DPO-finetuned models.
5. Computes per-illusion-type HEAS breakdown and response consistency.

Usage
-----
    python experiments/post_dpo_eval.py \\
        --device cuda \\
        --adapter-path results/rl_alignment/checkpoint-1000 \\
        --results-dir results/full \\
        --n-gradcam 6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics.heas import compute_heas
from src.analysis.plots import (
    plot_heas_comparison_bar,
    plot_gradcam_comparison,
    plot_attention_overlay,
    save_figure,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_cached_results(results_dir: Path, model_name: str) -> list[dict]:
    """Load all per-illusion result JSONs for a given model."""
    results = []
    for path in sorted(results_dir.glob(f"{model_name}_*_results.json")):
        with open(path) as fh:
            data = json.load(fh)
            results.extend(data)
    return results


def _compute_heas_per_category(
    results: list[dict],
    human_baselines: dict[str, float],
    ctrl_threshold: float,
) -> dict[str, float]:
    """Compute HEAS for each category from a flat results list."""
    df = pd.DataFrame(results)
    if df.empty:
        return {}
    heas_scores = {}
    for category in df["category"].unique():
        if category not in human_baselines:
            continue
        subset = df[df["category"] == category].to_dict("records")
        res = compute_heas(subset, human_baselines[category], ctrl_threshold, category=category)
        heas_scores[category] = res["heas"]
    return heas_scores


def _compute_heas_per_illusion_type(
    results: list[dict],
    human_baselines: dict[str, float],
    ctrl_threshold: float,
) -> dict[str, dict[str, Any]]:
    """Compute HEAS broken down by (category, illusion_type)."""
    df = pd.DataFrame(results)
    if df.empty:
        return {}
    breakdown = {}
    for (category, illusion_type), group in df.groupby(["category", "illusion_type"]):
        if category not in human_baselines:
            continue
        subset = group.to_dict("records")
        res = compute_heas(subset, human_baselines[category], ctrl_threshold, category=category)
        breakdown[f"{category}/{illusion_type}"] = {
            "heas": res["heas"],
            "p_model_illusory": res["p_model_illusory"],
            "n_stimuli": res["n_stimuli"],
        }
    return breakdown


def _compute_response_consistency(results_dir: Path, model_name: str) -> dict[str, float]:
    """Measure answer variance from audit logs (if available).

    Lower variance = more consistent across prompt orderings/framings.
    Returns {illusion_type: mean_std_illusory_rate} or empty dict.
    """
    audit_dir = results_dir / "audit" / model_name
    if not audit_dir.exists():
        return {}
    consistency = {}
    by_illusion = {}
    for path in audit_dir.glob("*.json"):
        with open(path) as fh:
            record = json.load(fh)
        itype = record.get("illusion_type", "unknown")
        illusory = record.get("illusory", 0.0)
        by_illusion.setdefault(itype, []).append(illusory)
    for itype, rates in by_illusion.items():
        consistency[itype] = float(np.std(rates))
    return consistency


# ──────────────────────────────────────────────────────────────────────────────
# GradCAM
# ──────────────────────────────────────────────────────────────────────────────

def _run_gradcam_comparison(
    adapter_path: str,
    device: str,
    stimuli_dir: Path,
    out_dir: Path,
    n_samples: int = 6,
) -> None:
    """Generate side-by-side GradCAM visualizations on Müller-Lyer stimuli.

    Picks samples at weak / medium / strong fin lengths from the manifest,
    then computes GradCAM for both the base and DPO-finetuned model.
    """
    import torch
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    from PIL import Image
    from src.analysis.vlm_saliency import compute_vlm_gradcam
    from src.models.vlm import VLMProber

    manifest_path = stimuli_dir / "geometric" / "muller_lyer" / "manifest.json"
    if not manifest_path.exists():
        logger.warning("Müller-Lyer manifest not found at %s — skipping GradCAM.", manifest_path)
        return

    with open(manifest_path) as fh:
        manifest = json.load(fh)

    # Select samples at evenly spaced fin lengths
    fin_lengths = sorted(set(e["params"]["fin_length"] for e in manifest))
    n_levels = len(fin_lengths)
    # Pick n_samples spread across weak → strong
    indices = np.linspace(0, n_levels - 1, n_samples, dtype=int)
    target_fins = [fin_lengths[i] for i in indices]

    # Pick one stimulus per target fin length (first match with fin_angle=30°)
    samples = []
    for fl in target_fins:
        for entry in manifest:
            if (abs(entry["params"]["fin_length"] - fl) < 0.1
                    and entry["params"].get("fin_angle_deg", 30.0) == 30.0):
                samples.append(entry)
                break
    if not samples:
        logger.warning("Could not find matching stimuli for GradCAM. Skipping.")
        return

    logger.info("Loading base LLaVA model for GradCAM ...")
    hf_model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(hf_model_id)

    base_model = LlavaForConditionalGeneration.from_pretrained(
        hf_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base_model.eval()

    # Build a standard prompt for GradCAM (neutral framing, fixed option order)
    prompt = VLMProber._build_prompt(
        question="Which of the two horizontal lines looks longer?",
        options=[
            ("A", "They are equal in length."),
            ("B", "The top line looks longer."),
            ("C", "The bottom line looks longer."),
        ],
        framing="neutral",
    )

    # Format as chat for LLaVA
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    gradcam_dir = out_dir / "figures" / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    # Compute base GradCAM for each sample
    base_cams = {}
    for entry in samples:
        sid = entry["stimulus_id"]
        image = Image.open(entry["illusion_path"]).convert("RGB")
        logger.info("  GradCAM [base] on %s (fin=%.1f) ...", sid, entry["params"]["fin_length"])
        try:
            cam = compute_vlm_gradcam(base_model, processor, image, formatted_prompt, target_token="A")
            base_cams[sid] = cam
            # Also save individual overlay
            fig = plot_attention_overlay(image, cam, title=f"Base — {sid} (fin={entry['params']['fin_length']})")
            save_figure(fig, gradcam_dir / f"{sid}_base.png")
        except Exception as e:
            logger.warning("  GradCAM [base] failed for %s: %s", sid, e)

    # Free base model, load DPO model
    del base_model
    torch.cuda.empty_cache()

    logger.info("Loading DPO-finetuned LLaVA model from %s ...", adapter_path)
    dpo_base = LlavaForConditionalGeneration.from_pretrained(
        hf_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Load trained projector weights if present
    projector_path = Path(adapter_path) / "multi_modal_projector.pt"
    if projector_path.exists():
        logger.info("Loading trained projector weights from %s ...", projector_path)
        projector_state = torch.load(str(projector_path), map_location="cpu", weights_only=True)
        dpo_base.model.multi_modal_projector.load_state_dict(projector_state)

    dpo_model = PeftModel.from_pretrained(dpo_base, adapter_path)
    dpo_model.eval()

    # Compute DPO GradCAM + comparison figures
    for entry in samples:
        sid = entry["stimulus_id"]
        image = Image.open(entry["illusion_path"]).convert("RGB")
        logger.info("  GradCAM [DPO] on %s (fin=%.1f) ...", sid, entry["params"]["fin_length"])
        try:
            cam_dpo = compute_vlm_gradcam(dpo_model, processor, image, formatted_prompt, target_token="A")
            # Individual overlay
            fig = plot_attention_overlay(image, cam_dpo, title=f"SymDPO — {sid} (fin={entry['params']['fin_length']})")
            save_figure(fig, gradcam_dir / f"{sid}_dpo.png")

            # Side-by-side comparison (if base CAM succeeded)
            if sid in base_cams:
                fin_len = entry["params"]["fin_length"]
                fig = plot_gradcam_comparison(
                    image, base_cams[sid], cam_dpo,
                    title=f"Müller-Lyer — fin_length={fin_len:.1f}px",
                )
                save_figure(fig, gradcam_dir / f"{sid}_comparison.png")
        except Exception as e:
            logger.warning("  GradCAM [DPO] failed for %s: %s", sid, e)

    del dpo_model, dpo_base
    torch.cuda.empty_cache()
    logger.info("GradCAM visualizations saved to %s", gradcam_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_post_dpo_eval(args: argparse.Namespace) -> None:
    with open(PROJECT_ROOT / "configs" / "experiments.yaml") as fh:
        cfg = yaml.safe_load(fh)

    human_baselines: dict[str, float] = cfg["human_baselines"]
    ctrl_threshold: float = cfg["control_ceiling_threshold"]
    results_dir = PROJECT_ROOT / cfg["full_eval"]["output_dir"]
    stimuli_dir = PROJECT_ROOT / cfg["paths"]["stimuli_dir"]
    out_dir = results_dir / "post_symMPO"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load cached results
    # ------------------------------------------------------------------
    logger.info("Loading cached llava_1.5 results ...")
    base_results = _load_cached_results(results_dir, "llava_1.5")
    logger.info("  Loaded %d base results.", len(base_results))

    logger.info("Loading llava_symDPO results ...")
    dpo_results = _load_cached_results(results_dir, "llava_symDPO")
    logger.info("  Loaded %d DPO results.", len(dpo_results))

    if not base_results:
        logger.error("No cached llava_1.5 results found in %s. Run full_eval.py first.", results_dir)
        sys.exit(1)
    if not dpo_results:
        logger.error("No llava_symDPO results found in %s. Run full_eval.py with --models llava_symDPO first.", results_dir)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. HEAS comparison (per category)
    # ------------------------------------------------------------------
    logger.info("Computing HEAS comparison ...")
    base_heas = _compute_heas_per_category(base_results, human_baselines, ctrl_threshold)
    dpo_heas = _compute_heas_per_category(dpo_results, human_baselines, ctrl_threshold)

    # Build comparison table
    categories = sorted(set(base_heas) | set(dpo_heas))
    comparison_rows = []
    for cat in categories:
        b = base_heas.get(cat, float("nan"))
        d = dpo_heas.get(cat, float("nan"))
        delta = d - b if not (np.isnan(b) or np.isnan(d)) else float("nan")
        comparison_rows.append({
            "category": cat,
            "heas_base": b,
            "heas_symDPO": d,
            "delta": delta,
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = out_dir / "heas_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info("HEAS comparison:\n%s", comparison_df.to_string(index=False))

    # Comparison bar chart
    fig = plot_heas_comparison_bar(base_heas, dpo_heas, categories)
    save_figure(fig, out_dir / "figures" / "heas_comparison_bar.png")

    # ------------------------------------------------------------------
    # 3. Per-illusion-type HEAS breakdown
    # ------------------------------------------------------------------
    logger.info("Computing per-illusion-type HEAS ...")
    base_breakdown = _compute_heas_per_illusion_type(base_results, human_baselines, ctrl_threshold)
    dpo_breakdown = _compute_heas_per_illusion_type(dpo_results, human_baselines, ctrl_threshold)

    all_keys = sorted(set(base_breakdown) | set(dpo_breakdown))
    breakdown_rows = []
    for key in all_keys:
        b = base_breakdown.get(key, {})
        d = dpo_breakdown.get(key, {})
        b_heas = b.get("heas", float("nan"))
        d_heas = d.get("heas", float("nan"))
        breakdown_rows.append({
            "illusion": key,
            "heas_base": b_heas,
            "p_illusory_base": b.get("p_model_illusory", float("nan")),
            "heas_symDPO": d_heas,
            "p_illusory_symDPO": d.get("p_model_illusory", float("nan")),
            "delta": d_heas - b_heas if not (np.isnan(b_heas) or np.isnan(d_heas)) else float("nan"),
            "n_stimuli_base": b.get("n_stimuli", 0),
            "n_stimuli_dpo": d.get("n_stimuli", 0),
        })
    breakdown_df = pd.DataFrame(breakdown_rows)
    breakdown_path = out_dir / "heas_per_illusion.csv"
    breakdown_df.to_csv(breakdown_path, index=False)
    logger.info("Per-illusion HEAS:\n%s", breakdown_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Response consistency (from audit logs)
    # ------------------------------------------------------------------
    logger.info("Computing response consistency ...")
    base_consistency = _compute_response_consistency(results_dir, "llava_1.5")
    dpo_consistency = _compute_response_consistency(results_dir, "llava_symDPO")

    if base_consistency or dpo_consistency:
        all_itypes = sorted(set(base_consistency) | set(dpo_consistency))
        consist_rows = []
        for itype in all_itypes:
            consist_rows.append({
                "illusion_type": itype,
                "std_illusory_base": base_consistency.get(itype, float("nan")),
                "std_illusory_symDPO": dpo_consistency.get(itype, float("nan")),
            })
        consist_df = pd.DataFrame(consist_rows)
        consist_path = out_dir / "response_consistency.csv"
        consist_df.to_csv(consist_path, index=False)
        logger.info("Response consistency:\n%s", consist_df.to_string(index=False))
    else:
        logger.info("No audit logs found — skipping response consistency.")

    # ------------------------------------------------------------------
    # 5. GradCAM visualizations (Müller-Lyer)
    # ------------------------------------------------------------------
    if args.n_gradcam > 0 and args.adapter_path:
        logger.info("Generating GradCAM visualizations (%d samples) ...", args.n_gradcam)
        _run_gradcam_comparison(
            adapter_path=args.adapter_path,
            device=args.device or "cuda",
            stimuli_dir=stimuli_dir,
            out_dir=out_dir,
            n_samples=args.n_gradcam,
        )
    else:
        logger.info("Skipping GradCAM (--n-gradcam 0 or no --adapter-path).")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Post-DPO evaluation complete. Outputs in %s", out_dir)
    logger.info("  - HEAS comparison:       %s", comparison_path)
    logger.info("  - Per-illusion HEAS:     %s", breakdown_path)
    if base_consistency or dpo_consistency:
        logger.info("  - Response consistency:  %s", out_dir / "response_consistency.csv")
    if args.n_gradcam > 0:
        logger.info("  - GradCAM figures:       %s", out_dir / "figures" / "gradcam")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-DPO evaluation: HEAS comparison + GradCAM visualizations.",
    )
    parser.add_argument("--device", default=None, help="cpu | cuda | mps")
    parser.add_argument(
        "--adapter-path",
        default="results/rl_alignment/checkpoint-1000",
        help="Path to the DPO LoRA adapter checkpoint.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing cached evaluation results (default: from config).",
    )
    parser.add_argument(
        "--n-gradcam",
        type=int,
        default=6,
        help="Number of Müller-Lyer stimuli for GradCAM comparison (0 to skip).",
    )
    args = parser.parse_args()
    run_post_dpo_eval(args)
