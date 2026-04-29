"""Qualitative analysis experiment: GradCAM and attention rollout.

For a selected set of stimuli and models the script:
  1. Loads saved model probes.
  2. Computes GradCAM (CNN) or attention rollout (ViT) saliency maps.
  3. Saves overlay figures to results/full/figures/saliency/.

Usage
-----
    python experiments/analysis.py [--device cpu|cuda|mps]
                                   [--n-stimuli 5]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stimuli.geometric import MullerLyerGenerator
from src.models.cnn import ResNetProber
from src.models.vit import ViTBProber
from src.analysis.gradcam import compute_gradcam
from src.analysis.attention import compute_attention_rollout
from src.analysis.plots import plot_attention_overlay, save_figure

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_analysis(args: argparse.Namespace) -> None:
    with open(PROJECT_ROOT / "configs" / "experiments.yaml") as fh:
        cfg = yaml.safe_load(fh)

    out_dir = PROJECT_ROOT / cfg["full_eval"]["output_dir"]
    sal_dir = out_dir / "figures" / "saliency"
    sal_dir.mkdir(parents=True, exist_ok=True)
    stimuli_dir = PROJECT_ROOT / cfg["paths"]["stimuli_dir"]

    # ------------------------------------------------------------------
    # Load Müller-Lyer stimuli
    # ------------------------------------------------------------------
    gen = MullerLyerGenerator()
    manifest_path = stimuli_dir / "geometric" / "muller_lyer" / "manifest.json"
    if not manifest_path.exists():
        logger.info("Generating Müller-Lyer stimuli for analysis …")
        gen.generate_dataset(stimuli_dir, manifest_path=manifest_path)

    pairs = gen.load_manifest(manifest_path)[: args.n_stimuli]
    logger.info("Loaded %d stimulus pairs.", len(pairs))

    # ------------------------------------------------------------------
    # ResNet-50: GradCAM
    # ------------------------------------------------------------------
    probe_path = out_dir / "resnet50_probe.pt"
    if probe_path.exists():
        resnet = ResNetProber(device=args.device, probe_path=probe_path)
        logger.info("Computing GradCAM for ResNet-50 …")
        for pair in pairs:
            for label, img in [("illusion", pair.illusion), ("control", pair.control)]:
                try:
                    cam = compute_gradcam(resnet, img, target_class=1)  # illusory class
                    fig = plot_attention_overlay(
                        img, cam,
                        title=f"ResNet-50 GradCAM | {pair.stimulus_id} | {label}",
                    )
                    save_figure(fig, sal_dir / f"gradcam_resnet50_{pair.stimulus_id}_{label}.png")
                except Exception as exc:
                    logger.warning("GradCAM failed for %s/%s: %s", pair.stimulus_id, label, exc)
    else:
        logger.warning("No ResNet-50 probe found at %s; skipping GradCAM.", probe_path)

    # ------------------------------------------------------------------
    # ViT-B/16: Attention rollout
    # ------------------------------------------------------------------
    vit_probe_path = out_dir / "vit_b_16_probe.pt"
    if vit_probe_path.exists():
        vit = ViTBProber(device=args.device, probe_path=vit_probe_path)
        logger.info("Computing attention rollout for ViT-B/16 …")
        for pair in pairs:
            for label, img in [("illusion", pair.illusion), ("control", pair.control)]:
                try:
                    rollout = compute_attention_rollout(vit, img)
                    fig = plot_attention_overlay(
                        img, rollout,
                        title=f"ViT-B/16 Attention | {pair.stimulus_id} | {label}",
                    )
                    save_figure(fig, sal_dir / f"attn_vitb16_{pair.stimulus_id}_{label}.png")
                except Exception as exc:
                    logger.warning("Attention rollout failed for %s/%s: %s", pair.stimulus_id, label, exc)
    else:
        logger.warning("No ViT-B/16 probe found at %s; skipping attention rollout.", vit_probe_path)

    logger.info("Analysis complete.  Figures saved to %s", sal_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saliency analysis experiment.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-stimuli", type=int, default=5,
                        help="Number of stimulus pairs to visualise.")
    args = parser.parse_args()
    run_analysis(args)
