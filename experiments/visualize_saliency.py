"""Generate saliency heatmaps (GradCAM/Attention) for presentation.

This script picks a representative stimulus for each illusion category
and generates saliency overlays for a CNN (ResNet-50) and a ViT (ViT-B/16).

Usage:
    python experiments/visualize_saliency.py [--device cuda|cpu]
"""

import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.cnn import ResNetProber
from src.models.vit import ViTBProber
from src.analysis.gradcam import compute_gradcam
from src.analysis.attention import compute_attention_rollout
from src.analysis.plots import plot_attention_overlay, save_figure
from src.probing.linear_probe import LinearProbeProtocol
from src.stimuli.geometric import MullerLyerGenerator
from src.stimuli.color import SimultaneousContrastGenerator
from src.stimuli.angle import ZollnerGenerator
from src.stimuli.motion import ScintillatingGridGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_visualization(device="cpu"):
    out_dir = PROJECT_ROOT / "results" / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup Models
    logger.info("Loading models on %s...", device)
    resnet = ResNetProber(device=device)
    vit = ViTBProber(device=device)
    
    # 2. Define illusions to visualize
    illusions = {
        "muller_lyer": (MullerLyerGenerator(), "geometric"),
        "simultaneous_contrast": (SimultaneousContrastGenerator(), "color"),
        "zollner": (ZollnerGenerator(), "angle"),
        "scintillating_grid": (ScintillatingGridGenerator(), "motion"),
    }
    
    for name, (gen, cat) in illusions.items():
        logger.info("=== Processing %s ===", name)
        
        # Generate multiple samples per illusion for better selection
        configs = []
        if name == "muller_lyer":
            configs = [
                ({"fin_length": 40, "fin_angle_deg": 30}, "mild"),
                ({"fin_length": 60, "fin_angle_deg": 20}, "strong"),
                ({"fin_length": 30, "fin_angle_deg": 45}, "weak")
            ]
        elif name == "simultaneous_contrast":
            configs = [({"dark_lum": 60, "bright_lum": 200}, "default")]
        elif name == "zollner":
            configs = [
                ({"hatch_angle_deg": 20}, "20deg"),
                ({"hatch_angle_deg": 10}, "10deg"),
                ({"hatch_angle_deg": 40}, "40deg")
            ]
        elif name == "scintillating_grid":
            configs = [({"disc_radius": 4}, "default")]
        else:
            configs = [({}, "default")]
            
        for config, label in configs:
            img_pair = gen.generate(config)
            img = img_pair.illusion
            
            # Try to load existing probes from full_eval results
            probe_dir = PROJECT_ROOT / "results" / "full"
            rn_probe_path = probe_dir / f"resnet50_{cat}_probe.pt"
            vit_probe_path = probe_dir / f"vit_b_16_{cat}_probe.pt"
            
            # ResNet - GradCAM
            if rn_probe_path.exists():
                lp = LinearProbeProtocol(resnet)
                lp.load(rn_probe_path)
                logger.info("Loaded ResNet probe for %s", cat)
                
                logger.info("Computing GradCAM for ResNet-50 (%s)...", label)
                resnet_cam = compute_gradcam(resnet, img, target_class=1) # target "illusory"
                fig_rn = plot_attention_overlay(img, resnet_cam, title=f"ResNet-50 GradCAM: {name} ({label})")
                save_figure(fig_rn, out_dir / f"{name}_{label}_resnet_gradcam.png")
            else:
                logger.warning("No ResNet probe found for %s. Skipping GradCAM.", cat)

            # ViT - Attention Rollout
            if vit_probe_path.exists():
                lp = LinearProbeProtocol(vit)
                lp.load(vit_probe_path)
                logger.info("Loaded ViT probe for %s", cat)
                
                logger.info("Computing Attention Rollout for ViT-B/16 (%s)...", label)
                vit_rollout = compute_attention_rollout(vit, img)
                fig_vit = plot_attention_overlay(img, vit_rollout, title=f"ViT-B/16 Attention: {name} ({label})")
                save_figure(fig_vit, out_dir / f"{name}_{label}_vit_attention.png")
            else:
                logger.warning("No ViT probe found for %s. Running rollout with base model.", cat)
                logger.info("Computing Attention Rollout for ViT-B/16 (%s)...", label)
                vit_rollout = compute_attention_rollout(vit, img)
                fig_vit = plot_attention_overlay(img, vit_rollout, title=f"ViT-B/16 Attention: {name} ({label})")
                save_figure(fig_vit, out_dir / f"{name}_{label}_vit_attention.png")

    logger.info("Visualizations saved to %s", out_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_visualization(device=args.device)
