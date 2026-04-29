"""Week 1 Proof-of-Concept experiment.

Tasks
-----
1. Generate Müller-Lyer parametric stimuli.
2. Run ResNet-50 (with linear probe) and CLIP ViT-B/32 (zero-shot) on them.
3. Compute HEAS and Spearman correlation for both models.
4. Write results to results/poc/ and produce a psychometric-curve plot.

Usage
-----
    python experiments/poc.py [--device cpu|cuda|mps] [--skip-generate]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stimuli.geometric import MullerLyerGenerator
from src.models.cnn import ResNetProber
from src.models.contrastive import CLIPProber
from src.probing.linear_probe import LinearProbeProtocol
from src.probing.zero_shot import ZeroShotProtocol
from src.probing.probe_data import ProbeDataGenerator
from src.metrics.heas import compute_heas
from src.metrics.psychometric import (
    psychometric_from_results,
    spearman_alignment,
    sign_alignment,
    PsychometricCurve,
)
from src.metrics.diagnostics import output_diagnostics, metric_definitions
from src.analysis.plots import plot_psychometric_curves, plot_heas_bar, save_figure

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "experiments.yaml") as fh:
        return yaml.safe_load(fh)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_probe_training_data(
    generator: MullerLyerGenerator, n_per_class: int = 300
) -> tuple[list, list[int]]:
    """Probe training data built from the Müller-Lyer generator itself.

    Class 0: control images (equal plain lines, no fins)
    Class 1: illusion images with strong fins (fin_length 25–80 px)
    Class 2: blank / single-line images (catch-all)

    Using the generator directly ensures zero domain gap between training
    and test images — ResNet features see the same visual format at both times.
    """
    pdg = ProbeDataGenerator(image_size=generator.image_size, n_per_class=n_per_class)
    return pdg.muller_lyer_from_generator(generator, seed=42)


def _plot_poc_diagnostics(diagnostics_df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures" / "diagnostics"
    fig_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(diagnostics_df))
    labels = diagnostics_df["model"].tolist()

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.38
    ax.bar(x - width / 2, diagnostics_df["accuracy"], width, label="illusion accuracy")
    ax.bar(x + width / 2, diagnostics_df["control_argmax_accuracy"], width, label="control argmax accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    ax.set_title("PoC Diagnostic Accuracy")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, fig_dir / "diagnostic_accuracy.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(diagnostics_df))
    for col, name in [
        ("p_pred_correct", "correct"),
        ("p_pred_illusory", "illusory"),
        ("p_pred_other", "other"),
    ]:
        ax.bar(x, diagnostics_df[col], bottom=bottom, label=name)
        bottom += diagnostics_df[col].fillna(0).to_numpy()
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of top predictions")
    ax.set_title("PoC Output Distribution")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, fig_dir / "diagnostic_pred_distribution.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_poc(args: argparse.Namespace) -> None:
    cfg = _load_config()
    poc_cfg = cfg["poc"]
    human_baselines = cfg["human_baselines"]
    # Per-PoC override takes priority over the global threshold
    ctrl_threshold = poc_cfg.get("control_ceiling_threshold", cfg["control_ceiling_threshold"])
    out_dir = PROJECT_ROOT / poc_cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate stimuli
    # ------------------------------------------------------------------
    stimuli_dir = PROJECT_ROOT / cfg["paths"]["stimuli_dir"]
    ml_dir = stimuli_dir / "geometric" / "muller_lyer"

    generator = MullerLyerGenerator()

    manifest_path = ml_dir / "manifest.json"

    if args.skip_generate and manifest_path.exists():
        logger.info("Loading existing Müller-Lyer stimuli from %s", ml_dir)
    else:
        logger.info("Generating Müller-Lyer stimuli …")
        generator.generate_dataset(stimuli_dir, manifest_path=manifest_path)
        logger.info("Generated stimuli, manifest at %s", manifest_path)

    pairs = generator.load_manifest(manifest_path)
    logger.info("Loaded %d stimulus pairs.", len(pairs))

    # ------------------------------------------------------------------
    # 2a. ResNet-50 (linear probe)
    # ------------------------------------------------------------------
    logger.info("Setting up ResNet-50 …")
    resnet = ResNetProber(device=args.device)
    probe_path = out_dir / "resnet50_probe.pt"

    protocol_rn = LinearProbeProtocol(resnet, epochs=50, control_ceiling_threshold=ctrl_threshold)

    if probe_path.exists() and not args.retrain:
        logger.info("Loading saved ResNet-50 probe from %s", probe_path)
        protocol_rn.load(probe_path)
    else:
        logger.info("Training ResNet-50 linear probe …")
        all_images, all_labels = _make_probe_training_data(generator)
        # 80/20 train/val split — stratified by class label
        import random as _random
        _rng = _random.Random(0)
        combined = list(zip(all_images, all_labels))
        _rng.shuffle(combined)
        split = int(0.8 * len(combined))
        train_set, val_set = combined[:split], combined[split:]
        train_imgs, train_lbls = zip(*train_set)
        val_imgs, val_lbls = zip(*val_set)
        logger.info(
            "Probe split: %d train / %d val (per-class ~%d/%d)",
            len(train_imgs), len(val_imgs), split // 3, len(combined) // 3 - split // 3,
        )
        protocol_rn.train(
            list(train_imgs), list(train_lbls),
            val_images=list(val_imgs), val_labels=list(val_lbls),
        )
        protocol_rn.save(probe_path)

    logger.info("Running ResNet-50 on Müller-Lyer stimuli …")
    resnet_results = resnet.probe_dataset(pairs)

    # Diagnostic: log control-image probabilities so we can assess the probe
    logger.info("ResNet-50 control-image probabilities (correct | illusory | other):")
    for r in resnet_results[:4]:
        cp = (r.get("raw") or {}).get("probs_control", [None, None, None])
        logger.info(
            "  fin_len=%-6s  ctrl=[%.3f, %.3f, %.3f]  ill=[%.3f, %.3f, %.3f]  pred=%s",
            r["params"].get("fin_length", "?"),
            *(cp if cp else [0, 0, 0]),
            r["correct"], r["illusory"], r["other"],
            r["predicted_label"],
        )

    # ------------------------------------------------------------------
    # 2b. CLIP ViT-B/32 (zero-shot)
    # ------------------------------------------------------------------
    logger.info("Setting up CLIP ViT-B/32 …")
    clip = CLIPProber(device=args.device)
    zs_protocol = ZeroShotProtocol(clip)

    logger.info("Running CLIP on Müller-Lyer stimuli …")
    clip_results = zs_protocol.probe_dataset(pairs)

    # ------------------------------------------------------------------
    # 3. HEAS and psychometric analysis
    # ------------------------------------------------------------------
    human_rate = human_baselines["geometric"]
    param_key = "fin_length"
    # Unique fin_length levels rounded to 2 dp (matches param_grid rounding)
    param_values = sorted(set(round(float(r["params"][param_key]), 2) for r in resnet_results))

    summary: dict[str, dict] = {}
    all_results_combined = []

    for model_name, results in [("resnet50", resnet_results), ("clip_vit_b32", clip_results)]:
        heas_result = compute_heas(results, human_rate, ctrl_threshold)
        # Continuous illusory probability — smoother and more informative
        rates = psychometric_from_results(results, param_key, param_values, use_continuous=True)
        rho, pval = spearman_alignment(rates, np.full_like(rates, human_rate))
        sign_score = sign_alignment(rates, np.full_like(rates, human_rate))

        # Fit sigmoid to get a threshold estimate
        try:
            curve = PsychometricCurve.fit(np.array(param_values), rates)
            threshold_px = curve.threshold()
            r2 = curve.r_squared
        except Exception:
            threshold_px, r2 = float("nan"), float("nan")

        summary[model_name] = {
            **heas_result,
            **output_diagnostics(results),
            "spearman_rho": rho,
            "spearman_pval": pval,
            "sign_alignment": sign_score,
            "psychometric_threshold_px": threshold_px,
            "psychometric_r2": r2,
        }
        logger.info(
            "%-20s  HEAS=%.3f  ρ=%.3f  sign=%.3f  threshold=%.1f px  R²=%.3f",
            model_name, heas_result["heas"], rho, sign_score, threshold_px, r2,
        )
        for r in results:
            r["model"] = model_name
            all_results_combined.append(r)

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    summary_path = out_dir / "poc_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Summary saved to %s", summary_path)

    results_path = out_dir / "poc_results.json"
    with open(results_path, "w") as fh:
        json.dump(all_results_combined, fh, indent=2, default=str)

    diagnostics_path = out_dir / "diagnostics_table.csv"
    diagnostics_df = pd.DataFrame(
        [
            {"model": model_name, **output_diagnostics(results)}
            for model_name, results in [("resnet50", resnet_results), ("clip_vit_b32", clip_results)]
        ]
    )
    diagnostics_df.to_csv(diagnostics_path, index=False)
    _plot_poc_diagnostics(diagnostics_df, out_dir)

    with open(out_dir / "metric_definitions.json", "w") as fh:
        json.dump(metric_definitions(), fh, indent=2)

    # Also save a snapshot of the config used
    cfg_snapshot_path = out_dir / "config_snapshot.yaml"
    with open(cfg_snapshot_path, "w") as fh:
        yaml.dump(cfg, fh)

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    model_curves = {}
    for model_name, results in [("resnet50", resnet_results), ("clip_vit_b32", clip_results)]:
        model_curves[model_name] = psychometric_from_results(
            results, param_key, param_values, use_continuous=True
        )

    fig = plot_psychometric_curves(
        param_values=np.array(param_values),
        model_curves=model_curves,
        human_rates=np.full(len(param_values), human_rate),
        category="geometric",
        illusion_type="muller_lyer",
        param_label="Fin length (px)",
    )
    save_figure(fig, out_dir / "figures" / "poc_psychometric.png")

    heas_df = pd.DataFrame(
        {m: {"geometric": v["heas"]} for m, v in summary.items()}
    ).T
    heas_df.index.name = "model"
    heas_df.columns.name = "category"
    heas_df = heas_df.T  # categories × models

    fig2 = plot_heas_bar(heas_df, "geometric", title="PoC — HEAS (Müller-Lyer)")
    save_figure(fig2, out_dir / "figures" / "poc_heas_bar.png")

    logger.info("PoC complete.  Results in %s", out_dir)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 1 PoC experiment.")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps")
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip stimulus generation if manifest already exists.",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Re-train the linear probe even if a saved one exists.",
    )
    args = parser.parse_args()
    run_poc(args)
