"""Full evaluation experiment: all categories × all models.

For each (stimulus category, model) combination the script:
  1. Generates stimuli (or loads from cache).
  2. Runs the appropriate probing protocol.
  3. Computes per-category HEAS.
  4. Assembles the full category × model HEAS table.
  5. Writes results as CSV + JSON, saves figures.

Usage
-----
    python experiments/full_eval.py [--device cpu|cuda|mps]
                                    [--skip-generate]
                                    [--models resnet50 clip_vit_b32 ...]
                                    [--categories geometric color angle motion]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stimuli.geometric import MullerLyerGenerator, PonzoGenerator, EbbinghausGenerator
from src.stimuli.color import SimultaneousContrastGenerator, WhiteIllusionGenerator
from src.stimuli.angle import ZollnerGenerator, PoggendorffGenerator
from src.stimuli.motion import ScintillatingGridGenerator, RotatingSnakesGenerator
from src.stimuli.base import StimulusGenerator
from src.stimuli.impossible import ExternalDatasetLoader
from src.models.cnn import ResNetProber, ConvNeXtProber
from src.models.vit import ViTBProber, ViTLProber
from src.models.contrastive import CLIPProber, DINOv2Prober
from src.models.vlm import LLaVAProber, QwenVLProber
from src.probing.linear_probe import LinearProbeProtocol
from src.probing.zero_shot import ZeroShotProtocol
from src.probing.vlm_protocol import VLMProtocol
from src.probing.probe_data import ProbeDataGenerator
from src.metrics.heas import heas_table, compute_heas
from src.metrics.psychometric import psychometric_from_results, spearman_alignment
from src.metrics.diagnostics import output_diagnostics, metric_definitions
from src.analysis.plots import plot_heas_table, plot_psychometric_curves, save_figure

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

GENERATOR_REGISTRY: dict[str, type[StimulusGenerator]] = {
    "muller_lyer": MullerLyerGenerator,
    "ponzo": PonzoGenerator,
    "ebbinghaus": EbbinghausGenerator,
    "simultaneous_contrast": SimultaneousContrastGenerator,
    "whites_illusion": WhiteIllusionGenerator,
    "zollner": ZollnerGenerator,
    "poggendorff": PoggendorffGenerator,
    "scintillating_grid": ScintillatingGridGenerator,
    "rotating_snakes": RotatingSnakesGenerator,
}

EXTERNAL_DATASETS = {"illusion_vqa", "hallusion_bench"}

# Primary sweep param per illusion type (for psychometric curves)
SWEEP_PARAM: dict[str, str] = {
    "muller_lyer": "fin_length",
    "ponzo": "convergence_deg",
    "ebbinghaus": "large_sat_radius",
    "simultaneous_contrast": "contrast_delta",
    "whites_illusion": "stripe_height",
    "zollner": "hatch_angle_deg",
    "poggendorff": "occluder_width",
    "scintillating_grid": "disc_radius",
    "rotating_snakes": "wheel_radius",
}


def _build_model(model_name: str, device: str | None, out_dir: Path) -> object:
    """Instantiate a model prober by name."""
    registry = {
        "resnet50": lambda: ResNetProber(device=device),
        "convnext_base": lambda: ConvNeXtProber(device=device),
        "vit_b_16": lambda: ViTBProber(device=device),
        "vit_l_16": lambda: ViTLProber(device=device),
        "clip_vit_b32": lambda: CLIPProber(device=device),
        "dinov2_vit_b14": lambda: DINOv2Prober(device=device),
        "llava_1.5": lambda: LLaVAProber(device=device),
        "llava_1.5_dpo": lambda: LLaVAProber(device=device, adapter_path="results/rl_alignment/checkpoint-1000"),
        "qwen_vl": lambda: QwenVLProber(device=device),
    }
    if model_name not in registry:
        raise ValueError(f"Unknown model: {model_name!r}")
    return registry[model_name]()


def _needs_linear_probe(model_name: str) -> bool:
    return model_name in {"resnet50", "convnext_base", "vit_b_16", "vit_l_16"}


def _is_vlm(model_name: str) -> bool:
    return model_name in {"llava_1.5", "qwen_vl"}


def _make_probe_train_data(category: str, n_per_class: int = 400) -> tuple[list, list[int]]:
    """Generate synthetic non-illusion probe training data for a given category."""
    try:
        pdg = ProbeDataGenerator(n_per_class=n_per_class)
        if category == "geometric":
            # MullerLyerGenerator is the primary geometric stimulus
            gen = MullerLyerGenerator()
            return pdg.muller_lyer_from_generator(gen)
        return pdg.for_category(category)
    except Exception as e:
        logger.warning("Falling back to generic probe data for %s: %s", category, e)
        pdg = ProbeDataGenerator(n_per_class=n_per_class)
        return pdg.geometric_length(seed=0)



def _plot_diagnostics(diagnostics_df: pd.DataFrame, out_dir: Path) -> None:
    """Save quick-look diagnostic figures."""
    if diagnostics_df.empty:
        return

    fig_dir = out_dir / "figures" / "diagnostics"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = diagnostics_df.copy()
    df["label"] = df["model"] + "\n" + df["illusion_type"]

    # 1) Physical accuracy and control sanity check
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(df)), 4))
    x = np.arange(len(df))
    width = 0.38
    ax.bar(x - width / 2, df["accuracy"], width, label="illusion accuracy")
    ax.bar(x + width / 2, df["control_argmax_accuracy"], width, label="control argmax accuracy")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Diagnostic Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=90)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, fig_dir / "diagnostic_accuracy.png")

    # 2) Hard-label output distribution
    pred_cols = ["p_pred_correct", "p_pred_illusory", "p_pred_other"]
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(df)), 4))
    bottom = np.zeros(len(df))
    for col, name in zip(pred_cols, ["correct", "illusory", "other"]):
        ax.bar(x, df[col], bottom=bottom, label=name)
        bottom += df[col].fillna(0).to_numpy()
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of top predictions")
    ax.set_title("Hard-Label Output Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=90)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, fig_dir / "diagnostic_pred_distribution.png")

    # 3) Mean probability distribution
    prob_cols = ["mean_prob_correct", "mean_prob_illusory", "mean_prob_other"]
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(df)), 4))
    bottom = np.zeros(len(df))
    for col, name in zip(prob_cols, ["correct", "illusory", "other"]):
        ax.bar(x, df[col], bottom=bottom, label=name)
        bottom += df[col].fillna(0).to_numpy()
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean assigned probability")
    ax.set_title("Mean Probability Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=90)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, fig_dir / "diagnostic_prob_distribution.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_full_eval(args: argparse.Namespace) -> None:
    with open(PROJECT_ROOT / "configs" / "experiments.yaml") as fh:
        cfg = yaml.safe_load(fh)

    eval_cfg = cfg["full_eval"]
    human_baselines: dict[str, float] = cfg["human_baselines"]
    ctrl_threshold: float = cfg["control_ceiling_threshold"]
    out_dir = PROJECT_ROOT / eval_cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    stimuli_dir = PROJECT_ROOT / cfg["paths"]["stimuli_dir"]

    model_names: list[str] = args.models or eval_cfg["models"]
    requested_cats: set[str] = set(args.categories) if args.categories else set()

    # Filter stimulus list if --categories is specified
    stim_list = [
        s for s in eval_cfg["stimuli"]
        if not requested_cats or s["category"] in requested_cats
    ]

    all_results: list[dict] = []

    # ------------------------------------------------------------------
    # Per-model loop (load once, run all stimuli)
    # ------------------------------------------------------------------
    for model_name in model_names:
        logger.info("=== Model: %s ===", model_name)
        prober = _build_model(model_name, args.device, out_dir)

        for stim_spec in stim_list:
            category = stim_spec["category"]
            illusion_type = stim_spec["illusion_type"]

            if illusion_type not in GENERATOR_REGISTRY and illusion_type not in EXTERNAL_DATASETS:
                logger.warning("Skipping unknown illusion type: %s", illusion_type)
                continue
            if illusion_type in EXTERNAL_DATASETS and _needs_linear_probe(model_name):
                logger.warning(
                    "Skipping %s with %s: linear probes are not defined for external VQA datasets.",
                    illusion_type, model_name,
                )
                continue

            # Load or generate stimuli
            manifest_path = stimuli_dir / category / illusion_type / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            if illusion_type in EXTERNAL_DATASETS:
                external_root = PROJECT_ROOT / cfg["paths"]["data_root"] / "external" / illusion_type
                if not external_root.exists():
                    logger.warning(
                        "Skipping %s: expected external dataset at %s",
                        illusion_type, external_root,
                    )
                    continue
                generator = ExternalDatasetLoader(source=illusion_type, data_root=external_root)
            else:
                gen_cls = GENERATOR_REGISTRY[illusion_type]
                generator = gen_cls()

            if args.skip_generate and manifest_path.exists():
                pairs = generator.load_manifest(manifest_path)
            else:
                logger.info("Generating %s / %s stimuli …", category, illusion_type)
                generator.generate_dataset(stimuli_dir, manifest_path=manifest_path)
                pairs = generator.load_manifest(manifest_path)

            results_path = out_dir / f"{model_name}_{illusion_type}_results.json"
            if results_path.exists():
                logger.info("Loading existing evaluated results for %s on %s from %s", model_name, illusion_type, results_path)
                with open(results_path, "r") as fh:
                    results = json.load(fh)
                all_results.extend(results)
                continue

            logger.info("Probing %d pairs with %s …", len(pairs), model_name)

            if _is_vlm(model_name):
                protocol = VLMProtocol(prober, output_dir=out_dir / "audit" / model_name)
                results = protocol.probe_dataset(pairs, verbose=args.verbose)
            elif _needs_linear_probe(model_name):
                probe_path = out_dir / f"{model_name}_{category}_probe.pt"
                lp = LinearProbeProtocol(prober, epochs=20, control_ceiling_threshold=ctrl_threshold)
                if probe_path.exists():
                    logger.info("Loading %s probe for %s from %s", category, model_name, probe_path)
                    lp.load(probe_path)
                else:
                    logger.info("Training %s probe for %s …", category, model_name)
                    train_images, train_labels = _make_probe_train_data(category)
                    lp.train(train_images, train_labels)
                    lp.save(probe_path)
                results = prober.probe_dataset(pairs, verbose=args.verbose)
            else:
                zs = ZeroShotProtocol(prober)
                results = zs.probe_dataset(pairs, verbose=args.verbose)

            with open(results_path, "w") as fh:
                json.dump(results, fh, indent=2, default=str)

            all_results.extend(results)

        # Free GPU memory between models
        del prober
        if args.device and "cuda" in args.device:
            import torch
            torch.cuda.empty_cache()

        # Free disk space to avoid cluster disk quota limits with large VLMs
        if _is_vlm(model_name):
            import gc
            import shutil
            from huggingface_hub import constants
            gc.collect()
            try:
                hf_cache = Path(constants.HF_HUB_CACHE)
                for p in hf_cache.glob("models--*"):
                    if "llava" in p.name.lower() or "qwen" in p.name.lower():
                        logger.info("Removing HuggingFace cache directory to free disk space: %s", p)
                        shutil.rmtree(p, ignore_errors=True)
            except Exception as e:
                logger.warning("Error removing HF cache: %s", e)

    # ------------------------------------------------------------------
    # HEAS table
    # ------------------------------------------------------------------
    logger.info("Computing HEAS table …")
    table = heas_table(all_results, human_baselines, ctrl_threshold)
    table_path = out_dir / "heas_table.csv"
    table.to_csv(table_path)
    logger.info("HEAS table saved to %s\n%s", table_path, table.to_string())

    # ------------------------------------------------------------------
    # Diagnostic table: accuracy + output distribution
    # ------------------------------------------------------------------
    diag_rows = []
    if all_results:
        for (category, illusion_type, model), subset in pd.DataFrame(all_results).groupby(
            ["category", "illusion_type", "model"]
        ):
            row = {
                "category": category,
                "illusion_type": illusion_type,
                "model": model,
                **output_diagnostics(subset.to_dict("records")),
            }
            diag_rows.append(row)
    diagnostics_df = pd.DataFrame(diag_rows)
    diagnostics_path = out_dir / "diagnostics_table.csv"
    diagnostics_df.to_csv(diagnostics_path, index=False)
    logger.info("Diagnostics table saved to %s", diagnostics_path)
    _plot_diagnostics(diagnostics_df, out_dir)

    # ------------------------------------------------------------------
    # Psychometric analysis (per illusion type)
    # ------------------------------------------------------------------
    psychometric_rows = []
    for stim_spec in stim_list:
        illusion_type = stim_spec["illusion_type"]
        category = stim_spec["category"]
        sweep_key = SWEEP_PARAM.get(illusion_type)
        if sweep_key is None:
            continue
        cat_results = [r for r in all_results if r["illusion_type"] == illusion_type]
        param_vals = sorted(set(float(r["params"].get(sweep_key, 0)) for r in cat_results))
        if not param_vals:
            continue
        model_curves = {}
        for model_name in model_names:
            m_results = [r for r in cat_results if r["model"] == model_name]
            if not m_results:
                continue
            rates = psychometric_from_results(m_results, sweep_key, param_vals)
            model_curves[model_name] = rates
            h_rate = human_baselines.get(category, 0.5)
            rho, pval = spearman_alignment(rates, np.full(len(param_vals), h_rate))
            psychometric_rows.append(
                {
                    "illusion_type": illusion_type,
                    "category": category,
                    "model": model_name,
                    "spearman_rho": rho,
                    "spearman_pval": pval,
                }
            )

        fig = plot_psychometric_curves(
            param_values=np.array(param_vals),
            model_curves=model_curves,
            human_rates=np.full(len(param_vals), human_baselines.get(category, 0.5)),
            category=category,
            illusion_type=illusion_type,
            param_label=sweep_key,
        )
        save_figure(fig, out_dir / "figures" / f"psychometric_{illusion_type}.png")

    if psychometric_rows:
        pd.DataFrame(psychometric_rows).to_csv(out_dir / "spearman_table.csv", index=False)

    # ------------------------------------------------------------------
    # HEAS heatmap figure
    # ------------------------------------------------------------------
    if not table.empty:
        fig_hm = plot_heas_table(table, title="Human Error Alignment Score (HEAS)")
        save_figure(fig_hm, out_dir / "figures" / "heas_heatmap.png")

    # ------------------------------------------------------------------
    # Raw results + config snapshot
    # ------------------------------------------------------------------
    raw_path = out_dir / "all_results.json"
    with open(raw_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)

    with open(out_dir / "metric_definitions.json", "w") as fh:
        json.dump(metric_definitions(), fh, indent=2)

    with open(out_dir / "config_snapshot.yaml", "w") as fh:
        yaml.dump(cfg, fh)

    logger.info("Full evaluation complete.  Results in %s", out_dir)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full evaluation experiment.")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of model names to run.")
    parser.add_argument("--categories", nargs="+", default=None, help="Subset of categories to evaluate.")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    run_full_eval(args)
