#!/usr/bin/env python3
"""Export aggregated JSON for the project website (small files for the browser).

Reads fig/all_results.json, fig/heas_table.csv, fig/diagnostics_table.csv
and writes website/data/*.json.

Usage:
    python scripts/export_website_data.py
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.psychometric import psychometric_from_results, spearman_alignment

FIG_DIR = PROJECT_ROOT / "fig"
OUT_DIR = PROJECT_ROOT / "website" / "data"

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

PROGRAMMATIC_ILLUSIONS = set(SWEEP_PARAM.keys())

STIMULI_IMG_DIR = PROJECT_ROOT / "website" / "assets" / "stimuli"

ILLUSION_GALLERY_META: dict[str, dict] = {
    "muller_lyer": {
        "label": "Müller-Lyer",
        "category": "geometric",
        "description": "Two equal horizontal shafts; arrow fins bias perceived length.",
        "human_effect": "Humans typically judge the top line longer when fins point inward on top and outward on bottom.",
        "sweep_param": "fin_length",
    },
    "ponzo": {
        "label": "Ponzo",
        "category": "geometric",
        "description": "Equal bars between converging rails (depth cue).",
        "human_effect": "The upper bar appears longer because of perspective context.",
        "sweep_param": "convergence_deg",
    },
    "ebbinghaus": {
        "label": "Ebbinghaus",
        "category": "geometric",
        "description": "Central discs with different surrounding circle sizes.",
        "human_effect": "The disc with larger surrounds appears smaller than an equal disc with small surrounds.",
        "sweep_param": "large_sat_radius",
    },
    "simultaneous_contrast": {
        "label": "Simultaneous contrast",
        "category": "color",
        "description": "Identical grey patches on dark vs. bright surrounds.",
        "human_effect": "The patch on the dark surround looks lighter than the patch on the bright surround.",
        "sweep_param": "contrast_delta",
    },
    "whites_illusion": {
        "label": "White's illusion",
        "category": "color",
        "description": "Grey patches embedded in black/white stripe surrounds.",
        "human_effect": "Patches in white stripes appear darker than patches in black stripes (same physical grey).",
        "sweep_param": "stripe_height",
    },
    "zollner": {
        "label": "Zöllner",
        "category": "angle",
        "description": "Parallel main lines with alternating oblique hatch marks.",
        "human_effect": "Hatched lines appear non-parallel; strength grows with hatch angle.",
        "sweep_param": "hatch_angle_deg",
    },
    "poggendorff": {
        "label": "Poggendorff",
        "category": "angle",
        "description": "A single oblique line interrupted by a rectangular occluder.",
        "human_effect": "The visible segments appear misaligned across the occluder.",
        "sweep_param": "occluder_width",
    },
    "scintillating_grid": {
        "label": "Scintillating grid",
        "category": "motion",
        "description": "Grey discs at grid intersections on a black grid.",
        "human_effect": "Flickering dark spots appear at intersections (motion from static).",
        "sweep_param": "disc_radius",
    },
    "rotating_snakes": {
        "label": "Rotating snakes",
        "category": "motion",
        "description": "Coloured annular segments in a wheel layout.",
        "human_effect": "Peripheral wheels appear to rotate (motion from static).",
        "sweep_param": "wheel_radius",
    },
    "illusion_vqa": {
        "label": "IllusionVQA (external)",
        "category": "impossible",
        "description": "Scene-level impossible or ambiguous objects from IllusionVQA.",
        "human_effect": "Humans report perceptual conflict or impossible interpretations.",
        "external": True,
    },
    "hallusion_bench": {
        "label": "HallusionBench (external)",
        "category": "impossible",
        "description": (
            "Fixed example from HallusionBench VD/illusion: two related frames used in their "
            "illusion-vs-reference protocol (not parametric)."
        ),
        "human_effect": (
            "Humans typically accept the illusory reading on these scene items; VLMs are scored with MCQ."
        ),
        "external": True,
    },
}

# Representative mid-sweep parameters for gallery thumbnails.
GALLERY_PARAMS: dict[str, dict] = {
    "muller_lyer": {"fin_length": 40.0, "fin_angle_deg": 30.0},
    "ponzo": {"convergence_deg": 37.5},
    "ebbinghaus": {"large_sat_radius": 35, "small_sat_radius": 12},
    "simultaneous_contrast": {
        "dark_lum": 46,
        "bright_lum": 174,
        "contrast_delta": 82,
    },
    "whites_illusion": {"stripe_height": 48},
    "zollner": {"main_angle_deg": 30.0, "hatch_angle_deg": 47.5},
    "poggendorff": {"occluder_width": 105},
    "scintillating_grid": {"disc_radius": 9},
    "rotating_snakes": {"wheel_radius": 54},
}

# param_grid() kwargs per illusion (match generator defaults used in full eval).
GALLERY_GRID_KWARGS: dict[str, dict] = {
    "muller_lyer": {"n_fin_levels": 20, "n_repeats": 1, "jitter_seed": 42},
    "ponzo": {"n_levels": 16, "n_repeats": 1, "jitter_seed": 43},
    "ebbinghaus": {"n_levels": 16, "n_repeats": 1, "jitter_seed": 44},
    "simultaneous_contrast": {"n_levels": 16, "n_repeats": 1, "jitter_seed": 45},
    "whites_illusion": {"n_levels": 16, "n_repeats": 1, "jitter_seed": 46},
    "zollner": {"n_levels": 16, "n_repeats": 1, "jitter_seed": 47},
    "poggendorff": {"n_levels": 16, "n_repeats": 1, "jitter_seed": 48},
    "scintillating_grid": {"n_levels": 16, "n_repeats": 1, "jitter_seed": 49},
    "rotating_snakes": {"n_levels": 16, "n_repeats": 1, "jitter_seed": 50},
}

# Params derived in JS from others (same rules as param_grid coupling).
GALLERY_DERIVED_PARAMS: dict[str, list[str]] = {
    "ebbinghaus": ["small_sat_radius"],
    "simultaneous_contrast": ["dark_lum", "bright_lum"],
    "scintillating_grid": ["grid_line_width"],
}

# Shown under the gallery when control rendering differs from illusion (matches Python).
GALLERY_CONTROL_NOTES: dict[str, str] = {
    "ebbinghaus": (
        "Control uses the same satellite radius on both flanks (mean of large/small), "
        "so changing the sweep updates both panels—this matches src/stimuli/geometric.py, not a UI bug."
    ),
}

# Discrete params in param_grid (use select, not range slider).
GALLERY_ENUM_PARAMS: dict[str, dict[str, list]] = {
    "muller_lyer": {"fin_angle_deg": [20.0, 30.0, 45.0]},
    "zollner": {"main_angle_deg": [25.0, 30.0, 35.0]},
    "poggendorff": {"line_angle_deg": [25.0, 30.0, 35.0]},
    "scintillating_grid": {"grid_spacing": [44, 48, 52]},
    "rotating_snakes": {
        "n_rings": [3, 4],
        "segment_count": [40, 48, 56],
        "wheel_grid": [3],
    },
}

GALLERY_PARAM_LABELS: dict[str, str] = {
    "fin_length": "Fin length (px)",
    "fin_angle_deg": "Fin angle (°)",
    "x_jitter": "Horizontal jitter (px)",
    "y_jitter": "Vertical jitter (px)",
    "shaft_scale": "Shaft scale",
    "convergence_deg": "Rail convergence (°)",
    "y_shift": "Vertical shift (px)",
    "bar_scale": "Bar scale",
    "vp_y_frac": "Vanishing-point height (frac.)",
    "large_sat_radius": "Large satellite radius (px)",
    "small_sat_radius": "Small satellite radius (px)",
    "center_radius": "Centre disc radius (px)",
    "satellite_distance": "Satellite orbit (px)",
    "contrast_delta": "Surround contrast (Δ)",
    "target_luminance": "Patch grey level",
    "patch_size": "Patch size (px)",
    "dark_lum": "Dark surround",
    "bright_lum": "Bright surround",
    "stripe_height": "Stripe height (px)",
    "phase_offset": "Stripe phase (px)",
    "patch_width": "Patch width (px)",
    "hatch_angle_deg": "Hatch angle (°)",
    "main_angle_deg": "Main line angle (°)",
    "hatch_spacing": "Hatch spacing (px)",
    "hatch_length": "Hatch length (px)",
    "occluder_width": "Occluder width (px)",
    "line_angle_deg": "Line angle (°)",
    "disc_radius": "Disc radius (px)",
    "grid_spacing": "Grid spacing (px)",
    "grid_line_width": "Grid line width (px)",
    "wheel_radius": "Wheel radius (px)",
    "n_rings": "Number of rings",
    "segment_count": "Segments per ring",
    "phase": "Wheel phase (rad)",
    "wheel_grid": "Wheels per side",
}


def _gallery_slider_step(param: str, lo, hi) -> float:
    """HTML range step: must allow fractional values when param_grid uses floats."""
    if param in ("shaft_scale", "bar_scale", "vp_y_frac"):
        return 0.001
    if param == "phase":
        return 0.01
    if isinstance(lo, float) or isinstance(hi, float):
        if param.endswith("_deg"):
            return 0.5
        span = float(hi) - float(lo)
        if span <= 0:
            return 0.001
        return max(0.001, round(span / 100, 4))
    return 1


def _grid_param_ranges(gen_cls, grid_kwargs: dict) -> dict[str, dict]:
    """Min/max per param key from one param_grid() call (mirrors eval sweep)."""
    grid = gen_cls().param_grid(**grid_kwargs)
    ranges: dict[str, dict] = {}
    for entry in grid:
        for key, val in entry.items():
            if key == "repeat":
                continue
            if key not in ranges:
                ranges[key] = {"min": val, "max": val}
            else:
                ranges[key]["min"] = min(ranges[key]["min"], val)
                ranges[key]["max"] = max(ranges[key]["max"], val)
    return ranges


def _build_gallery_controls(illusion_id: str, gen_cls) -> list[dict]:
    """Build UI controls from param_grid ranges (excludes derived/coupled params)."""
    grid_kw = GALLERY_GRID_KWARGS[illusion_id]
    ranges = _grid_param_ranges(gen_cls, grid_kw)
    derived = set(GALLERY_DERIVED_PARAMS.get(illusion_id, []))
    enums = GALLERY_ENUM_PARAMS.get(illusion_id, {})
    controls: list[dict] = []

    for param in sorted(ranges.keys()):
        if param in derived:
            continue
        label = GALLERY_PARAM_LABELS.get(param, param)
        if param in enums:
            if len(enums[param]) <= 1:
                continue
            controls.append(
                {
                    "param": param,
                    "label": label,
                    "type": "select",
                    "values": enums[param],
                }
            )
            continue
        lo, hi = ranges[param]["min"], ranges[param]["max"]
        step = _gallery_slider_step(param, lo, hi)
        controls.append(
            {
                "param": param,
                "label": label,
                "type": "range",
                "min": lo,
                "max": hi,
                "step": step,
            }
        )
    return controls


def _load_human_baselines() -> dict[str, float]:
    cfg_path = PROJECT_ROOT / "configs" / "experiments.yaml"
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("human_baselines", {})


def export_heas_table() -> dict:
    csv_path = FIG_DIR / "heas_table.csv"
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    models = [c for c in df.columns if c != "category"]
    return {
        "categories": df["category"].tolist(),
        "models": models,
        "values": {
            row["category"]: {m: _nan_to_none(row.get(m)) for m in models}
            for row in records
        },
        "human_baselines": _load_human_baselines(),
    }


def _nan_to_none(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


def export_psychometric_curves(all_results: list[dict]) -> dict:
    human_baselines = _load_human_baselines()
    by_illusion: dict[str, list] = defaultdict(list)
    for r in all_results:
        it = r.get("illusion_type")
        if it in PROGRAMMATIC_ILLUSIONS:
            by_illusion[it].append(r)

    illusions_out = {}
    spearman_rows = []

    for illusion_type, cat_results in sorted(by_illusion.items()):
        sweep_key = SWEEP_PARAM[illusion_type]
        category = cat_results[0]["category"]
        param_vals = sorted(
            set(round(float(r["params"].get(sweep_key, 0)), 2) for r in cat_results)
        )
        human_rate = human_baselines.get(category, 0.5)
        models_out = {}

        model_names = sorted(set(r["model"] for r in cat_results))
        for model_name in model_names:
            m_results = [r for r in cat_results if r["model"] == model_name]
            rates = psychometric_from_results(m_results, sweep_key, param_vals)
            points = [
                {"param": float(pv), "illusory_rate": _nan_to_none(rate)}
                for pv, rate in zip(param_vals, rates)
            ]
            models_out[model_name] = points

            h_curve = np.full(len(param_vals), human_rate)
            rho, pval = spearman_alignment(rates, h_curve)
            if len(param_vals) >= 3 and not np.all(np.isnan(rates)):
                spearman_rows.append({
                    "illusion_type": illusion_type,
                    "category": category,
                    "model": model_name,
                    "spearman_rho": _nan_to_none(rho),
                    "spearman_pval": _nan_to_none(pval),
                })

        illusions_out[illusion_type] = {
            "category": category,
            "sweep_param": sweep_key,
            "human_rate": float(human_rate),
            "param_values": [float(p) for p in param_vals],
            "models": models_out,
        }

    return {
        "illusions": illusions_out,
        "spearman": spearman_rows,
    }


def export_diagnostics_summary() -> list[dict]:
    csv_path = FIG_DIR / "diagnostics_table.csv"
    df = pd.read_csv(csv_path)
    cols = [
        "category", "illusion_type", "model", "n", "accuracy",
        "p_pred_correct", "p_pred_illusory", "p_pred_other",
        "control_argmax_accuracy",
    ]
    rows = []
    for _, row in df.iterrows():
        rec = {}
        for c in cols:
            v = row.get(c)
            if pd.isna(v):
                rec[c] = None
            elif c in ("category", "illusion_type", "model"):
                rec[c] = str(v)
            elif c == "n":
                rec[c] = int(v)
            else:
                rec[c] = float(v)
        rows.append(rec)
    return rows


def export_heas_cell_details(heas: dict, diagnostics: list[dict]) -> dict:
    """Per (category, model) drill-down: p_model estimate + per-illusion breakdown."""
    human = heas["human_baselines"]
    by_cell: dict[str, dict] = defaultdict(lambda: {"illusions": []})

    for row in diagnostics:
        cat = row["category"]
        model = row["model"]
        key = f"{cat}|{model}"
        heas_val = heas["values"].get(cat, {}).get(model)
        p_ill = row.get("p_pred_illusory")
        by_cell[key]["category"] = cat
        by_cell[key]["model"] = model
        by_cell[key]["heas"] = heas_val
        by_cell[key]["p_human"] = human.get(cat)
        by_cell[key]["illusions"].append({
            "illusion_type": row["illusion_type"],
            "p_illusory": p_ill,
            "p_correct": row.get("p_pred_correct"),
            "p_other": row.get("p_pred_other"),
            "control_pass_rate": row.get("control_argmax_accuracy"),
        })

    for key, cell in by_cell.items():
        illusions = cell["illusions"]
        rates = [x["p_illusory"] for x in illusions if x["p_illusory"] is not None]
        cell["p_model"] = float(np.mean(rates)) if rates else None
        if cell["heas"] is not None and cell["p_human"] is not None and cell["p_model"] is not None:
            gap = abs(cell["p_model"] - cell["p_human"])
            cell["alignment_gap"] = gap
            if gap < 0.05:
                cell["interpretation"] = "near_human"
            elif cell["p_model"] < cell["p_human"] - 0.15:
                cell["interpretation"] = "under_illusory"
            elif cell["p_model"] > cell["p_human"] + 0.15:
                cell["interpretation"] = "over_illusory"
            else:
                cell["interpretation"] = "moderate_mismatch"

    return dict(by_cell)


MODEL_FAMILIES = {
    "supervised": ["resnet50", "convnext_base", "vit_b_16", "vit_l_16"],
    "contrastive": ["clip_vit_b32", "dinov2_vit_b14"],
    "vlm": ["llava_1.5"],
    "vlm_dpo": ["llava_1.5_dpo"],
}

ILLUSION_LABELS = {
    "muller_lyer": "Müller-Lyer",
    "ponzo": "Ponzo",
    "ebbinghaus": "Ebbinghaus",
    "simultaneous_contrast": "Simultaneous contrast",
    "whites_illusion": "White's illusion",
    "zollner": "Zöllner",
    "poggendorff": "Poggendorff",
    "scintillating_grid": "Scintillating grid",
    "rotating_snakes": "Rotating snakes",
    "illusion_vqa": "IllusionVQA",
    "hallusion_bench": "HallusionBench",
}


def export_rq_summary(heas: dict, diagnostics: list[dict], dpo_delta: list[dict]) -> dict:
    """RQ-structured tables and explorer presets for the website."""
    human = heas["human_baselines"]
    categories = heas["categories"]
    models = heas["models"]
    values = heas["values"]

    rq1_rows = []
    for cat in categories:
        row_vals = {m: values[cat].get(m) for m in models}
        valid = {m: v for m, v in row_vals.items() if v is not None}
        best_model = max(valid, key=valid.get) if valid else None
        non_vlm = {m: v for m, v in valid.items() if not m.startswith("llava")}
        best_non_vlm = max(non_vlm, key=non_vlm.get) if non_vlm else None
        rq1_rows.append({
            "category": cat,
            "p_human": human.get(cat),
            "best_model": best_model,
            "best_heas": valid.get(best_model) if best_model else None,
            "best_non_vlm": best_non_vlm,
            "best_non_vlm_heas": non_vlm.get(best_non_vlm) if best_non_vlm else None,
        })

    family_means = []
    for fam_name, fam_models in MODEL_FAMILIES.items():
        per_cat = {}
        all_scores = []
        for cat in categories:
            scores = [values[cat].get(m) for m in fam_models if values[cat].get(m) is not None]
            if scores:
                mean_s = float(np.mean(scores))
                per_cat[cat] = round(mean_s, 3)
                all_scores.extend(scores)
        family_means.append({
            "family": fam_name,
            "mean_heas_over_cells": round(float(np.mean(all_scores)), 3) if all_scores else None,
            "per_category": per_cat,
        })

    failure_examples = [
        {
            "mode": "correct_avoidant",
            "model": "resnet50",
            "illusion_type": "simultaneous_contrast",
            "category": "color",
            "label": "Correct-avoidant",
            "description": "Predicts physically correct on illusion trials; humans report illusory ~92%.",
        },
        {
            "mode": "anti_illusory_geometric",
            "model": "clip_vit_b32",
            "illusion_type": "muller_lyer",
            "category": "geometric",
            "label": "Anti-illusory (geometric)",
            "description": "P(illusory)≈0 on Müller-Lyer despite control success → HEAS 0.12.",
        },
        {
            "mode": "other_collapse",
            "model": "resnet50",
            "illusion_type": "poggendorff",
            "category": "angle",
            "label": "Other-collapse",
            "description": "~87% 'other' labels on Poggendorff—not human-like convergence.",
        },
        {
            "mode": "illusory_without_control",
            "model": "llava_1.5",
            "illusion_type": "zollner",
            "category": "angle",
            "label": "Illusory saturation (weak control)",
            "description": "P(illusory)≈97% but control pass ≈11% pre-DPO; HEAS needs control filtering.",
        },
        {
            "mode": "dpo_rescued",
            "model": "llava_1.5_dpo",
            "illusion_type": "zollner",
            "category": "angle",
            "label": "DPO-rescued interpretability",
            "description": "Control pass ~90% while retaining illusory dominance on Zöllner.",
        },
    ]
    for ex in failure_examples:
        d = next(
            (
                r
                for r in diagnostics
                if r["model"] == ex["model"] and r["illusion_type"] == ex["illusion_type"]
            ),
            None,
        )
        if d:
            ex["p_illusory"] = d.get("p_pred_illusory")
            ex["p_correct"] = d.get("p_pred_correct")
            ex["p_other"] = d.get("p_pred_other")
            ex["control_pass"] = d.get("control_argmax_accuracy")

    explorer_presets = [
        {"id": "clip_geometric", "label": "CLIP × geometric (anti-human)", "category": "geometric", "model": "clip_vit_b32"},
        {"id": "resnet_color", "label": "ResNet × color (correct-avoidant)", "category": "color", "model": "resnet50"},
        {"id": "dpo_color", "label": "LLaVA+DPO × color (aligned)", "category": "color", "model": "llava_1.5_dpo"},
        {"id": "vitl_geometric", "label": "ViT-L × geometric (best CNN/ViT)", "category": "geometric", "model": "vit_l_16"},
        {"id": "llava_angle", "label": "LLaVA base × angle", "category": "angle", "model": "llava_1.5"},
    ]

    return {
        "rq1_category_summary": rq1_rows,
        "rq2_family_means": family_means,
        "rq3_failure_modes": failure_examples,
        "dpo_delta": dpo_delta,
        "explorer_presets": explorer_presets,
        "coverage_gaps": [
            "ResNet-50: not evaluated on motion or impossible (empty HEAS cells).",
            "CNN/ViT families: not evaluated on impossible scene-level stimuli.",
            "HEAS uses one reference illusory-response rate per category.",
        ],
    }


def export_dpo_delta(heas: dict) -> list[dict]:
    """Per-category HEAS gain: llava_1.5_dpo minus llava_1.5."""
    values = heas["values"]
    out = []
    for cat, row in values.items():
        base = row.get("llava_1.5")
        dpo = row.get("llava_1.5_dpo")
        if base is not None and dpo is not None:
            out.append({
                "category": cat,
                "llava_base": base,
                "llava_dpo": dpo,
                "delta": dpo - base,
                "human_baseline": heas["human_baselines"].get(cat),
            })
    return out


def export_symmpo_delta(heas: dict) -> dict:
    """SymMPO comparison payload for the website.

    When ``llava_1.5_symmpo`` is absent from the HEAS table, returns
    ``{"status": "pending", "rows": []}`` so the site shows a placeholder.
    """
    values = heas["values"]
    has_symmpo = any(
        row.get("llava_1.5_symmpo") is not None for row in values.values()
    )
    if not has_symmpo:
        return {"status": "pending", "rows": []}

    rows = []
    for cat, row in values.items():
        base = row.get("llava_1.5")
        dpo = row.get("llava_1.5_dpo")
        symmpo = row.get("llava_1.5_symmpo")
        if base is None and dpo is None and symmpo is None:
            continue
        rows.append({
            "category": cat,
            "llava_base": base,
            "llava_dpo": dpo,
            "llava_symmpo": symmpo,
            "human_baseline": heas["human_baselines"].get(cat),
        })
    return {"status": "ready", "rows": rows}


def export_stimulus_gallery() -> dict:
    """Render example illusion/control pairs for the website stimulus gallery."""
    from PIL import Image

    from src.stimuli import (
        EbbinghausGenerator,
        MullerLyerGenerator,
        PoggendorffGenerator,
        PonzoGenerator,
        RotatingSnakesGenerator,
        ScintillatingGridGenerator,
        SimultaneousContrastGenerator,
        WhiteIllusionGenerator,
        ZollnerGenerator,
    )
    from src.stimuli.impossible import ExternalDatasetLoader

    generators = {
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

    STIMULI_IMG_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []

    for illusion_type, gen_cls in generators.items():
        meta = ILLUSION_GALLERY_META[illusion_type]
        params = GALLERY_PARAMS[illusion_type]
        pair = gen_cls().generate(params)
        subdir = STIMULI_IMG_DIR / illusion_type
        subdir.mkdir(parents=True, exist_ok=True)
        pair.illusion.save(subdir / "illusion.png")
        pair.control.save(subdir / "control.png")
        entries.append(
            {
                "id": illusion_type,
                "label": meta["label"],
                "category": meta["category"],
                "description": meta["description"],
                "human_effect": meta["human_effect"],
                "params": params,
                "sweep_param": meta.get("sweep_param"),
                "programmatic": True,
                "controls": _build_gallery_controls(illusion_type, gen_cls),
                "derived_params": GALLERY_DERIVED_PARAMS.get(illusion_type, []),
                "illusion_image": f"./assets/stimuli/{illusion_type}/illusion.png",
                "control_image": f"./assets/stimuli/{illusion_type}/control.png",
                "control_note": GALLERY_CONTROL_NOTES.get(illusion_type),
            }
        )

    # IllusionVQA: use first manifest item if downloaded.
    ivqa_root = PROJECT_ROOT / "data" / "external" / "illusion_vqa"
    try:
        loader = ExternalDatasetLoader("illusion_vqa", ivqa_root)
        manifest = loader._load_manifest()
        if manifest:
            entry = manifest[0]
            subdir = STIMULI_IMG_DIR / "illusion_vqa"
            subdir.mkdir(parents=True, exist_ok=True)
            ill = Image.open(entry["illusion_path"]).convert("RGB")
            ctrl = Image.open(entry["control_path"]).convert("RGB")
            ill.save(subdir / "illusion.png")
            ctrl.save(subdir / "control.png")
            same_ctrl = entry["illusion_path"] == entry["control_path"]
            meta = ILLUSION_GALLERY_META["illusion_vqa"]
            entries.append(
                {
                    "id": "illusion_vqa",
                    "label": meta["label"],
                    "category": meta["category"],
                    "description": meta["description"],
                    "human_effect": meta["human_effect"],
                    "params": entry.get("params", {}),
                    "external": True,
                    "illusion_image": "./assets/stimuli/illusion_vqa/illusion.png",
                    "control_image": "./assets/stimuli/illusion_vqa/control.png",
                    "control_note": (
                        "No separate control image in manifest; same frame shown."
                        if same_ctrl
                        else None
                    ),
                }
            )
    except (FileNotFoundError, OSError):
        pass

    # HallusionBench: sample from bundled zip if present.
    zip_path = PROJECT_ROOT / "hallusion_bench.zip"
    if zip_path.exists():
        # VD/illusion/10: gallery preview pair (illusion vs. reference frame).
        inner_ill = "hallusion_bench/VD/illusion/18_0.png"
        inner_ctrl = "hallusion_bench/VD/illusion/18_1.png"
        subdir = STIMULI_IMG_DIR / "hallusion_bench"
        subdir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            ill = Image.open(io.BytesIO(zf.read(inner_ill))).convert("RGB")
            ctrl = Image.open(io.BytesIO(zf.read(inner_ctrl))).convert("RGB")
            ill.save(subdir / "illusion.png")
            ctrl.save(subdir / "control.png")
        meta = ILLUSION_GALLERY_META["hallusion_bench"]
        entries.append(
            {
                "id": "hallusion_bench",
                "label": meta["label"],
                "category": meta["category"],
                "description": meta["description"],
                "human_effect": meta["human_effect"],
                "params": {"source_illusion": inner_ill, "source_control": inner_ctrl},
                "external": True,
                "illusion_image": "./assets/stimuli/hallusion_bench/illusion.png",
                "control_image": "./assets/stimuli/hallusion_bench/control.png",
                "control_note": (
                    "Static preview: HallusionBench item 18 (illusion frame vs. reference frame). "
                    "Evaluation uses the full downloaded manifest, not this single pair."
                ),
            }
        )

    category_order = ["geometric", "color", "angle", "motion", "impossible"]
    entries.sort(
        key=lambda e: (
            category_order.index(e["category"]),
            e["label"],
        )
    )
    return {"illusions": entries, "category_order": category_order}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results_path = FIG_DIR / "all_results.json"
    print(f"Loading {all_results_path} ...")
    with open(all_results_path) as fh:
        all_results = json.load(fh)
    print(f"  {len(all_results)} records")

    heas = export_heas_table()
    with open(OUT_DIR / "heas_table.json", "w") as fh:
        json.dump(heas, fh, indent=2)

    psych = export_psychometric_curves(all_results)
    with open(OUT_DIR / "psychometric_curves.json", "w") as fh:
        json.dump(psych, fh, indent=2)

    diag = export_diagnostics_summary()
    with open(OUT_DIR / "diagnostics_summary.json", "w") as fh:
        json.dump(diag, fh, indent=2)

    heas_cells = export_heas_cell_details(heas, diag)
    with open(OUT_DIR / "heas_cell_details.json", "w") as fh:
        json.dump(heas_cells, fh, indent=2)

    dpo_delta = export_dpo_delta(heas)
    with open(OUT_DIR / "dpo_delta.json", "w") as fh:
        json.dump(dpo_delta, fh, indent=2)

    symmpo_delta = export_symmpo_delta(heas)
    with open(OUT_DIR / "symmpo_delta.json", "w") as fh:
        json.dump(symmpo_delta, fh, indent=2)

    rq_summary = export_rq_summary(heas, diag, dpo_delta)
    with open(OUT_DIR / "rq_summary.json", "w") as fh:
        json.dump(rq_summary, fh, indent=2)

    gallery = export_stimulus_gallery()
    with open(OUT_DIR / "illusion_gallery.json", "w") as fh:
        json.dump(gallery, fh, indent=2)
    print(f"  Stimulus gallery: {len(gallery['illusions'])} illusion types → {STIMULI_IMG_DIR}")

    meta = {
        "model_labels": {
            "resnet50": "ResNet-50",
            "convnext_base": "ConvNeXt-B",
            "vit_b_16": "ViT-B/16",
            "vit_l_16": "ViT-L/16",
            "clip_vit_b32": "CLIP ViT-B/32",
            "dinov2_vit_b14": "DINOv2 ViT-B/14",
            "llava_1.5": "LLaVA-1.5",
            "llava_1.5_dpo": "LLaVA-1.5 + DPO",
            "llava_1.5_symmpo": "LLaVA-1.5 + SymMPO",
        },
        "category_labels": {
            "geometric": "Geometric / length",
            "color": "Color / brightness",
            "angle": "Angle / orientation",
            "motion": "Motion-from-static",
            "impossible": "Impossible / scene-level",
        },
    }
    with open(OUT_DIR / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Wrote JSON to {OUT_DIR}")


if __name__ == "__main__":
    main()
