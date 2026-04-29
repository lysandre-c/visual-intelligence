"""Reusable matplotlib visualisation functions.

All functions return a ``matplotlib.figure.Figure`` so the caller controls
whether to save, display, or further customise the output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# HEAS table heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_heas_table(
    heas_df: pd.DataFrame,
    title: str = "Human Error Alignment Score (HEAS)",
    figsize: tuple[float, float] | None = None,
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
    annotate: bool = True,
) -> plt.Figure:
    """Plot the category × model HEAS matrix as a colour-coded heatmap.

    Parameters
    ----------
    heas_df :
        DataFrame with categories as rows and model names as columns,
        values in [0, 1] (NaN for excluded entries).
    title   : Figure title.
    figsize : (width, height) in inches.
    cmap    : Matplotlib colour map name.
    vmin, vmax : Colour scale limits.
    annotate : Whether to write HEAS values inside cells.

    Returns
    -------
    matplotlib.figure.Figure
    """
    nrows, ncols = heas_df.shape
    if figsize is None:
        figsize = (max(6, ncols * 1.4), max(3, nrows * 0.9))

    fig, ax = plt.subplots(figsize=figsize)
    data = heas_df.values.astype(float)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(heas_df.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(heas_df.index, fontsize=9)
    ax.set_title(title, fontsize=11, pad=12)

    if annotate:
        for i in range(nrows):
            for j in range(ncols):
                val = data[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < 0.4 or val > 0.85 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=text_color)
                else:
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=7, color="grey")

    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="HEAS")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Psychometric curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_psychometric_curves(
    param_values: np.ndarray,
    model_curves: dict[str, np.ndarray],
    human_rates: np.ndarray | None = None,
    category: str = "",
    illusion_type: str = "",
    param_label: str = "Stimulus intensity",
    figsize: tuple[float, float] = (7, 4),
) -> plt.Figure:
    """Plot psychometric response curves for multiple models + human baseline.

    Parameters
    ----------
    param_values : 1-D array of stimulus parameter values (x-axis).
    model_curves : dict mapping model name → 1-D array of illusory rates.
    human_rates  : Optional human illusory rates at the same param values.
    category     : Illusion category (for the title).
    illusion_type: Specific illusion (for the title).
    param_label  : X-axis label.
    figsize      : Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for idx, (model_name, rates) in enumerate(model_curves.items()):
        ax.plot(param_values, rates, marker="o", markersize=4,
                color=colors[idx % len(colors)], label=model_name, linewidth=1.5)

    if human_rates is not None:
        ax.plot(param_values, human_rates, marker="s", markersize=5,
                color="black", linestyle="--", linewidth=2, label="Human baseline")

    ax.set_xlabel(param_label, fontsize=10)
    ax.set_ylabel("Illusory response rate", fontsize=10)
    title_parts = [p for p in [category, illusion_type] if p]
    ax.set_title("  ·  ".join(title_parts) if title_parts else "Psychometric curves", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.7)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Attention / GradCAM overlay
# ──────────────────────────────────────────────────────────────────────────────

def plot_attention_overlay(
    image: Any,  # PIL.Image.Image
    saliency: np.ndarray,
    title: str = "",
    alpha: float = 0.5,
    cmap: str = "jet",
    figsize: tuple[float, float] = (5, 5),
) -> plt.Figure:
    """Overlay a saliency map on top of an image.

    Parameters
    ----------
    image   : PIL image (the illusion stimulus).
    saliency: 2-D float32 array in [0, 1], same spatial size as ``image``.
    title   : Subplot title.
    alpha   : Transparency of the saliency overlay.
    cmap    : Colourmap for the saliency.
    figsize : Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    h, w = np.array(image).shape[:2]
    if saliency.shape != (h, w):
        from PIL import Image as PImage
        sal_img = PImage.fromarray((saliency * 255).astype(np.uint8)).resize(
            (w, h), resample=PImage.BICUBIC
        )
        saliency = np.array(sal_img, dtype=np.float32) / 255.0
    ax.imshow(saliency, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Bar chart: per-category HEAS comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_heas_bar(
    heas_df: pd.DataFrame,
    category: str,
    figsize: tuple[float, float] = (6, 4),
    title: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of HEAS scores for all models in one category.

    Parameters
    ----------
    heas_df  : DataFrame indexed by category, columns = model names.
    category : The category row to visualise.
    figsize  : Figure size.
    title    : Optional title override.

    Returns
    -------
    matplotlib.figure.Figure
    """
    row = heas_df.loc[category].dropna().sort_values()
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#d73027" if v < 0.5 else "#4dac26" for v in row.values]
    ax.barh(row.index, row.values, color=colors, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("HEAS", fontsize=10)
    ax.set_title(title or f"HEAS — {category}", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: save figure
# ──────────────────────────────────────────────────────────────────────────────

def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save ``fig`` to ``path``, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
