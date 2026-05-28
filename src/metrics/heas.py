"""Human Error Alignment Score (HEAS) computation.

Definition (from the proposal)
--------------------------------
    HEAS(m, c) = 1 - |p_model_illusory(m, c) - p_human_illusory(c)|

where
  - ``m``  is the model identifier,
  - ``c``  is the illusion category,
  - ``p_model_illusory`` is the fraction of stimuli where the model chose
    the human-illusory answer (after filtering by control accuracy if applicable),
  - ``p_human_illusory`` is the corresponding human proportion from published
    psychophysics data.

A score of 1.0 means the model fails exactly as often as humans.
A score of 0.0 means the model's susceptibility rate is as far as possible
from the human baseline.

Note: HEAS is traditionally meaningful when control accuracy exceeds a ceiling
threshold. We apply this filter to programmatic categories but skip it for
the 'impossible' category.

Functions
---------
compute_heas(model_results, human_illusory_rate, control_ceiling_threshold, category)
    Compute HEAS for a single (model, category) pair.

heas_table(all_results, human_baselines, control_ceiling_threshold)
    Compute the full category × model HEAS matrix as a pandas DataFrame.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def compute_heas(
    model_results: list[dict[str, Any]],
    human_illusory_rate: float,
    control_ceiling_threshold: float = 0.80,
    category: str | None = None,
) -> dict[str, Any]:
    """Compute HEAS for one (model, category) slice of results.

    Parameters
    ----------
    model_results :
        List of per-stimulus result dicts (as returned by ``ModelProber.probe_dataset``).
        Each dict must have keys: ``correct``, ``illusory``, ``other``,
        ``predicted_label``, and optionally ``probs_control``.
    human_illusory_rate :
        The proportion of humans who give the illusory answer for this category
        (from published psychophysics, range [0, 1]).
    control_ceiling_threshold :
        Stimuli whose control-image ``correct`` probability is below this
        threshold are excluded.  Defaults to 0.80.  Set to -1 for argmax mode.
    category :
        The illusion category. If "impossible", the control check is skipped.

    Returns
    -------
    dict with keys:
        ``heas``                 — the HEAS score (float, [0, 1]).
        ``p_model_illusory``     — model illusory rate (float).
        ``p_human_illusory``     — human illusory rate (float, echoed).
        ``n_stimuli``            — number of stimuli included after filtering.
        ``n_excluded``           — number of stimuli excluded (low control acc).
    """
    included = []
    excluded = 0

    # Skip control check for impossible category as no physical controls exist
    # (control_path defaults to illusion_path in the loader).
    skip_check = (category == "impossible")

    for r in model_results:
        if skip_check:
            included.append(r)
            continue

        raw = r.get("raw")
        if not isinstance(raw, dict):
            raw = {}
        ctrl_probs = raw.get("probs_control") or None

        if control_ceiling_threshold < 0:
            # Argmax mode (threshold=-1): pass if the control image's most
            # confident class is "correct" (index 0).
            if ctrl_probs is not None:
                # Use cached argmax if available, otherwise compute it
                passes = raw.get("ctrl_argmax_correct", int(ctrl_probs.index(max(ctrl_probs))) == 0)
            else:
                passes = r.get("correct", 0.0) >= r.get("illusory", 0.0)
        else:
            # Standard threshold mode: ctrl_correct probability must exceed threshold.
            if ctrl_probs is not None:
                ctrl_correct = ctrl_probs[0]
            else:
                ctrl_correct = r.get("correct", 0.0)
            passes = ctrl_correct >= control_ceiling_threshold

        if not passes:
            excluded += 1
            continue
        included.append(r)

    n = len(included)
    if n == 0:
        return {
            "heas": float("nan"),
            "p_model_illusory": float("nan"),
            "p_human_illusory": human_illusory_rate,
            "n_stimuli": 0,
            "n_excluded": excluded,
        }

    n_illusory = sum(1 for r in included if r["predicted_label"] == "illusory")
    p_model = n_illusory / n
    heas = 1.0 - abs(p_model - human_illusory_rate)

    return {
        "heas": heas,
        "p_model_illusory": p_model,
        "p_human_illusory": human_illusory_rate,
        "n_stimuli": n,
        "n_excluded": excluded,
    }


def heas_table(
    all_results: list[dict[str, Any]],
    human_baselines: dict[str, float],
    control_ceiling_threshold: float = 0.80,
) -> pd.DataFrame:
    """Build the full category × model HEAS table.

    Parameters
    ----------
    all_results :
        Concatenated results from all models and all categories.
        Each dict must have ``model``, ``category``, and ``predicted_label``.
    human_baselines :
        Mapping ``{category: p_human_illusory}``.
        Categories absent from this dict are skipped.
    control_ceiling_threshold :
        Forwarded to :func:`compute_heas`.

    Returns
    -------
    pd.DataFrame
        Index = category, columns = model name, values = HEAS score.
    """
    df = pd.DataFrame(all_results)
    if df.empty:
        return pd.DataFrame()

    models = df["model"].unique().tolist()
    categories = [c for c in df["category"].unique() if c in human_baselines]

    rows = []
    for category in categories:
        human_rate = human_baselines[category]
        row = {"category": category}
        for model in models:
            subset_df = df[(df["category"] == category) & (df["model"] == model)]
            if subset_df.empty:
                row[model] = float("nan")
            else:
                subset_list = subset_df.to_dict("records")
                res = compute_heas(
                    subset_list, 
                    human_rate, 
                    control_ceiling_threshold,
                    category=category
                )
                row[model] = res["heas"]
        rows.append(row)

    return pd.DataFrame(rows).set_index("category")
