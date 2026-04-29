"""Psychometric curve fitting and alignment metrics.

A psychometric curve describes how a model's (or human's) illusory response
rate varies as a function of a continuously-varying stimulus parameter (e.g.
Müller-Lyer fin length, Ponzo convergence angle).

Functions / classes
-------------------
PsychometricCurve
    Fits and evaluates a sigmoidal psychometric function.

spearman_alignment(model_curve, human_curve, param_values)
    Spearman ρ between model and human response rates at matched param points.

sign_alignment(model_curve, human_curve, param_values)
    Proportion of param points where model and human curves agree in their
    direction of change (both increasing or both decreasing).

psychometric_from_results(results, param_key, param_values)
    Utility: compute per-param illusory rates from a result list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Logistic psychometric function
# ──────────────────────────────────────────────────────────────────────────────

def _logistic(x: np.ndarray, alpha: float, beta: float, gamma: float, delta: float) -> np.ndarray:
    """4-parameter logistic: γ + (δ - γ) / (1 + exp(-β(x - α)))."""
    return gamma + (delta - gamma) / (1.0 + np.exp(-beta * (x - alpha)))


@dataclass
class PsychometricCurve:
    """A fitted 4-parameter psychometric function.

    Attributes
    ----------
    alpha : Threshold (x-value at 50% between floor and ceiling).
    beta  : Slope.
    gamma : Lower asymptote (floor).
    delta : Upper asymptote (ceiling).
    popt  : Raw scipy optimised parameters.
    pcov  : Parameter covariance matrix.
    r_squared : Goodness of fit.
    """

    alpha: float
    beta: float
    gamma: float
    delta: float
    popt: np.ndarray
    pcov: np.ndarray
    r_squared: float

    @classmethod
    def fit(
        cls,
        param_values: np.ndarray,
        response_rates: np.ndarray,
        p0: list[float] | None = None,
    ) -> "PsychometricCurve":
        """Fit a logistic psychometric curve.

        Parameters
        ----------
        param_values   : Stimulus intensity values (x-axis).
        response_rates : Illusory-response proportions (y-axis, range [0, 1]).
        p0             : Initial parameter guess [alpha, beta, gamma, delta].

        Returns
        -------
        PsychometricCurve
        """
        x = np.asarray(param_values, dtype=float)
        y = np.asarray(response_rates, dtype=float)

        if p0 is None:
            p0 = [float(np.median(x)), 0.1, float(y.min()), float(y.max())]

        try:
            popt, pcov = curve_fit(
                _logistic, x, y, p0=p0,
                bounds=([x.min(), -np.inf, 0.0, 0.0], [x.max(), np.inf, 1.0, 1.0]),
                maxfev=10_000,
            )
        except RuntimeError:
            logger.warning("Psychometric fit did not converge; returning linear interpolation proxy.")
            popt = np.array(p0)
            pcov = np.full((4, 4), np.nan)

        y_pred = _logistic(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return cls(
            alpha=float(popt[0]),
            beta=float(popt[1]),
            gamma=float(popt[2]),
            delta=float(popt[3]),
            popt=popt,
            pcov=pcov,
            r_squared=r2,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _logistic(np.asarray(x, dtype=float), *self.popt)

    def threshold(self) -> float:
        """Return alpha, the stimulus intensity at half-maximal response."""
        return self.alpha


# ──────────────────────────────────────────────────────────────────────────────
# Alignment metrics
# ──────────────────────────────────────────────────────────────────────────────

def spearman_alignment(
    model_rates: np.ndarray,
    human_rates: np.ndarray,
) -> tuple[float, float]:
    """Spearman ρ between model and human illusory-response curves.

    Parameters
    ----------
    model_rates : Per-parameter illusory rates for the model.
    human_rates : Matched per-parameter illusory rates for humans.

    Returns
    -------
    (rho, p_value)
    """
    model_rates = np.asarray(model_rates)
    human_rates = np.asarray(human_rates)
    if len(model_rates) < 3:
        logger.warning("Too few data points for Spearman correlation; returning (nan, nan).")
        return float("nan"), float("nan")
    rho, pval = spearmanr(model_rates, human_rates)
    return float(rho), float(pval)


def sign_alignment(
    model_rates: np.ndarray,
    human_rates: np.ndarray,
) -> float:
    """Proportion of consecutive pairs where model and human agree in direction.

    For each adjacent pair of stimulus intensities (i, i+1) we check whether
    both curves increase (or both decrease / stay flat).  Returns the fraction
    of such pairs that agree.

    Parameters
    ----------
    model_rates : Per-parameter illusory rates for the model (ordered by param).
    human_rates : Matched per-parameter illusory rates for humans.

    Returns
    -------
    float in [0, 1]
    """
    model_rates = np.asarray(model_rates)
    human_rates = np.asarray(human_rates)
    if len(model_rates) < 2:
        return float("nan")
    m_diff = np.sign(np.diff(model_rates))
    h_diff = np.sign(np.diff(human_rates))
    return float(np.mean(m_diff == h_diff))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def psychometric_from_results(
    results: list[dict[str, Any]],
    param_key: str,
    param_values: list[float],
    use_continuous: bool = True,
) -> np.ndarray:
    """Compute per-parameter illusory rates from a flat results list.

    Parameters
    ----------
    results     : Per-stimulus result dicts with ``params``, ``predicted_label``,
                  and ``illusory`` (continuous probability) fields.
    param_key   : The key inside ``params`` used to group stimuli.
    param_values: Ordered list of unique parameter values to consider.
    use_continuous :
        If True (default), use the mean continuous ``illusory`` probability
        as the response rate — this produces a smooth psychometric curve and
        avoids the step-function artefact of discrete labels.
        If False, use the fraction of stimuli with ``predicted_label="illusory"``.

    Returns
    -------
    np.ndarray of shape (len(param_values),)
    """
    rates = []
    for pv in param_values:
        # Match on rounded fin_length to tolerate float precision differences
        subset = [
            r for r in results
            if round(float(r["params"].get(param_key, -1)), 2) == round(float(pv), 2)
        ]
        if not subset:
            rates.append(float("nan"))
            continue
        if use_continuous:
            rates.append(float(np.mean([r.get("illusory", 0.0) for r in subset])))
        else:
            illusory_count = sum(1 for r in subset if r["predicted_label"] == "illusory")
            rates.append(illusory_count / len(subset))
    return np.array(rates)
