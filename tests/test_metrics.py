"""Unit tests for HEAS and psychometric metrics.

Covers
------
- HEAS edge cases: perfect alignment, perfect misalignment, NaN when no stimuli pass filter.
- heas_table: correct shape and model names.
- psychometric_from_results: correct rate computation.
- spearman_alignment: expected ρ on trivially correlated / anti-correlated data.
- sign_alignment: expected score on monotone data.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.metrics.heas import compute_heas, heas_table
from src.metrics.psychometric import (
    PsychometricCurve,
    psychometric_from_results,
    sign_alignment,
    spearman_alignment,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_results(
    n: int,
    illusory_fraction: float,
    ctrl_correct: float = 1.0,
    model: str = "test_model",
    category: str = "geometric",
    illusion_type: str = "muller_lyer",
) -> list[dict]:
    """Create synthetic result dicts with controlled illusory rate."""
    results = []
    for i in range(n):
        is_illusory = i < int(n * illusory_fraction)
        label = "illusory" if is_illusory else "correct"
        results.append(
            {
                "stimulus_id": f"stim_{i:04d}",
                "category": category,
                "illusion_type": illusion_type,
                "params": {"fin_length": float(i % 10)},
                "model": model,
                "correct": 1.0 - float(is_illusory) * 0.9,
                "illusory": float(is_illusory) * 0.9,
                "other": 0.0,
                "predicted_label": label,
                "raw": {"probs_control": [ctrl_correct, 1.0 - ctrl_correct, 0.0]},
            }
        )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# compute_heas tests
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeHeas:
    def test_perfect_alignment(self):
        """Model matches human rate exactly → HEAS = 1.0."""
        human_rate = 0.75
        results = _make_results(100, human_rate)
        out = compute_heas(results, human_rate)
        assert math.isclose(out["heas"], 1.0, abs_tol=1e-2)

    def test_zero_alignment(self):
        """Model rate is 0 when human rate is 1 → HEAS = 0.0."""
        human_rate = 1.0
        results = _make_results(100, 0.0)
        out = compute_heas(results, human_rate)
        assert math.isclose(out["heas"], 0.0, abs_tol=1e-2)

    def test_heas_bounds(self):
        """HEAS must always be in [0, 1]."""
        for model_rate in np.linspace(0, 1, 11):
            results = _make_results(50, float(model_rate))
            out = compute_heas(results, human_illusory_rate=0.5)
            assert 0.0 <= out["heas"] <= 1.0 or math.isnan(out["heas"])

    def test_returns_nan_when_no_stimuli_pass_filter(self):
        """If all control accuracies are below threshold, HEAS should be NaN."""
        results = _make_results(20, 0.5, ctrl_correct=0.0)
        out = compute_heas(results, 0.5, control_ceiling_threshold=0.8)
        assert math.isnan(out["heas"])
        assert out["n_stimuli"] == 0
        assert out["n_excluded"] == 20

    def test_excluded_count(self):
        """Half the stimuli have low control accuracy."""
        results = _make_results(20, 0.5, ctrl_correct=1.0)
        # Override half to low control correct
        for r in results[:10]:
            r["raw"]["probs_control"] = [0.1, 0.9, 0.0]
        out = compute_heas(results, 0.5, control_ceiling_threshold=0.8)
        assert out["n_excluded"] == 10
        assert out["n_stimuli"] == 10

    def test_p_model_illusory_correct(self):
        """p_model_illusory should match the requested fraction."""
        results = _make_results(100, 0.6)
        out = compute_heas(results, 0.5)
        assert math.isclose(out["p_model_illusory"], 0.6, abs_tol=0.02)


# ──────────────────────────────────────────────────────────────────────────────
# heas_table tests
# ──────────────────────────────────────────────────────────────────────────────

class TestHeasTable:
    def test_table_shape(self):
        models = ["model_a", "model_b"]
        all_results = []
        for m in models:
            all_results.extend(_make_results(50, 0.5, model=m, category="geometric"))
            all_results.extend(_make_results(50, 0.5, model=m, category="color"))
        human_baselines = {"geometric": 0.75, "color": 0.80}
        table = heas_table(all_results, human_baselines)
        assert table.shape == (2, 2)  # 2 categories × 2 models

    def test_table_columns_are_models(self):
        all_results = _make_results(50, 0.5, model="model_x")
        table = heas_table(all_results, {"geometric": 0.75})
        assert "model_x" in table.columns

    def test_missing_model_category_is_nan(self):
        """If a model has no results for a category, HEAS should be NaN."""
        results_a = _make_results(50, 0.5, model="a", category="geometric")
        results_b = _make_results(50, 0.5, model="b", category="color")
        table = heas_table(results_a + results_b, {"geometric": 0.75, "color": 0.80})
        assert math.isnan(table.loc["geometric", "b"])
        assert math.isnan(table.loc["color", "a"])


# ──────────────────────────────────────────────────────────────────────────────
# psychometric_from_results tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPsychometricFromResults:
    def test_rate_computation(self):
        results = []
        for i, fl in enumerate([10.0, 20.0, 30.0, 40.0]):
            for j in range(10):
                # All illusory for fl≥20
                label = "illusory" if fl >= 20.0 else "correct"
                results.append({
                    "params": {"fin_length": fl},
                    "predicted_label": label,
                })
        param_values = [10.0, 20.0, 30.0, 40.0]
        rates = psychometric_from_results(results, "fin_length", param_values)
        assert rates[0] == pytest.approx(0.0)
        assert rates[1] == pytest.approx(1.0)
        assert rates[2] == pytest.approx(1.0)
        assert rates[3] == pytest.approx(1.0)

    def test_missing_param_returns_nan(self):
        results = [{"params": {"fin_length": 10.0}, "predicted_label": "correct"}]
        rates = psychometric_from_results(results, "fin_length", [10.0, 99.0])
        assert math.isnan(rates[1])


# ──────────────────────────────────────────────────────────────────────────────
# spearman_alignment tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSpearmanAlignment:
    def test_perfect_positive_correlation(self):
        x = np.arange(10, dtype=float)
        rho, _ = spearman_alignment(x, x)
        assert math.isclose(rho, 1.0, abs_tol=1e-9)

    def test_perfect_negative_correlation(self):
        x = np.arange(10, dtype=float)
        rho, _ = spearman_alignment(x, x[::-1])
        assert math.isclose(rho, -1.0, abs_tol=1e-9)

    def test_short_input_returns_nan(self):
        rho, pval = spearman_alignment(np.array([0.5, 0.6]), np.array([0.5, 0.7]))
        assert math.isnan(rho)


# ──────────────────────────────────────────────────────────────────────────────
# sign_alignment tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSignAlignment:
    def test_identical_curves_perfect_score(self):
        x = np.array([0.1, 0.3, 0.6, 0.8])
        assert sign_alignment(x, x) == pytest.approx(1.0)

    def test_opposite_trends_zero_score(self):
        increasing = np.array([0.1, 0.3, 0.6, 0.8])
        decreasing = increasing[::-1]
        assert sign_alignment(increasing, decreasing) == pytest.approx(0.0)

    def test_single_point_returns_nan(self):
        assert math.isnan(sign_alignment(np.array([0.5]), np.array([0.5])))


# ──────────────────────────────────────────────────────────────────────────────
# PsychometricCurve tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPsychometricCurve:
    def test_fit_monotone_data(self):
        """Fitting a true logistic curve should recover near-perfect R²."""
        x = np.linspace(0, 10, 20)
        alpha, beta, gamma, delta = 5.0, 1.5, 0.05, 0.95
        y = gamma + (delta - gamma) / (1 + np.exp(-beta * (x - alpha)))
        y += np.random.default_rng(0).normal(0, 0.01, len(y))
        y = np.clip(y, 0, 1)
        curve = PsychometricCurve.fit(x, y)
        assert curve.r_squared > 0.95

    def test_predict_within_bounds(self):
        x = np.linspace(0, 10, 10)
        y = np.linspace(0.1, 0.9, 10)
        curve = PsychometricCurve.fit(x, y)
        preds = curve.predict(x)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)
