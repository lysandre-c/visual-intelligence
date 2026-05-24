"""Diagnostic metrics for model outputs.

These metrics are descriptive rather than the main scientific endpoint.
They help interpret HEAS and psychometric curves by showing whether a model
is mostly choosing ``correct``, ``illusory``, or ``other``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


LABELS = ("correct", "illusory", "other")


def output_diagnostics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise accuracy and output distribution for a result slice.

    Accuracy here means physical correctness on the illusion image:
    ``predicted_label == "correct"``.  This is not the main alignment metric,
    but it is useful context.  A human-like illusion error lowers accuracy while
    increasing illusory response.
    """
    n = len(results)
    if n == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "p_pred_correct": float("nan"),
            "p_pred_illusory": float("nan"),
            "p_pred_other": float("nan"),
            "mean_prob_correct": float("nan"),
            "mean_prob_illusory": float("nan"),
            "mean_prob_other": float("nan"),
            "control_argmax_accuracy": float("nan"),
        }

    pred_counts = Counter(r.get("predicted_label", "other") for r in results)
    mean_probs = {
        label: sum(float(r.get(label, 0.0)) for r in results) / n
        for label in LABELS
    }

    ctrl_ok = 0
    ctrl_total = 0
    for r in results:
        raw = r.get("raw")
        if not isinstance(raw, dict):
            raw = {}
        ctrl_probs = raw.get("probs_control")
        if ctrl_probs is None:
            continue
        ctrl_total += 1
        ctrl_ok += int(int(ctrl_probs.index(max(ctrl_probs))) == 0)

    return {
        "n": n,
        "accuracy": pred_counts["correct"] / n,
        "p_pred_correct": pred_counts["correct"] / n,
        "p_pred_illusory": pred_counts["illusory"] / n,
        "p_pred_other": pred_counts["other"] / n,
        "mean_prob_correct": mean_probs["correct"],
        "mean_prob_illusory": mean_probs["illusory"],
        "mean_prob_other": mean_probs["other"],
        "control_argmax_accuracy": ctrl_ok / ctrl_total if ctrl_total else float("nan"),
    }


def metric_definitions() -> dict[str, str]:
    """Human-readable definitions saved with experiment outputs."""
    return {
        "accuracy": (
            "Fraction of illusion images where predicted_label == 'correct'. "
            "This measures physical correctness, not human-like alignment."
        ),
        "illusory_response_rate": (
            "Mean probability assigned to the canonical human illusory answer. "
            "On psychometric plots this is averaged at each stimulus parameter value."
        ),
        "p_pred_illusory": (
            "Fraction of images whose top predicted label is 'illusory'. "
            "This is a hard-label version of illusory_response_rate."
        ),
        "HEAS": (
            "Human Error Alignment Score: 1 - abs(p_model_illusory - "
            "p_human_illusory). Higher means the model makes the canonical "
            "illusory error at a similar rate to humans."
        ),
        "control_argmax_accuracy": (
            "Fraction of control images whose top prediction is 'correct'. "
            "Used as a sanity check that the task is understood without the illusion cue."
        ),
        "psychometric_threshold": (
            "Stimulus strength where the fitted illusory-response curve reaches "
            "its midpoint. Meaningful only when the curve has a graded transition."
        ),
    }
