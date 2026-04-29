"""Base abstractions for model probers.

Every model wrapper in this project must implement ``ModelProber``, which
takes an illusion / control image pair and returns a ``ResponseDistribution``
— the probability (or soft score) the model assigns to each of the three
answer categories: *correct*, *illusory*, and *other*.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Response types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ResponseDistribution:
    """Soft assignment of a model's response to the three answer categories.

    Attributes
    ----------
    correct : float
        Probability / score for the physically correct answer.
    illusory : float
        Probability / score for the human-illusory answer.
    other : float
        Probability / score for all remaining options.
    raw : dict
        Free-form dict for storing model-specific extra information (logits,
        chosen text, etc.).
    """

    correct: float
    illusory: float
    other: float
    raw: dict[str, Any] | None = None

    def to_label(self) -> str:
        """Return the label with the highest score."""
        scores = {"correct": self.correct, "illusory": self.illusory, "other": self.other}
        return max(scores, key=scores.__getitem__)

    def normalise(self) -> "ResponseDistribution":
        """Return a copy with scores summing to 1 (no-op if already so)."""
        total = self.correct + self.illusory + self.other
        if total == 0:
            return ResponseDistribution(1 / 3, 1 / 3, 1 / 3, self.raw)
        return ResponseDistribution(
            self.correct / total,
            self.illusory / total,
            self.other / total,
            self.raw,
        )

    def as_dict(self) -> dict[str, float]:
        return {"correct": self.correct, "illusory": self.illusory, "other": self.other}


# ──────────────────────────────────────────────────────────────────────────────
# Abstract prober
# ──────────────────────────────────────────────────────────────────────────────

class ModelProber(ABC):
    """Abstract base class for all model probers.

    Subclasses must implement :meth:`probe_pair`, which receives the illusion
    and control images alongside the metadata needed to construct a
    task-appropriate question.

    The base class handles device management and common utilities.
    """

    # Human-readable model name; set by each subclass.
    model_name: str = ""

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def probe_pair(
        self,
        illusion: Image.Image,
        control: Image.Image,
        correct_answer: str,
        illusory_answer: str,
        category: str,
        illusion_type: str,
        extra: dict[str, Any] | None = None,
    ) -> ResponseDistribution:
        """Probe the model on a single illusion / control pair.

        The implementation must:
        1. Feed the images through the model using the appropriate probing
           protocol (linear probe, zero-shot, VLM prompting).
        2. Map model outputs to (correct, illusory, other) scores.
        3. Verify that control accuracy exceeds the sanity-check threshold
           before returning a meaningful distribution.

        Parameters
        ----------
        illusion : The image containing the illusory cue.
        control  : The matched control image with no illusion.
        correct_answer : The physically correct answer label.
        illusory_answer: The human-illusory answer label.
        category : Illusion category string.
        illusion_type: Specific illusion name.
        extra : Optional additional metadata (e.g. VLM question text).

        Returns
        -------
        ResponseDistribution
            Scores for the three answer categories.
        """

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def probe_dataset(
        self,
        pairs: list,  # list[StimulusPair] — avoid circular import
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Run :meth:`probe_pair` over a list of ``StimulusPair`` objects.

        Returns a list of result dicts, one per pair, including the
        ``ResponseDistribution`` as well as the pair metadata.
        """
        from tqdm import tqdm
        results = []
        iterator = tqdm(pairs, desc=self.model_name, disable=not verbose)
        for pair in iterator:
            dist = self.probe_pair(
                illusion=pair.illusion,
                control=pair.control,
                correct_answer=pair.correct_answer,
                illusory_answer=pair.illusory_answer,
                category=pair.category,
                illusion_type=pair.illusion_type,
            )
            results.append(
                {
                    "stimulus_id": pair.stimulus_id,
                    "category": pair.category,
                    "illusion_type": pair.illusion_type,
                    "params": pair.params,
                    "model": self.model_name,
                    **dist.normalise().as_dict(),
                    "predicted_label": dist.to_label(),
                    "raw": dist.raw,
                }
            )
        return results
