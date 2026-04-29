"""Zero-shot probing protocol for contrastive models (CLIP, DINOv2).

For CLIP: image–text cosine similarity.
For DINOv2: image–image cosine similarity with reference exemplars.

The ``ZeroShotProtocol`` class centralises:
  - Text prompt construction / injection for CLIP.
  - Reference image construction for DINOv2.
  - Batch evaluation with progress reporting.
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# Per-category text prompts (neutral framing).
# These are used as defaults; illusion-type-specific prompts override them.
_TEXT_PROMPTS: dict[str, dict[str, str]] = {
    "geometric": {
        "correct": "Two horizontal lines that look the same length.",
        "illusory": "Two horizontal lines where one looks shorter than the other.",
        "other": "A geometric diagram where line lengths cannot be compared.",
    },
    "color": {
        "correct": "Two grey squares that look exactly the same shade of grey.",
        "illusory": "Two grey squares where one looks darker than the other.",
        "other": "A pattern of grey shapes with no clear brightness comparison.",
    },
    "angle": {
        "correct": "A set of lines that look perfectly parallel to each other.",
        "illusory": "A set of lines that appear tilted or non-parallel.",
        "other": "A pattern of lines with no clear parallel relationship.",
    },
    "motion": {
        "correct": "A grid with bright white dots at every intersection.",
        "illusory": "A grid where dark spots appear to flash at the intersections.",
        "other": "A grid pattern.",
    },
    "impossible": {
        "correct": "A three-dimensional object that could exist in the real world.",
        "illusory": "A three-dimensional object that is physically impossible.",
        "other": "An ambiguous or unclear three-dimensional shape.",
    },
}

# Per-illusion-type prompts override the category defaults.
# The framing targets the specific perceptual mechanism of each illusion
# so that the control image (no context) and illusion image (with context)
# give clearly different probability profiles.
_ILLUSION_TYPE_PROMPTS: dict[str, dict[str, str]] = {
    # Geometric / length
    "muller_lyer": {
        "correct": "Two plain horizontal lines with no arrows, both the same length.",
        "illusory": "Two horizontal lines with arrowhead fins at their ends, making one line look shorter.",
        "other": "An abstract diagram of arrows or shapes.",
    },
    "ponzo": {
        "correct": "Two horizontal bars of the same length between parallel rails.",
        "illusory": "Two horizontal bars between converging lines, where the upper bar looks longer.",
        "other": "A diagram of lines with no clear depth cue.",
    },
    "ebbinghaus": {
        "correct": "Two circles of the same size surrounded by similarly-sized circles.",
        "illusory": "Two circles of the same size where one looks larger due to surrounding small circles.",
        "other": "A pattern of circles with no clear size comparison.",
    },
    # Color / brightness
    "simultaneous_contrast": {
        "correct": "Two grey patches of identical brightness on different backgrounds.",
        "illusory": "Two grey patches where one looks lighter because of a darker surround.",
        "other": "A pattern of grey squares.",
    },
    "whites_illusion": {
        "correct": "Two grey bars that look the same shade of grey.",
        "illusory": "Two grey bars on striped backgrounds where they appear different shades.",
        "other": "A striped pattern with grey inserts.",
    },
    # Angle / orientation
    "zollner": {
        "correct": "A set of long diagonal lines that are truly parallel.",
        "illusory": "A set of long lines with short cross-hatches that make the main lines look tilted.",
        "other": "A pattern of diagonal hatching.",
    },
    "poggendorff": {
        "correct": "A diagonal line passing behind a rectangle, with the two visible segments perfectly aligned.",
        "illusory": "A diagonal line passing behind a rectangle, where the two segments look misaligned.",
        "other": "A rectangle with line segments on either side.",
    },
}


class ZeroShotProtocol:
    """Encapsulates the zero-shot probing protocol for contrastive models.

    Parameters
    ----------
    prober :
        A ``CLIPProber`` or ``DINOv2Prober`` instance.
    custom_prompts :
        Optional override for the default text prompts dict.
    """

    def __init__(
        self,
        prober: Any,
        custom_prompts: dict[str, dict[str, str]] | None = None,
        custom_type_prompts: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.prober = prober
        self.prompts = custom_prompts or _TEXT_PROMPTS
        self.custom_type_prompts = custom_type_prompts

    def get_prompts(self, category: str, illusion_type: str | None = None) -> dict[str, str]:
        """Return the text prompts for a given illusion category / type.

        Resolution order:
        1. illusion-type-specific prompts (most specific)
        2. category-level prompts
        3. "geometric" fallback
        """
        if illusion_type is not None:
            type_prompts = (self.custom_type_prompts or _ILLUSION_TYPE_PROMPTS).get(illusion_type)
            if type_prompts:
                return type_prompts
        return self.prompts.get(category, self.prompts.get("geometric", {}))

    def probe_stimulus(
        self,
        illusion: Image.Image,
        control: Image.Image,
        category: str,
        illusion_type: str,
        correct_answer: str,
        illusory_answer: str,
        extra: dict[str, Any] | None = None,
    ):
        """Probe a single stimulus pair using zero-shot text similarity."""
        merged_extra: dict[str, Any] = {"text_prompts": self.get_prompts(category, illusion_type)}
        if extra:
            merged_extra.update(extra)
        return self.prober.probe_pair(
            illusion=illusion,
            control=control,
            correct_answer=correct_answer,
            illusory_answer=illusory_answer,
            category=category,
            illusion_type=illusion_type,
            extra=merged_extra,
        )

    def probe_dataset(self, pairs: list, verbose: bool = True) -> list[dict[str, Any]]:
        """Run zero-shot probing over all stimulus pairs."""
        from tqdm import tqdm

        results = []
        iterator = tqdm(pairs, desc=f"ZeroShot[{self.prober.model_name}]", disable=not verbose)
        for pair in iterator:
            dist = self.probe_stimulus(
                illusion=pair.illusion,
                control=pair.control,
                category=pair.category,
                illusion_type=pair.illusion_type,
                correct_answer=pair.correct_answer,
                illusory_answer=pair.illusory_answer,
            )
            results.append(
                {
                    "stimulus_id": pair.stimulus_id,
                    "category": pair.category,
                    "illusion_type": pair.illusion_type,
                    "params": pair.params,
                    "model": self.prober.model_name,
                    **dist.normalise().as_dict(),
                    "predicted_label": dist.to_label(),
                    "raw": dist.raw,
                }
            )
        return results
