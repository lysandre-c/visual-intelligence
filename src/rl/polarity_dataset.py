"""Generate Symmetric Polarity-Inverted DPO training data.

For each visual illusion image produced by the existing stimulus generators,
this script creates a *polarity pair*: two opposite textual prompts on the
exact same static image, with chosen/rejected labels that flip according
to the linguistic polarity.

Additionally, ~20 % of samples are non-illusion **control VQA** entries
that ask factual questions about the image (e.g. "How many lines are in
this image?") to serve as a catastrophic-forgetting safeguard.

Output: ``data/rl/dataset.jsonl``
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image

# ── Project root on sys.path ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stimuli.geometric import (
    MullerLyerGenerator,
    PonzoGenerator,
    EbbinghausGenerator,
)
from src.stimuli.color import SimultaneousContrastGenerator, WhiteIllusionGenerator
from src.stimuli.angle import ZollnerGenerator, PoggendorffGenerator
from src.stimuli.motion import ScintillatingGridGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# Polarity templates per illusion category
# ────────────────────────────────────────────────────────────────────────

# Each template defines:
#   question_orig / question_inv   : the polarity-inverted prompt pair
#   chosen_orig / rejected_orig    : human-aligned + non-aligned for x
#   chosen_inv  / rejected_inv     : flipped pair for x'
#   control_questions              : factual non-illusion VQA questions

POLARITY_TEMPLATES: dict[str, dict[str, Any]] = {
    # ── Geometric (Müller-Lyer, Ponzo, Ebbinghaus) ──────────────────────
    "geometric": {
        "question_orig": (
            "Which of the two horizontal lines appears longer?\n"
            "Options:\n  {A}. The top line looks longer.\n"
            "  {B}. The bottom line looks longer.\n"
            "  {C}. They are equal in length.\n"
            "Reply with the letter of your answer only."
        ),
        "question_inv": (
            "Which of the two horizontal lines appears shorter?\n"
            "Options:\n  {A}. The top line looks shorter.\n"
            "  {B}. The bottom line looks shorter.\n"
            "  {C}. They are equal in length.\n"
            "Reply with the letter of your answer only."
        ),
        # For Müller-Lyer: top line has inward fins → appears shorter,
        # bottom line has outward fins → appears longer.
        # Human illusory answer to "longer?" → bottom line.
        "chosen_orig": "The bottom line looks longer.",
        "rejected_orig": "They are equal in length.",
        "chosen_inv": "The top line looks shorter.",
        "rejected_inv": "They are equal in length.",
        "control_questions": [
            "How many horizontal lines are shown in this image?",
            "What colour is the background of this image?",
            "Are there any diagonal marks at the ends of the lines?",
        ],
        "control_answers": [
            "Two horizontal lines.",
            "The background is white.",
            "Yes, there are diagonal marks.",
        ],
    },
    # ── Colour (Simultaneous Contrast, White's) ─────────────────────────
    "color": {
        "question_orig": (
            "Which of the two grey patches appears brighter?\n"
            "Options:\n  {A}. The left patch looks brighter.\n"
            "  {B}. The right patch looks brighter.\n"
            "  {C}. They are the same brightness.\n"
            "Reply with the letter of your answer only."
        ),
        "question_inv": (
            "Which of the two grey patches appears dimmer?\n"
            "Options:\n  {A}. The left patch looks dimmer.\n"
            "  {B}. The right patch looks dimmer.\n"
            "  {C}. They are the same brightness.\n"
            "Reply with the letter of your answer only."
        ),
        # Simultaneous contrast: left patch on dark surround looks brighter.
        "chosen_orig": "The left patch looks brighter.",
        "rejected_orig": "They are the same brightness.",
        "chosen_inv": "The right patch looks dimmer.",
        "rejected_inv": "They are the same brightness.",
        "control_questions": [
            "How many grey patches are visible in this image?",
            "Is the background uniform across the whole image?",
        ],
        "control_answers": [
            "Two grey patches.",
            "No, the background differs between left and right.",
        ],
    },
    # ── Angle (Zöllner, Poggendorff) ────────────────────────────────────
    "angle": {
        "question_orig": (
            "Do the long diagonal lines appear parallel to each other?\n"
            "Options:\n  {A}. No, they appear to converge or diverge.\n"
            "  {B}. Yes, they are parallel.\n"
            "  {C}. It is unclear.\n"
            "Reply with the letter of your answer only."
        ),
        "question_inv": (
            "Do the long diagonal lines appear non-parallel?\n"
            "Options:\n  {A}. Yes, they appear non-parallel.\n"
            "  {B}. No, they look parallel.\n"
            "  {C}. It is unclear.\n"
            "Reply with the letter of your answer only."
        ),
        # Zöllner: lines appear non-parallel (illusory).
        "chosen_orig": "No, they appear to converge or diverge.",
        "rejected_orig": "Yes, they are parallel.",
        "chosen_inv": "Yes, they appear non-parallel.",
        "rejected_inv": "No, they look parallel.",
        "control_questions": [
            "How many long lines are drawn in this image?",
            "Are there short hatching marks crossing the main lines?",
        ],
        "control_answers": [
            "Several long lines.",
            "Yes, short hatch marks cross the lines.",
        ],
    },
    # ── Motion (Scintillating Grid) ─────────────────────────────────────
    "motion": {
        "question_orig": (
            "Does the static pattern appear to move, rotate, or flicker?\n"
            "Options:\n  {A}. Yes, the pattern appears to move or flicker.\n"
            "  {B}. No, the pattern appears still.\n"
            "  {C}. It is unclear.\n"
            "Reply with the letter of your answer only."
        ),
        "question_inv": (
            "Does the static pattern appear completely still?\n"
            "Options:\n  {A}. Yes, the pattern appears still.\n"
            "  {B}. No, the pattern appears to move or flicker.\n"
            "  {C}. It is unclear.\n"
            "Reply with the letter of your answer only."
        ),
        # Scintillating grid: pattern appears to flicker (illusory).
        "chosen_orig": "Yes, the pattern appears to move or flicker.",
        "rejected_orig": "No, the pattern appears still.",
        "chosen_inv": "No, the pattern appears to move or flicker.",
        "rejected_inv": "Yes, the pattern appears still.",
        "control_questions": [
            "What shape is the overall grid structure?",
            "Are there small dots at the intersections of the grid?",
        ],
        "control_answers": [
            "The grid is a rectangular lattice.",
            "Yes, there are dots at the intersections.",
        ],
    },
}


# ────────────────────────────────────────────────────────────────────────
# Generator registry
# ────────────────────────────────────────────────────────────────────────

# Maps (category, illusion_type) → generator class.
GENERATOR_REGISTRY: list[tuple[str, str, type]] = [
    ("geometric", "muller_lyer", MullerLyerGenerator),
    ("geometric", "ponzo", PonzoGenerator),
    ("geometric", "ebbinghaus", EbbinghausGenerator),
    ("color", "simultaneous_contrast", SimultaneousContrastGenerator),
    ("color", "whites_illusion", WhiteIllusionGenerator),
    ("angle", "zollner", ZollnerGenerator),
    ("angle", "poggendorff", PoggendorffGenerator),
    ("motion", "scintillating_grid", ScintillatingGridGenerator),
]


def _save_image(image: Image.Image, path: Path) -> None:
    """Save a PIL image, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path))


def _build_polarity_entry(
    image_path: str,
    category: str,
    illusion_type: str,
    template: dict[str, Any],
    stimulus_id: str,
) -> dict[str, Any]:
    """Build one JSONL record for a polarity-inverted DPO pair."""
    return {
        "image_path": image_path,
        "category": category,
        "illusion_type": illusion_type,
        "stimulus_id": stimulus_id,
        "original_prompt": template["question_orig"],
        "original_chosen": template["chosen_orig"],
        "original_rejected": template["rejected_orig"],
        "inverted_prompt": template["question_inv"],
        "inverted_chosen": template["chosen_inv"],
        "inverted_rejected": template["rejected_inv"],
        "is_control": False,
    }


def _build_control_entry(
    image_path: str,
    category: str,
    illusion_type: str,
    question: str,
    answer: str,
    stimulus_id: str,
) -> dict[str, Any]:
    """Build a control VQA JSONL record (no DPO — standard SFT-style).

    For control entries the "chosen" fields on both polarities are the
    same factual answer, and the "rejected" is a generic wrong answer.
    This makes the DPO gradient push toward the correct factual response
    without any polarity structure.
    """
    wrong_answer = "I cannot determine the answer."
    return {
        "image_path": image_path,
        "category": category,
        "illusion_type": illusion_type,
        "stimulus_id": f"{stimulus_id}_ctrl",
        "original_prompt": f"{question}\nAnswer in one sentence.",
        "original_chosen": answer,
        "original_rejected": wrong_answer,
        "inverted_prompt": f"{question}\nAnswer in one sentence.",
        "inverted_chosen": answer,
        "inverted_rejected": wrong_answer,
        "is_control": True,
    }


# ────────────────────────────────────────────────────────────────────────
# Main dataset builder
# ────────────────────────────────────────────────────────────────────────


def generate_dataset(
    output_dir: Path | None = None,
    max_per_illusion: int = 60,
    control_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    """Generate the symmetric polarity DPO dataset.

    Parameters
    ----------
    output_dir : Where to write images and the JSONL.
        Defaults to ``PROJECT_ROOT / "data" / "rl"``.
    max_per_illusion : Maximum number of stimulus pairs to sample per
        illusion type (to keep dataset size manageable).
    control_ratio : Fraction of the total dataset reserved for non-
        illusion VQA control entries (catastrophic-forgetting safeguard).
    seed : Random seed for reproducible sampling.

    Returns
    -------
    Path to the generated ``dataset.jsonl``.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "rl"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"

    rng = random.Random(seed)
    all_entries: list[dict[str, Any]] = []
    control_pool: list[dict[str, Any]] = []

    for category, illusion_type, gen_cls in GENERATOR_REGISTRY:
        logger.info("Processing %s / %s ...", category, illusion_type)
        gen = gen_cls()
        grid = gen.param_grid()

        # Sub-sample if grid is very large
        if len(grid) > max_per_illusion:
            grid = rng.sample(grid, max_per_illusion)

        template = POLARITY_TEMPLATES[category]

        for idx, params in enumerate(grid):
            pair = gen.generate(params)
            stim_id = f"{illusion_type}_{idx:04d}"

            # Save illusion image
            img_path = images_dir / category / f"{stim_id}.png"
            _save_image(pair.illusion, img_path)
            rel_path = str(img_path.relative_to(PROJECT_ROOT))

            # Polarity-inverted DPO entry
            entry = _build_polarity_entry(
                image_path=rel_path,
                category=category,
                illusion_type=illusion_type,
                template=template,
                stimulus_id=stim_id,
            )
            all_entries.append(entry)

            # Control VQA candidates (we'll sample from these later)
            for q, a in zip(
                template["control_questions"], template["control_answers"]
            ):
                ctrl = _build_control_entry(
                    image_path=rel_path,
                    category=category,
                    illusion_type=illusion_type,
                    question=q,
                    answer=a,
                    stimulus_id=stim_id,
                )
                control_pool.append(ctrl)

    # ── Sample control entries ──────────────────────────────────────────
    n_control = max(1, int(len(all_entries) * control_ratio / (1 - control_ratio)))
    n_control = min(n_control, len(control_pool))
    control_entries = rng.sample(control_pool, n_control)
    all_entries.extend(control_entries)

    # Shuffle the final dataset
    rng.shuffle(all_entries)

    # ── Write JSONL ─────────────────────────────────────────────────────
    jsonl_path = output_dir / "dataset.jsonl"
    with open(jsonl_path, "w") as fh:
        for entry in all_entries:
            fh.write(json.dumps(entry) + "\n")

    n_dpo = sum(1 for e in all_entries if not e["is_control"])
    n_ctrl = sum(1 for e in all_entries if e["is_control"])
    logger.info(
        "Dataset written to %s  (%d DPO pairs + %d control = %d total)",
        jsonl_path,
        n_dpo,
        n_ctrl,
        len(all_entries),
    )
    return jsonl_path


# ────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Symmetric Polarity DPO dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/rl).",
    )
    parser.add_argument(
        "--max-per-illusion",
        type=int,
        default=60,
        help="Max stimulus pairs per illusion type.",
    )
    parser.add_argument(
        "--control-ratio",
        type=float,
        default=0.2,
        help="Fraction of dataset reserved for control VQA.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        max_per_illusion=args.max_per_illusion,
        control_ratio=args.control_ratio,
        seed=args.seed,
    )
