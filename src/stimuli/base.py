"""Base classes for parametric illusion stimulus generation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from PIL import Image


# Three mutually-exclusive answer labels used throughout the project.
ANSWER_CORRECT = "correct"       # Physically / geometrically correct answer
ANSWER_ILLUSORY = "illusory"     # The direction humans typically err
ANSWER_OTHER = "other"           # Any remaining option


@dataclass
class StimulusPair:
    """A single illusion / control pair with its ground-truth labels.

    Attributes
    ----------
    illusion:
        The image that contains the illusory cue.
    control:
        A matched image with the illusory cue stripped; the correct answer
        should be trivially visible here.
    category:
        One of ``geometric``, ``color``, ``angle``, ``motion``, ``impossible``.
    illusion_type:
        Specific illusion name, e.g. ``muller_lyer``.
    params:
        The parameter dict used to generate this stimulus (for psychometric
        sweep bookkeeping).
    correct_answer:
        The physically correct answer label.
    illusory_answer:
        The answer a typical human would give under the illusion.
    stimulus_id:
        Unique identifier for this pair within the dataset.
    """

    illusion: Image.Image
    control: Image.Image
    category: str
    illusion_type: str
    params: dict[str, Any]
    correct_answer: str
    illusory_answer: str
    stimulus_id: str = ""

    def save(self, directory: Path) -> dict[str, Any]:
        """Persist both images and return a metadata dict for the manifest."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        illusion_path = directory / f"{self.stimulus_id}_illusion.png"
        control_path = directory / f"{self.stimulus_id}_control.png"
        self.illusion.save(illusion_path)
        self.control.save(control_path)
        meta = {
            "stimulus_id": self.stimulus_id,
            "category": self.category,
            "illusion_type": self.illusion_type,
            "params": self.params,
            "correct_answer": self.correct_answer,
            "illusory_answer": self.illusory_answer,
            "illusion_path": str(illusion_path),
            "control_path": str(control_path),
        }
        return meta

    @classmethod
    def load(cls, meta: dict[str, Any]) -> "StimulusPair":
        """Reconstruct a StimulusPair from a manifest entry."""
        return cls(
            illusion=Image.open(meta["illusion_path"]).convert("RGB"),
            control=Image.open(meta["control_path"]).convert("RGB"),
            category=meta["category"],
            illusion_type=meta["illusion_type"],
            params=meta["params"],
            correct_answer=meta["correct_answer"],
            illusory_answer=meta["illusory_answer"],
            stimulus_id=meta["stimulus_id"],
        )


class StimulusGenerator(ABC):
    """Abstract base class for parametric illusion generators.

    Subclasses implement :meth:`generate` and :meth:`param_grid`.
    The base class handles stimulus ID assignment and manifest I/O.
    """

    category: str = ""       # Set by subclasses, e.g. "geometric"
    illusion_type: str = ""  # Set by subclasses, e.g. "muller_lyer"

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def generate(self, params: dict[str, Any]) -> StimulusPair:
        """Generate an illusion / control pair for the given ``params``.

        Parameters
        ----------
        params:
            Keyword arguments specific to the illusion (e.g. flanker length).

        Returns
        -------
        StimulusPair
            The illusion and its matched control, with answer labels set.
        """

    @abstractmethod
    def param_grid(self) -> list[dict[str, Any]]:
        """Return the full parameter sweep for psychometric analysis.

        Each dict in the list is passed individually to :meth:`generate`.
        The ordering should correspond to increasing illusion strength so
        that psychometric curves are monotone by construction.
        """

    # ------------------------------------------------------------------ #
    # Concrete helpers                                                     #
    # ------------------------------------------------------------------ #

    def generate_dataset(
        self,
        output_dir: Path,
        manifest_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Generate all stimuli defined by :meth:`param_grid` and save them.

        Parameters
        ----------
        output_dir:
            Root directory under which images are written.
        manifest_path:
            If provided, the manifest JSON is written here.  Defaults to
            ``output_dir / "manifest.json"``.

        Returns
        -------
        list[dict]
            The manifest entries (one per StimulusPair).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_path or output_dir / "manifest.json"

        manifest: list[dict[str, Any]] = []
        for idx, params in enumerate(self.param_grid()):
            pair = self.generate(params)
            pair.stimulus_id = f"{self.illusion_type}_{idx:04d}"
            subdir = output_dir / self.category / self.illusion_type
            meta = pair.save(subdir)
            manifest.append(meta)

        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

        return manifest

    @staticmethod
    def load_manifest(manifest_path: Path) -> list[StimulusPair]:
        """Load all stimulus pairs recorded in a manifest file."""
        with open(manifest_path) as fh:
            entries = json.load(fh)
        return [StimulusPair.load(e) for e in entries]
