"""Loader for external illusion datasets (category 5: impossible objects / scene-level).

Supported sources
-----------------
- IllusionVQA   : Shahgir et al. (COLM 2024)
- HallusionBench: Guan et al. (CVPR 2024)

Both datasets are expected to be downloaded manually into
``data/external/<dataset_name>/`` and then indexed via this loader.
The loader normalises entries into ``StimulusPair`` objects so the rest of
the pipeline is dataset-agnostic.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from .base import StimulusGenerator, StimulusPair, ANSWER_CORRECT, ANSWER_ILLUSORY


_SUPPORTED_SOURCES = {"illusion_vqa", "hallusion_bench"}


class ExternalDatasetLoader(StimulusGenerator):
    """Load illusion pairs from IllusionVQA or HallusionBench.

    Unlike programmatic generators, this class reads pre-existing images and
    metadata from disk.  The ``param_grid`` method returns one entry per
    dataset item; ``generate`` reconstructs the ``StimulusPair`` from the
    manifest entry produced by ``build_manifest``.

    Parameters
    ----------
    source : ``"illusion_vqa"`` or ``"hallusion_bench"``
    data_root : Root directory for the dataset (must contain the manifest).
    """

    category = "impossible"

    def __init__(self, source: str, data_root: Path) -> None:
        if source not in _SUPPORTED_SOURCES:
            raise ValueError(f"source must be one of {_SUPPORTED_SOURCES}, got {source!r}")
        self.source = source
        self.illusion_type = source
        self.data_root = Path(data_root)
        self._manifest: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Manifest construction (run once after raw dataset download)
    # ------------------------------------------------------------------

    def build_manifest_illusion_vqa(self) -> list[dict[str, Any]]:
        """Parse IllusionVQA directory structure into a manifest list.

        Expects the following layout::

            data/external/illusion_vqa/
                images/
                    <id>_illusion.png
                    <id>_control.png   (or absent → use illusion as control)
                metadata.json          # list of {id, question, answer, human_error}
        """
        meta_path = self.data_root / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"IllusionVQA metadata not found at {meta_path}. "
                "Please download the dataset first."
            )
        with open(meta_path) as fh:
            raw = json.load(fh)

        manifest: list[dict[str, Any]] = []
        for item in raw:
            sid = str(item["id"])
            illusion_path = self.data_root / "images" / f"{sid}_illusion.png"
            control_path = self.data_root / "images" / f"{sid}_control.png"
            if not illusion_path.exists():
                continue
            if not control_path.exists():
                control_path = illusion_path  # fallback: use same image
            manifest.append(
                {
                    "stimulus_id": f"illusion_vqa_{sid}",
                    "category": self.category,
                    "illusion_type": self.illusion_type,
                    "params": {"source_id": sid},
                    "correct_answer": ANSWER_CORRECT,
                    "illusory_answer": ANSWER_ILLUSORY,
                    "illusion_path": str(illusion_path),
                    "control_path": str(control_path),
                    "question": item.get("question", ""),
                    "source_correct": item.get("answer", ""),
                    "human_error": item.get("human_error", ""),
                }
            )
        return manifest

    def build_manifest_hallusion_bench(self) -> list[dict[str, Any]]:
        """Parse HallusionBench into a manifest.

        Expects::

            data/external/hallusion_bench/
                HallusionBench.json   # official file with image/question pairs
                images/               # images referenced in the JSON
        """
        meta_path = self.data_root / "HallusionBench.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"HallusionBench JSON not found at {meta_path}. "
                "Please download the dataset first."
            )
        with open(meta_path) as fh:
            raw = json.load(fh)

        manifest: list[dict[str, Any]] = []
        for idx, item in enumerate(raw):
            # HallusionBench JSON stores paths as "./hallusion_bench/VD/illusion/0_0.png"
            fname = item.get("filename") or item.get("figure_id") or item.get("image", "")
            # Strip leading "./" so Path joining works correctly
            fname = fname.lstrip("./")
            illusion_path = self.data_root / fname
            if not illusion_path.exists():
                continue
            manifest.append(
                {
                    "stimulus_id": f"hallusion_{idx:05d}",
                    "category": self.category,
                    "illusion_type": self.illusion_type,
                    "params": {"source_idx": idx, "figure_id": fname},
                    "correct_answer": ANSWER_CORRECT,
                    "illusory_answer": ANSWER_ILLUSORY,
                    "illusion_path": str(illusion_path),
                    "control_path": str(illusion_path),
                    "question": item.get("question", ""),
                    "source_correct": str(item.get("gt_answer", "")),
                    "human_error": "",
                }
            )
        return manifest

    # ------------------------------------------------------------------
    # StimulusGenerator interface
    # ------------------------------------------------------------------

    def _load_manifest(self) -> list[dict[str, Any]]:
        if self._manifest is not None:
            return self._manifest
        manifest_path = self.data_root / "manifest.json"
        if not manifest_path.exists():
            if self.source == "illusion_vqa":
                self._manifest = self.build_manifest_illusion_vqa()
            else:
                self._manifest = self.build_manifest_hallusion_bench()
            with open(manifest_path, "w") as fh:
                json.dump(self._manifest, fh, indent=2)
        else:
            with open(manifest_path) as fh:
                self._manifest = json.load(fh)
        return self._manifest

    def param_grid(self) -> list[dict[str, Any]]:
        """Return all entries from the manifest as params dicts."""
        return [{"_entry": e} for e in self._load_manifest()]

    def generate(self, params: dict[str, Any]) -> StimulusPair:
        """Reconstruct a StimulusPair from a manifest entry stored in params."""
        entry = params["_entry"]
        illusion = Image.open(entry["illusion_path"]).convert("RGB")
        control = Image.open(entry["control_path"]).convert("RGB")
        return StimulusPair(
            illusion=illusion,
            control=control,
            category=entry["category"],
            illusion_type=entry["illusion_type"],
            params=entry["params"],
            correct_answer=entry["correct_answer"],
            illusory_answer=entry["illusory_answer"],
            stimulus_id=entry["stimulus_id"],
        )

    def __iter__(self) -> Iterator[StimulusPair]:
        for params in self.param_grid():
            yield self.generate(params)
