"""VLM probing protocol: multiple-choice with anti-bias controls.

This module provides ``VLMProtocol``, a higher-level wrapper around
``VLMProber`` that applies the full experimental protocol from the paper:
  1. Multiple prompt orderings to cancel position bias.
  2. Two prompt framings: neutral and name-blind.
  3. Structured logging of per-response data for audit.

It also provides ``build_probe_dataset`` which iterates a stimulus list and
writes per-stimulus results to a JSONL file for reproducibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


class VLMProtocol:
    """Full VLM probing protocol with position-bias controls.

    Parameters
    ----------
    prober :
        A ``VLMProber`` subclass instance (``LLaVAProber`` or ``QwenVLProber``).
    output_dir :
        Directory for per-stimulus JSONL audit logs.
    """

    def __init__(self, prober: Any, output_dir: Path | None = None) -> None:
        self.prober = prober
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def probe_stimulus(
        self,
        illusion: Image.Image,
        control: Image.Image,
        category: str,
        illusion_type: str,
        correct_answer: str,
        illusory_answer: str,
        stimulus_id: str = "",
        extra: dict[str, Any] | None = None,
    ):
        """Probe one stimulus pair and optionally write an audit record."""
        dist = self.prober.probe_pair(
            illusion=illusion,
            control=control,
            correct_answer=correct_answer,
            illusory_answer=illusory_answer,
            category=category,
            illusion_type=illusion_type,
            extra=extra,
        )

        if self.output_dir and stimulus_id:
            record = {
                "stimulus_id": stimulus_id,
                "model": self.prober.model_name,
                "category": category,
                "illusion_type": illusion_type,
                **dist.normalise().as_dict(),
                "raw_responses": dist.raw.get("raw_responses", []) if dist.raw else [],
            }
            log_path = self.output_dir / f"{stimulus_id}.json"
            with open(log_path, "w") as fh:
                json.dump(record, fh, indent=2)

        return dist

    def probe_dataset(
        self,
        pairs: list,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Run the protocol over a list of ``StimulusPair`` objects."""
        from tqdm import tqdm

        results = []
        iterator = tqdm(pairs, desc=f"VLM[{self.prober.model_name}]", disable=not verbose)
        for pair in iterator:
            dist = self.probe_stimulus(
                illusion=pair.illusion,
                control=pair.control,
                category=pair.category,
                illusion_type=pair.illusion_type,
                correct_answer=pair.correct_answer,
                illusory_answer=pair.illusory_answer,
                stimulus_id=pair.stimulus_id,
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
