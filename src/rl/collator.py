"""Custom data collator with Symbol Demonstration and Option Shuffling.

This collator implements two key anti-shortcut mechanisms:

1. **Symbol Demonstration (SymDPO)**: Replaces standard option letters
   (A, B, C) with semantically neutral Unicode symbols (♣, ♠, ♦) in
   both prompts and completions.  This prevents the model from learning
   positional biases tied to the Latin alphabet.

2. **Dynamic Option Shuffling**: Randomly permutes the order of
   multiple-choice options within each batch item, erasing any
   positional shortcuts (e.g. "the correct answer is always option A").

Each batch item is expanded into eight tokenised sequences:
    - original   chosen   (prompt + completion)
    - original   rejected (prompt + completion)
    - inverted   chosen   (prompt + completion)
    - inverted   rejected (prompt + completion)
    × {policy, reference}  ← handled at training time, not here
"""

from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image


# ── Symbol sets ─────────────────────────────────────────────────────────
# Standard letters → neutral symbols mapping.
SYMBOL_MAP: dict[str, str] = {"A": "♣", "B": "♠", "C": "♦"}
SYMBOLS: list[str] = list(SYMBOL_MAP.values())


def _apply_symbol_demo(text: str, letter_to_symbol: dict[str, str]) -> str:
    """Replace option letters with assigned symbols throughout *text*.

    Handles patterns like ``{A}.``, ``Option A``, bare ``A`` at option
    boundaries, etc.
    """
    result = text
    for letter, symbol in letter_to_symbol.items():
        # Replace "{A}" placeholder form
        result = result.replace(f"{{{letter}}}", symbol)
        # Replace "A." at the start of an option line
        result = re.sub(
            rf"(?m)^(\s*){letter}\.",
            rf"\1{symbol}.",
            result,
        )
        # Replace "Option A" references
        result = re.sub(
            rf"\bOption {letter}\b",
            f"Option {symbol}",
            result,
        )
    return result


def _shuffle_options_in_prompt(
    prompt: str,
    chosen: str,
    rejected: str,
    rng: random.Random,
) -> tuple[str, str, str, dict[str, str]]:
    """Shuffle the multiple-choice options and update chosen/rejected.

    This function:
    1. Extracts option lines from the prompt.
    2. Creates a random permutation of the *symbols* (not the text).
    3. Reassigns symbols to shuffled option texts.
    4. Remaps chosen/rejected answer strings accordingly.

    Returns (new_prompt, new_chosen, new_rejected, letter_to_symbol).
    """
    # Parse option lines:  "  ♣. The top line looks longer."
    option_pattern = re.compile(r"^\s*([♣♠♦A-C])\.\s*(.+)$", re.MULTILINE)
    matches = list(option_pattern.finditer(prompt))

    if len(matches) < 2:
        # Not enough options to shuffle — return as-is with identity map
        return prompt, chosen, rejected, {k: k for k in SYMBOL_MAP}

    # Extract (symbol, text) pairs
    option_texts = [(m.group(1), m.group(2).strip()) for m in matches]

    # Shuffle the *texts*, reassign symbols in order
    texts_only = [t for _, t in option_texts]
    symbols_used = [s for s, _ in option_texts]
    rng.shuffle(texts_only)

    # Build new mapping: new_symbol → original_text
    new_prompt = prompt
    old_to_new_text: dict[str, str] = {}
    for i, (sym, _old_text) in enumerate(option_texts):
        new_text = texts_only[i]
        # Replace the old option line with the new text
        old_line_pattern = re.compile(
            rf"(\s*){re.escape(sym)}\.\s*{re.escape(_old_text)}"
        )
        new_prompt = old_line_pattern.sub(
            rf"\g<1>{sym}. {new_text}", new_prompt, count=1
        )

    # Remap chosen/rejected: find which symbol now has the chosen text
    new_chosen = chosen
    new_rejected = rejected
    for i, sym in enumerate(symbols_used):
        if texts_only[i] == chosen.strip() or chosen.strip() in texts_only[i]:
            new_chosen = texts_only[i]
        if texts_only[i] == rejected.strip() or rejected.strip() in texts_only[i]:
            new_rejected = texts_only[i]

    letter_to_symbol = dict(zip(["A", "B", "C"][: len(symbols_used)], symbols_used))
    return new_prompt, new_chosen, new_rejected, letter_to_symbol


class SymmetricPolarityCollator:
    """Collator for Symmetric Polarity DPO training.

    Handles image loading, symbol demonstration, option shuffling, and
    tokenisation of the four prompt-completion pairs per sample.

    Parameters
    ----------
    processor : The HuggingFace ``AutoProcessor`` for LLaVA.
    project_root : Absolute path to the project root (image paths in the
        JSONL are relative to this).
    max_length : Maximum total sequence length (prompt + completion).
    symbol_demo : If True, apply symbol demonstration.
    option_shuffle : If True, shuffle option order per sample.
    seed : Random seed for shuffling.
    """

    def __init__(
        self,
        processor: Any,
        project_root: Path,
        max_length: int = 1024,
        symbol_demo: bool = True,
        option_shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.processor = processor
        self.project_root = Path(project_root)
        self.max_length = max_length
        self.symbol_demo = symbol_demo
        self.option_shuffle = option_shuffle
        self._rng = random.Random(seed)

    def _load_image(self, rel_path: str) -> Image.Image:
        """Load an image from a path relative to project root."""
        abs_path = self.project_root / rel_path
        return Image.open(str(abs_path)).convert("RGB")

    def _prepare_text(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> tuple[str, str, str]:
        """Apply symbol demo and option shuffling to one prompt pair."""
        # Step 1: Symbol Demonstration
        if self.symbol_demo:
            # Build a consistent letter→symbol mapping for this call
            letters = list(SYMBOL_MAP.keys())
            symbols = list(SYMBOL_MAP.values())
            self._rng.shuffle(symbols)
            l2s = dict(zip(letters, symbols))
            prompt = _apply_symbol_demo(prompt, l2s)
            chosen = _apply_symbol_demo(chosen, l2s)
            rejected = _apply_symbol_demo(rejected, l2s)

        # Step 2: Option Shuffling
        if self.option_shuffle:
            prompt, chosen, rejected, _ = _shuffle_options_in_prompt(
                prompt, chosen, rejected, self._rng
            )

        return prompt, chosen, rejected

    def _tokenize_pair(
        self,
        image: Image.Image,
        prompt: str,
        completion: str,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a (prompt, completion) pair with image.

        Returns dict with input_ids, attention_mask, labels, and
        pixel_values.  Labels are set to -100 for prompt tokens so
        that only completion tokens contribute to the loss.
        """
        # Build the conversation in LLaVA chat format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": completion},
                ],
            },
        ]

        # Format using the chat template
        full_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False
        )

        # Tokenize the full sequence with image
        encoding = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Build labels: mask the prompt portion with -100
        # To find the prompt/completion boundary, tokenize prompt-only
        prompt_conv = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        prompt_text = self.processor.apply_chat_template(
            prompt_conv, add_generation_prompt=True
        )
        prompt_encoding = self.processor.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Mask prompt tokens
        # Also mask padding
        labels[attention_mask == 0] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if "pixel_values" in encoding:
            result["pixel_values"] = encoding["pixel_values"].squeeze(0)

        return result

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        """Collate a batch of JSONL entries into padded tensors.

        Each entry produces four sequences:
        - orig_chosen, orig_rejected, inv_chosen, inv_rejected

        Returns a dict with keys prefixed by ``orig_chosen_``,
        ``orig_rejected_``, ``inv_chosen_``, ``inv_rejected_``.
        """
        prefixes = [
            ("orig_chosen", "original_prompt", "original_chosen"),
            ("orig_rejected", "original_prompt", "original_rejected"),
            ("inv_chosen", "inverted_prompt", "inverted_chosen"),
            ("inv_rejected", "inverted_prompt", "inverted_rejected"),
        ]

        collated: dict[str, list[torch.Tensor]] = {
            f"{pfx}_{key}": []
            for pfx, _, _ in prefixes
            for key in ["input_ids", "attention_mask", "labels", "pixel_values"]
        }

        for sample in batch:
            image = self._load_image(sample["image_path"])

            for prefix, prompt_key, completion_key in prefixes:
                prompt, chosen_or_rej, _unused = self._prepare_text(
                    sample[prompt_key],
                    sample[completion_key],
                    "",  # We don't need the other completion here
                )
                tok = self._tokenize_pair(image, prompt, chosen_or_rej)

                for key in ["input_ids", "attention_mask", "labels"]:
                    collated[f"{prefix}_{key}"].append(tok[key])
                if "pixel_values" in tok:
                    collated[f"{prefix}_pixel_values"].append(tok["pixel_values"])

        # Stack into batch tensors
        result: dict[str, torch.Tensor] = {}
        for key, tensors in collated.items():
            if tensors:
                result[key] = torch.stack(tensors, dim=0)

        return result
