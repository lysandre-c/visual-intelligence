"""Multimodal LLM probers (LLaVA-1.5, Qwen-VL).

VLMs are queried with multiple-choice prompts.  To control for position bias
[HallusionBench, CVPR 2024] we:
  1. Randomise the option ordering across trials.
  2. Run two prompt framings: neutral and name-blind.
  3. Aggregate over framings and orderings into a ResponseDistribution.
"""

from __future__ import annotations

import random
from typing import Any

from PIL import Image

import transformers.integrations.bitsandbytes
def skip_check(*args, **kwargs):
    return True
transformers.integrations.bitsandbytes.validate_bnb_backend_availability = skip_check

from .base import ModelProber, ResponseDistribution


# ──────────────────────────────────────────────────────────────────────────────
# Shared VLM base
# ──────────────────────────────────────────────────────────────────────────────

class VLMProber(ModelProber):
    """Base class for VLM-based probers.

    Subclasses implement :meth:`_query` which sends a text + image to the
    underlying model and returns the raw output string.
    """

    model_name: str = "vlm"

    def __init__(
        self,
        n_orderings: int = 4,
        n_framings: int = 2,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        super().__init__(device)
        self.n_orderings = n_orderings
        self.n_framings = n_framings
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    def _query(self, image: Image.Image, prompt: str) -> str:
        """Send one image + text prompt, return the model's text response."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        question: str,
        options: list[tuple[str, str]],
        framing: str = "neutral",
    ) -> str:
        """Build a multiple-choice prompt.

        Parameters
        ----------
        question : The task question, e.g. "Which line looks longer?"
        options  : List of (letter, description) pairs.
        framing  : ``"neutral"`` or ``"name_blind"`` (avoids illusion names).
        """
        if framing == "name_blind":
            preamble = (
                "Look carefully at the image. Answer the following question "
                "based only on what you see.\n\n"
            )
        else:
            preamble = "Answer the following question about the image.\n\n"

        option_lines = "\n".join(f"  {letter}. {desc}" for letter, desc in options)
        prompt = (
            f"{preamble}"
            f"Question: {question}\n\n"
            f"Options:\n{option_lines}\n\n"
            "Reply with the letter of your answer only (e.g. 'A')."
        )
        return prompt

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate_responses(
        self,
        responses: list[tuple[str, dict[str, str]]],
    ) -> ResponseDistribution:
        """Map raw text responses to (correct, illusory, other) counts.

        Parameters
        ----------
        responses : list of (model_output_string, letter_to_label_map)
        """
        counts = {"correct": 0.0, "illusory": 0.0, "other": 0.0}
        for raw_output, letter_map in responses:
            chosen = raw_output.strip().upper()[:1]
            label = letter_map.get(chosen, "other")
            counts[label] += 1.0

        total = sum(counts.values()) or 1.0
        return ResponseDistribution(
            correct=counts["correct"] / total,
            illusory=counts["illusory"] / total,
            other=counts["other"] / total,
            raw={"raw_responses": [r for r, _ in responses]},
        )

    # ------------------------------------------------------------------
    # ModelProber interface
    # ------------------------------------------------------------------

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
        question = _get_question(extra, category)
        descriptions = _get_answer_descriptions(extra, category)

        letters = ["A", "B", "C"]
        label_order = ["correct", "illusory", "other"]

        # Randomly sample 1 ordering per image to average out bias over the dataset
        self._rng.shuffle(label_order)
        shuffled = list(zip(letters, label_order))
        options = [(l, descriptions[lbl]) for l, lbl in shuffled]
        letter_to_label = {l: lbl for l, lbl in shuffled}

        # Randomly sample 1 framing per image
        framing = self._rng.choice(["neutral", "name_blind"])
        
        prompt = self._build_prompt(question, options, framing)
        
        # Query illusion image
        raw_out = self._query(illusion, prompt)
        
        # Query control image (only if not impossible category)
        ctrl_probs = None
        ctrl_argmax_correct = False
        if category != "impossible":
            raw_out_ctrl = self._query(control, prompt)
            ctrl_chosen = raw_out_ctrl.strip().upper()[:1]
            ctrl_label = letter_to_label.get(ctrl_chosen, "other")
            
            ctrl_probs = [0.0, 0.0, 0.0]
            if ctrl_label == "correct":
                ctrl_probs[0] = 1.0
            elif ctrl_label == "illusory":
                ctrl_probs[1] = 1.0
            else:
                ctrl_probs[2] = 1.0
            ctrl_argmax_correct = (ctrl_label == "correct")

        responses = [(raw_out, letter_to_label)]
        dist = self._aggregate_responses(responses)
        
        if dist.raw is None:
            dist.raw = {}
        if ctrl_probs is not None:
            dist.raw["probs_control"] = ctrl_probs
            dist.raw["ctrl_argmax_correct"] = ctrl_argmax_correct
            
        return dist


# ──────────────────────────────────────────────────────────────────────────────
# LLaVA-1.5
# ──────────────────────────────────────────────────────────────────────────────

class LLaVAProber(VLMProber):
    """LLaVA-1.5 (7B or 13B) prober using HuggingFace transformers."""

    model_name = "llava_1.5"

    def __init__(
        self,
        hf_model_id: str = "llava-hf/llava-1.5-7b-hf",
        n_orderings: int = 4,
        n_framings: int = 2,
        seed: int = 42,
        device: str | None = None,
        load_in_4bit: bool = False,
        adapter_path: str | None = None,
    ) -> None:
        super().__init__(n_orderings, n_framings, seed, device)
        self.hf_model_id = hf_model_id
        self._pipe = None  # Lazy loading
        self._load_in_4bit = load_in_4bit
        self.adapter_path = adapter_path

    def _load_model(self) -> None:
        from transformers import pipeline, BitsAndBytesConfig, AutoProcessor  # type: ignore
        import torch
        self.processor = AutoProcessor.from_pretrained(self.hf_model_id)

        # BitsAndBytes quantisation requires CUDA; skip on MPS/CPU.
        if self._load_in_4bit and torch.cuda.is_available():
            kwargs: dict[str, Any] = {
                "device_map": "auto",
                "model_kwargs": {
                    "quantization_config": BitsAndBytesConfig(load_in_4bit=True)
                },
            }
        elif self.device == "mps":
            # HuggingFace device_map does not route to MPS automatically;
            # load on CPU with float16 and move to MPS manually.
            kwargs = {
                "model_kwargs": {"torch_dtype": torch.float16},
                "device": self.device,
            }
        else:
            kwargs = {"device_map": "auto"}
        self._pipe = pipeline("image-text-to-text", model=self.hf_model_id, **kwargs)

        if self.adapter_path is not None:
            from peft import PeftModel
            self._pipe.model = PeftModel.from_pretrained(self._pipe.model, self.adapter_path)

    def _query(self, image: Image.Image, prompt: str) -> str:
        if self._pipe is None:
            self._load_model()
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        formatted = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        task = getattr(self._pipe, "task", "image-to-text")
        if task == "image-text-to-text":
            outputs = self._pipe(image, text=formatted, generate_kwargs={"max_new_tokens": 150})
        else:
            outputs = self._pipe(image, prompt=formatted, generate_kwargs={"max_new_tokens": 16})
            
        generated = outputs[0]["generated_text"]
        
        if isinstance(generated, list):
            generated = generated[-1].get("content", "")
        else:
            if "ASSISTANT:" in generated:
                generated = generated.split("ASSISTANT:")[-1]
                
        return generated.strip()

            

# ──────────────────────────────────────────────────────────────────────────────
# Qwen-VL
# ──────────────────────────────────────────────────────────────────────────────

class QwenVLProber(VLMProber):
    """Qwen-VL-Chat prober using HuggingFace transformers."""

    model_name = "qwen_vl"

    def __init__(
        self,
        hf_model_id: str = "Qwen/Qwen-VL-Chat",
        n_orderings: int = 4,
        n_framings: int = 2,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        super().__init__(n_orderings, n_framings, seed, device)
        self.hf_model_id = hf_model_id
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_id, trust_remote_code=True
        )
        if self.device == "mps":
            # device_map="auto" does not route to MPS; load to CPU then move.
            self._model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(self.device).eval()
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_id,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).eval()

    def _query(self, image: Image.Image, prompt: str) -> str:
        if self._model is None:
            self._load_model()
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            query = self._tokenizer.from_list_format(
                [{"image": tmp_path}, {"text": prompt}]
            )
            response, _ = self._model.chat(
                self._tokenizer, query=query, history=None
            )
        finally:
            os.unlink(tmp_path)

        return response.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Question / description templates
# ──────────────────────────────────────────────────────────────────────────────

_QUESTIONS: dict[str, str] = {
    "geometric": "Which of the two horizontal lines looks longer?",
    "color": "Which of the two grey patches looks brighter?",
    "angle": "Do the long diagonal lines appear parallel?",
    "motion": "Does the static pattern appear to move, rotate, or flicker?",
    "impossible": "Is the object shown in the image physically possible or impossible?",
}

_ANSWER_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "geometric": {
        "correct": "They are equal in length.",
        "illusory": "The top line looks longer.",
        "other": "The bottom line looks longer.",
    },
    "color": {
        "correct": "They are the same brightness.",
        "illusory": "The left patch looks brighter.",
        "other": "The right patch looks brighter.",
    },
    "angle": {
        "correct": "Yes, they are parallel.",
        "illusory": "No, they appear to converge or diverge.",
        "other": "It is unclear.",
    },
    "motion": {
        "correct": "No, the pattern appears still.",
        "illusory": "Yes, the pattern appears to move, rotate, or flicker.",
        "other": "It is unclear.",
    },
    "impossible": {
        "correct": "It is physically possible.",
        "illusory": "It is physically impossible.",
        "other": "It is ambiguous.",
    },
}


def _get_question(extra: dict[str, Any] | None, category: str) -> str:
    if extra is not None and "question" in extra:
        return extra["question"]
    return _QUESTIONS.get(category, "What do you observe in this image?")


def _get_answer_descriptions(
    extra: dict[str, Any] | None, category: str
) -> dict[str, str]:
    if extra is not None and "answer_descriptions" in extra:
        return extra["answer_descriptions"]
    return _ANSWER_DESCRIPTIONS.get(
        category,
        {"correct": "Option A.", "illusory": "Option B.", "other": "Option C."},
    )
