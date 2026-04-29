"""Color / brightness illusion generators.

Illusions implemented
---------------------
- Simultaneous contrast : a grey patch on a dark background appears lighter
  than the same grey patch on a bright background.
- White's illusion       : grey bars embedded in black vs. white stripes
  appear to differ in brightness despite being identical.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .base import StimulusGenerator, StimulusPair, ANSWER_CORRECT, ANSWER_ILLUSORY


# ──────────────────────────────────────────────────────────────────────────────
# Simultaneous Contrast
# ──────────────────────────────────────────────────────────────────────────────

class SimultaneousContrastGenerator(StimulusGenerator):
    """Simultaneous brightness contrast illusion.

    Left half: target grey patch on a dark surround.
    Right half: identical grey patch on a bright surround.
    The left patch appears lighter; humans say the patches differ (illusory),
    whereas they are physically the same.
    Sweep variable: surround luminance difference (dark_lum vs bright_lum).
    """

    category = "color"
    illusion_type = "simultaneous_contrast"

    def __init__(
        self,
        image_size: tuple[int, int] = (512, 256),
        patch_size: int = 80,
        target_luminance: int = 128,
    ) -> None:
        self.image_size = image_size
        self.patch_size = patch_size
        self.target_luminance = target_luminance

    def _make_pair(
        self,
        dark_lum: int,
        bright_lum: int,
        patch_size: int | None = None,
        target_luminance: int | None = None,
        x_jitter: int = 0,
        y_jitter: int = 0,
    ) -> tuple[Image.Image, Image.Image]:
        w, h = self.image_size
        half_w = w // 2
        ps = self.patch_size if patch_size is None else patch_size
        tl = self.target_luminance if target_luminance is None else target_luminance

        illusion_arr = np.ones((h, w, 3), dtype=np.uint8)
        illusion_arr[:, :half_w] = dark_lum
        illusion_arr[:, half_w:] = bright_lum

        # Centre the target patches
        cy = h // 2 + y_jitter
        cx_left = half_w // 2 + x_jitter
        cx_right = half_w + half_w // 2 + x_jitter
        for cx in [cx_left, cx_right]:
            r0, r1 = cy - ps // 2, cy + ps // 2
            c0, c1 = cx - ps // 2, cx + ps // 2
            illusion_arr[r0:r1, c0:c1] = tl

        illusion = Image.fromarray(illusion_arr, mode="RGB")

        # Control: mid-grey surround on both sides → no perceived difference
        mid_lum = (dark_lum + bright_lum) // 2
        control_arr = np.full((h, w, 3), mid_lum, dtype=np.uint8)
        for cx in [cx_left, cx_right]:
            r0, r1 = cy - ps // 2, cy + ps // 2
            c0, c1 = cx - ps // 2, cx + ps // 2
            control_arr[r0:r1, c0:c1] = tl
        control = Image.fromarray(control_arr, mode="RGB")

        return illusion, control

    def generate(self, params: dict[str, Any]) -> StimulusPair:
        illusion, control = self._make_pair(
            int(params["dark_lum"]), int(params["bright_lum"]),
            patch_size=int(params.get("patch_size", self.patch_size)),
            target_luminance=int(params.get("target_luminance", self.target_luminance)),
            x_jitter=int(params.get("x_jitter", 0)),
            y_jitter=int(params.get("y_jitter", 0)),
        )
        return StimulusPair(
            illusion=illusion,
            control=control,
            category=self.category,
            illusion_type=self.illusion_type,
            params=params,
            correct_answer=ANSWER_CORRECT,   # patches are equal
            illusory_answer=ANSWER_ILLUSORY,  # humans say left is lighter
        )

    def param_grid(
        self,
        n_levels: int = 16,
        n_repeats: int = 30,
        jitter_seed: int = 45,
    ) -> list[dict[str, Any]]:
        """Sweep clearly visible surround contrast with repeated trials."""
        rng = np.random.default_rng(jitter_seed)
        # delta=50 already produces a visible brightness-contrast setup;
        # smaller values are too close to neutral for robust human reports.
        deltas = np.linspace(50, 115, n_levels).astype(int).tolist()
        grid: list[dict[str, Any]] = []
        for delta in deltas:
            dark = max(0, self.target_luminance - delta)
            bright = min(255, self.target_luminance + delta)
            for rep in range(n_repeats):
                grid.append({
                    "dark_lum": int(dark),
                    "bright_lum": int(bright),
                    "contrast_delta": int(delta),
                    "repeat": rep,
                    "patch_size": int(rng.integers(68, 93)),
                    "target_luminance": int(rng.integers(120, 137)),
                    "x_jitter": int(rng.integers(-12, 13)),
                    "y_jitter": int(rng.integers(-10, 11)),
                })
        return grid


# ──────────────────────────────────────────────────────────────────────────────
# White's Illusion
# ──────────────────────────────────────────────────────────────────────────────

class WhiteIllusionGenerator(StimulusGenerator):
    """White's illusion.

    Horizontal black-and-white stripes; grey patches embedded in the black
    stripes (left group) vs. in the white stripes (right group) appear to
    differ in brightness even though they are the same grey.

    Sweep variable: stripe width (affects illusion strength).
    """

    category = "color"
    illusion_type = "whites_illusion"

    def __init__(
        self,
        image_size: tuple[int, int] = (512, 512),
        target_luminance: int = 128,
        n_stripes: int = 8,
    ) -> None:
        self.image_size = image_size
        self.target_luminance = target_luminance
        self.n_stripes = n_stripes

    def _make_pair(
        self,
        stripe_height: int,
        phase_offset: int = 0,
        patch_width: int = 40,
    ) -> tuple[Image.Image, Image.Image]:
        w, h = self.image_size
        tl = self.target_luminance

        def stripe_spans() -> list[tuple[int, int, int]]:
            spans = []
            for i in range(h // stripe_height + 2):
                raw_y0 = i * stripe_height - phase_offset
                raw_y1 = raw_y0 + stripe_height
                if raw_y1 <= 0 or raw_y0 >= h:
                    continue
                y0 = max(0, raw_y0)
                y1 = min(raw_y1, h)
                spans.append((i, y0, y1))
            return spans

        def make_striped() -> np.ndarray:
            arr = np.zeros((h, w, 3), dtype=np.uint8)
            for i, y0, y1 in stripe_spans():
                left_lum = 0 if i % 2 == 0 else 255
                right_lum = 255 if i % 2 == 0 else 0
                arr[y0:y1, :half_w] = left_lum
                arr[y0:y1, half_w:] = right_lum
            return arr

        half_w = w // 2
        patch_w = patch_width
        illusion_arr = make_striped()

        def place_patches(arr: np.ndarray) -> None:
            placed = 0
            for i, y0, y1 in stripe_spans():
                if i % 2 != 0:
                    continue
                left_cx = half_w // 2
                right_cx = half_w + half_w // 2
                arr[y0:y1, left_cx - patch_w // 2 : left_cx + patch_w // 2] = tl
                arr[y0:y1, right_cx - patch_w // 2 : right_cx + patch_w // 2] = tl
                placed += 1
                if placed == 3:
                    break

        place_patches(illusion_arr)
        illusion = Image.fromarray(illusion_arr, mode="RGB")

        # Control: uniform grey surround on both sides
        control_arr = np.full((h, w, 3), 128, dtype=np.uint8)
        place_patches(control_arr)
        control = Image.fromarray(control_arr, mode="RGB")

        return illusion, control

    def generate(self, params: dict[str, Any]) -> StimulusPair:
        illusion, control = self._make_pair(
            int(params["stripe_height"]),
            phase_offset=int(params.get("phase_offset", 0)),
            patch_width=int(params.get("patch_width", 40)),
        )
        return StimulusPair(
            illusion=illusion,
            control=control,
            category=self.category,
            illusion_type=self.illusion_type,
            params=params,
            correct_answer=ANSWER_CORRECT,
            illusory_answer=ANSWER_ILLUSORY,
        )

    def param_grid(
        self,
        n_levels: int = 16,
        n_repeats: int = 30,
        jitter_seed: int = 46,
    ) -> list[dict[str, Any]]:
        """Sweep stripe height in a range where White's illusion remains visible."""
        rng = np.random.default_rng(jitter_seed)
        stripe_heights = np.linspace(20, 76, n_levels).astype(int).tolist()
        grid: list[dict[str, Any]] = []
        for stripe_height in stripe_heights:
            for rep in range(n_repeats):
                grid.append({
                    "stripe_height": int(stripe_height),
                    "repeat": rep,
                    "phase_offset": int(rng.integers(0, max(1, stripe_height // 2))),
                    "patch_width": int(rng.integers(34, 49)),
                })
        return grid
