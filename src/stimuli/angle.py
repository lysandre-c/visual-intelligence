"""Angle / orientation illusion generators.

Illusions implemented
---------------------
- Zöllner    : parallel lines appear non-parallel due to crossing hatches.
- Poggendorff: a straight line behind an occluder appears misaligned.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .base import StimulusGenerator, StimulusPair, ANSWER_CORRECT, ANSWER_ILLUSORY


# ──────────────────────────────────────────────────────────────────────────────
# Zöllner
# ──────────────────────────────────────────────────────────────────────────────

class ZollnerGenerator(StimulusGenerator):
    """Parametric Zöllner illusion.

    Parallel diagonal long-lines are crossed by short hatches at an angle,
    making them appear non-parallel.  Sweep: hatch angle relative to the
    long-line direction.
    """

    category = "angle"
    illusion_type = "zollner"

    def __init__(
        self,
        image_size: tuple[int, int] = (512, 512),
        n_main_lines: int = 5,
        hatch_spacing: int = 20,
        hatch_length: int = 15,
        line_width: int = 3,
        background: tuple[int, int, int] = (255, 255, 255),
        foreground: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.image_size = image_size
        self.n_main_lines = n_main_lines
        self.hatch_spacing = hatch_spacing
        self.hatch_length = hatch_length
        self.line_width = line_width
        self.background = background
        self.foreground = foreground

    def _draw_zollner(
        self,
        draw: ImageDraw.ImageDraw,
        w: int,
        h: int,
        main_angle_deg: float,
        hatch_angle_deg: float,
        alternate: bool,
    ) -> None:
        main_rad = math.radians(main_angle_deg)
        spacing_y = h // (self.n_main_lines + 1)
        for i in range(1, self.n_main_lines + 1):
            cy = i * spacing_y
            # Project line across the image
            half_len = w * 0.7
            x0 = int(w / 2 - half_len * math.cos(main_rad))
            y0 = int(cy - half_len * math.sin(main_rad))
            x1 = int(w / 2 + half_len * math.cos(main_rad))
            y1 = int(cy + half_len * math.sin(main_rad))
            draw.line([(x0, y0), (x1, y1)], fill=self.foreground, width=self.line_width)

            if hatch_angle_deg == 0.0:
                continue

            # Hatches along the line
            n_hatches = int(2 * half_len / self.hatch_spacing)
            hatch_rad = math.radians(
                main_angle_deg + hatch_angle_deg * (1 if (i % 2 == 0 and alternate) else -1)
            )
            for j in range(n_hatches):
                t = -half_len + j * self.hatch_spacing
                hx = int(w / 2 + t * math.cos(main_rad))
                hy = int(cy + t * math.sin(main_rad))
                hl = self.hatch_length / 2
                hx0 = int(hx - hl * math.cos(hatch_rad))
                hy0 = int(hy - hl * math.sin(hatch_rad))
                hx1 = int(hx + hl * math.cos(hatch_rad))
                hy1 = int(hy + hl * math.sin(hatch_rad))
                draw.line([(hx0, hy0), (hx1, hy1)], fill=self.foreground, width=self.line_width)

    def _make_pair(
        self,
        main_angle_deg: float,
        hatch_angle_deg: float,
        hatch_spacing: int | None = None,
        hatch_length: int | None = None,
    ) -> tuple[Image.Image, Image.Image]:
        w, h = self.image_size
        old_spacing, old_length = self.hatch_spacing, self.hatch_length
        if hatch_spacing is not None:
            self.hatch_spacing = hatch_spacing
        if hatch_length is not None:
            self.hatch_length = hatch_length
        illusion = Image.new("RGB", self.image_size, self.background)
        d = ImageDraw.Draw(illusion)
        self._draw_zollner(d, w, h, main_angle_deg, hatch_angle_deg, alternate=True)

        control = Image.new("RGB", self.image_size, self.background)
        dc = ImageDraw.Draw(control)
        self._draw_zollner(dc, w, h, main_angle_deg, hatch_angle_deg=0.0, alternate=False)
        self.hatch_spacing, self.hatch_length = old_spacing, old_length

        return illusion, control

    def generate(self, params: dict[str, Any]) -> StimulusPair:
        illusion, control = self._make_pair(
            params.get("main_angle_deg", 30.0), params["hatch_angle_deg"],
            hatch_spacing=int(params.get("hatch_spacing", self.hatch_spacing)),
            hatch_length=int(params.get("hatch_length", self.hatch_length)),
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
        jitter_seed: int = 47,
    ) -> list[dict[str, Any]]:
        rng = np.random.default_rng(jitter_seed)
        hatch_angles = np.linspace(20, 75, n_levels).tolist()
        grid: list[dict[str, Any]] = []
        for hatch_angle in hatch_angles:
            for rep in range(n_repeats):
                grid.append({
                    "main_angle_deg": float(round(rng.choice([25.0, 30.0, 35.0]), 1)),
                    "hatch_angle_deg": float(round(hatch_angle, 2)),
                    "repeat": rep,
                    "hatch_spacing": int(rng.integers(17, 24)),
                    "hatch_length": int(rng.integers(13, 21)),
                })
        return grid


# ──────────────────────────────────────────────────────────────────────────────
# Poggendorff
# ──────────────────────────────────────────────────────────────────────────────

class PoggendorffGenerator(StimulusGenerator):
    """Parametric Poggendorff illusion.

    A diagonal line is interrupted by a vertical occluder band.  The two
    visible segments appear misaligned, even though they are collinear.
    Sweep: occluder width.
    """

    category = "angle"
    illusion_type = "poggendorff"

    def __init__(
        self,
        image_size: tuple[int, int] = (512, 256),
        line_angle_deg: float = 30.0,
        line_width: int = 3,
        background: tuple[int, int, int] = (255, 255, 255),
        foreground: tuple[int, int, int] = (0, 0, 0),
        occluder_color: tuple[int, int, int] = (200, 200, 200),
    ) -> None:
        self.image_size = image_size
        self.line_angle_deg = line_angle_deg
        self.line_width = line_width
        self.background = background
        self.foreground = foreground
        self.occluder_color = occluder_color

    def _make_pair(
        self,
        occluder_width: int,
        line_angle_deg: float | None = None,
        y_shift: int = 0,
    ) -> tuple[Image.Image, Image.Image]:
        w, h = self.image_size
        cx = w // 2
        angle_deg = self.line_angle_deg if line_angle_deg is None else line_angle_deg
        angle_rad = math.radians(angle_deg)

        def y_at_x(x: int) -> int:
            return int(h // 2 + y_shift + (x - cx) * math.tan(angle_rad))

        x_left_end = cx - occluder_width // 2
        x_right_start = cx + occluder_width // 2

        illusion = Image.new("RGB", self.image_size, self.background)
        d = ImageDraw.Draw(illusion)
        d.line([(0, y_at_x(0)), (w, y_at_x(w))], fill=self.foreground, width=self.line_width)
        d.rectangle(
            [(x_left_end, 0), (x_right_start, h)],
            fill=self.occluder_color,
        )

        # Control: no occluder → full uninterrupted line
        control = Image.new("RGB", self.image_size, self.background)
        dc = ImageDraw.Draw(control)
        dc.line([(0, y_at_x(0)), (w, y_at_x(w))], fill=self.foreground, width=self.line_width)

        return illusion, control

    def generate(self, params: dict[str, Any]) -> StimulusPair:
        illusion, control = self._make_pair(
            int(params["occluder_width"]),
            line_angle_deg=float(params.get("line_angle_deg", self.line_angle_deg)),
            y_shift=int(params.get("y_shift", 0)),
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
        jitter_seed: int = 48,
    ) -> list[dict[str, Any]]:
        rng = np.random.default_rng(jitter_seed)
        occluder_widths = np.linspace(45, 165, n_levels).astype(int).tolist()
        grid: list[dict[str, Any]] = []
        for width in occluder_widths:
            for rep in range(n_repeats):
                grid.append({
                    "occluder_width": int(width),
                    "repeat": rep,
                    "line_angle_deg": float(round(rng.choice([25.0, 30.0, 35.0]), 1)),
                    "y_shift": int(rng.integers(-12, 13)),
                })
        return grid
