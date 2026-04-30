"""Motion-from-static illusion generators.

Illusions implemented
---------------------
- Scintillating Grid : bright spots at grid intersections appear to scintillate
  (flicker dark) when the image is viewed statically.
  Follows Sun & Dekel (2021) [doi:10.1167/jov.21.11.15].
- Rotating Snakes    : asymmetric luminance/color cycles appear to rotate in
  peripheral vision, despite being static.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .base import StimulusGenerator, StimulusPair, ANSWER_CORRECT, ANSWER_ILLUSORY


# ──────────────────────────────────────────────────────────────────────────────
# Scintillating Grid
# ──────────────────────────────────────────────────────────────────────────────

class ScintillatingGridGenerator(StimulusGenerator):
    """Parametric Scintillating Grid.

    A dark grey background with bright grid lines and white discs at
    intersections.  When fixating off-centre the discs appear to turn dark.
    The control has the discs removed (plain grid, no illusion).
    Sweep variable: disc radius (affects perceived scintillation strength).
    """

    category = "motion"
    illusion_type = "scintillating_grid"

    def __init__(
        self,
        image_size: tuple[int, int] = (512, 512),
        grid_spacing: int = 48,
        grid_line_width: int = 8,
        background_lum: int = 64,
        grid_lum: int = 200,
        disc_lum: int = 255,
    ) -> None:
        self.image_size = image_size
        self.grid_spacing = grid_spacing
        self.grid_line_width = grid_line_width
        self.background_lum = background_lum
        self.grid_lum = grid_lum
        self.disc_lum = disc_lum

    def _make_pair(
        self,
        disc_radius: int,
        grid_spacing: int | None = None,
        grid_line_width: int | None = None,
    ) -> tuple[Image.Image, Image.Image]:
        w, h = self.image_size
        bg = self.background_lum
        gl = self.grid_lum
        dl = self.disc_lum
        gs = self.grid_spacing if grid_spacing is None else grid_spacing
        glw = self.grid_line_width if grid_line_width is None else grid_line_width

        grid_arr = np.full((h, w, 3), bg, dtype=np.uint8)

        # Draw grid lines
        xs = list(range(gs, w, gs))
        ys = list(range(gs, h, gs))
        half = glw // 2
        for x in xs:
            grid_arr[:, max(0, x - half) : x + half + 1] = gl
        for y in ys:
            grid_arr[max(0, y - half) : y + half + 1, :] = gl

        illusion_arr = grid_arr.copy()
        illusion_img = Image.fromarray(illusion_arr, mode="RGB")
        d = ImageDraw.Draw(illusion_img)
        for x in xs:
            for y in ys:
                r = disc_radius
                d.ellipse([(x - r, y - r), (x + r, y + r)], fill=(dl, dl, dl))

        control_arr = grid_arr.copy()
        control = Image.fromarray(control_arr, mode="RGB")

        return illusion_img, control

    def generate(self, params: dict[str, Any]) -> StimulusPair:
        illusion, control = self._make_pair(
            int(params["disc_radius"]),
            grid_spacing=int(params.get("grid_spacing", self.grid_spacing)),
            grid_line_width=int(params.get("grid_line_width", self.grid_line_width)),
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
        jitter_seed: int = 49,
    ) -> list[dict[str, Any]]:
        rng = np.random.default_rng(jitter_seed)
        # Keep dots in a perceptual range where scintillation remains visible.
        # Very large dots can suppress the effect, so cap the upper radius.
        disc_radii = np.linspace(6, 13, n_levels).astype(int).tolist()
        grid: list[dict[str, Any]] = []
        for radius in disc_radii:
            for rep in range(n_repeats):
                # Couple line thickness to dot size to preserve local contrast
                # at intersections for larger radii.
                line_base = int(round(0.65 * radius + 3))
                line_width = int(np.clip(line_base + rng.integers(-1, 2), 6, 12))
                grid.append({
                    "disc_radius": int(radius),
                    "repeat": rep,
                    "grid_spacing": int(rng.choice([44, 48, 52])),
                    "grid_line_width": line_width,
                })
        return grid


# ──────────────────────────────────────────────────────────────────────────────
# Rotating Snakes
# ──────────────────────────────────────────────────────────────────────────────

class RotatingSnakesGenerator(StimulusGenerator):
    """Parametric Rotating Snakes illusion.

    Circular "snakes" are built from repeated asymmetric luminance/color
    cycles.  The ordered sequence creates a strong apparent rotation in
    peripheral vision.  The control keeps the same circular segmentation but
    uses a symmetric sequence without a consistent motion direction.

    Sweep variable: wheel radius (larger snakes produce stronger peripheral
    drift while preserving the same local pattern).
    """

    category = "motion"
    illusion_type = "rotating_snakes"

    def __init__(
        self,
        image_size: tuple[int, int] = (512, 512),
        wheel_radius: int = 58,
        ring_width: int = 10,
        ring_gap: int = 2,
        n_rings: int = 4,
        segment_count: int = 48,
        background: tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        self.image_size = image_size
        self.wheel_radius = wheel_radius
        self.ring_width = ring_width
        self.ring_gap = ring_gap
        self.n_rings = n_rings
        self.segment_count = segment_count
        self.background = background

    def _draw_luminance_ring(
        self,
        draw: ImageDraw.ImageDraw,
        cx: int,
        cy: int,
        outer_radius: int,
        phase_deg: float,
        direction: int,
        illusion: bool,
    ) -> None:
        palette = [
            (0, 0, 0),
            (45, 80, 210),
            (255, 255, 255),
            (245, 205, 30),
        ]
        control_palette = [
            (0, 0, 0),
            (255, 255, 255),
            (0, 0, 0),
            (255, 255, 255),
        ]
        colors = palette if illusion else control_palette
        step = 360 / self.segment_count
        bbox = [
            cx - outer_radius,
            cy - outer_radius,
            cx + outer_radius,
            cy + outer_radius,
        ]
        for segment_index in range(self.segment_count):
            color_index = (direction * segment_index) % len(colors)
            start = phase_deg + segment_index * step
            draw.pieslice(bbox, start=start, end=start + step + 0.4, fill=colors[color_index])

        inner_radius = max(0, outer_radius - self.ring_width)
        draw.ellipse(
            [
                cx - inner_radius,
                cy - inner_radius,
                cx + inner_radius,
                cy + inner_radius,
            ],
            fill=self.background,
        )

    def _draw_wheel(
        self,
        draw: ImageDraw.ImageDraw,
        cx: int,
        cy: int,
        wheel_radius: int,
        phase_deg: float,
        direction: int,
        illusion: bool,
    ) -> None:
        for ring_index in range(self.n_rings):
            outer_radius = wheel_radius - ring_index * (self.ring_width + self.ring_gap)
            if outer_radius <= self.ring_width:
                break
            ring_phase = phase_deg + ring_index * 0.5 * (360 / self.segment_count)
            self._draw_luminance_ring(
                draw,
                cx,
                cy,
                outer_radius,
                ring_phase,
                direction if ring_index % 2 == 0 else -direction,
                illusion,
            )

    def _make_pair(
        self,
        wheel_radius: int | None = None,
        n_rings: int | None = None,
        segment_count: int | None = None,
        phase: float = 0.0,
        wheel_grid: int = 3,
    ) -> tuple[Image.Image, Image.Image]:
        w, h = self.image_size
        old_n_rings, old_segment_count = self.n_rings, self.segment_count
        if n_rings is not None:
            self.n_rings = n_rings
        if segment_count is not None:
            self.segment_count = segment_count
        radius = self.wheel_radius if wheel_radius is None else wheel_radius
        phase_deg = math.degrees(phase)

        illusion = Image.new("RGB", self.image_size, self.background)
        d = ImageDraw.Draw(illusion)
        control = Image.new("RGB", self.image_size, self.background)
        dc = ImageDraw.Draw(control)

        xs = np.linspace(radius + 18, w - radius - 18, wheel_grid)
        ys = np.linspace(radius + 18, h - radius - 18, wheel_grid)
        for row, cy in enumerate(ys):
            for col, cx in enumerate(xs):
                direction = 1 if (row + col) % 2 == 0 else -1
                local_phase = phase_deg + (row * wheel_grid + col) * 17
                self._draw_wheel(
                    d,
                    int(cx),
                    int(cy),
                    radius,
                    local_phase,
                    direction,
                    illusion=True,
                )
                self._draw_wheel(
                    dc,
                    int(cx),
                    int(cy),
                    radius,
                    local_phase,
                    direction,
                    illusion=False,
                )

        self.n_rings, self.segment_count = old_n_rings, old_segment_count

        return illusion, control

    def generate(self, params: dict[str, Any]) -> StimulusPair:
        illusion, control = self._make_pair(
            wheel_radius=int(params.get("wheel_radius", self.wheel_radius)),
            n_rings=int(params.get("n_rings", self.n_rings)),
            segment_count=int(params.get("segment_count", self.segment_count)),
            phase=float(params.get("phase", 0.0)),
            wheel_grid=int(params.get("wheel_grid", 3)),
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
        jitter_seed: int = 50,
    ) -> list[dict[str, Any]]:
        rng = np.random.default_rng(jitter_seed)
        wheel_radii = np.linspace(46, 62, n_levels).astype(int).tolist()
        grid: list[dict[str, Any]] = []
        for radius in wheel_radii:
            for rep in range(n_repeats):
                grid.append({
                    "wheel_radius": int(radius),
                    "repeat": rep,
                    "n_rings": int(rng.choice([3, 4])),
                    "segment_count": int(rng.choice([40, 48, 56])),
                    "phase": float(round(rng.uniform(0, 2 * math.pi), 3)),
                    "wheel_grid": 3,
                })
        return grid
