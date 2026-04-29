"""Motion-from-static illusion generators.

Illusions implemented
---------------------
- Scintillating Grid : bright spots at grid intersections appear to scintillate
  (flicker dark) when the image is viewed statically.
  Follows Sun & Dekel (2021) [doi:10.1167/jov.21.11.15].
- Fraser Spiral      : concentric arcs that appear to form a spiral but are
  actually circles.
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
        # Avoid tiny discs: the scintillation effect is weak or absent.
        disc_radii = np.linspace(7, 18, n_levels).astype(int).tolist()
        grid: list[dict[str, Any]] = []
        for radius in disc_radii:
            for rep in range(n_repeats):
                grid.append({
                    "disc_radius": int(radius),
                    "repeat": rep,
                    "grid_spacing": int(rng.choice([44, 48, 52])),
                    "grid_line_width": int(rng.choice([7, 8, 9])),
                })
        return grid


# ──────────────────────────────────────────────────────────────────────────────
# Fraser Spiral
# ──────────────────────────────────────────────────────────────────────────────

class FraserSpiralGenerator(StimulusGenerator):
    """Fraser spiral (twisted cord) illusion.

    Concentric annuli are divided into alternating black/white *slanted*
    tiles.  The true contours are circular, but the local slant cues create
    a strong apparent spiral.  The control shows the same concentric layout
    without slanted texture.

    Sweep variable: number of rings (more rings → stronger spiral percept).
    """

    category = "motion"
    illusion_type = "fraser_spiral"

    def __init__(
        self,
        image_size: tuple[int, int] = (512, 512),
        tile_arc_length: int = 34,
        ring_width: int = 16,
        ring_gap: int = 8,
        line_width: int = 4,
        background: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self.image_size = image_size
        self.tile_arc_length = tile_arc_length
        self.ring_width = ring_width
        self.ring_gap = ring_gap
        self.line_width = line_width
        self.background = background

    @staticmethod
    def _annular_tile_points(
        cx: int,
        cy: int,
        inner_r: float,
        outer_r: float,
        inner_a0: float,
        inner_a1: float,
        outer_a0: float,
        outer_a1: float,
        n_steps: int = 4,
    ) -> list[tuple[int, int]]:
        """Polygon points for one slanted annular tile."""
        points: list[tuple[int, int]] = []
        for a in np.linspace(outer_a0, outer_a1, n_steps):
            points.append((int(cx + outer_r * math.cos(a)), int(cy + outer_r * math.sin(a))))
        for a in np.linspace(inner_a1, inner_a0, n_steps):
            points.append((int(cx + inner_r * math.cos(a)), int(cy + inner_r * math.sin(a))))
        return points

    def _draw_twisted_ring(
        self,
        draw: ImageDraw.ImageDraw,
        cx: int,
        cy: int,
        inner_r: float,
        outer_r: float,
        tile_arc_length: int,
        slant_rad: float,
        phase: float = 0.0,
        black_fraction: float = 0.52,
    ) -> None:
        """Draw one ring of alternating slanted black tiles.

        Only black tiles are drawn; white tiles are the background.  The outer
        angular coordinates are shifted relative to the inner coordinates,
        producing the twisted-cord cue that drives the Fraser illusion.
        """
        mid_r = (inner_r + outer_r) / 2
        n_tiles = max(24, int(2 * math.pi * mid_r / tile_arc_length))
        step = 2 * math.pi / n_tiles
        for i in range(n_tiles):
            if i % 2 == 1:
                continue
            inner_a0 = phase + i * step
            inner_a1 = inner_a0 + black_fraction * step
            outer_a0 = inner_a0 + slant_rad
            outer_a1 = inner_a1 + slant_rad
            draw.polygon(
                self._annular_tile_points(cx, cy, inner_r, outer_r, inner_a0, inner_a1, outer_a0, outer_a1),
                fill=(0, 0, 0),
            )

    def _make_pair(
        self,
        n_circles: int,
        tile_arc_length: int | None = None,
        phase: float = 0.0,
        radius_scale: float = 1.0,
        slant_deg: float = 22.0,
        ring_phase_step: float = 0.22,
    ) -> tuple[Image.Image, Image.Image]:
        w, h = self.image_size
        cx, cy = w // 2, h // 2
        max_radius = min(cx, cy) - 8
        tile_len = self.tile_arc_length if tile_arc_length is None else tile_arc_length
        slant_rad = math.radians(slant_deg)

        illusion = Image.new("RGB", self.image_size, self.background)
        d = ImageDraw.Draw(illusion)
        outer_max = int(max_radius * radius_scale)
        total_band = n_circles * self.ring_width + (n_circles - 1) * self.ring_gap
        inner_start = max(8, outer_max - total_band)
        for idx in range(n_circles):
            inner_r = inner_start + idx * (self.ring_width + self.ring_gap)
            outer_r = inner_r + self.ring_width
            ring_phase = phase + idx * ring_phase_step
            self._draw_twisted_ring(
                d, cx, cy, inner_r, outer_r,
                tile_arc_length=tile_len,
                slant_rad=slant_rad,
                phase=ring_phase,
            )
            # Thin circular boundaries reinforce that the physical contour is circular.
            d.ellipse(
                [(cx - outer_r, cy - outer_r), (cx + outer_r, cy + outer_r)],
                outline=(0, 0, 0),
                width=max(1, self.line_width // 2),
            )

        control = Image.new("RGB", self.image_size, self.background)
        dc = ImageDraw.Draw(control)
        for idx in range(n_circles):
            inner_r = inner_start + idx * (self.ring_width + self.ring_gap)
            outer_r = inner_r + self.ring_width
            dc.ellipse(
                [(cx - outer_r, cy - outer_r), (cx + outer_r, cy + outer_r)],
                outline=(0, 0, 0),
                width=self.line_width,
            )

        return illusion, control

    def generate(self, params: dict[str, Any]) -> StimulusPair:
        illusion, control = self._make_pair(
            int(params["n_circles"]),
            tile_arc_length=int(params.get("tile_arc_length", self.tile_arc_length)),
            phase=float(params.get("phase", 0.0)),
            radius_scale=float(params.get("radius_scale", 1.0)),
            slant_deg=float(params.get("slant_deg", 22.0)),
            ring_phase_step=float(params.get("ring_phase_step", 0.22)),
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
        n_repeats: int = 30,
        jitter_seed: int = 50,
    ) -> list[dict[str, Any]]:
        rng = np.random.default_rng(jitter_seed)
        # Start at 7 rings; fewer are often too sparse to produce a strong
        # spiral percept.  The sweep remains interpretable as "more rings".
        n_circles_values = [7, 9, 11, 13, 15, 17]
        grid: list[dict[str, Any]] = []
        for n_circles in n_circles_values:
            for rep in range(n_repeats):
                grid.append({
                    "n_circles": int(n_circles),
                    "repeat": rep,
                    "tile_arc_length": int(rng.choice([30, 34, 38])),
                    "phase": float(round(rng.uniform(0, 2 * math.pi), 3)),
                    "radius_scale": float(round(rng.uniform(0.95, 1.0), 3)),
                    "slant_deg": float(round(rng.uniform(18, 28), 2)),
                    "ring_phase_step": float(round(rng.uniform(0.16, 0.28), 3)),
                })
        return grid
