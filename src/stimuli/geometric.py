"""Geometric / length illusion generators.

Illusions implemented
---------------------
- Müller-Lyer   : arrowhead fins modulate perceived line length.
- Ponzo         : converging rails modulate perceived bar length.
- Ebbinghaus    : surrounding circles modulate perceived disc size.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .base import StimulusGenerator, StimulusPair, ANSWER_CORRECT, ANSWER_ILLUSORY


# ──────────────────────────────────────────────────────────────────────────────
# Müller-Lyer
# ──────────────────────────────────────────────────────────────────────────────

class MullerLyerGenerator(StimulusGenerator):
    """Parametric Müller-Lyer stimulus generator.

    Each stimulus presents two horizontal line shafts of equal physical length.
    The top shaft has inward-pointing fins (arrow → makes line look *shorter*),
    the bottom shaft has outward-pointing fins (arrow-tail → makes line look
    *longer*).  The control image shows the same layout with fins removed.

    Parameters
    ----------
    image_size : (width, height) in pixels.
    shaft_length : Length of each shaft in pixels.
    fin_length : Length of each fin arm in pixels.  The sweep variable.
    fin_angle_deg : Opening angle of the fins.
    line_width : Stroke width in pixels.
    background : Background RGB colour.
    foreground : Line RGB colour.
    """

    category = "geometric"
    illusion_type = "muller_lyer"

    def __init__(
        self,
        image_size: tuple[int, int] = (512, 256),
        shaft_length: int = 200,
        line_width: int = 3,
        background: tuple[int, int, int] = (255, 255, 255),
        foreground: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.image_size = image_size
        self.shaft_length = shaft_length
        self.line_width = line_width
        self.background = background
        self.foreground = foreground

    # ------------------------------------------------------------------
    def _draw_shaft_with_fins(
        self,
        draw: ImageDraw.ImageDraw,
        x0: int,
        y: int,
        length: int,
        fin_length: float,
        fin_angle_deg: float,
        inward: bool,
    ) -> None:
        """Draw a horizontal shaft with arrow fins at both ends."""
        x1 = x0 + length
        draw.line([(x0, y), (x1, y)], fill=self.foreground, width=self.line_width)
        angle_rad = math.radians(fin_angle_deg)
        for tip_x, base_direction in [(x0, 1), (x1, -1)]:
            if inward:
                fin_dir = base_direction
            else:
                fin_dir = -base_direction
            for sign in (+1, -1):
                dx = fin_dir * fin_length * math.cos(angle_rad)
                dy = sign * fin_length * math.sin(angle_rad)
                draw.line(
                    [(tip_x, y), (int(tip_x + dx), int(y + dy))],
                    fill=self.foreground,
                    width=self.line_width,
                )

    def _make_pair(
        self,
        fin_length: float,
        fin_angle_deg: float,
        x_jitter: int = 0,
        y_jitter: int = 0,
        shaft_scale: float = 1.0,
    ) -> tuple[Image.Image, Image.Image]:
        """Render one stimulus pair.

        Parameters
        ----------
        x_jitter, y_jitter : Pixel offsets applied to both shafts.
        shaft_scale        : Multiplicative scale on shaft_length (keeps fins
                             proportionally the same but changes absolute size).
        """
        w, h = self.image_size
        shaft_length = max(40, int(self.shaft_length * shaft_scale))
        x0 = (w - shaft_length) // 2 + x_jitter
        # Clamp so shaft never leaves the canvas
        x0 = max(10, min(w - shaft_length - 10, x0))
        y_top = h // 3 + y_jitter
        y_bot = 2 * h // 3 + y_jitter

        illusion = Image.new("RGB", self.image_size, self.background)
        d = ImageDraw.Draw(illusion)
        self._draw_shaft_with_fins(d, x0, y_top, shaft_length, fin_length, fin_angle_deg, inward=True)
        self._draw_shaft_with_fins(d, x0, y_bot, shaft_length, fin_length, fin_angle_deg, inward=False)

        control = Image.new("RGB", self.image_size, self.background)
        dc = ImageDraw.Draw(control)
        dc.line([(x0, y_top), (x0 + shaft_length, y_top)], fill=self.foreground, width=self.line_width)
        dc.line([(x0, y_bot), (x0 + shaft_length, y_bot)], fill=self.foreground, width=self.line_width)

        return illusion, control

    # ------------------------------------------------------------------
    def generate(self, params: dict[str, Any]) -> StimulusPair:
        fin_length = params["fin_length"]
        fin_angle_deg = params.get("fin_angle_deg", 30.0)
        # Apply per-trial jitter stored in params (zero by default)
        illusion, control = self._make_pair(
            fin_length, fin_angle_deg,
            x_jitter=params.get("x_jitter", 0),
            y_jitter=params.get("y_jitter", 0),
            shaft_scale=params.get("shaft_scale", 1.0),
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
        n_fin_levels: int = 20,
        fin_angles: list[float] | None = None,
        n_repeats: int = 10,
        jitter_seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Build a trial grid with proper statistical replication.

        Structure
        ---------
        fin_length   : ``n_fin_levels`` log-spaced values from 6 to 80 px.
                       Log-spacing gives more resolution at the psychometric
                       threshold while avoiding near-invisible fins that would
                       not produce a meaningful human illusion.
        fin_angle_deg: ``fin_angles`` (default 3 values: 20°, 30°, 45°).
        repeat       : ``n_repeats`` independent trials per (fin_length, angle)
                       pair, each with unique nuisance jitter (x/y offset,
                       shaft scale).  Nuisance variation is irrelevant to the
                       illusion strength but makes each trial visually distinct,
                       so the model cannot memorise a single image.

        Total trials = n_fin_levels × len(fin_angles) × n_repeats.
        Default: 20 × 3 × 10 = 600 pairs → 30 measurements per fin_length level.

        With n=30, the 95 % CI on a proportion is ±0.18 (worst case p=0.5),
        giving meaningful error bars on the psychometric curve.
        """
        if fin_angles is None:
            fin_angles = [20.0, 30.0, 45.0]

        # Log-spaced fin lengths: denser near threshold, sparser at extremes
        fin_lengths = np.logspace(np.log10(6), np.log10(80), n_fin_levels).tolist()

        rng = np.random.default_rng(jitter_seed)
        grid: list[dict[str, Any]] = []

        for fl in fin_lengths:
            for fa in fin_angles:
                # Generate n_repeats independent jitter instances
                x_jitters = rng.integers(-30, 31, size=n_repeats).tolist()
                y_jitters = rng.integers(-15, 16, size=n_repeats).tolist()
                shaft_scales = rng.uniform(0.85, 1.15, size=n_repeats).tolist()
                for rep in range(n_repeats):
                    grid.append({
                        "fin_length": float(round(fl, 2)),
                        "fin_angle_deg": float(fa),
                        "repeat": rep,
                        "x_jitter": int(x_jitters[rep]),
                        "y_jitter": int(y_jitters[rep]),
                        "shaft_scale": float(round(shaft_scales[rep], 3)),
                    })
        return grid


# ──────────────────────────────────────────────────────────────────────────────
# Ponzo
# ──────────────────────────────────────────────────────────────────────────────

class PonzoGenerator(StimulusGenerator):
    """Parametric Ponzo (railway lines) illusion.

    Two horizontal bars of equal length are placed between converging lines.
    The upper bar appears longer due to depth cue from the converging context.
    The sweep parameter is the convergence angle of the rails.
    """

    category = "geometric"
    illusion_type = "ponzo"

    def __init__(
        self,
        image_size: tuple[int, int] = (512, 512),
        bar_length: int = 120,
        line_width: int = 3,
        background: tuple[int, int, int] = (255, 255, 255),
        foreground: tuple[int, int, int] = (0, 0, 0),
        bar_color: tuple[int, int, int] = (180, 0, 0),
    ) -> None:
        self.image_size = image_size
        self.bar_length = bar_length
        self.line_width = line_width
        self.background = background
        self.foreground = foreground
        self.bar_color = bar_color

    def _make_pair(
        self,
        convergence_deg: float,
        y_shift: int = 0,
        bar_scale: float = 1.0,
        vp_y_frac: float = 0.1,
    ) -> tuple[Image.Image, Image.Image]:
        w, h = self.image_size
        cx = w // 2

        # Vanishing point at the top centre
        vp_y = int(h * vp_y_frac)
        angle_rad = math.radians(convergence_deg / 2)
        bar_length = max(40, int(self.bar_length * bar_scale))

        def rail_x(y: int, side: int) -> int:
            dist = y - vp_y
            return int(cx + side * dist * math.tan(angle_rad))

        illusion = Image.new("RGB", self.image_size, self.background)
        d = ImageDraw.Draw(illusion)
        bot_y = int(h * 0.9)
        d.line([(cx, vp_y), (rail_x(bot_y, -1), bot_y)], fill=self.foreground, width=self.line_width)
        d.line([(cx, vp_y), (rail_x(bot_y, +1), bot_y)], fill=self.foreground, width=self.line_width)

        # Upper and lower bars
        for bar_y_frac, color in [(0.35, self.bar_color), (0.65, self.bar_color)]:
            by = int(h * bar_y_frac) + y_shift
            bx0 = cx - bar_length // 2
            bx1 = cx + bar_length // 2
            d.line([(bx0, by), (bx1, by)], fill=color, width=self.line_width + 2)

        control = Image.new("RGB", self.image_size, self.background)
        dc = ImageDraw.Draw(control)
        # Parallel rails (no convergence)
        margin = bar_length // 2 + 20
        dc.line([(cx - margin, vp_y), (cx - margin, bot_y)], fill=self.foreground, width=self.line_width)
        dc.line([(cx + margin, vp_y), (cx + margin, bot_y)], fill=self.foreground, width=self.line_width)
        for bar_y_frac in [0.35, 0.65]:
            by = int(h * bar_y_frac) + y_shift
            bx0 = cx - bar_length // 2
            bx1 = cx + bar_length // 2
            dc.line([(bx0, by), (bx1, by)], fill=self.bar_color, width=self.line_width + 2)

        return illusion, control

    def generate(self, params: dict[str, Any]) -> StimulusPair:
        illusion, control = self._make_pair(
            params["convergence_deg"],
            y_shift=params.get("y_shift", 0),
            bar_scale=params.get("bar_scale", 1.0),
            vp_y_frac=params.get("vp_y_frac", 0.1),
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
        jitter_seed: int = 43,
    ) -> list[dict[str, Any]]:
        """Meaningful Ponzo sweep with repeated nuisance variation.

        We avoid very small convergence angles because they are visually weak
        and unlikely to induce a reliable human illusion.  Each convergence
        level has ``n_repeats`` independent layouts.
        """
        rng = np.random.default_rng(jitter_seed)
        convergence_degs = np.linspace(15, 60, n_levels).tolist()
        grid: list[dict[str, Any]] = []
        for conv in convergence_degs:
            for rep in range(n_repeats):
                grid.append({
                    "convergence_deg": float(round(conv, 2)),
                    "repeat": rep,
                    "y_shift": int(rng.integers(-18, 19)),
                    "bar_scale": float(round(rng.uniform(0.85, 1.15), 3)),
                    "vp_y_frac": float(round(rng.uniform(0.07, 0.13), 3)),
                })
        return grid


# ──────────────────────────────────────────────────────────────────────────────
# Ebbinghaus
# ──────────────────────────────────────────────────────────────────────────────

class EbbinghausGenerator(StimulusGenerator):
    """Parametric Ebbinghaus (Titchener circles) illusion.

    A central disc is surrounded by either large or small satellite circles,
    making it appear smaller or larger respectively.  Left disc surrounded by
    large circles, right disc surrounded by small circles.
    The sweep parameter is the satellite radius.
    """

    category = "geometric"
    illusion_type = "ebbinghaus"

    def __init__(
        self,
        image_size: tuple[int, int] = (512, 256),
        center_radius: int = 30,
        n_satellites: int = 6,
        satellite_distance: int = 70,
        min_gap: int = 10,
        background: tuple[int, int, int] = (255, 255, 255),
        foreground: tuple[int, int, int] = (0, 0, 0),
        center_fill: tuple[int, int, int] = (80, 80, 80),
        satellite_fill: tuple[int, int, int] = (200, 200, 200),
    ) -> None:
        self.image_size = image_size
        self.center_radius = center_radius
        self.n_satellites = n_satellites
        self.satellite_distance = satellite_distance
        self.min_gap = min_gap
        self.background = background
        self.foreground = foreground
        self.center_fill = center_fill
        self.satellite_fill = satellite_fill

    def _draw_ebbinghaus_config(
        self,
        draw: ImageDraw.ImageDraw,
        w: int,
        h: int,
        cx: int,
        cy: int,
        satellite_radius: int,
        center_radius: int | None = None,
        satellite_distance: int | None = None,
        min_gap: int | None = None,
    ) -> None:
        """Draw central disc + ring of satellites."""
        r = self.center_radius if center_radius is None else center_radius
        gap = self.min_gap if min_gap is None else min_gap
        base_distance = self.satellite_distance if satellite_distance is None else satellite_distance
        # Keep satellites from overlapping center, each other, and image bounds.
        n_sat = self.n_satellites
        max_canvas_distance = max(8, min(cx, w - cx, cy, h - cy) - satellite_radius - 2)
        while n_sat > 3:
            neighbor_min = satellite_radius / max(1e-6, math.sin(math.pi / n_sat)) + gap
            center_min = r + satellite_radius + gap
            needed = max(base_distance, center_min, neighbor_min)
            if needed <= max_canvas_distance:
                break
            n_sat -= 1
        neighbor_min = satellite_radius / max(1e-6, math.sin(math.pi / n_sat)) + gap
        center_min = r + satellite_radius + gap
        distance = min(max_canvas_distance, max(base_distance, center_min, neighbor_min))
        draw.ellipse(
            [(cx - r, cy - r), (cx + r, cy + r)],
            fill=self.center_fill,
            outline=self.foreground,
            width=2,
        )
        for i in range(n_sat):
            angle = 2 * math.pi * i / n_sat
            sx = int(cx + distance * math.cos(angle))
            sy = int(cy + distance * math.sin(angle))
            sr = satellite_radius
            draw.ellipse(
                [(sx - sr, sy - sr), (sx + sr, sy + sr)],
                fill=self.satellite_fill,
                outline=self.foreground,
                width=2,
            )

    def _make_pair(
        self,
        large_sat_radius: int,
        small_sat_radius: int,
        center_radius: int | None = None,
        satellite_distance: int | None = None,
        x_jitter: int = 0,
        y_jitter: int = 0,
    ) -> tuple[Image.Image, Image.Image]:
        w, h = self.image_size
        cy = h // 2 + y_jitter
        cx_left = w // 4 + x_jitter
        cx_right = 3 * w // 4 + x_jitter

        illusion = Image.new("RGB", self.image_size, self.background)
        d = ImageDraw.Draw(illusion)
        self._draw_ebbinghaus_config(d, w, h, cx_left, cy, large_sat_radius, center_radius, satellite_distance)
        self._draw_ebbinghaus_config(d, w, h, cx_right, cy, small_sat_radius, center_radius, satellite_distance)

        control = Image.new("RGB", self.image_size, self.background)
        dc = ImageDraw.Draw(control)
        # Same-size satellites on both sides → no illusion
        mid_radius = (large_sat_radius + small_sat_radius) // 2
        self._draw_ebbinghaus_config(dc, w, h, cx_left, cy, mid_radius, center_radius, satellite_distance)
        self._draw_ebbinghaus_config(dc, w, h, cx_right, cy, mid_radius, center_radius, satellite_distance)

        return illusion, control

    def generate(self, params: dict[str, Any]) -> StimulusPair:
        illusion, control = self._make_pair(
            int(params["large_sat_radius"]), int(params["small_sat_radius"]),
            center_radius=int(params.get("center_radius", self.center_radius)),
            satellite_distance=int(params.get("satellite_distance", self.satellite_distance)),
            x_jitter=int(params.get("x_jitter", 0)),
            y_jitter=int(params.get("y_jitter", 0)),
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
        jitter_seed: int = 44,
    ) -> list[dict[str, Any]]:
        """Sweep only perceptually meaningful surround-size differences."""
        rng = np.random.default_rng(jitter_seed)
        # Stay in a range that preserves clear Ebbinghaus context without
        # geometric crowding on the default canvas.
        large_radii = np.linspace(30, 40, n_levels).astype(int).tolist()
        small_radii = np.linspace(8, 16, n_levels).astype(int).tolist()
        grid: list[dict[str, Any]] = []
        for large, small in zip(large_radii, small_radii):
            for rep in range(n_repeats):
                grid.append({
                    "large_sat_radius": int(large),
                    "small_sat_radius": int(small),
                    "repeat": rep,
                    "center_radius": int(rng.integers(27, 34)),
                    "satellite_distance": int(rng.integers(70, 88)),
                    "x_jitter": int(rng.integers(-8, 9)),
                    "y_jitter": int(rng.integers(-8, 9)),
                })
        return grid
