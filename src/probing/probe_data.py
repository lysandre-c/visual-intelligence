"""Synthetic probe training data generators.

For each illusion category the probe must be trained on *non-illusion* images
where the ground-truth label is unambiguous.  Using solid-colour images teaches
the probe nothing about the visual task; using images that are structurally
similar to the test stimuli but with explicit, unambiguous content gives the
backbone features a chance to learn the discrimination.

Labels throughout
-----------------
0 = correct  (the physically accurate percept is easy to verify)
1 = illusory  (the image clearly shows the "wrong" percept direction)
2 = other    (ambiguous / no clear comparison)

Usage
-----
    from src.probing.probe_data import ProbeDataGenerator
    gen = ProbeDataGenerator(image_size=(512, 256), n_per_class=100)
    images, labels = gen.geometric_length(seed=42)
"""

from __future__ import annotations

import math
import random
from typing import Any, TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw


class ProbeDataGenerator:
    """Generate synthetic training images for each illusion category.

    Parameters
    ----------
    image_size : (width, height) in pixels — should match the test stimuli.
    n_per_class : How many images per class to generate (total = 3 × n_per_class).
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (512, 256),
        n_per_class: int = 100,
    ) -> None:
        self.image_size = image_size
        self.n_per_class = n_per_class

    # ------------------------------------------------------------------ #
    # Category: geometric / length                                         #
    # ------------------------------------------------------------------ #

    def geometric_length(
        self, seed: int = 42
    ) -> tuple[list[Image.Image], list[int]]:
        """Training data for geometric / length illusion probes.

        The images are designed to match the Müller-Lyer test stimuli as closely
        as possible: pure white background, black lines at h/3 and 2h/3, same
        stroke width range.  This minimises domain shift between probe training
        and test evaluation.

        Class 0 — two horizontal lines of identical length (equal, obvious).
        Class 1 — two horizontal lines with a clear length difference (≥30 %).
        Class 2 — single line, crossed lines, or blank (no length comparison).
        """
        rng = random.Random(seed)
        images: list[Image.Image] = []
        labels: list[int] = []
        w, h = self.image_size

        # Fixed y-positions that mirror the Müller-Lyer generator exactly
        y_top_base = h // 3
        y_bot_base = 2 * h // 3

        def _draw_hline(draw, y, x0, length, lw=3, color=(0, 0, 0)):
            draw.line([(x0, y), (x0 + length, y)], fill=color, width=lw)

        # ---- Class 0: equal-length lines --------------------------------
        for _ in range(self.n_per_class):
            img = Image.new("RGB", self.image_size, (255, 255, 255))
            d = ImageDraw.Draw(img)
            length = rng.randint(100, 280)
            x0 = (w - length) // 2 + rng.randint(-20, 20)
            x0 = max(10, min(w - length - 10, x0))
            y_top = y_top_base + rng.randint(-10, 10)
            y_bot = y_bot_base + rng.randint(-10, 10)
            lw = rng.randint(2, 4)
            _draw_hline(d, y_top, x0, length, lw)
            _draw_hline(d, y_bot, x0, length, lw)
            images.append(img)
            labels.append(0)

        # ---- Class 1: clearly different-length lines --------------------
        for _ in range(self.n_per_class):
            img = Image.new("RGB", self.image_size, (255, 255, 255))
            d = ImageDraw.Draw(img)
            long_len = rng.randint(150, 280)
            # Short line is 35–65 % of the long one — clearly different
            ratio = rng.uniform(0.35, 0.65)
            short_len = int(long_len * ratio)
            x0_long = (w - long_len) // 2 + rng.randint(-10, 10)
            x0_long = max(10, min(w - long_len - 10, x0_long))
            x0_short = (w - short_len) // 2 + rng.randint(-10, 10)
            x0_short = max(10, min(w - short_len - 10, x0_short))
            y_top = y_top_base + rng.randint(-10, 10)
            y_bot = y_bot_base + rng.randint(-10, 10)
            lw = rng.randint(2, 4)
            if rng.random() < 0.5:
                _draw_hline(d, y_top, x0_long, long_len, lw)
                _draw_hline(d, y_bot, x0_short, short_len, lw)
            else:
                _draw_hline(d, y_top, x0_short, short_len, lw)
                _draw_hline(d, y_bot, x0_long, long_len, lw)
            images.append(img)
            labels.append(1)

        # ---- Class 2: other (no clear length comparison) ----------------
        for _ in range(self.n_per_class):
            img = Image.new("RGB", self.image_size, (255, 255, 255))
            d = ImageDraw.Draw(img)
            choice = rng.randint(0, 2)
            lw = rng.randint(2, 4)
            if choice == 0:
                # Single horizontal line at center
                length = rng.randint(80, 260)
                x0 = (w - length) // 2
                _draw_hline(d, h // 2, x0, length, lw)
            elif choice == 1:
                # Two angled (non-horizontal) lines
                for _ in range(2):
                    x0 = rng.randint(50, w // 2)
                    y0 = rng.randint(40, h - 40)
                    angle = rng.uniform(20, 70)
                    length = rng.randint(80, 180)
                    x1 = int(x0 + length * math.cos(math.radians(angle)))
                    y1 = int(y0 + length * math.sin(math.radians(angle)))
                    d.line([(x0, y0), (x1, y1)], fill=(0, 0, 0), width=lw)
            else:
                # Cross at center
                cx, cy = w // 2, h // 2
                arm = rng.randint(50, 100)
                d.line([(cx - arm, cy), (cx + arm, cy)], fill=(0, 0, 0), width=lw)
                d.line([(cx, cy - arm), (cx, cy + arm)], fill=(0, 0, 0), width=lw)
            images.append(img)
            labels.append(2)

        combined = list(zip(images, labels))
        rng.shuffle(combined)
        images, labels = zip(*combined)
        return list(images), list(labels)

    # ------------------------------------------------------------------ #
    # Category: color / brightness                                         #
    # ------------------------------------------------------------------ #

    def color_brightness(
        self, seed: int = 42
    ) -> tuple[list[Image.Image], list[int]]:
        """Training data for color / brightness illusion probes.

        Class 0 — two grey patches of identical luminance on identical surrounds.
        Class 1 — two grey patches with clearly different luminance (≥40 lum units).
        Class 2 — single patch, no patch, or uniform field.
        """
        rng = random.Random(seed)
        images: list[Image.Image] = []
        labels: list[int] = []
        w, h = self.image_size
        half_w = w // 2
        ps = 60  # patch size

        def _make_patch_image(left_bg, right_bg, left_lum, right_lum):
            arr = np.ones((h, w, 3), dtype=np.uint8)
            arr[:, :half_w] = left_bg
            arr[:, half_w:] = right_bg
            cy = h // 2
            cx_l = half_w // 2
            cx_r = half_w + half_w // 2
            for cx, lum in [(cx_l, left_lum), (cx_r, right_lum)]:
                arr[cy - ps // 2 : cy + ps // 2, cx - ps // 2 : cx + ps // 2] = lum
            return Image.fromarray(arr, mode="RGB")

        # Class 0: identical patches
        for _ in range(self.n_per_class):
            lum = rng.randint(80, 180)
            left_bg = rng.randint(20, 240)
            right_bg = left_bg + rng.randint(-20, 20)
            right_bg = max(0, min(255, right_bg))
            images.append(_make_patch_image(left_bg, right_bg, lum, lum))
            labels.append(0)

        # Class 1: clearly different patches
        for _ in range(self.n_per_class):
            lum_a = rng.randint(60, 120)
            lum_b = lum_a + rng.randint(60, 120)
            lum_b = min(255, lum_b)
            mid_bg = rng.randint(100, 160)
            if rng.random() < 0.5:
                images.append(_make_patch_image(mid_bg, mid_bg, lum_a, lum_b))
            else:
                images.append(_make_patch_image(mid_bg, mid_bg, lum_b, lum_a))
            labels.append(1)

        # Class 2: single patch or uniform field
        for _ in range(self.n_per_class):
            arr = np.full((h, w, 3), rng.randint(100, 200), dtype=np.uint8)
            if rng.random() < 0.5:
                cy, cx = h // 2, w // 2
                arr[cy - ps // 2 : cy + ps // 2, cx - ps // 2 : cx + ps // 2] = rng.randint(50, 220)
            images.append(Image.fromarray(arr, mode="RGB"))
            labels.append(2)

        combined = list(zip(images, labels))
        rng.shuffle(combined)
        images, labels = zip(*combined)
        return list(images), list(labels)

    # ------------------------------------------------------------------ #
    # Category: angle / orientation                                        #
    # ------------------------------------------------------------------ #

    def angle_orientation(
        self, seed: int = 42
    ) -> tuple[list[Image.Image], list[int]]:
        """Training data for angle / orientation illusion probes.

        Class 0 — clearly parallel lines.
        Class 1 — clearly non-parallel lines (converging or diverging).
        Class 2 — single line or crossed lines.
        """
        rng = random.Random(seed)
        images: list[Image.Image] = []
        labels: list[int] = []
        w, h = self.image_size

        def _angled_line(draw, y_center, angle_deg, length, lw, color=(0, 0, 0)):
            rad = math.radians(angle_deg)
            x0 = int(w / 2 - length / 2 * math.cos(rad))
            y0 = int(y_center - length / 2 * math.sin(rad))
            x1 = int(w / 2 + length / 2 * math.cos(rad))
            y1 = int(y_center + length / 2 * math.sin(rad))
            draw.line([(x0, y0), (x1, y1)], fill=color, width=lw)

        for _ in range(self.n_per_class):
            bg = rng.randint(220, 255)
            img = Image.new("RGB", (w, h), (bg, bg, bg))
            d = ImageDraw.Draw(img)
            angle = rng.uniform(-40, 40)
            lw = rng.randint(2, 4)
            for y_frac in [0.33, 0.67]:
                _angled_line(d, int(h * y_frac), angle, rng.randint(150, 280), lw)
            images.append(img)
            labels.append(0)

        for _ in range(self.n_per_class):
            bg = rng.randint(220, 255)
            img = Image.new("RGB", (w, h), (bg, bg, bg))
            d = ImageDraw.Draw(img)
            base = rng.uniform(-30, 30)
            diverge = rng.uniform(15, 45)
            lw = rng.randint(2, 4)
            _angled_line(d, int(h * 0.33), base, rng.randint(150, 280), lw)
            _angled_line(d, int(h * 0.67), base + diverge, rng.randint(150, 280), lw)
            images.append(img)
            labels.append(1)

        for _ in range(self.n_per_class):
            bg = rng.randint(220, 255)
            img = Image.new("RGB", (w, h), (bg, bg, bg))
            d = ImageDraw.Draw(img)
            angle = rng.uniform(-60, 60)
            lw = rng.randint(2, 4)
            _angled_line(d, h // 2, angle, rng.randint(100, 250), lw)
            images.append(img)
            labels.append(2)

        combined = list(zip(images, labels))
        rng.shuffle(combined)
        images, labels = zip(*combined)
        return list(images), list(labels)

    # ------------------------------------------------------------------ #
    # Illusion-specific: Müller-Lyer                                       #
    # ------------------------------------------------------------------ #

    def muller_lyer_from_generator(
        self,
        generator: Any,
        seed: int = 42,
    ) -> tuple[list[Image.Image], list[int]]:
        """Probe training data built from the actual Müller-Lyer generator.

        This is the key fix for the "fins → other" problem: instead of training
        on plain lines we train on the SAME visual format as the test stimuli.

        Class 0 (correct) — Müller-Lyer CONTROL images: two plain equal lines
            with no fins.  The model must learn "this means equal."
        Class 1 (illusory) — Müller-Lyer ILLUSION images with strong fins
            (fin_length ∈ [25, 80]).  The model must learn "fins cause the
            illusory percept."
        Class 2 (other)   — Blank images + single-line images so the probe has
            a catch-all class that doesn't absorb the illusion stimuli.

        Parameters
        ----------
        generator :
            A ``MullerLyerGenerator`` instance (duck-typed to avoid a hard
            import cycle; only ``_make_pair`` and ``image_size`` are used).
        """
        rng = random.Random(seed)
        images: list[Image.Image] = []
        labels: list[int] = []
        w, h = generator.image_size

        fin_angles = [20.0, 30.0, 45.0]
        strong_fin_lengths = [fl for fl in np.linspace(25, 80, 30).tolist()]

        # ---- Class 0: control images (no fins, equal lines) -------------
        for i in range(self.n_per_class):
            x_jitter = rng.randint(-25, 25)
            y_jitter = rng.randint(-12, 12)
            shaft_scale = rng.uniform(0.85, 1.15)
            _, ctrl = generator._make_pair(
                fin_length=0.0,
                fin_angle_deg=rng.choice(fin_angles),
                x_jitter=x_jitter, y_jitter=y_jitter, shaft_scale=shaft_scale,
            )
            images.append(ctrl)
            labels.append(0)

        # ---- Class 1: illusion images with strong fins ------------------
        for i in range(self.n_per_class):
            fl = rng.choice(strong_fin_lengths)
            fa = rng.choice(fin_angles)
            x_jitter = rng.randint(-25, 25)
            y_jitter = rng.randint(-12, 12)
            shaft_scale = rng.uniform(0.85, 1.15)
            ill, _ = generator._make_pair(
                fin_length=fl, fin_angle_deg=fa,
                x_jitter=x_jitter, y_jitter=y_jitter, shaft_scale=shaft_scale,
            )
            images.append(ill)
            labels.append(1)

        # ---- Class 2: other (blank / single line) -----------------------
        for _ in range(self.n_per_class):
            img = Image.new("RGB", (w, h), (255, 255, 255))
            d = ImageDraw.Draw(img)
            choice = rng.randint(0, 2)
            if choice == 0:
                pass  # blank white image
            elif choice == 1:
                length = rng.randint(60, 220)
                x0 = (w - length) // 2
                d.line([(x0, h // 2), (x0 + length, h // 2)], fill=(0, 0, 0), width=3)
            else:
                cx, cy = w // 2, h // 2
                arm = rng.randint(40, 90)
                d.line([(cx - arm, cy), (cx + arm, cy)], fill=(0, 0, 0), width=3)
                d.line([(cx, cy - arm), (cx, cy + arm)], fill=(0, 0, 0), width=3)
            images.append(img)
            labels.append(2)

        combined = list(zip(images, labels))
        rng.shuffle(combined)
        images, labels = zip(*combined)
        return list(images), list(labels)

    # ------------------------------------------------------------------ #
    # Dispatcher                                                           #
    # ------------------------------------------------------------------ #

    def for_category(
        self, category: str, seed: int = 42
    ) -> tuple[list[Image.Image], list[int]]:
        """Return training data for the given illusion category."""
        dispatch = {
            "geometric": self.geometric_length,
            "color": self.color_brightness,
            "angle": self.angle_orientation,
        }
        if category not in dispatch:
            raise ValueError(
                f"No probe training data generator for category {category!r}. "
                f"Available: {list(dispatch.keys())}"
            )
        return dispatch[category](seed=seed)
