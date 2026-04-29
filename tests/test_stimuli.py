"""Unit tests for stimulus generators.

Tests verify:
  - Each generator produces correctly-typed StimulusPair objects.
  - Illusion and control images have the expected size and mode.
  - param_grid() returns a non-empty, consistent list of dicts.
  - generate_dataset() writes images + manifest to a temp directory.
  - StimulusPair.save() / load() round-trips correctly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.stimuli.base import StimulusPair, ANSWER_CORRECT, ANSWER_ILLUSORY
from src.stimuli.geometric import MullerLyerGenerator, PonzoGenerator, EbbinghausGenerator
from src.stimuli.color import SimultaneousContrastGenerator, WhiteIllusionGenerator
from src.stimuli.angle import ZollnerGenerator, PoggendorffGenerator
from src.stimuli.motion import ScintillatingGridGenerator, FraserSpiralGenerator


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

ALL_GENERATORS = [
    MullerLyerGenerator,
    PonzoGenerator,
    EbbinghausGenerator,
    SimultaneousContrastGenerator,
    WhiteIllusionGenerator,
    ZollnerGenerator,
    PoggendorffGenerator,
    ScintillatingGridGenerator,
    FraserSpiralGenerator,
]


# ──────────────────────────────────────────────────────────────────────────────
# param_grid tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("GeneratorClass", ALL_GENERATORS)
def test_param_grid_nonempty(GeneratorClass):
    gen = GeneratorClass()
    grid = gen.param_grid()
    assert isinstance(grid, list)
    assert len(grid) > 0, f"{GeneratorClass.__name__}.param_grid() returned empty list"


@pytest.mark.parametrize("GeneratorClass", ALL_GENERATORS)
def test_param_grid_dicts(GeneratorClass):
    gen = GeneratorClass()
    grid = gen.param_grid()
    for entry in grid:
        assert isinstance(entry, dict), f"Expected dict, got {type(entry)}"


# ──────────────────────────────────────────────────────────────────────────────
# generate() tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("GeneratorClass", ALL_GENERATORS)
def test_generate_returns_stimulus_pair(GeneratorClass):
    gen = GeneratorClass()
    params = gen.param_grid()[0]
    pair = gen.generate(params)
    assert isinstance(pair, StimulusPair)


@pytest.mark.parametrize("GeneratorClass", ALL_GENERATORS)
def test_generate_image_types(GeneratorClass):
    gen = GeneratorClass()
    params = gen.param_grid()[0]
    pair = gen.generate(params)
    assert isinstance(pair.illusion, Image.Image)
    assert isinstance(pair.control, Image.Image)


@pytest.mark.parametrize("GeneratorClass", ALL_GENERATORS)
def test_generate_image_mode_rgb(GeneratorClass):
    gen = GeneratorClass()
    params = gen.param_grid()[0]
    pair = gen.generate(params)
    assert pair.illusion.mode == "RGB", f"illusion mode={pair.illusion.mode}"
    assert pair.control.mode == "RGB", f"control mode={pair.control.mode}"


@pytest.mark.parametrize("GeneratorClass", ALL_GENERATORS)
def test_generate_image_sizes_match(GeneratorClass):
    gen = GeneratorClass()
    params = gen.param_grid()[0]
    pair = gen.generate(params)
    assert pair.illusion.size == pair.control.size, (
        f"illusion {pair.illusion.size} != control {pair.control.size}"
    )


@pytest.mark.parametrize("GeneratorClass", ALL_GENERATORS)
def test_generate_answer_labels(GeneratorClass):
    gen = GeneratorClass()
    params = gen.param_grid()[0]
    pair = gen.generate(params)
    assert pair.correct_answer == ANSWER_CORRECT
    assert pair.illusory_answer == ANSWER_ILLUSORY


@pytest.mark.parametrize("GeneratorClass", ALL_GENERATORS)
def test_generate_category_set(GeneratorClass):
    gen = GeneratorClass()
    params = gen.param_grid()[0]
    pair = gen.generate(params)
    assert pair.category != "", "category should not be empty"
    assert pair.illusion_type != "", "illusion_type should not be empty"


# ──────────────────────────────────────────────────────────────────────────────
# generate_dataset / manifest tests
# ──────────────────────────────────────────────────────────────────────────────

def test_generate_dataset_writes_files():
    gen = MullerLyerGenerator()
    # Only generate 2 stimuli to keep the test fast
    original_grid = gen.param_grid
    gen.param_grid = lambda: original_grid()[:2]

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        manifest = gen.generate_dataset(out)
        assert len(manifest) == 2
        manifest_file = out / "manifest.json"
        assert manifest_file.exists()
        for entry in manifest:
            assert Path(entry["illusion_path"]).exists()
            assert Path(entry["control_path"]).exists()


def test_stimulus_pair_save_load_roundtrip():
    gen = MullerLyerGenerator()
    pair = gen.generate(gen.param_grid()[0])
    pair.stimulus_id = "test_0000"

    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        meta = pair.save(d)
        restored = StimulusPair.load(meta)

    assert restored.stimulus_id == pair.stimulus_id
    assert restored.category == pair.category
    assert restored.illusion_type == pair.illusion_type
    assert restored.correct_answer == pair.correct_answer
    assert restored.illusory_answer == pair.illusory_answer
    assert restored.illusion.size == pair.illusion.size
    assert restored.control.size == pair.control.size


# ──────────────────────────────────────────────────────────────────────────────
# Illusion-specific sanity checks
# ──────────────────────────────────────────────────────────────────────────────

def test_muller_lyer_control_differs_from_illusion():
    """Control image should not be identical to illusion image (fins removed)."""
    import numpy as np
    gen = MullerLyerGenerator()
    pair = gen.generate({"fin_length": 40.0, "fin_angle_deg": 30.0})
    ill_arr = np.array(pair.illusion)
    ctrl_arr = np.array(pair.control)
    assert not np.array_equal(ill_arr, ctrl_arr), "Illusion and control should differ"


def test_simultaneous_contrast_patches_equal():
    """Both target patches in the illusion must be the same grey value."""
    import numpy as np
    gen = SimultaneousContrastGenerator(patch_size=40)
    pair = gen.generate({"dark_lum": 30, "bright_lum": 220})
    arr = np.array(pair.illusion)
    w, h = gen.image_size
    half_w = w // 2
    cy = h // 2
    ps = 40
    left_patch = arr[cy - ps // 2 : cy + ps // 2, half_w // 2 - ps // 2 : half_w // 2 + ps // 2]
    right_cx = half_w + half_w // 2
    right_patch = arr[cy - ps // 2 : cy + ps // 2, right_cx - ps // 2 : right_cx + ps // 2]
    assert np.allclose(left_patch.mean(), right_patch.mean(), atol=5), (
        "Both patches should have the same mean luminance"
    )
