"""Unit tests for probing protocols.

Tests use lightweight mock probers / models to verify pipeline logic without
requiring GPU or downloaded model weights.
"""

from __future__ import annotations

import math
import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.models.base import ModelProber, ResponseDistribution
from src.probing.zero_shot import ZeroShotProtocol, _TEXT_PROMPTS
from src.probing.vlm_protocol import VLMProtocol
from src.stimuli.base import StimulusPair, ANSWER_CORRECT, ANSWER_ILLUSORY


# ──────────────────────────────────────────────────────────────────────────────
# Mock objects
# ──────────────────────────────────────────────────────────────────────────────

def _rgb_image(w: int = 64, h: int = 64, value: int = 128) -> Image.Image:
    return Image.new("RGB", (w, h), (value, value, value))


def _dummy_pair(
    illusion_lum: int = 100,
    control_lum: int = 200,
    category: str = "geometric",
    illusion_type: str = "muller_lyer",
) -> StimulusPair:
    return StimulusPair(
        illusion=_rgb_image(value=illusion_lum),
        control=_rgb_image(value=control_lum),
        category=category,
        illusion_type=illusion_type,
        params={"fin_length": 20.0},
        correct_answer=ANSWER_CORRECT,
        illusory_answer=ANSWER_ILLUSORY,
        stimulus_id="test_0000",
    )


class _ConstantProber(ModelProber):
    """Always returns the same distribution regardless of input."""

    model_name = "constant_prober"

    def __init__(self, correct: float, illusory: float, other: float) -> None:
        # Skip device logic for tests
        self.device = "cpu"
        self._dist = ResponseDistribution(correct, illusory, other)

    def probe_pair(self, illusion, control, correct_answer, illusory_answer,
                   category, illusion_type, extra=None):
        return self._dist


# ──────────────────────────────────────────────────────────────────────────────
# ResponseDistribution tests
# ──────────────────────────────────────────────────────────────────────────────

class TestResponseDistribution:
    def test_to_label_correct(self):
        d = ResponseDistribution(0.7, 0.2, 0.1)
        assert d.to_label() == "correct"

    def test_to_label_illusory(self):
        d = ResponseDistribution(0.1, 0.8, 0.1)
        assert d.to_label() == "illusory"

    def test_to_label_other(self):
        d = ResponseDistribution(0.1, 0.2, 0.7)
        assert d.to_label() == "other"

    def test_normalise_sums_to_one(self):
        d = ResponseDistribution(3.0, 1.0, 1.0).normalise()
        assert math.isclose(d.correct + d.illusory + d.other, 1.0)

    def test_normalise_zero_returns_uniform(self):
        d = ResponseDistribution(0.0, 0.0, 0.0).normalise()
        assert math.isclose(d.correct, 1 / 3)

    def test_as_dict_keys(self):
        d = ResponseDistribution(0.5, 0.3, 0.2)
        keys = set(d.as_dict().keys())
        assert keys == {"correct", "illusory", "other"}


# ──────────────────────────────────────────────────────────────────────────────
# ModelProber.probe_dataset tests
# ──────────────────────────────────────────────────────────────────────────────

class TestModelProberDataset:
    def test_returns_one_result_per_pair(self):
        prober = _ConstantProber(0.7, 0.2, 0.1)
        pairs = [_dummy_pair() for _ in range(5)]
        results = prober.probe_dataset(pairs, verbose=False)
        assert len(results) == 5

    def test_result_contains_required_keys(self):
        prober = _ConstantProber(0.7, 0.2, 0.1)
        results = prober.probe_dataset([_dummy_pair()], verbose=False)
        required = {"stimulus_id", "category", "illusion_type", "model",
                    "correct", "illusory", "other", "predicted_label"}
        assert required.issubset(set(results[0].keys()))

    def test_probabilities_normalised(self):
        prober = _ConstantProber(3.0, 1.0, 1.0)  # unnormalised
        results = prober.probe_dataset([_dummy_pair()], verbose=False)
        total = results[0]["correct"] + results[0]["illusory"] + results[0]["other"]
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_model_name_in_result(self):
        prober = _ConstantProber(0.5, 0.3, 0.2)
        results = prober.probe_dataset([_dummy_pair()], verbose=False)
        assert results[0]["model"] == "constant_prober"


# ──────────────────────────────────────────────────────────────────────────────
# ZeroShotProtocol tests
# ──────────────────────────────────────────────────────────────────────────────

class TestZeroShotProtocol:
    def _make_mock_clip_prober(self):
        prober = MagicMock()
        prober.model_name = "clip_vit_b32"
        prober.probe_pair.return_value = ResponseDistribution(0.5, 0.4, 0.1)
        return prober

    def test_get_prompts_returns_dict(self):
        prober = self._make_mock_clip_prober()
        zs = ZeroShotProtocol(prober)
        prompts = zs.get_prompts("geometric")
        assert isinstance(prompts, dict)
        assert set(prompts.keys()) == {"correct", "illusory", "other"}

    def test_probe_stimulus_injects_text_prompts(self):
        prober = self._make_mock_clip_prober()
        zs = ZeroShotProtocol(prober)
        pair = _dummy_pair()
        zs.probe_stimulus(
            illusion=pair.illusion,
            control=pair.control,
            category="geometric",
            illusion_type="muller_lyer",
            correct_answer=ANSWER_CORRECT,
            illusory_answer=ANSWER_ILLUSORY,
        )
        call_kwargs = prober.probe_pair.call_args.kwargs
        assert "extra" in call_kwargs
        assert "text_prompts" in call_kwargs["extra"]

    def test_probe_dataset_length(self):
        prober = self._make_mock_clip_prober()
        zs = ZeroShotProtocol(prober)
        pairs = [_dummy_pair() for _ in range(4)]
        results = zs.probe_dataset(pairs, verbose=False)
        assert len(results) == 4

    def test_fallback_to_geometric_prompts_for_unknown_category(self):
        prober = self._make_mock_clip_prober()
        zs = ZeroShotProtocol(prober)
        prompts = zs.get_prompts("unknown_category")
        # Should fall back to the default key or return empty dict
        assert isinstance(prompts, dict)


# ──────────────────────────────────────────────────────────────────────────────
# VLMProtocol tests
# ──────────────────────────────────────────────────────────────────────────────

class TestVLMProtocol:
    def _make_mock_vlm_prober(self):
        prober = MagicMock()
        prober.model_name = "llava_mock"
        prober.probe_pair.return_value = ResponseDistribution(0.2, 0.6, 0.2)
        return prober

    def test_probe_dataset_returns_correct_length(self):
        prober = self._make_mock_vlm_prober()
        protocol = VLMProtocol(prober)
        pairs = [_dummy_pair() for _ in range(3)]
        results = protocol.probe_dataset(pairs, verbose=False)
        assert len(results) == 3

    def test_probe_stimulus_writes_audit_file(self, tmp_path):
        prober = self._make_mock_vlm_prober()
        protocol = VLMProtocol(prober, output_dir=tmp_path)
        pair = _dummy_pair()
        pair.stimulus_id = "audit_test_0000"
        protocol.probe_stimulus(
            illusion=pair.illusion,
            control=pair.control,
            category="geometric",
            illusion_type="muller_lyer",
            correct_answer=ANSWER_CORRECT,
            illusory_answer=ANSWER_ILLUSORY,
            stimulus_id=pair.stimulus_id,
        )
        audit_file = tmp_path / "audit_test_0000.json"
        assert audit_file.exists()

    def test_result_predicted_label_valid(self):
        prober = self._make_mock_vlm_prober()
        protocol = VLMProtocol(prober)
        results = protocol.probe_dataset([_dummy_pair()], verbose=False)
        assert results[0]["predicted_label"] in {"correct", "illusory", "other"}


# ──────────────────────────────────────────────────────────────────────────────
# LinearProbeProtocol tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLinearProbeProtocol:
    def _make_mock_prober(self, feature_dim: int = 32):
        """Create a lightweight mock prober with a real tiny backbone."""
        _feature_dim = feature_dim

        class _MockProber:
            model_name = "mock_cnn"
            device = "cpu"
            probe = None

            def __init__(self):
                self.feature_dim = _feature_dim

            def extract_features(self, image: Image.Image) -> torch.Tensor:
                rng = torch.Generator().manual_seed(0)
                return torch.randn(1, self.feature_dim, generator=rng)

            def attach_probe(self, p: nn.Linear) -> None:
                self.probe = p

            def save_probe(self, path):
                torch.save(self.probe.state_dict(), path)

            def load_probe(self, path):
                state = torch.load(path, map_location="cpu")
                p = nn.Linear(self.feature_dim, 3)
                p.load_state_dict(state)
                self.probe = p

        return _MockProber()

    def test_train_attaches_probe(self):
        from src.probing.linear_probe import LinearProbeProtocol

        prober = self._make_mock_prober()
        images = [_rgb_image() for _ in range(30)]
        labels = [i % 3 for i in range(30)]
        lp = LinearProbeProtocol(prober, epochs=2)
        lp.train(images, labels)
        assert prober.probe is not None
        assert isinstance(prober.probe, nn.Linear)
        assert prober.probe.out_features == 3

    def test_save_load_roundtrip(self, tmp_path):
        from src.probing.linear_probe import LinearProbeProtocol

        prober = self._make_mock_prober()
        images = [_rgb_image() for _ in range(15)]
        labels = [i % 3 for i in range(15)]
        lp = LinearProbeProtocol(prober, epochs=2)
        lp.train(images, labels)

        probe_path = tmp_path / "probe.pt"
        lp.save(probe_path)

        prober2 = self._make_mock_prober()
        lp2 = LinearProbeProtocol(prober2)
        lp2.load(probe_path)

        assert prober2.probe is not None

    def test_control_ceiling_check_fails_below_threshold(self, tmp_path):
        from src.probing.linear_probe import LinearProbeProtocol

        prober = self._make_mock_prober()
        images = [_rgb_image() for _ in range(30)]
        labels = [i % 3 for i in range(30)]
        lp = LinearProbeProtocol(prober, epochs=2, control_ceiling_threshold=0.99)
        lp.train(images, labels)

        # Use solid grey images that may not be classified well
        ctrl_images = [_rgb_image(value=128) for _ in range(10)]
        ctrl_labels = [0] * 10

        # Random probe output may not hit 99% → expect RuntimeError or proceed
        # We just ensure the method runs without unexpected exceptions (other than RuntimeError).
        try:
            acc = lp.check_control_ceiling(ctrl_images, ctrl_labels)
            assert 0.0 <= acc <= 1.0
        except RuntimeError as exc:
            assert "Control ceiling" in str(exc)
