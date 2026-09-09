# Real-Time ML Inference Server - Model Registry Tests
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import ModelConfig
from src.model_registry import ModelRegistry


class TestModelRegistry:
    """Tests for model loading, listing, and hot-reload detection."""

    def test_load_and_predict(self, demo_model_path: Path) -> None:
        """Loading a sklearn model and running predict should return an array."""
        reg = ModelRegistry()
        cfg = ModelConfig(name="test", path=str(demo_model_path), backend="sklearn")
        reg.load(cfg)

        model = reg.get("test")
        assert model is not None

        X = np.random.randn(5, 10)
        preds = model.predict(X)
        assert preds.shape == (5,)

    def test_list_models(self, demo_model_path: Path) -> None:
        """list_models should return names of all loaded models."""
        reg = ModelRegistry()
        cfg = ModelConfig(name="m1", path=str(demo_model_path), backend="sklearn")
        reg.load(cfg)
        assert "m1" in reg.list_models()

    def test_unload(self, demo_model_path: Path) -> None:
        """Unloading a model should remove it from the registry."""
        reg = ModelRegistry()
        cfg = ModelConfig(name="m1", path=str(demo_model_path), backend="sklearn")
        reg.load(cfg)
        assert reg.unload("m1") is True
        assert reg.get("m1") is None
        assert reg.unload("m1") is False

    def test_hot_reload_no_change(self, demo_model_path: Path) -> None:
        """check_for_updates should return empty when no file changed."""
        reg = ModelRegistry()
        cfg = ModelConfig(name="m1", path=str(demo_model_path), backend="sklearn")
        reg.load(cfg)
        reloaded = reg.check_for_updates()
        assert reloaded == []

    def test_stats(self, demo_model_path: Path) -> None:
        """stats should include request counts and latency info."""
        reg = ModelRegistry()
        cfg = ModelConfig(name="m1", path=str(demo_model_path), backend="sklearn")
        reg.load(cfg)

        model = reg.get("m1")
        assert model is not None
        model.predict(np.random.randn(1, 10))

        s = reg.stats()
        assert s["m1"]["request_count"] == 1
        assert s["m1"]["avg_latency_ms"] >= 0
