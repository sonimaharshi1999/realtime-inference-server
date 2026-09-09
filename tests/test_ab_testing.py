# Real-Time ML Inference Server - A/B Testing Tests
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import ABTestConfig, ModelConfig
from src.ab_testing import ABRouter
from src.model_registry import ModelRegistry


class TestABRouter:
    """Tests for the A/B traffic router."""

    def _make_router(
        self,
        demo_model_path: Path,
        challenger_model_path: Path,
        split: float = 0.5,
    ) -> ABRouter:
        reg = ModelRegistry()
        reg.load(ModelConfig(name="a", path=str(demo_model_path), backend="sklearn"))
        reg.load(ModelConfig(name="b", path=str(challenger_model_path), backend="sklearn"))
        cfg = ABTestConfig(model_a="a", model_b="b", traffic_split=split)
        return ABRouter(config=cfg, registry=reg)

    def test_route_returns_predictions(
        self, demo_model_path: Path, challenger_model_path: Path
    ) -> None:
        """route() should return a model name and predictions array."""
        router = self._make_router(demo_model_path, challenger_model_path)
        X = np.random.randn(3, 10)
        name, preds = router.route(X)
        assert name in ("a", "b")
        assert preds.shape == (3,)

    def test_traffic_split_bias(
        self, demo_model_path: Path, challenger_model_path: Path
    ) -> None:
        """With split=1.0 all traffic should go to model_a."""
        router = self._make_router(demo_model_path, challenger_model_path, split=1.0)
        X = np.random.randn(1, 10)
        names = set()
        for _ in range(20):
            name, _ = router.route(X)
            names.add(name)
        assert names == {"a"}

    def test_report_structure(
        self, demo_model_path: Path, challenger_model_path: Path
    ) -> None:
        """report() should contain both variant summaries."""
        router = self._make_router(demo_model_path, challenger_model_path)
        X = np.random.randn(1, 10)
        router.route(X)
        report = router.report()
        assert "model_a" in report
        assert "variants" in report
        assert "a" in report["variants"]
        assert "b" in report["variants"]
