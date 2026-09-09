# Real-Time ML Inference Server - Server Integration Tests
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import json
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient


@pytest.mark.anyio
class TestHealthEndpoint:
    """Tests for /health."""

    async def test_health_returns_200(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "demo" in body["models_loaded"]


@pytest.mark.anyio
class TestPredictEndpoint:
    """Tests for /predict REST endpoint."""

    async def test_predict_single_sample(self, client: AsyncClient) -> None:
        """POST /predict with one row should return one prediction."""
        payload = {"features": [[0.1] * 10]}
        resp = await client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["predictions"]) == 1
        assert body["latency_ms"] >= 0

    async def test_predict_batch(self, client: AsyncClient) -> None:
        """POST /predict with multiple rows should return matching count."""
        payload = {"features": [[float(i)] * 10 for i in range(5)]}
        resp = await client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 5

    async def test_predict_with_explicit_model(self, client: AsyncClient) -> None:
        """Specifying model_name should route to that model."""
        payload = {"features": [[0.5] * 10], "model_name": "demo"}
        resp = await client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["model_name"] == "demo"

    async def test_predict_unknown_model_returns_404(self, client: AsyncClient) -> None:
        """Requesting a nonexistent model should return 404."""
        payload = {"features": [[0.1] * 10], "model_name": "nonexistent"}
        resp = await client.post("/predict", json=payload)
        assert resp.status_code == 404

    async def test_predict_caching(self, client: AsyncClient) -> None:
        """Repeated requests with the same cache_key should be served from cache."""
        payload = {"features": [[0.2] * 10], "cache_key": "test-key-1"}
        resp1 = await client.post("/predict", json=payload)
        assert resp1.status_code == 200
        assert resp1.json()["cached"] is False

        resp2 = await client.post("/predict", json=payload)
        assert resp2.status_code == 200
        assert resp2.json()["cached"] is True
        assert resp2.json()["latency_ms"] == 0.0


@pytest.mark.anyio
class TestMetricsEndpoint:
    """Tests for /metrics."""

    async def test_metrics_returns_200(self, client: AsyncClient) -> None:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "throughput_rps" in body
        assert "cache_stats" in body


@pytest.mark.anyio
class TestModelsEndpoint:
    """Tests for /models and /models/reload."""

    async def test_list_models(self, client: AsyncClient) -> None:
        resp = await client.get("/models")
        assert resp.status_code == 200
        names = [m["name"] for m in resp.json()]
        assert "demo" in names

    async def test_reload_models(self, client: AsyncClient) -> None:
        resp = await client.post("/models/reload")
        assert resp.status_code == 200
        body = resp.json()
        assert "message" in body


@pytest.mark.anyio
class TestABTestEndpoints:
    """Tests for A/B test configuration and reporting."""

    async def test_ab_report(self, client: AsyncClient) -> None:
        resp = await client.get("/ab/report")
        assert resp.status_code == 200
        body = resp.json()
        assert "model_a" in body

    async def test_configure_ab(self, client: AsyncClient) -> None:
        payload = {
            "model_a": "demo",
            "model_b": "challenger",
            "traffic_split": 0.7,
        }
        resp = await client.post("/ab/configure", json=payload)
        assert resp.status_code == 200
        assert resp.json()["config"]["traffic_split"] == 0.7


@pytest.mark.anyio
class TestCacheEndpoint:
    """Tests for /cache management."""

    async def test_clear_cache(self, client: AsyncClient) -> None:
        resp = await client.delete("/cache")
        assert resp.status_code == 200
        assert resp.json()["message"] == "Cache cleared"
