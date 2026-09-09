# Real-Time ML Inference Server - Monitoring Tests
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

from src.monitoring import MetricsCollector


class TestMetricsCollector:
    """Tests for the sliding-window metrics collector."""

    def test_empty_snapshot(self) -> None:
        """An empty collector should return zeroed metrics."""
        mc = MetricsCollector(window_seconds=5.0)
        snap = mc.snapshot()
        assert snap["requests_in_window"] == 0
        assert snap["throughput_rps"] == 0.0

    def test_record_and_snapshot(self) -> None:
        """Recording requests should be reflected in the snapshot."""
        mc = MetricsCollector(window_seconds=60.0)
        mc.record(latency_ms=10.0, model_name="m1", batch_size=1)
        mc.record(latency_ms=20.0, model_name="m1", batch_size=1)
        snap = mc.snapshot()
        assert snap["requests_in_window"] == 2
        assert snap["total_requests"] == 2
        assert snap["latency_p50_ms"] > 0

    def test_per_model_breakdown(self) -> None:
        """per_model should separate metrics by model name."""
        mc = MetricsCollector(window_seconds=60.0)
        mc.record(latency_ms=5.0, model_name="a")
        mc.record(latency_ms=15.0, model_name="b")
        breakdown = mc.per_model()
        assert "a" in breakdown
        assert "b" in breakdown
        assert breakdown["a"]["requests"] == 1

    def test_error_rate(self) -> None:
        """error_rate should reflect the proportion of error requests."""
        mc = MetricsCollector(window_seconds=60.0)
        mc.record(latency_ms=5.0, model_name="m1", status="ok")
        mc.record(latency_ms=5.0, model_name="m1", status="error")
        snap = mc.snapshot()
        assert snap["error_rate"] == 0.5
