# Real-Time ML Inference Server - Health Monitoring
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List


@dataclass
class RequestRecord:
    """Lightweight record of a completed request."""

    timestamp: float
    latency_ms: float
    model_name: str
    batch_size: int
    status: str  # "ok" or "error"


class MetricsCollector:
    """Collects latency, throughput, and error metrics with a sliding window."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self._window: float = window_seconds
        self._records: Deque[RequestRecord] = deque()
        self._lock: threading.Lock = threading.Lock()
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._started_at: float = time.time()

    def record(
        self,
        latency_ms: float,
        model_name: str,
        batch_size: int = 1,
        status: str = "ok",
    ) -> None:
        """Log a completed request."""

        rec = RequestRecord(
            timestamp=time.time(),
            latency_ms=latency_ms,
            model_name=model_name,
            batch_size=batch_size,
            status=status,
        )
        with self._lock:
            self._records.append(rec)
            self._total_requests += 1
            if status == "error":
                self._total_errors += 1
            self._prune()

    def snapshot(self) -> Dict:
        """Return current metrics over the sliding window."""

        with self._lock:
            self._prune()
            records = list(self._records)

        if not records:
            return {
                "window_seconds": self._window,
                "requests_in_window": 0,
                "throughput_rps": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0,
                "error_rate": 0.0,
                "total_requests": self._total_requests,
                "uptime_seconds": round(time.time() - self._started_at, 1),
            }

        latencies = sorted(r.latency_ms for r in records)
        n = len(latencies)
        window_duration = min(
            self._window,
            time.time() - records[0].timestamp if n > 1 else self._window,
        )
        throughput = n / window_duration if window_duration > 0 else 0.0

        return {
            "window_seconds": self._window,
            "requests_in_window": n,
            "throughput_rps": round(throughput, 2),
            "latency_p50_ms": round(latencies[int(n * 0.50)], 3),
            "latency_p95_ms": round(latencies[min(int(n * 0.95), n - 1)], 3),
            "latency_p99_ms": round(latencies[min(int(n * 0.99), n - 1)], 3),
            "error_rate": round(
                sum(1 for r in records if r.status == "error") / n, 4
            ),
            "total_requests": self._total_requests,
            "uptime_seconds": round(time.time() - self._started_at, 1),
        }

    def per_model(self) -> Dict[str, Dict]:
        """Break down window metrics by model."""

        with self._lock:
            self._prune()
            records = list(self._records)

        by_model: Dict[str, List[RequestRecord]] = {}
        for r in records:
            by_model.setdefault(r.model_name, []).append(r)

        result: Dict[str, Dict] = {}
        for name, recs in by_model.items():
            latencies = sorted(r.latency_ms for r in recs)
            n = len(latencies)
            result[name] = {
                "requests": n,
                "avg_latency_ms": round(sum(latencies) / n, 3),
                "p99_latency_ms": round(latencies[min(int(n * 0.99), n - 1)], 3),
            }
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        cutoff = time.time() - self._window
        while self._records and self._records[0].timestamp < cutoff:
            self._records.popleft()
