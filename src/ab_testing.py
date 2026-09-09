# Real-Time ML Inference Server - A/B Testing Router
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from src.config import ABTestConfig
from src.model_registry import LoadedModel, ModelRegistry


@dataclass
class ABMetrics:
    """Per-variant running metrics for an A/B experiment."""

    requests: int = 0
    total_latency_ms: float = 0.0
    predictions: list = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        if self.requests == 0:
            return 0.0
        return round(self.total_latency_ms / self.requests, 3)


class ABRouter:
    """Routes inference requests between two models according to a traffic split."""

    def __init__(
        self,
        config: ABTestConfig,
        registry: ModelRegistry,
    ) -> None:
        self._config: ABTestConfig = config
        self._registry: ModelRegistry = registry
        self._metrics: Dict[str, ABMetrics] = {
            config.model_a: ABMetrics(),
            config.model_b: ABMetrics(),
        }
        self._started_at: float = time.time()

    @property
    def config(self) -> ABTestConfig:
        return self._config

    def update_split(self, split: float) -> None:
        """Dynamically adjust the traffic split (0.0 - 1.0 toward model_a)."""

        self._config = self._config.model_copy(update={"traffic_split": split})

    def route(self, X: np.ndarray) -> Tuple[str, np.ndarray]:
        """Pick a model variant and return (model_name, predictions)."""

        if random.random() < self._config.traffic_split:
            chosen = self._config.model_a
        else:
            chosen = self._config.model_b

        model: Optional[LoadedModel] = self._registry.get(chosen)
        if model is None:
            raise ValueError(f"Model '{chosen}' not found in registry")

        start = time.perf_counter()
        preds = model.predict(X)
        elapsed_ms = (time.perf_counter() - start) * 1000

        metrics = self._metrics[chosen]
        metrics.requests += 1
        metrics.total_latency_ms += elapsed_ms

        return chosen, preds

    def report(self) -> Dict:
        """Return a summary of the running experiment."""

        return {
            "model_a": self._config.model_a,
            "model_b": self._config.model_b,
            "traffic_split": self._config.traffic_split,
            "duration_seconds": round(time.time() - self._started_at, 1),
            "variants": {
                name: {
                    "requests": m.requests,
                    "avg_latency_ms": m.avg_latency_ms,
                }
                for name, m in self._metrics.items()
            },
        }
