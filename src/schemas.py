# Real-Time ML Inference Server - Pydantic Request / Response Schemas
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# HTTP REST schemas
# ------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Single or batch prediction request via the REST API."""

    features: List[List[float]] = Field(
        ..., description="2-D feature matrix (rows = samples, cols = features)"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Explicit model name. When omitted, A/B routing or the default model is used.",
    )
    cache_key: Optional[str] = Field(
        default=None,
        description="Optional cache key for the feature set.",
    )


class PredictResponse(BaseModel):
    """Prediction results."""

    predictions: List[float]
    model_name: str
    latency_ms: float
    cached: bool = False


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str = "healthy"
    models_loaded: List[str] = Field(default_factory=list)
    uptime_seconds: float = 0.0


class MetricsResponse(BaseModel):
    """Monitoring dashboard metrics."""

    window_seconds: float
    requests_in_window: int
    throughput_rps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    error_rate: float
    total_requests: int
    uptime_seconds: float
    per_model: Dict[str, Any] = Field(default_factory=dict)
    cache_stats: Dict[str, Any] = Field(default_factory=dict)
    ab_test: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    """Metadata for a loaded model."""

    name: str
    version: str
    backend: str
    request_count: int
    avg_latency_ms: float


class ReloadResponse(BaseModel):
    """Response after a hot-reload check."""

    reloaded: List[str]
    message: str


# ------------------------------------------------------------------
# WebSocket schemas
# ------------------------------------------------------------------

class WSPredictRequest(BaseModel):
    """WebSocket prediction payload (JSON)."""

    features: List[float] = Field(
        ..., description="1-D feature vector for a single sample"
    )
    model_name: Optional[str] = None
    request_id: Optional[str] = None


class WSPredictResponse(BaseModel):
    """WebSocket prediction result (JSON)."""

    predictions: List[float]
    model_name: str
    latency_ms: float
    request_id: Optional[str] = None
