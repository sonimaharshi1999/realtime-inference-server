# Real-Time ML Inference Server - Configuration
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field


BASE_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = BASE_DIR / "models"


class ModelConfig(BaseModel):
    """Configuration for a single model backend."""

    name: str = Field(..., description="Unique model identifier")
    path: str = Field(..., description="File path to the serialised model")
    backend: str = Field(
        default="sklearn", description="Backend type: sklearn or onnx"
    )
    version: str = Field(default="1.0.0", description="Semantic version tag")


class ABTestConfig(BaseModel):
    """A/B testing traffic-split configuration."""

    model_a: str = Field(..., description="Primary model name")
    model_b: str = Field(..., description="Challenger model name")
    traffic_split: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of traffic routed to model_a (rest goes to model_b)",
    )


class CacheConfig(BaseModel):
    """In-memory feature cache settings."""

    max_size: int = Field(default=10_000, description="Maximum cached entries")
    ttl_seconds: float = Field(
        default=300.0, description="Time-to-live per entry in seconds"
    )


class BatchConfig(BaseModel):
    """Batched inference settings."""

    max_batch_size: int = Field(default=32, description="Max items per batch")
    max_wait_ms: float = Field(
        default=50.0,
        description="Max milliseconds to wait before flushing a partial batch",
    )


class ServerConfig(BaseModel):
    """Top-level server configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    ab_test: Optional[ABTestConfig] = None
    cache: CacheConfig = Field(default_factory=CacheConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)


def default_config() -> ServerConfig:
    """Return a sensible default configuration with the bundled demo model."""

    default_model_path = str(MODELS_DIR / "demo_model.joblib")
    return ServerConfig(
        models={
            "demo": ModelConfig(
                name="demo",
                path=default_model_path,
                backend="sklearn",
                version="1.0.0",
            ),
        },
    )
