# Real-Time ML Inference Server - Test Fixtures
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import sys
from pathlib import Path
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ABTestConfig,
    ModelConfig,
    ServerConfig,
    CacheConfig,
    BatchConfig,
    MODELS_DIR,
)
from src.train import train_demo_model, train_challenger_model


def _ensure_models() -> None:
    """Train demo models once before the test session."""
    demo = MODELS_DIR / "demo_model.joblib"
    if not demo.exists():
        train_demo_model()
    challenger = MODELS_DIR / "challenger_model.joblib"
    if not challenger.exists():
        train_challenger_model()


# Run model training at import time so every test has models available.
_ensure_models()


@pytest.fixture(scope="session")
def demo_model_path() -> Path:
    return MODELS_DIR / "demo_model.joblib"


@pytest.fixture(scope="session")
def challenger_model_path() -> Path:
    return MODELS_DIR / "challenger_model.joblib"


@pytest.fixture()
def server_config(demo_model_path: Path, challenger_model_path: Path) -> ServerConfig:
    return ServerConfig(
        models={
            "demo": ModelConfig(
                name="demo",
                path=str(demo_model_path),
                backend="sklearn",
                version="1.0.0",
            ),
            "challenger": ModelConfig(
                name="challenger",
                path=str(challenger_model_path),
                backend="sklearn",
                version="0.5.0",
            ),
        },
        ab_test=ABTestConfig(
            model_a="demo",
            model_b="challenger",
            traffic_split=0.5,
        ),
        cache=CacheConfig(max_size=100, ttl_seconds=10.0),
        batch=BatchConfig(max_batch_size=8, max_wait_ms=20.0),
    )


@pytest_asyncio.fixture()
async def client(server_config: ServerConfig) -> AsyncGenerator[AsyncClient, None]:
    from src.server import create_app, init_state

    app = create_app(server_config)

    # ASGITransport does not fire ASGI lifespan events, so we must
    # initialise models and A/B state manually for the test suite.
    init_state(app.state.s)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
