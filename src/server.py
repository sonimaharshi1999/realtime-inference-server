# Real-Time ML Inference Server - FastAPI Application
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request

from src.ab_testing import ABRouter
from src.batch_queue import BatchQueue
from src.cache import FeatureCache
from src.config import ServerConfig, default_config, ABTestConfig
from src.model_registry import ModelRegistry
from src.monitoring import MetricsCollector
from src.schemas import (
    HealthResponse,
    MetricsResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    ReloadResponse,
    WSPredictRequest,
    WSPredictResponse,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# State container
# ------------------------------------------------------------------

class AppState:
    """Holds all mutable server state in one object attached to app.state."""

    def __init__(self, config: ServerConfig) -> None:
        self.config: ServerConfig = config
        self.registry: ModelRegistry = ModelRegistry()
        self.cache: FeatureCache = FeatureCache(
            max_size=config.cache.max_size,
            ttl_seconds=config.cache.ttl_seconds,
        )
        self.metrics: MetricsCollector = MetricsCollector()
        self.batch_queue: Optional[BatchQueue] = None
        self.ab_router: Optional[ABRouter] = None


def init_state(state: AppState) -> None:
    """Synchronously load models and configure A/B routing.

    Called both during the ASGI lifespan *and* directly from tests
    (where ASGITransport does not fire lifespan events).
    """

    for model_cfg in state.config.models.values():
        try:
            state.registry.load(model_cfg)
        except Exception:
            logger.exception("Failed to load model %s", model_cfg.name)

    if state.config.ab_test is not None:
        state.ab_router = ABRouter(
            config=state.config.ab_test, registry=state.registry
        )


async def start_batch_queue(state: AppState) -> None:
    """Initialise and start the batch queue for the first configured model."""

    first_name = next(iter(state.config.models), "demo")
    first_model = state.registry.get(first_name)
    if first_model is not None:
        state.batch_queue = BatchQueue(
            model=first_model,
            max_batch_size=state.config.batch.max_batch_size,
            max_wait_ms=state.config.batch.max_wait_ms,
        )
        await state.batch_queue.start()


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------

def create_app(config: Optional[ServerConfig] = None) -> FastAPI:
    """Public factory used by the main module and tests."""

    cfg = config or default_config()
    app_state = AppState(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        init_state(app_state)
        await start_batch_queue(app_state)
        yield
        if app_state.batch_queue is not None:
            await app_state.batch_queue.stop()

    app = FastAPI(
        title="Real-Time ML Inference Server",
        version="1.0.0",
        lifespan=lifespan,
    )
    # Attach state so routes (and tests) can reach it.
    app.state.s = app_state
    _register_routes(app)
    return app


# ------------------------------------------------------------------
# Helper to get state from request
# ------------------------------------------------------------------

def _s(request_or_app: Any) -> AppState:
    """Resolve the AppState from a Request, WebSocket, or FastAPI instance."""

    if isinstance(request_or_app, FastAPI):
        return request_or_app.state.s  # type: ignore[return-value]
    return request_or_app.app.state.s  # type: ignore[return-value]


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

def _register_routes(app: FastAPI) -> None:
    """Attach all HTTP and WebSocket endpoints to *app*."""

    # Health --------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        st = _s(request)
        snap = st.metrics.snapshot()
        return HealthResponse(
            status="healthy",
            models_loaded=st.registry.list_models(),
            uptime_seconds=snap["uptime_seconds"],
        )

    # Metrics / dashboard ------------------------------------------

    @app.get("/metrics", response_model=MetricsResponse)
    async def get_metrics(request: Request) -> MetricsResponse:
        st = _s(request)
        snap = st.metrics.snapshot()
        return MetricsResponse(
            **snap,
            per_model=st.metrics.per_model(),
            cache_stats=st.cache.stats,
            ab_test=st.ab_router.report() if st.ab_router else None,
        )

    # Model info & hot-reload --------------------------------------

    @app.get("/models", response_model=list[ModelInfo])
    async def list_models(request: Request) -> list[ModelInfo]:
        st = _s(request)
        stats = st.registry.stats()
        return [
            ModelInfo(
                name=name,
                version=s["version"],
                backend=s["backend"],
                request_count=s["request_count"],
                avg_latency_ms=s["avg_latency_ms"],
            )
            for name, s in stats.items()
        ]

    @app.post("/models/reload", response_model=ReloadResponse)
    async def reload_models(request: Request) -> ReloadResponse:
        st = _s(request)
        reloaded = st.registry.check_for_updates()
        if reloaded:
            return ReloadResponse(
                reloaded=reloaded,
                message=f"Reloaded {len(reloaded)} model(s).",
            )
        return ReloadResponse(reloaded=[], message="All models up to date.")

    # REST prediction ----------------------------------------------

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest, request: Request) -> PredictResponse:
        st = _s(request)

        # Check cache
        if req.cache_key:
            cached_result = st.cache.get(req.cache_key)
            if cached_result is not None:
                return PredictResponse(
                    predictions=cached_result["predictions"],
                    model_name=cached_result["model_name"],
                    latency_ms=0.0,
                    cached=True,
                )

        X = np.array(req.features)
        start = time.perf_counter()

        # A/B routing
        if st.ab_router and req.model_name is None:
            model_name, preds = st.ab_router.route(X)
        else:
            model_name = req.model_name or next(iter(st.config.models), "demo")
            model = st.registry.get(model_name)
            if model is None:
                raise HTTPException(
                    status_code=404, detail=f"Model '{model_name}' not found"
                )
            preds = model.predict(X)

        latency_ms = (time.perf_counter() - start) * 1000
        predictions = preds.tolist()

        # Store in cache
        if req.cache_key:
            st.cache.put(
                req.cache_key,
                {"predictions": predictions, "model_name": model_name},
            )

        st.metrics.record(
            latency_ms=latency_ms,
            model_name=model_name,
            batch_size=len(req.features),
        )

        return PredictResponse(
            predictions=predictions,
            model_name=model_name,
            latency_ms=round(latency_ms, 3),
        )

    # WebSocket streaming ------------------------------------------

    @app.websocket("/ws/predict")
    async def ws_predict(ws: WebSocket) -> None:
        st = _s(ws)
        await ws.accept()
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    payload = WSPredictRequest.model_validate_json(raw)
                except Exception:
                    await ws.send_text(
                        json.dumps({"error": "Invalid JSON payload"})
                    )
                    continue

                X = np.array(payload.features).reshape(1, -1)
                start = time.perf_counter()

                # A/B routing
                if st.ab_router and payload.model_name is None:
                    model_name, preds = st.ab_router.route(X)
                else:
                    model_name = payload.model_name or next(
                        iter(st.config.models), "demo"
                    )
                    model = st.registry.get(model_name)
                    if model is None:
                        await ws.send_text(
                            json.dumps(
                                {"error": f"Model '{model_name}' not found"}
                            )
                        )
                        continue
                    preds = model.predict(X)

                latency_ms = (time.perf_counter() - start) * 1000

                st.metrics.record(
                    latency_ms=latency_ms,
                    model_name=model_name,
                    batch_size=1,
                )

                resp = WSPredictResponse(
                    predictions=preds.flatten().tolist(),
                    model_name=model_name,
                    latency_ms=round(latency_ms, 3),
                    request_id=payload.request_id,
                )
                await ws.send_text(resp.model_dump_json())
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")

    # A/B test management ------------------------------------------

    @app.post("/ab/configure")
    async def configure_ab(config: ABTestConfig, request: Request) -> dict:
        st = _s(request)
        st.ab_router = ABRouter(config=config, registry=st.registry)
        return {"message": "A/B test configured", "config": config.model_dump()}

    @app.get("/ab/report")
    async def ab_report(request: Request) -> dict:
        st = _s(request)
        if st.ab_router is None:
            return {"message": "No A/B test active"}
        return st.ab_router.report()

    # Cache management ---------------------------------------------

    @app.delete("/cache")
    async def clear_cache(request: Request) -> dict:
        st = _s(request)
        st.cache.clear()
        return {"message": "Cache cleared"}
