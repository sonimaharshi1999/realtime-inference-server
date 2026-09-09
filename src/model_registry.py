# Real-Time ML Inference Server - Model Registry & Hot-Reload
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import joblib
import numpy as np

from src.config import ModelConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Backend protocol
# ------------------------------------------------------------------

class ModelBackend(Protocol):
    """Minimal predict interface every backend must satisfy."""

    def predict(self, X: np.ndarray) -> np.ndarray: ...


# ------------------------------------------------------------------
# sklearn wrapper
# ------------------------------------------------------------------

class SklearnBackend:
    """Wraps a scikit-learn estimator loaded from a joblib file."""

    def __init__(self, model: Any) -> None:
        self._model: Any = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


# ------------------------------------------------------------------
# ONNX wrapper (optional dependency)
# ------------------------------------------------------------------

class OnnxBackend:
    """Wraps an ONNX Runtime InferenceSession."""

    def __init__(self, session: Any) -> None:
        self._session: Any = session
        self._input_name: str = session.get_inputs()[0].name

    def predict(self, X: np.ndarray) -> np.ndarray:
        result = self._session.run(None, {self._input_name: X.astype(np.float32)})
        return np.array(result[0])


# ------------------------------------------------------------------
# Loaded model handle
# ------------------------------------------------------------------

class LoadedModel:
    """Bundles a backend with its config and a file checksum for hot-reload."""

    def __init__(
        self,
        config: ModelConfig,
        backend: ModelBackend,
        file_hash: str,
    ) -> None:
        self.config: ModelConfig = config
        self.backend: ModelBackend = backend
        self.file_hash: str = file_hash
        self.loaded_at: float = time.time()
        self.request_count: int = 0
        self.total_latency_ms: float = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        result = self.backend.predict(X)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.request_count += 1
        self.total_latency_ms += elapsed_ms
        return result

    @property
    def avg_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return round(self.total_latency_ms / self.request_count, 3)


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_backend(config: ModelConfig) -> ModelBackend:
    if config.backend == "sklearn":
        model = joblib.load(config.path)
        return SklearnBackend(model)
    elif config.backend == "onnx":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNX backends. "
                "Install it with: pip install onnxruntime"
            ) from exc
        session = ort.InferenceSession(config.path)
        return OnnxBackend(session)
    else:
        raise ValueError(f"Unsupported backend: {config.backend}")


class ModelRegistry:
    """Manages model loading, lookup, and hot-reloading."""

    def __init__(self) -> None:
        self._models: Dict[str, LoadedModel] = {}
        self._lock: threading.Lock = threading.Lock()

    def load(self, config: ModelConfig) -> LoadedModel:
        """Load (or reload) a model from disk."""

        backend = _load_backend(config)
        fhash = _file_hash(config.path)
        loaded = LoadedModel(config=config, backend=backend, file_hash=fhash)
        with self._lock:
            self._models[config.name] = loaded
        logger.info("Loaded model %s (v%s) [%s]", config.name, config.version, fhash[:8])
        return loaded

    def get(self, name: str) -> Optional[LoadedModel]:
        with self._lock:
            return self._models.get(name)

    def list_models(self) -> List[str]:
        with self._lock:
            return list(self._models.keys())

    def unload(self, name: str) -> bool:
        with self._lock:
            if name in self._models:
                del self._models[name]
                logger.info("Unloaded model %s", name)
                return True
            return False

    def check_for_updates(self) -> List[str]:
        """Compare file hashes and reload models whose files changed on disk.

        Returns the list of model names that were reloaded.
        """

        reloaded: List[str] = []
        with self._lock:
            snapshot = list(self._models.items())

        for name, loaded in snapshot:
            path = loaded.config.path
            if not os.path.exists(path):
                continue
            current_hash = _file_hash(path)
            if current_hash != loaded.file_hash:
                logger.info("Detected change in %s, hot-reloading...", name)
                self.load(loaded.config)
                reloaded.append(name)

        return reloaded

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                name: {
                    "version": m.config.version,
                    "backend": m.config.backend,
                    "request_count": m.request_count,
                    "avg_latency_ms": m.avg_latency_ms,
                    "loaded_at": m.loaded_at,
                    "file_hash": m.file_hash[:8],
                }
                for name, m in self._models.items()
            }
