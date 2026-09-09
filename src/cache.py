# Real-Time ML Inference Server - In-Memory Feature Cache
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import time
import threading
from collections import OrderedDict
from typing import Any, Optional, Tuple


class FeatureCache:
    """Thread-safe in-memory LRU cache with per-entry TTL.

    Eviction happens on access (lazy) and when the cache exceeds *max_size*.
    """

    def __init__(self, max_size: int = 10_000, ttl_seconds: float = 300.0) -> None:
        self._max_size: int = max_size
        self._ttl: float = ttl_seconds
        self._store: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._lock: threading.Lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value, returning ``None`` on miss or expiry."""

        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            ts, value = entry
            if time.monotonic() - ts > self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            # Move to end (most-recently used)
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        """Insert or update a cache entry."""

        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (time.monotonic(), value)
            self._evict_if_needed()

    def invalidate(self, key: str) -> bool:
        """Remove a single key. Returns ``True`` if the key existed."""

        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Drop all entries."""

        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                "current_size": len(self._store),
                "max_size": self._max_size,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Remove the oldest entries until we are within *max_size*."""

        while len(self._store) > self._max_size:
            self._store.popitem(last=False)
