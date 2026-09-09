# Real-Time ML Inference Server - Cache Tests
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import time

from src.cache import FeatureCache


class TestFeatureCache:
    """Tests for the in-memory LRU feature cache."""

    def test_put_and_get(self) -> None:
        """Storing a value and retrieving it should return the same object."""
        cache = FeatureCache(max_size=10, ttl_seconds=60)
        cache.put("k1", [1.0, 2.0, 3.0])
        assert cache.get("k1") == [1.0, 2.0, 3.0]

    def test_miss_returns_none(self) -> None:
        """A cache miss should return None."""
        cache = FeatureCache(max_size=10, ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self) -> None:
        """Entries older than TTL should be treated as misses."""
        cache = FeatureCache(max_size=10, ttl_seconds=0.05)
        cache.put("k1", "value")
        time.sleep(0.1)
        assert cache.get("k1") is None

    def test_lru_eviction(self) -> None:
        """When the cache is full, the oldest entry should be evicted."""
        cache = FeatureCache(max_size=3, ttl_seconds=60)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("d") == 4

    def test_invalidate(self) -> None:
        """Invalidating a key should remove it from the cache."""
        cache = FeatureCache(max_size=10, ttl_seconds=60)
        cache.put("k1", "val")
        assert cache.invalidate("k1") is True
        assert cache.get("k1") is None
        assert cache.invalidate("k1") is False

    def test_clear(self) -> None:
        """Clearing the cache should remove all entries."""
        cache = FeatureCache(max_size=10, ttl_seconds=60)
        for i in range(5):
            cache.put(f"k{i}", i)
        cache.clear()
        assert cache.size == 0

    def test_stats_tracking(self) -> None:
        """Hit/miss counters should update correctly."""
        cache = FeatureCache(max_size=10, ttl_seconds=60)
        cache.put("k1", "v")
        cache.get("k1")  # hit
        cache.get("k2")  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
