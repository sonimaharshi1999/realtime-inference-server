# Real-Time ML Inference Server - Request Queuing & Batched Inference
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from src.model_registry import LoadedModel

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """A single inference request waiting in the batch queue."""

    features: np.ndarray
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    enqueued_at: float = field(default_factory=time.perf_counter)


class BatchQueue:
    """Collects individual requests and flushes them as a single batch.

    The queue waits up to *max_wait_ms* for a full batch of *max_batch_size*
    items before invoking the model once.
    """

    def __init__(
        self,
        model: LoadedModel,
        max_batch_size: int = 32,
        max_wait_ms: float = 50.0,
    ) -> None:
        self._model: LoadedModel = model
        self._max_batch_size: int = max_batch_size
        self._max_wait_seconds: float = max_wait_ms / 1000.0
        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._batches_processed: int = 0
        self._total_items: int = 0

    async def start(self) -> None:
        """Start the background flush loop."""

        self._running = True
        self._task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Gracefully drain and stop the flush loop."""

        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Flush any remaining items
        await self._flush()

    async def submit(self, features: np.ndarray) -> np.ndarray:
        """Enqueue a request and await its result."""

        loop = asyncio.get_running_loop()
        req = InferenceRequest(
            features=features,
            future=loop.create_future(),
        )
        await self._queue.put(req)
        return await req.future

    @property
    def stats(self) -> dict:
        return {
            "batches_processed": self._batches_processed,
            "total_items": self._total_items,
            "pending": self._queue.qsize(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        while self._running:
            try:
                await asyncio.wait_for(
                    self._collect_and_flush(),
                    timeout=self._max_wait_seconds,
                )
            except asyncio.TimeoutError:
                await self._flush()
            except asyncio.CancelledError:
                break

    async def _collect_and_flush(self) -> None:
        """Wait for the first item, then collect up to max_batch_size."""

        first = await self._queue.get()
        batch: List[InferenceRequest] = [first]

        # Try to fill the batch without blocking
        while len(batch) < self._max_batch_size:
            try:
                item = self._queue.get_nowait()
                batch.append(item)
            except asyncio.QueueEmpty:
                break

        await self._run_batch(batch)

    async def _flush(self) -> None:
        """Flush whatever is in the queue right now."""

        batch: List[InferenceRequest] = []
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if batch:
            await self._run_batch(batch)

    async def _run_batch(self, batch: List[InferenceRequest]) -> None:
        if not batch:
            return

        X = np.vstack([r.features for r in batch])
        try:
            predictions = await asyncio.get_running_loop().run_in_executor(
                None, self._model.predict, X
            )
            for i, req in enumerate(batch):
                if not req.future.done():
                    req.future.set_result(predictions[i : i + 1])
        except Exception as exc:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)

        self._batches_processed += 1
        self._total_items += len(batch)
