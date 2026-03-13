# Copyright (c) Microsoft. All rights reserved.

"""Simplified in-memory store for mini-agent-lightning.

The full library ships in-memory, SQLite, and MongoDB backends together with
complex eviction logic and HTTP server bridging.  The mini store keeps only
what the core training loop needs: a FIFO task queue, per-rollout span
storage, and a shared resource snapshot.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .types import (
    Attempt,
    AttemptedRollout,
    AttemptStatus,
    NamedResources,
    Rollout,
    RolloutMode,
    RolloutStatus,
    Span,
    TaskInput,
)

__all__ = ["InMemoryStore"]

logger = logging.getLogger(__name__)


class InMemoryStore:
    """Thread-safe, async-compatible in-memory store.

    Responsibilities:
    - Accept rollouts enqueued by algorithms.
    - Hand out rollouts to runners via ``claim_rollout``.
    - Buffer spans emitted by runners.
    - Store a single *latest* named-resource snapshot for runners to fetch.

    Unlike the production ``InMemoryLightningStore`` this class does **not**:
    - Evict spans when memory is low.
    - Manage retries or configurable timeouts.
    - Support OTLP/HTTP span ingestion.
    - Bridge to SQLite or MongoDB.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

        # rollout_id -> Rollout
        self._rollouts: Dict[str, Rollout] = {}
        # rollout_id -> [Attempt, ...]
        self._attempts: Dict[str, List[Attempt]] = {}
        # rollout_id -> [Span, ...]
        self._spans: Dict[str, List[Span]] = {}

        # FIFO queue of rollout_ids waiting to be claimed
        self._queue: asyncio.Queue[str] = asyncio.Queue()

        # Latest named-resource snapshot shared with runners
        self._resources: Optional[NamedResources] = None

        # completion events keyed by rollout_id
        self._done_events: Dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    async def set_resources(self, resources: NamedResources) -> None:
        """Publish a new named-resource snapshot for runners to fetch.

        Args:
            resources: Key-value mapping (prompts, model paths, …).
        """
        async with self._lock:
            self._resources = dict(resources)
        logger.debug("Resources updated: keys=%s", list(resources.keys()))

    async def get_resources(self) -> Optional[NamedResources]:
        """Return the latest named-resource snapshot, or ``None`` when none has been set.

        Returns:
            A shallow copy of the current resource mapping, or ``None``.
        """
        async with self._lock:
            return dict(self._resources) if self._resources is not None else None

    # ------------------------------------------------------------------
    # Rollout lifecycle
    # ------------------------------------------------------------------

    async def enqueue_rollout(
        self,
        input: TaskInput,
        *,
        mode: Optional[RolloutMode] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Rollout:
        """Create a rollout and add it to the execution queue.

        Args:
            input: Task payload that the runner will pass to the agent.
            mode: Optional semantic mode (``"train"``, ``"val"``, or ``"test"``).
            metadata: Arbitrary key-value pairs persisted with the rollout.

        Returns:
            The newly created :class:`Rollout`.
        """
        resources = await self.get_resources()
        rollout = Rollout(
            input=input,
            mode=mode,
            metadata=metadata,
            resources=resources,
        )
        async with self._lock:
            self._rollouts[rollout.rollout_id] = rollout
            self._spans[rollout.rollout_id] = []
            self._attempts[rollout.rollout_id] = []
            self._done_events[rollout.rollout_id] = asyncio.Event()
        await self._queue.put(rollout.rollout_id)
        logger.debug("Enqueued rollout %s (mode=%s)", rollout.rollout_id, mode)
        return rollout

    async def claim_rollout(self, *, worker_id: Optional[str] = None, timeout: float = 1.0) -> Optional[AttemptedRollout]:
        """Claim the next queued rollout and create its first attempt.

        Args:
            worker_id: Optional identifier of the claiming worker.
            timeout: Seconds to wait for a queued rollout before returning ``None``.

        Returns:
            An :class:`AttemptedRollout` or ``None`` when the queue is empty.
        """
        try:
            rollout_id = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

        async with self._lock:
            rollout = self._rollouts.get(rollout_id)
            if rollout is None:
                logger.error("Claimed rollout_id=%s not found in store", rollout_id)
                return None

            attempt = Attempt(rollout_id=rollout_id, worker_id=worker_id)
            self._attempts[rollout_id].append(attempt)

            rollout.status = "running"
            rollout.resources = dict(self._resources) if self._resources is not None else None

        logger.debug("Worker %s claimed rollout %s", worker_id, rollout_id)
        return AttemptedRollout(**rollout.model_dump(), attempt=attempt)

    async def finish_rollout(
        self,
        rollout_id: str,
        attempt_id: str,
        *,
        status: RolloutStatus = "succeeded",
        attempt_status: AttemptStatus = "succeeded",
    ) -> None:
        """Mark a rollout (and its active attempt) as finished.

        Args:
            rollout_id: Identifier of the rollout to close.
            attempt_id: Identifier of the attempt to close.
            status: Terminal rollout status (``"succeeded"`` or ``"failed"``).
            attempt_status: Terminal attempt status.
        """
        async with self._lock:
            rollout = self._rollouts.get(rollout_id)
            if rollout:
                rollout.status = status
                rollout.end_time = time.time()

            for attempt in self._attempts.get(rollout_id, []):
                if attempt.attempt_id == attempt_id:
                    attempt.status = attempt_status
                    attempt.end_time = time.time()
                    break

            event = self._done_events.get(rollout_id)
            if event:
                event.set()
        logger.debug("Finished rollout %s with status=%s", rollout_id, status)

    # ------------------------------------------------------------------
    # Span ingestion & querying
    # ------------------------------------------------------------------

    async def add_spans(self, rollout_id: str, spans: List[Span]) -> None:
        """Append spans to the per-rollout buffer.

        Args:
            rollout_id: Rollout the spans belong to.
            spans: OpenTelemetry ``ReadableSpan`` objects to store.
        """
        async with self._lock:
            bucket = self._spans.setdefault(rollout_id, [])
            bucket.extend(spans)
        logger.debug("Added %d span(s) to rollout %s", len(spans), rollout_id)

    async def query_spans(self, rollout_id: str) -> List[Span]:
        """Return all spans recorded for a rollout.

        Args:
            rollout_id: Identifier of the rollout to query.

        Returns:
            List of :class:`Span` objects in insertion order.
        """
        async with self._lock:
            return list(self._spans.get(rollout_id, []))

    # ------------------------------------------------------------------
    # Rollout querying
    # ------------------------------------------------------------------

    async def get_rollout(self, rollout_id: str) -> Optional[Rollout]:
        """Return the rollout with the given identifier.

        Args:
            rollout_id: Identifier of the rollout to fetch.

        Returns:
            The :class:`Rollout`, or ``None`` when not found.
        """
        async with self._lock:
            return self._rollouts.get(rollout_id)

    async def list_rollouts(self, *, status: Optional[RolloutStatus] = None) -> List[Rollout]:
        """Return all rollouts, optionally filtered by status.

        Args:
            status: When provided only rollouts with this status are returned.

        Returns:
            List of :class:`Rollout` objects.
        """
        async with self._lock:
            rollouts = list(self._rollouts.values())
        if status is not None:
            rollouts = [r for r in rollouts if r.status == status]
        return rollouts

    async def wait_for_rollout(self, rollout_id: str, *, timeout: Optional[float] = None) -> Optional[Rollout]:
        """Block until the rollout reaches a terminal state.

        Args:
            rollout_id: Identifier of the rollout to wait for.
            timeout: Maximum seconds to wait; ``None`` means wait indefinitely.

        Returns:
            The finished :class:`Rollout`, or ``None`` when the timeout expires.
        """
        async with self._lock:
            event = self._done_events.get(rollout_id)

        if event is None:
            return None

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

        return await self.get_rollout(rollout_id)
