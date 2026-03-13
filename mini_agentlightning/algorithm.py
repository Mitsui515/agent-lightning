# Copyright (c) Microsoft. All rights reserved.

"""Algorithm base class for mini-agent-lightning.

The full library ships ``Algorithm``, ``FastAlgorithm``, ``Baseline``,
``APO``, and VERL integrations.  The mini algorithm module provides only the
abstract base that custom algorithms extend, together with lightweight
accessors for the store and resource management.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .store import InMemoryStore
from .types import NamedResources, Rollout, Span, Triplet

if TYPE_CHECKING:
    from .trainer import MiniTrainer

__all__ = ["Algorithm"]

logger = logging.getLogger(__name__)


class Algorithm:
    """Base class for training algorithms.

    Subclasses override :meth:`run` (sync) or :meth:`run_async` (async) to
    implement a learning strategy.  The algorithm is given access to the
    :class:`~mini_agentlightning.store.InMemoryStore` so it can:

    - Enqueue rollout tasks with :meth:`enqueue`.
    - Wait for rollouts to complete with :meth:`wait_for_rollout`.
    - Query collected spans with :meth:`query_spans`.
    - Publish updated resources (e.g. revised prompts) with :meth:`set_resources`.

    Example::

        class MyAlgorithm(Algorithm):
            async def run_async(self) -> None:
                for task in self.dataset:
                    await self.enqueue(task)
                rollouts = await self.wait_all()
                for r in rollouts:
                    spans = await self.query_spans(r.rollout_id)
                    reward = find_final_reward(spans)
                    self.update(reward)
    """

    def __init__(self) -> None:
        self._store: Optional[InMemoryStore] = None
        self._trainer_ref: Optional[weakref.ReferenceType[MiniTrainer]] = None
        # Track rollout IDs enqueued in this run for convenience.
        self._enqueued_ids: List[str] = []

    # ------------------------------------------------------------------
    # Framework-internal wiring (called by MiniTrainer)
    # ------------------------------------------------------------------

    def _attach(self, store: InMemoryStore, trainer: MiniTrainer) -> None:
        self._store = store
        self._trainer_ref = weakref.ref(trainer)

    # ------------------------------------------------------------------
    # Public helpers for subclasses
    # ------------------------------------------------------------------

    def get_store(self) -> InMemoryStore:
        """Return the :class:`~mini_agentlightning.store.InMemoryStore` managed by the trainer.

        Returns:
            The shared :class:`~mini_agentlightning.store.InMemoryStore`.

        Raises:
            RuntimeError: When the algorithm has not been attached to a trainer.
        """
        if self._store is None:
            raise RuntimeError("Algorithm not attached to a trainer. Call via MiniTrainer.fit().")
        return self._store

    async def enqueue(self, task: Any, *, mode: str = "train") -> Rollout:
        """Enqueue a single task for execution.

        Args:
            task: Arbitrary task payload forwarded to the agent as ``rollout.input``.
            mode: Rollout mode (``"train"``, ``"val"``, or ``"test"``).

        Returns:
            The created :class:`~mini_agentlightning.types.Rollout`.
        """
        rollout = await self.get_store().enqueue_rollout(
            task,
            mode=mode,  # type: ignore[arg-type]
        )
        self._enqueued_ids.append(rollout.rollout_id)
        return rollout

    async def wait_for_rollout(self, rollout_id: str, *, timeout: Optional[float] = None) -> Optional[Rollout]:
        """Block until a rollout reaches a terminal state.

        Args:
            rollout_id: Identifier of the rollout to wait for.
            timeout: Maximum seconds to wait; ``None`` means indefinite.

        Returns:
            The finished :class:`~mini_agentlightning.types.Rollout` or ``None``
            when the timeout expires.
        """
        return await self.get_store().wait_for_rollout(rollout_id, timeout=timeout)

    async def wait_all(self, *, timeout: Optional[float] = None) -> List[Rollout]:
        """Wait for all rollouts enqueued during the current run to finish.

        Args:
            timeout: Per-rollout timeout in seconds; ``None`` means indefinite.

        Returns:
            List of finished :class:`~mini_agentlightning.types.Rollout` objects
            in the order they were enqueued.
        """
        results: List[Rollout] = []
        for rid in self._enqueued_ids:
            rollout = await self.wait_for_rollout(rid, timeout=timeout)
            if rollout is not None:
                results.append(rollout)
        return results

    async def query_spans(self, rollout_id: str) -> List[Span]:
        """Return all spans recorded for a rollout.

        Args:
            rollout_id: Identifier of the rollout to query.

        Returns:
            List of :class:`~mini_agentlightning.types.Span` objects.
        """
        return await self.get_store().query_spans(rollout_id)

    async def set_resources(self, resources: NamedResources) -> None:
        """Publish an updated named-resource snapshot for runners to use.

        The resources will be attached to all *subsequently* enqueued rollouts.

        Args:
            resources: Key-value mapping (prompts, model paths, …).
        """
        await self.get_store().set_resources(resources)

    # ------------------------------------------------------------------
    # Override one of these
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Synchronous training entry point.

        Override this method to implement a synchronous algorithm.  When both
        ``run`` and ``run_async`` are overridden, ``run_async`` takes precedence.
        """
        raise NotImplementedError("Override run() or run_async().")

    async def run_async(self) -> None:
        """Asynchronous training entry point.

        Override this for async-native algorithms (recommended).
        """
        await asyncio.to_thread(self.run)
