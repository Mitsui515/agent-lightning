# Copyright (c) Microsoft. All rights reserved.

"""Agent base class and runner for mini-agent-lightning.

The full library ships ``LitAgent``, ``LitAgentRunner``, and multiple
execution strategies (shared memory, client/server, inter-process).
The mini agent module collapses these into two focused classes:

- :class:`MiniAgent` – base class users extend to implement task execution.
- :class:`MiniRunner` – polls the store for work and drives ``MiniAgent``.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

from .store import InMemoryStore
from .tracer import MiniTracer
from .types import AttemptedRollout, NamedResources

if TYPE_CHECKING:
    from .trainer import MiniTrainer

__all__ = ["MiniAgent", "MiniRunner"]

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MiniAgent(Generic[T]):
    """Base class for implementing agent rollouts.

    Users subclass ``MiniAgent`` and override :meth:`rollout` (or
    :meth:`rollout_async` for async code) to define the agent's behaviour for
    a single task.

    Example::

        class MyAgent(MiniAgent[dict]):
            async def rollout_async(self, task: dict) -> None:
                result = await call_llm(task["prompt"])
                emit_reward(score(result))
    """

    def __init__(self) -> None:
        self._trainer_ref: weakref.ReferenceType[MiniTrainer] | None = None
        self._runner_ref: weakref.ReferenceType[MiniRunner[Any]] | None = None

    # ------------------------------------------------------------------
    # Override one of these
    # ------------------------------------------------------------------

    def rollout(self, task: T) -> None:
        """Process a single task synchronously.

        Override this method to implement synchronous agent logic.  The runner
        calls this inside an :meth:`asyncio.to_thread` wrapper so blocking I/O
        does not stall the event loop.

        Args:
            task: Task payload provided by the algorithm via the store.
        """
        raise NotImplementedError("Override rollout() or rollout_async().")

    async def rollout_async(self, task: T) -> None:
        """Process a single task asynchronously.

        Override this for async-native agent code.  When both methods are
        overridden, ``rollout_async`` takes precedence.

        Args:
            task: Task payload provided by the algorithm via the store.
        """
        # Default: run the sync version in a thread pool.
        await asyncio.to_thread(self.rollout, task)

    # ------------------------------------------------------------------
    # Convenience accessors set by the runner
    # ------------------------------------------------------------------

    def get_resources(self) -> Optional[NamedResources]:
        """Return the latest resource snapshot from the trainer's store.

        Returns:
            The named-resource mapping or ``None`` when no resources have been
            published yet.
        """
        runner = self._runner_ref() if self._runner_ref else None
        if runner is None:
            return None
        return runner.current_resources

    def get_trainer(self) -> MiniTrainer:
        """Return the :class:`~mini_agentlightning.trainer.MiniTrainer` that manages this agent.

        Returns:
            The :class:`~mini_agentlightning.trainer.MiniTrainer` instance.

        Raises:
            ValueError: When the trainer reference is not set or has been garbage-collected.
        """
        if self._trainer_ref is None:
            raise ValueError("Trainer not set. Is the agent attached to a MiniTrainer?")
        trainer = self._trainer_ref()
        if trainer is None:
            raise ValueError("Trainer has been garbage-collected.")
        return trainer


class MiniRunner(Generic[T]):
    """Polls an :class:`~mini_agentlightning.store.InMemoryStore` for work and drives a :class:`MiniAgent`.

    The runner is intentionally simple: it runs a polling loop inside an
    asyncio task, claims rollouts one-by-one, executes the agent inside a
    ``MiniTracer.trace_context``, writes resulting spans back to the store, and
    marks the rollout as finished.

    Args:
        tracer: :class:`~mini_agentlightning.tracer.MiniTracer` used to capture spans.
        poll_interval: Seconds between store polls when the queue is empty.
        worker_id: Human-readable identifier for this worker.
    """

    def __init__(
        self,
        tracer: MiniTracer,
        poll_interval: float = 0.5,
        worker_id: str = "worker-0",
    ) -> None:
        self._tracer = tracer
        self._poll_interval = poll_interval
        self._worker_id = worker_id

        self._agent: Optional[MiniAgent[T]] = None
        self._store: Optional[InMemoryStore] = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self.current_resources: Optional[NamedResources] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self, agent: MiniAgent[T], store: InMemoryStore) -> None:
        """Attach the agent and store to this runner.

        Args:
            agent: :class:`MiniAgent` instance to execute.
            store: :class:`~mini_agentlightning.store.InMemoryStore` to poll for work.
        """
        self._agent = agent
        self._agent._runner_ref = weakref.ref(self)
        self._store = store
        self._tracer.setup()

    def stop(self) -> None:
        """Signal the runner to exit after the current rollout finishes."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, max_rollouts: Optional[int] = None) -> None:
        """Run the polling loop until stopped or ``max_rollouts`` is reached.

        Args:
            max_rollouts: Optional cap on the number of rollouts to process.
                When ``None`` the loop runs until :meth:`stop` is called.
        """
        if self._agent is None or self._store is None:
            raise RuntimeError("Runner not initialised. Call init() first.")

        processed = 0
        while not self._stop_event.is_set():
            if max_rollouts is not None and processed >= max_rollouts:
                logger.debug("Worker %s reached max_rollouts=%d, stopping.", self._worker_id, max_rollouts)
                break

            rollout = await self._store.claim_rollout(
                worker_id=self._worker_id,
                timeout=self._poll_interval,
            )
            if rollout is None:
                continue  # nothing in the queue, keep polling

            await self._execute_rollout(rollout)
            processed += 1

        logger.debug("Worker %s exiting after %d rollout(s).", self._worker_id, processed)

    async def _execute_rollout(self, rollout: AttemptedRollout) -> None:
        """Run a single rollout: trace the agent call and persist results.

        Args:
            rollout: The claimed :class:`~mini_agentlightning.types.AttemptedRollout`.
        """
        assert self._agent is not None
        assert self._store is not None

        self.current_resources = rollout.resources

        rollout_id = rollout.rollout_id
        attempt_id = rollout.attempt.attempt_id
        status: str = "succeeded"

        logger.debug("Worker %s executing rollout %s", self._worker_id, rollout_id)
        try:
            async with self._tracer.trace_context(
                store=self._store,
                rollout_id=rollout_id,
                attempt_id=attempt_id,
            ):
                await self._agent.rollout_async(rollout.input)
        except Exception:
            logger.exception("Agent raised an exception during rollout %s", rollout_id)
            status = "failed"
        finally:
            await self._store.finish_rollout(
                rollout_id,
                attempt_id,
                status=status,  # type: ignore[arg-type]
                attempt_status=status,  # type: ignore[arg-type]
            )
