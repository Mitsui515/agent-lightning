# Copyright (c) Microsoft. All rights reserved.

"""High-level trainer for mini-agent-lightning.

The full library ships a ``Trainer`` that wires algorithm, runner, store,
tracer, execution strategy, LLM proxy, and adapter through a pluggable
component system.  The mini trainer collapses this into a single class that:

1. Creates and owns an :class:`~mini_agentlightning.store.InMemoryStore`.
2. Sets up one :class:`~mini_agentlightning.tracer.MiniTracer` per runner.
3. Runs *N* runner tasks concurrently inside a single ``asyncio`` event loop
   (shared-memory execution only).
4. Runs the algorithm concurrently with the runners.
5. Tears everything down cleanly when the algorithm exits.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import Any, Generic, List, Optional, TypeVar

from .agent import MiniAgent, MiniRunner
from .algorithm import Algorithm
from .store import InMemoryStore
from .tracer import MiniTracer
from .types import NamedResources

__all__ = ["MiniTrainer"]

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MiniTrainer(Generic[T]):
    """Orchestrates the mini-agent-lightning training loop.

    Args:
        agent: :class:`~mini_agentlightning.agent.MiniAgent` instance to execute.
        algorithm: :class:`~mini_agentlightning.algorithm.Algorithm` that enqueues
            tasks and learns from collected spans.
        n_runners: Number of concurrent runner tasks.
        poll_interval: Seconds between store polls when the queue is empty.
        initial_resources: Optional named-resource snapshot to publish before
            the algorithm starts.

    Example::

        trainer = MiniTrainer(
            agent=MyAgent(),
            algorithm=MyAlgorithm(),
            n_runners=2,
        )
        asyncio.run(trainer.fit())
    """

    def __init__(
        self,
        agent: MiniAgent[T],
        algorithm: Algorithm,
        *,
        n_runners: int = 1,
        poll_interval: float = 0.5,
        initial_resources: Optional[NamedResources] = None,
    ) -> None:
        self._agent = agent
        self._algorithm = algorithm
        self._n_runners = n_runners
        self._poll_interval = poll_interval
        self._initial_resources = initial_resources

        self.store = InMemoryStore()
        self._runners: List[MiniRunner[T]] = []
        self._runner_tasks: List[asyncio.Task[Any]] = []

        # Attach trainer reference to agent and algorithm.
        self._agent._trainer_ref = weakref.ref(self)
        self._algorithm._attach(self.store, self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fit(self) -> None:
        """Run the training loop until the algorithm exits.

        1. Publish ``initial_resources`` to the store (if any).
        2. Spawn *n_runners* runner tasks.
        3. Await the algorithm's ``run_async`` method.
        4. Signal all runners to stop and await their termination.
        """
        if self._initial_resources is not None:
            await self.store.set_resources(self._initial_resources)

        self._runners = [
            MiniRunner(
                tracer=MiniTracer(),
                poll_interval=self._poll_interval,
                worker_id=f"worker-{i}",
            )
            for i in range(self._n_runners)
        ]

        for runner in self._runners:
            runner.init(self._agent, self.store)

        # Start runner tasks
        self._runner_tasks = [
            asyncio.create_task(runner.run(), name=f"runner-{i}")
            for i, runner in enumerate(self._runners)
        ]

        try:
            await self._algorithm.run_async()
        finally:
            # Stop all runners and collect them.
            for runner in self._runners:
                runner.stop()

            if self._runner_tasks:
                done, pending = await asyncio.wait(self._runner_tasks, timeout=10.0)
                for task in pending:
                    task.cancel()
                    logger.warning("Runner task %s did not stop in time; cancelled.", task.get_name())

        logger.info("MiniTrainer.fit() finished.")

    async def dev(self, task: Any) -> List[Any]:
        """Execute a single rollout for rapid local development / debugging.

        Unlike :meth:`fit`, ``dev`` runs exactly one rollout synchronously
        (no algorithm involved) and returns the collected spans.

        Args:
            task: Task payload forwarded to the agent.

        Returns:
            List of :class:`~mini_agentlightning.types.Span` objects collected
            during the rollout.
        """
        if self._initial_resources is not None:
            await self.store.set_resources(self._initial_resources)

        tracer = MiniTracer()
        tracer.setup()

        runner = MiniRunner(tracer=tracer, poll_interval=0.1, worker_id="dev-worker")
        runner.init(self._agent, self.store)

        rollout = await self.store.enqueue_rollout(task, mode="test")
        # Run exactly one rollout then stop.
        await runner.run(max_rollouts=1)

        # Prefer the tracer's in-memory buffer which is always up-to-date.
        # Fall back to the store if the buffer is empty (e.g. when spans were
        # submitted from a different context).
        spans = tracer.get_last_trace()
        if not spans:
            spans = await self.store.query_spans(rollout.rollout_id)
        return spans
