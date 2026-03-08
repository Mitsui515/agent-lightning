# Copyright (c) Microsoft. All rights reserved.

"""SQLite-backed LightningStore for persistent single-process usage.

This backend stores all data in a local SQLite database file. It is
suitable for development, local experiments, and single-process training
loops that need persistence without a full MongoDB deployment.

Example usage::

    import asyncio
    from agentlightning.store.sqlite import SQLiteLightningStore

    store = SQLiteLightningStore(db_path="lightning.db")
    asyncio.run(store.initialize())
    # … use store …
    asyncio.run(store.close())
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional, Sequence, Union

from agentlightning.types import Attempt, AttemptedRollout, Rollout
from agentlightning.utils.metrics import MetricsBackend

from .base import LightningStoreCapabilities, is_finished
from .collection.sqlite import SQLiteLightningCollections
from .collection_based import CollectionBasedLightningStore, healthcheck_before, tracked

logger = logging.getLogger(__name__)

# Polling constants for wait_for_rollouts.
_MIN_SLEEP_SECONDS = 0.01
_MAX_SLEEP_SECONDS = 10.0
_DEADLINE_BUFFER_SECONDS = 0.1


class SQLiteLightningStore(CollectionBasedLightningStore[SQLiteLightningCollections]):
    """SQLite-backed implementation of :class:`~agentlightning.store.base.LightningStore`.

    All data is persisted in a single SQLite database file. The store
    supports a single asyncio event-loop and is **not** safe to share
    between multiple processes (use
    :class:`~agentlightning.store.mongo.MongoLightningStore` for that
    use case).

    Args:
        db_path: Path to the SQLite database file. Pass ``":memory:"`` to
            use an in-memory database (data is lost when the store closes).
        tracker: Optional metrics backend for Prometheus / console metrics.
        scan_debounce_seconds: Debounce time for the unhealthy-rollout scan.
            Set to ``0`` to disable debouncing.
    """

    def __init__(
        self,
        db_path: str = "agentlightning.db",
        *,
        tracker: MetricsBackend | None = None,
        scan_debounce_seconds: float = 10.0,
    ) -> None:
        super().__init__(
            collections=SQLiteLightningCollections(db_path=db_path, tracker=tracker),
            tracker=tracker,
            scan_debounce_seconds=scan_debounce_seconds,
        )
        self._db_path = db_path

    @property
    def capabilities(self) -> LightningStoreCapabilities:
        """Return the capabilities of the SQLite store."""
        return LightningStoreCapabilities(
            thread_safe=False,
            async_safe=True,
            zero_copy=False,
            otlp_traces=False,
        )

    async def initialize(self) -> None:
        """Create all required database tables.

        Must be awaited once before any store operation. Calling this
        method more than once is safe (idempotent).
        """
        await self.collections.initialize()
        logger.info("SQLiteLightningStore initialised. db_path=%r", self._db_path)

    async def close(self) -> None:
        """Close the underlying SQLite connection."""
        await self.collections.close()
        logger.info("SQLiteLightningStore closed. db_path=%r", self._db_path)

    @tracked("wait_for_rollouts_sqlite")
    @healthcheck_before
    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        """Poll until all requested rollouts have finished, or the timeout expires.

        Args:
            rollout_ids: IDs of rollouts to wait for.
            timeout: Maximum seconds to wait. ``None`` means wait indefinitely.

        Returns:
            Finished rollouts in the same order as *rollout_ids* (missing
            rollouts are omitted from the result).
        """
        deadline = time.time() + timeout if timeout is not None else None

        finished: Dict[str, Rollout] = {}
        pending = set(rollout_ids)

        while True:
            async with self.collections.atomic(
                mode="r", snapshot=self._read_snapshot, labels=["rollouts"]
            ) as collections:
                rollouts = await collections.rollouts.query(
                    filter={"rollout_id": {"within": list(pending)}}
                )
            for rollout in rollouts:
                if is_finished(rollout):
                    finished[rollout.rollout_id] = rollout
                    pending.discard(rollout.rollout_id)

            if not pending:
                break

            current_time = time.time()
            if deadline is not None and current_time >= deadline:
                break

            if deadline is not None:
                remaining = deadline - current_time - _DEADLINE_BUFFER_SECONDS
                rest = max(_MIN_SLEEP_SECONDS, min(_MAX_SLEEP_SECONDS, remaining))
            else:
                rest = _MAX_SLEEP_SECONDS
            await asyncio.sleep(rest)

        logger.debug(
            "wait_for_rollouts: finished=%d, still pending=%d",
            len(finished),
            len(pending),
        )
        return [finished[rid] for rid in rollout_ids if rid in finished]

    @tracked("_unlocked_many_rollouts_to_attempted_rollouts_sqlite")
    async def _unlocked_many_rollouts_to_attempted_rollouts(
        self, collections: SQLiteLightningCollections, rollouts: Sequence[Rollout]
    ) -> List[Union[Rollout, AttemptedRollout]]:
        """Attach the latest attempt to each rollout, if one is available."""
        async with collections.atomic(mode="r", snapshot=self._read_snapshot, labels=["attempts"]) as collections:
            attempts = await collections.attempts.query(
                filter={"rollout_id": {"within": [r.rollout_id for r in rollouts]}},
                sort={"name": "sequence_id", "order": "desc"},
            )
        latest: Dict[str, Attempt] = {}
        for attempt in attempts:
            if attempt.rollout_id not in latest:
                latest[attempt.rollout_id] = attempt

        result: List[Union[Rollout, AttemptedRollout]] = []
        for rollout in rollouts:
            if rollout.rollout_id in latest:
                result.append(AttemptedRollout(**rollout.model_dump(), attempt=latest[rollout.rollout_id]))
            else:
                result.append(rollout)
        return result
