# Copyright (c) Microsoft. All rights reserved.

"""Tests specific to the SQLite LightningStore backend.

These tests cover SQLite-specific behaviour such as persistence,
concurrent access, and correct handling of the ``aiosqlite`` connection
lifecycle that are not covered by the generic ``store_fixture`` tests in
``test_core.py``.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import pytest
import pytest_asyncio

from agentlightning.store.sqlite import SQLiteLightningStore
from agentlightning.types import LLM

pytestmark = [pytest.mark.store, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def sqlite_inmemory():
    """Fresh in-memory SQLite store."""
    store = SQLiteLightningStore(db_path=":memory:", scan_debounce_seconds=0)
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------


async def test_initialize_is_idempotent(sqlite_inmemory: SQLiteLightningStore) -> None:
    """Calling initialize() multiple times must not raise."""
    await sqlite_inmemory.initialize()
    await sqlite_inmemory.initialize()
    stats = await sqlite_inmemory.statistics()
    assert stats["total_rollouts"] == 0


async def test_capabilities(sqlite_inmemory: SQLiteLightningStore) -> None:
    """SQLite store reports its capabilities correctly."""
    caps = sqlite_inmemory.capabilities
    assert caps["async_safe"] is True
    assert caps["thread_safe"] is False
    assert caps["zero_copy"] is False
    assert caps["otlp_traces"] is False


async def test_class_name_in_statistics(sqlite_inmemory: SQLiteLightningStore) -> None:
    """statistics() must include the class name."""
    stats = await sqlite_inmemory.statistics()
    assert stats["name"] == "SQLiteLightningStore"


# ---------------------------------------------------------------------------
# Persistence across re-opens
# ---------------------------------------------------------------------------


async def test_data_persists_across_reopen() -> None:
    """Data written to a file-backed SQLite database must survive a close/reopen."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Create a rollout and close the store.
        store1 = SQLiteLightningStore(db_path=db_path, scan_debounce_seconds=0)
        await store1.initialize()
        rollout = await store1.enqueue_rollout(input={"task": "persist-check"})
        rollout_id = rollout.rollout_id
        await store1.close()

        # Reopen the same file and verify the rollout is still there.
        store2 = SQLiteLightningStore(db_path=db_path, scan_debounce_seconds=0)
        await store2.initialize()
        try:
            all_rollouts = await store2.query_rollouts()
            found_ids = [r.rollout_id for r in all_rollouts.items]
            assert rollout_id in found_ids, (
                f"Rollout {rollout_id!r} not found after reopening database. "
                f"Found: {found_ids}"
            )
            assert all_rollouts.items[0].status == "queuing"
        finally:
            await store2.close()
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Resource management
# ---------------------------------------------------------------------------


async def test_add_and_get_resources(sqlite_inmemory: SQLiteLightningStore) -> None:
    """Resources can be stored and retrieved from a SQLite store."""
    llm = LLM(endpoint="http://localhost:8000/v1", model="test-model")
    update = await sqlite_inmemory.add_resources({"llm": llm})

    assert update.resources_id is not None
    assert "llm" in update.resources
    assert update.resources["llm"].model == "test-model"  # type: ignore[union-attr]


async def test_update_resources_increments_version(sqlite_inmemory: SQLiteLightningStore) -> None:
    """Updating resources should increment the version counter."""
    llm = LLM(endpoint="http://localhost:8000/v1", model="v1-model")
    first = await sqlite_inmemory.add_resources({"llm": llm})

    llm_v2 = LLM(endpoint="http://localhost:8000/v1", model="v2-model")
    second = await sqlite_inmemory.update_resources(first.resources_id, {"llm": llm_v2})

    assert second.version > first.version
    assert second.resources["llm"].model == "v2-model"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Rollout queue
# ---------------------------------------------------------------------------


async def test_enqueue_and_dequeue_rollout(sqlite_inmemory: SQLiteLightningStore) -> None:
    """Enqueued rollouts can be dequeued by a worker."""
    await sqlite_inmemory.enqueue_rollout(input={"q": 1})
    await sqlite_inmemory.enqueue_rollout(input={"q": 2})

    first = await sqlite_inmemory.dequeue_rollout()
    assert first is not None
    assert first.input == {"q": 1}

    second = await sqlite_inmemory.dequeue_rollout()
    assert second is not None
    assert second.input == {"q": 2}

    empty = await sqlite_inmemory.dequeue_rollout()
    assert empty is None


async def test_start_rollout_is_immediately_running(sqlite_inmemory: SQLiteLightningStore) -> None:
    """start_rollout() should return an AttemptedRollout in preparing state."""
    result = await sqlite_inmemory.start_rollout(input={"immediate": True})
    assert result.status == "preparing"
    assert result.attempt is not None


# ---------------------------------------------------------------------------
# Wait for rollouts (timeout path)
# ---------------------------------------------------------------------------


async def test_wait_for_rollouts_timeout(sqlite_inmemory: SQLiteLightningStore) -> None:
    """wait_for_rollouts should return an empty list when all rollouts are still queued."""
    r = await sqlite_inmemory.enqueue_rollout(input={"w": True})
    finished = await sqlite_inmemory.wait_for_rollouts(rollout_ids=[r.rollout_id], timeout=0.05)
    # Not finished yet, so should return empty list.
    assert finished == []


async def test_wait_for_already_finished_rollout(sqlite_inmemory: SQLiteLightningStore) -> None:
    """wait_for_rollouts returns immediately when the rollout is already finished."""
    r = await sqlite_inmemory.start_rollout(input={"done": True})
    # Mark the attempt as succeeded and the rollout as succeeded.
    await sqlite_inmemory.update_attempt(r.rollout_id, r.attempt.attempt_id, status="succeeded")
    await sqlite_inmemory.update_rollout(rollout_id=r.rollout_id, status="succeeded")

    finished = await sqlite_inmemory.wait_for_rollouts(rollout_ids=[r.rollout_id], timeout=2.0)
    assert len(finished) == 1
    assert finished[0].rollout_id == r.rollout_id
    assert finished[0].status == "succeeded"


# ---------------------------------------------------------------------------
# Collection-level operations (SQLite-specific)
# ---------------------------------------------------------------------------


async def test_sqlite_collection_size(sqlite_inmemory: SQLiteLightningStore) -> None:
    """Collection size should reflect the number of stored items."""
    await sqlite_inmemory.enqueue_rollout(input={"a": 1})
    await sqlite_inmemory.enqueue_rollout(input={"b": 2})
    await sqlite_inmemory.enqueue_rollout(input={"c": 3})

    stats = await sqlite_inmemory.statistics()
    assert stats["total_rollouts"] == 3


async def test_sqlite_filter_by_status(sqlite_inmemory: SQLiteLightningStore) -> None:
    """Query should filter rollouts by status correctly."""
    await sqlite_inmemory.enqueue_rollout(input={"status_test": 1})
    await sqlite_inmemory.enqueue_rollout(input={"status_test": 2})
    r = await sqlite_inmemory.start_rollout(input={"status_test": 3})

    queued = await sqlite_inmemory.query_rollouts(status_in=["queuing"])
    preparing = await sqlite_inmemory.query_rollouts(status_in=["preparing"])

    assert queued.total == 2
    assert preparing.total == 1
    assert preparing.items[0].rollout_id == r.rollout_id


async def test_sqlite_worker_registration(sqlite_inmemory: SQLiteLightningStore) -> None:
    """Workers can be registered and retrieved."""
    await sqlite_inmemory.update_worker("worker-1", heartbeat_stats={"cpu": 0.3})
    await sqlite_inmemory.update_worker("worker-2", heartbeat_stats={"cpu": 0.6})

    stats = await sqlite_inmemory.statistics()
    assert stats["total_workers"] == 2
