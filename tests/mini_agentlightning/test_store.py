# Copyright (c) Microsoft. All rights reserved.

"""Tests for mini_agentlightning.store.InMemoryStore."""

import asyncio
from typing import List

import pytest

from mini_agentlightning.store import InMemoryStore
from mini_agentlightning.types import Rollout


@pytest.mark.asyncio
async def test_enqueue_and_claim_rollout():
    store = InMemoryStore()
    rollout = await store.enqueue_rollout({"task": "hello"})
    assert rollout.status == "queuing"
    assert rollout.rollout_id

    claimed = await store.claim_rollout(worker_id="w0")
    assert claimed is not None
    assert claimed.rollout_id == rollout.rollout_id
    assert claimed.status == "running"
    assert claimed.attempt.worker_id == "w0"


@pytest.mark.asyncio
async def test_claim_empty_queue_returns_none():
    store = InMemoryStore()
    claimed = await store.claim_rollout(timeout=0.05)
    assert claimed is None


@pytest.mark.asyncio
async def test_finish_rollout_sets_status():
    store = InMemoryStore()
    rollout = await store.enqueue_rollout("task")
    claimed = await store.claim_rollout()
    assert claimed is not None

    await store.finish_rollout(claimed.rollout_id, claimed.attempt.attempt_id, status="succeeded")
    fetched = await store.get_rollout(claimed.rollout_id)
    assert fetched is not None
    assert fetched.status == "succeeded"
    assert fetched.end_time is not None


@pytest.mark.asyncio
async def test_add_and_query_spans():
    from opentelemetry.sdk.trace import ReadableSpan

    store = InMemoryStore()
    rollout = await store.enqueue_rollout("task")

    # We create minimal fake ReadableSpan objects via MiniTracer
    from mini_agentlightning.tracer import MiniTracer

    tracer = MiniTracer()
    tracer.setup()

    async with tracer.trace_context(
        store=store, rollout_id=rollout.rollout_id, attempt_id="attempt-1"
    ) as otel_tracer:
        with otel_tracer.start_as_current_span("test-span"):
            pass

    spans = await store.query_spans(rollout.rollout_id)
    assert len(spans) >= 1
    assert any(s.name == "test-span" for s in spans)


@pytest.mark.asyncio
async def test_set_and_get_resources():
    store = InMemoryStore()
    resources = {"prompt": "You are a helpful assistant."}
    await store.set_resources(resources)
    fetched = await store.get_resources()
    assert fetched == resources


@pytest.mark.asyncio
async def test_resources_attached_to_rollout():
    store = InMemoryStore()
    resources = {"model": "gpt-4"}
    await store.set_resources(resources)

    rollout = await store.enqueue_rollout("task")
    claimed = await store.claim_rollout()
    assert claimed is not None
    assert claimed.resources == resources


@pytest.mark.asyncio
async def test_wait_for_rollout():
    store = InMemoryStore()
    rollout = await store.enqueue_rollout("task")
    claimed = await store.claim_rollout()
    assert claimed is not None

    async def finish_later():
        await asyncio.sleep(0.05)
        await store.finish_rollout(claimed.rollout_id, claimed.attempt.attempt_id, status="succeeded")

    asyncio.create_task(finish_later())
    finished = await store.wait_for_rollout(rollout.rollout_id, timeout=2.0)
    assert finished is not None
    assert finished.status == "succeeded"


@pytest.mark.asyncio
async def test_wait_for_rollout_timeout():
    store = InMemoryStore()
    rollout = await store.enqueue_rollout("task")
    _ = await store.claim_rollout()

    # Rollout never finishes → should time out.
    result = await store.wait_for_rollout(rollout.rollout_id, timeout=0.05)
    assert result is None


@pytest.mark.asyncio
async def test_list_rollouts_by_status():
    store = InMemoryStore()
    r1 = await store.enqueue_rollout("a")
    r2 = await store.enqueue_rollout("b")

    # Claim r1 but not r2
    c1 = await store.claim_rollout()
    assert c1 is not None
    await store.finish_rollout(c1.rollout_id, c1.attempt.attempt_id, status="succeeded")

    succeeded = await store.list_rollouts(status="succeeded")
    queuing = await store.list_rollouts(status="queuing")

    succeeded_ids = {r.rollout_id for r in succeeded}
    queuing_ids = {r.rollout_id for r in queuing}

    assert r1.rollout_id in succeeded_ids
    assert r2.rollout_id in queuing_ids
