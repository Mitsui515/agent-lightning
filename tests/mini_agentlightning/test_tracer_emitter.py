# Copyright (c) Microsoft. All rights reserved.

"""Tests for mini_agentlightning.tracer.MiniTracer and mini_agentlightning.emitter."""

import pytest

from mini_agentlightning.emitter import emit_reward, find_final_reward, get_reward_value
from mini_agentlightning.store import InMemoryStore
from mini_agentlightning.tracer import MiniTracer


@pytest.mark.asyncio
async def test_tracer_captures_spans():
    tracer = MiniTracer()
    tracer.setup()

    async with tracer.trace_context() as otel_tracer:
        with otel_tracer.start_as_current_span("span-a"):
            pass
        with otel_tracer.start_as_current_span("span-b"):
            pass

    spans = tracer.get_last_trace()
    names = [s.name for s in spans]
    assert "span-a" in names
    assert "span-b" in names


@pytest.mark.asyncio
async def test_tracer_resets_between_traces():
    tracer = MiniTracer()
    tracer.setup()

    async with tracer.trace_context() as otel_tracer:
        with otel_tracer.start_as_current_span("first"):
            pass

    first_spans = tracer.get_last_trace()
    assert any(s.name == "first" for s in first_spans)

    async with tracer.trace_context() as otel_tracer:
        with otel_tracer.start_as_current_span("second"):
            pass

    second_spans = tracer.get_last_trace()
    # Buffer should only contain spans from the most recent trace.
    assert not any(s.name == "first" for s in second_spans)
    assert any(s.name == "second" for s in second_spans)


@pytest.mark.asyncio
async def test_emit_reward_captured_in_trace():
    tracer = MiniTracer()
    tracer.setup()

    async with tracer.trace_context() as _otel_tracer:
        emit_reward(0.75)

    spans = tracer.get_last_trace()
    reward = find_final_reward(spans)
    assert reward == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_emit_reward_integer_coercion():
    tracer = MiniTracer()
    tracer.setup()

    async with tracer.trace_context():
        emit_reward(1)  # int should be accepted

    spans = tracer.get_last_trace()
    reward = find_final_reward(spans)
    assert reward == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_emit_reward_bool_coercion():
    tracer = MiniTracer()
    tracer.setup()

    async with tracer.trace_context():
        emit_reward(True)  # bool should be coerced to 1.0

    spans = tracer.get_last_trace()
    reward = find_final_reward(spans)
    assert reward == pytest.approx(1.0)


def test_emit_reward_invalid_type_raises():
    with pytest.raises(TypeError):
        emit_reward("not-a-number")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_find_final_reward_multiple_rewards():
    tracer = MiniTracer()
    tracer.setup()

    async with tracer.trace_context():
        emit_reward(0.5)
        emit_reward(0.9)

    spans = tracer.get_last_trace()
    reward = find_final_reward(spans)
    # The last emitted reward should be returned.
    assert reward == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_find_final_reward_no_reward_spans():
    tracer = MiniTracer()
    tracer.setup()

    async with tracer.trace_context() as otel_tracer:
        with otel_tracer.start_as_current_span("no-reward"):
            pass

    spans = tracer.get_last_trace()
    assert find_final_reward(spans) is None


@pytest.mark.asyncio
async def test_tracer_writes_spans_to_store():
    store = InMemoryStore()
    rollout = await store.enqueue_rollout("task")

    tracer = MiniTracer()
    tracer.setup()

    async with tracer.trace_context(
        store=store, rollout_id=rollout.rollout_id, attempt_id="attempt-1"
    ) as otel_tracer:
        with otel_tracer.start_as_current_span("stored-span"):
            pass
        emit_reward(1.0)

    # Give the background I/O thread a moment to flush.
    import asyncio
    await asyncio.sleep(0.1)

    spans = await store.query_spans(rollout.rollout_id)
    assert any(s.name == "stored-span" for s in spans)
