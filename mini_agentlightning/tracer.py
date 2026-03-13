# Copyright (c) Microsoft. All rights reserved.

"""Simplified OpenTelemetry tracer for mini-agent-lightning.

The full library supports AgentOps, Weave, OTEL, and Dummy backends and ships
complex span processors with background I/O loops.  The mini tracer keeps only
what the training loop needs: span capture and optional forwarding to an
:class:`~mini_agentlightning.store.InMemoryStore`.

**Design notes**

OpenTelemetry only allows the global :class:`TracerProvider` to be set once
per process.  Rather than fight that constraint, ``MiniTracer`` keeps its own
*local* provider (never registered as the global one) and exposes the active
tracer through a :class:`contextvars.ContextVar`.  The companion
:func:`~mini_agentlightning.emitter.emit_reward` helper reads from that
context variable, so spans always land in the correct rollout's buffer.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, List, Optional

import opentelemetry.trace as otel_trace_api
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace import TracerProvider as TracerProviderImpl

from .store import InMemoryStore
from .types import Span

__all__ = ["MiniTracer", "_active_otel_tracer"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Process-wide context variable: the *local* OTEL Tracer for the current
# async task.  MiniTracer.trace_context() sets this before yielding so that
# emit_reward() knows which provider to call start_as_current_span() on.
# ---------------------------------------------------------------------------

_active_otel_tracer: contextvars.ContextVar[Optional[otel_trace_api.Tracer]] = contextvars.ContextVar(
    "mini_agl_active_otel_tracer", default=None
)


# ---------------------------------------------------------------------------
# Span processor
# ---------------------------------------------------------------------------


class _BufferingSpanProcessor(SpanProcessor):
    """Span processor that buffers completed spans and optionally forwards
    them to an :class:`~mini_agentlightning.store.InMemoryStore`."""

    def __init__(self) -> None:
        self._spans: List[Span] = []
        self._store: Optional[InMemoryStore] = None
        self._rollout_id: Optional[str] = None
        self._lock = threading.Lock()

        # Fallback background I/O loop for when no asyncio loop is running
        # (e.g. rollout executed from a plain sync thread).
        self._io_loop: Optional[asyncio.AbstractEventLoop] = None
        self._io_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # SpanProcessor contract
    # ------------------------------------------------------------------

    def on_start(self, span: Any, parent_context: Any = None) -> None:  # noqa: ANN001
        pass

    def on_end(self, span: ReadableSpan) -> None:
        with self._lock:
            self._spans.append(span)
            store = self._store
            rollout_id = self._rollout_id

        if store is not None and rollout_id is not None:
            self._submit_to_store(store, rollout_id, [span])

    def shutdown(self) -> None:
        if self._io_loop is not None and not self._io_loop.is_closed():
            self._io_loop.call_soon_threadsafe(self._io_loop.stop)
        if self._io_thread is not None:
            self._io_thread.join(timeout=2.0)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:  # noqa: ARG002
        return True

    # ------------------------------------------------------------------
    # Helpers called by MiniTracer.trace_context
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the span buffer for the next trace."""
        with self._lock:
            self._spans.clear()

    def set_store_context(
        self,
        store: Optional[InMemoryStore],
        rollout_id: Optional[str],
    ) -> None:
        with self._lock:
            self._store = store
            self._rollout_id = rollout_id

    def get_spans(self) -> List[Span]:
        with self._lock:
            return list(self._spans)

    # ------------------------------------------------------------------
    # Span submission to store
    # ------------------------------------------------------------------

    def _submit_to_store(self, store: InMemoryStore, rollout_id: str, spans: List[Span]) -> None:
        """Schedule `store.add_spans` on the asyncio loop if one is running,
        otherwise fall back to a dedicated background thread."""
        try:
            loop = asyncio.get_running_loop()
            # We're being called from within an asyncio task (on the event loop
            # thread).  Schedule the coroutine as a task; it will run as soon as
            # the current coroutine next yields.
            loop.create_task(store.add_spans(rollout_id, spans))
        except RuntimeError:
            # No running loop (pure sync context or different thread).
            io_loop = self._ensure_io_loop()
            asyncio.run_coroutine_threadsafe(store.add_spans(rollout_id, spans), io_loop)

    def _ensure_io_loop(self) -> asyncio.AbstractEventLoop:
        if self._io_loop is None or self._io_loop.is_closed():
            self._io_loop = asyncio.new_event_loop()
            self._io_thread = threading.Thread(
                target=self._io_loop.run_forever, daemon=True, name="mini-agl-span-io"
            )
            self._io_thread.start()
        return self._io_loop


# ---------------------------------------------------------------------------
# MiniTracer
# ---------------------------------------------------------------------------


class MiniTracer:
    """Lightweight, self-contained OpenTelemetry tracer for mini-agent-lightning.

    Each ``MiniTracer`` instance owns its own :class:`TracerProvider` (never
    registered as the global OTEL provider), so multiple tracer instances can
    coexist safely inside the same process—important for tests and for
    running multiple runners in a single Python process.

    The active tracer is published to :data:`_active_otel_tracer` so that
    :func:`~mini_agentlightning.emitter.emit_reward` and any code that
    calls :func:`opentelemetry.trace.get_tracer` via the mini-library's own
    helpers can always find the right tracer without touching global OTEL state.

    Example::

        tracer = MiniTracer()
        tracer.setup()

        async with tracer.trace_context(store=store, rollout_id=rid, attempt_id=aid):
            await run_agent(task)
            emit_reward(1.0)

        spans = tracer.get_last_trace()
    """

    def __init__(self) -> None:
        self._processor: Optional[_BufferingSpanProcessor] = None
        self._provider: Optional[TracerProviderImpl] = None

    def setup(self) -> None:
        """Initialise the local tracer provider (call once per runner)."""
        if self._provider is not None:
            return

        provider = TracerProviderImpl()
        processor = _BufferingSpanProcessor()
        provider.add_span_processor(processor)

        self._provider = provider
        self._processor = processor
        logger.debug("MiniTracer: local provider initialised.")

    @asynccontextmanager
    async def trace_context(
        self,
        name: Optional[str] = None,
        *,
        store: Optional[InMemoryStore] = None,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> AsyncGenerator[otel_trace_api.Tracer, None]:
        """Async context manager delimiting a single rollout trace.

        Spans emitted inside this block (via :func:`~mini_agentlightning.emitter.emit_reward`
        or any OTEL instrumentation that uses the local tracer) are buffered and
        can be retrieved with :meth:`get_last_trace`.  When ``store`` and
        ``rollout_id`` are provided the spans are also written to the store.

        Args:
            name: Optional label passed to :meth:`TracerProvider.get_tracer`.
            store: :class:`~mini_agentlightning.store.InMemoryStore` to write spans to.
            rollout_id: Rollout identifier used for store ingestion.
            attempt_id: Attempt identifier (stored in context for tracing but not
                used for span routing in the mini store).

        Yields:
            The local :class:`opentelemetry.trace.Tracer` instance.
        """
        if self._processor is None or self._provider is None:
            raise RuntimeError("MiniTracer not initialised.  Call setup() first.")

        self._processor.reset()
        self._processor.set_store_context(store, rollout_id)

        otel_tracer = self._provider.get_tracer(name or __name__)
        token = _active_otel_tracer.set(otel_tracer)
        try:
            yield otel_tracer
        finally:
            _active_otel_tracer.reset(token)
            self._processor.set_store_context(None, None)
            # Yield to the event loop so that any store.add_spans() tasks that were
            # scheduled via loop.create_task() inside on_end() get a chance to run
            # before the caller continues.
            await asyncio.sleep(0)

    def get_last_trace(self) -> List[Span]:
        """Return spans captured during the most recent :meth:`trace_context` block.

        Returns:
            List of :class:`~mini_agentlightning.types.Span` objects in completion order.
        """
        if self._processor is None:
            return []
        return self._processor.get_spans()
