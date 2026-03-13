# Copyright (c) Microsoft. All rights reserved.

"""Reward emission helpers for mini-agent-lightning.

The full library supports multi-dimensional rewards, AgentOps integration,
link attributes, and tag attributes.  The mini emitter focuses on the two
operations that every training loop needs:

- :func:`emit_reward` – record a scalar reward as an OTEL span attribute.
- :func:`find_final_reward` – extract the last reward value from a trace.

``emit_reward`` uses the *local* tracer published by
:class:`~mini_agentlightning.tracer.MiniTracer` via
:data:`~mini_agentlightning.tracer._active_otel_tracer`, so it is fully
independent of the global OpenTelemetry provider.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from .types import Span

__all__ = ["emit_reward", "find_final_reward", "get_reward_value"]

logger = logging.getLogger(__name__)

# Attribute key used to store the reward value inside a span.
_REWARD_ATTR = "agentlightning.reward.value"
_REWARD_SPAN_NAME = "agentlightning.reward"


def emit_reward(value: float) -> None:
    """Record a scalar reward as an OpenTelemetry span.

    The span is emitted on the *local* tracer registered by the enclosing
    :meth:`~mini_agentlightning.tracer.MiniTracer.trace_context` block.
    Call this inside such a block so the span is captured and associated with
    the correct rollout.

    Args:
        value: Numeric reward to record.  Integers and booleans are accepted
            and automatically converted to ``float``.

    Raises:
        TypeError: When ``value`` is not a numeric type.

    Example::

        async with tracer.trace_context(store=store, rollout_id=rid, attempt_id=aid):
            await run_agent(task)
            emit_reward(1.0)
    """
    from .tracer import _active_otel_tracer

    if isinstance(value, bool):
        value = float(value)
    elif isinstance(value, int):
        value = float(value)
    elif not isinstance(value, float):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"Reward must be a numeric value, got {type(value)!r}")

    tracer = _active_otel_tracer.get()
    if tracer is None:
        logger.warning(
            "emit_reward() called outside a MiniTracer.trace_context() block. "
            "The reward span will not be captured."
        )
        return

    with tracer.start_as_current_span(_REWARD_SPAN_NAME) as span:
        span.set_attribute(_REWARD_ATTR, value)

    logger.debug("Emitted reward: %s", value)


def get_reward_value(span: Span) -> Optional[float]:
    """Extract the reward scalar from a span, if present.

    Args:
        span: An OpenTelemetry :class:`~opentelemetry.sdk.trace.ReadableSpan`.

    Returns:
        The reward value or ``None`` when the span is not a reward span.
    """
    if span.attributes is None:
        return None
    value = span.attributes.get(_REWARD_ATTR)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        logger.warning("Unexpected reward attribute type %s for span %s", type(value), span.name)
        return None
    return float(value)


def find_final_reward(spans: Sequence[Span]) -> Optional[float]:
    """Return the last reward value found in a list of spans.

    Iterates the span sequence in reverse so that the most recently emitted
    reward is returned first.

    Args:
        spans: Sequence of :class:`~mini_agentlightning.types.Span` objects.

    Returns:
        The scalar reward from the last reward span, or ``None`` when none
        are present.
    """
    for span in reversed(spans):
        reward = get_reward_value(span)
        if reward is not None:
            return reward
    return None
