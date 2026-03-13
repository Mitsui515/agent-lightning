# Copyright (c) Microsoft. All rights reserved.

"""Span-to-triplet adapter for mini-agent-lightning.

The full library ships a sophisticated ``TracerTraceToTriplet`` that walks
the OTEL span tree, matches rewards to LLM call spans, handles multi-turn
conversations, and supports customisable reward-match policies.  The mini
adapter does just enough to extract a single ``Triplet`` (or a list of
triplets) from a flat list of spans so that algorithms have structured data
to work with.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from .emitter import find_final_reward, get_reward_value
from .types import Span, Triplet

__all__ = ["spans_to_triplets", "spans_to_triplet"]

logger = logging.getLogger(__name__)

# Common Gen-AI semantic convention keys (subset used by LiteLLM / OpenAI)
_PROMPT_KEYS = (
    "gen_ai.prompt",
    "llm.prompts",
    "input",
)
_RESPONSE_KEYS = (
    "gen_ai.completion",
    "llm.completions",
    "output",
)


def _try_get_attr(span: Span, keys: tuple[str, ...]) -> Optional[Any]:
    """Return the first matching attribute value from a span."""
    if span.attributes is None:
        return None
    for key in keys:
        val = span.attributes.get(key)
        if val is not None:
            return val
    return None


def _is_llm_span(span: Span) -> bool:
    """Return ``True`` when the span looks like an LLM call."""
    if span.attributes is None:
        return False
    # Both OpenAI instrumentation and LiteLLM use "gen_ai.system".
    return any(
        k in span.attributes
        for k in ("gen_ai.system", "llm.model", "gen_ai.request.model", "llm.prompts")
    )


def spans_to_triplets(spans: Sequence[Span]) -> List[Triplet]:
    """Convert a flat list of spans into a list of :class:`~mini_agentlightning.types.Triplet` objects.

    Each *LLM call span* in the trace becomes one triplet.  The reward is taken
    from the *last* reward span in the entire trace (i.e. the same reward is
    replicated across all triplets when there is more than one LLM call).

    If the trace contains no LLM call spans the function falls back to creating
    a single triplet with ``prompt=None`` and ``response=None`` carrying the
    final reward.

    Args:
        spans: Sequence of :class:`~mini_agentlightning.types.Span` objects from
            a completed rollout.

    Returns:
        Non-empty list of :class:`~mini_agentlightning.types.Triplet` instances.
    """
    reward = find_final_reward(spans)
    llm_spans = [s for s in spans if _is_llm_span(s)]

    if not llm_spans:
        logger.debug("No LLM spans found; creating a single triplet with reward=%s", reward)
        return [Triplet(prompt=None, response=None, reward=reward)]

    triplets: List[Triplet] = []
    for span in llm_spans:
        prompt = _try_get_attr(span, _PROMPT_KEYS)
        response = _try_get_attr(span, _RESPONSE_KEYS)
        span_reward = get_reward_value(span)
        triplets.append(
            Triplet(
                prompt=prompt,
                response=response,
                reward=span_reward if span_reward is not None else reward,
                metadata={"span_name": span.name},
            )
        )
    return triplets


def spans_to_triplet(spans: Sequence[Span]) -> Triplet:
    """Convert a flat list of spans into a *single* :class:`~mini_agentlightning.types.Triplet`.

    When the trace contains multiple LLM call spans only the **first** one is
    used as the prompt/response.  The reward is always taken from the last
    reward span.

    Args:
        spans: Sequence of :class:`~mini_agentlightning.types.Span` objects.

    Returns:
        A single :class:`~mini_agentlightning.types.Triplet`.
    """
    triplets = spans_to_triplets(spans)
    return triplets[0]
