# Copyright (c) Microsoft. All rights reserved.

"""Core data models for mini-agent-lightning.

This module intentionally keeps only the fields and types required by the
training loop, discarding retry configs, memory-management bookkeeping, and
all deprecated legacy shapes present in the full library.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel, Field

__all__ = [
    "Span",
    "Triplet",
    "RolloutStatus",
    "AttemptStatus",
    "RolloutMode",
    "Attempt",
    "Rollout",
    "AttemptedRollout",
    "TaskInput",
    "NamedResources",
]

# ---------------------------------------------------------------------------
# Span alias – we work directly with OpenTelemetry ReadableSpan objects so
# that the full library's span serialisation/deserialization round-trip is
# unnecessary.
# ---------------------------------------------------------------------------

Span = ReadableSpan
"""A single OpenTelemetry span captured during agent execution."""

TaskInput = Any
"""Arbitrary task payload provided by an algorithm to drive a rollout."""

NamedResources = Dict[str, Any]
"""Key-value mapping of named resources (e.g. prompts, model paths) shared
between the algorithm and runners."""

# ---------------------------------------------------------------------------
# Status literals – kept minimal; only statuses actually used in the loop.
# ---------------------------------------------------------------------------

RolloutStatus = Literal[
    "queuing",
    "running",
    "succeeded",
    "failed",
]
"""Lifecycle states of a rollout in the mini training loop."""

AttemptStatus = Literal[
    "running",
    "succeeded",
    "failed",
]
"""Lifecycle states of an execution attempt."""

RolloutMode = Literal["train", "val", "test"]
"""Semantic mode attached to a rollout for downstream analytics."""


# ---------------------------------------------------------------------------
# Core Pydantic models
# ---------------------------------------------------------------------------


class Triplet(BaseModel):
    """Single interaction turn captured during reinforcement learning.

    A triplet pairs a model *prompt* with the *response* it produced and an
    optional scalar *reward* assigned after evaluation.
    """

    prompt: Any
    """The input context presented to the model."""
    response: Any
    """The model output for the given prompt."""
    reward: Optional[float] = None
    """Scalar feedback signal; `None` when no reward has been assigned yet."""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Free-form key-value pairs for downstream processing."""


class Attempt(BaseModel):
    """Single execution attempt associated with a rollout."""

    rollout_id: str
    """Parent rollout identifier."""
    attempt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this attempt."""
    start_time: float = Field(default_factory=time.time)
    """Wall-clock timestamp (seconds) when the attempt started."""
    end_time: Optional[float] = None
    """Wall-clock timestamp (seconds) when the attempt finished."""
    status: AttemptStatus = "running"
    """Current lifecycle status."""
    worker_id: Optional[str] = None
    """Identifier of the worker executing this attempt."""


class Rollout(BaseModel):
    """Training rollout managed by the store and executed by a runner."""

    rollout_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for the rollout."""
    input: TaskInput
    """Task payload provided by the algorithm."""
    start_time: float = Field(default_factory=time.time)
    """Wall-clock timestamp (seconds) when the rollout was created."""
    end_time: Optional[float] = None
    """Wall-clock timestamp (seconds) when the rollout finished."""
    mode: Optional[RolloutMode] = None
    """Semantic mode: `"train"`, `"val"`, or `"test"`."""
    status: RolloutStatus = "queuing"
    """Current lifecycle status."""
    resources: Optional[NamedResources] = None
    """Snapshot of resources that should be used during execution."""
    metadata: Optional[Dict[str, Any]] = None
    """Arbitrary key-value pairs attached to the rollout."""


class AttemptedRollout(Rollout):
    """A rollout paired with its active execution attempt.

    Returned by [`InMemoryStore.claim_rollout`][mini_agentlightning.store.InMemoryStore.claim_rollout]
    so that a runner has everything it needs to start execution in one object.
    """

    attempt: Attempt
    """The currently active attempt for this rollout."""
