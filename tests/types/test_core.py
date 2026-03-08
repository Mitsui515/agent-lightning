# Copyright (c) Microsoft. All rights reserved.

"""Tests for agentlightning.types.core module.

Covers:
- Triplet model
- PaginatedResult sequence behaviour
- Rollout / Attempt / AttemptedRollout construction and validation
- RolloutConfig defaults and constraints
- Worker model
- FilterField / FilterOptions helpers (used by collection backends)
"""

from __future__ import annotations

import time
from typing import Any, List

import pytest
from pydantic import ValidationError

from agentlightning.types import (
    Attempt,
    AttemptedRollout,
    PaginatedResult,
    Rollout,
    RolloutConfig,
    Worker,
)
from agentlightning.types.core import Triplet


# ---------------------------------------------------------------------------
# Triplet
# ---------------------------------------------------------------------------


class TestTriplet:
    def test_basic_creation(self) -> None:
        t = Triplet(prompt="Hello", response="World")
        assert t.prompt == "Hello"
        assert t.response == "World"
        assert t.reward is None
        assert t.metadata == {}

    def test_with_reward(self) -> None:
        t = Triplet(prompt="Q", response="A", reward=0.9)
        assert t.reward == pytest.approx(0.9)

    def test_metadata_roundtrip(self) -> None:
        t = Triplet(prompt="x", response="y", metadata={"step": 1, "ok": True})
        assert t.metadata["step"] == 1

    def test_prompt_and_response_accept_any_type(self) -> None:
        t = Triplet(prompt={"role": "user", "content": "hi"}, response=[1, 2, 3])
        assert isinstance(t.prompt, dict)
        assert isinstance(t.response, list)

    def test_serialisation_roundtrip(self) -> None:
        t = Triplet(prompt="p", response="r", reward=1.0)
        data = t.model_dump()
        recovered = Triplet.model_validate(data)
        assert recovered == t


# ---------------------------------------------------------------------------
# PaginatedResult
# ---------------------------------------------------------------------------


class TestPaginatedResult:
    def _make(self, items: List[Any], total: int = 10) -> PaginatedResult[Any]:
        return PaginatedResult(items=items, limit=len(items), offset=0, total=total)

    def test_len(self) -> None:
        result = self._make(["a", "b", "c"])
        assert len(result) == 3

    def test_getitem_int(self) -> None:
        result = self._make([10, 20, 30])
        assert result[0] == 10
        assert result[-1] == 30

    def test_getitem_slice(self) -> None:
        result = self._make([1, 2, 3, 4])
        assert list(result[1:3]) == [2, 3]

    def test_iter(self) -> None:
        result = self._make([1, 2, 3])
        assert list(result) == [1, 2, 3]

    def test_repr_with_items(self) -> None:
        result = self._make(["a", "b", "c"], total=5)
        r = repr(result)
        assert "PaginatedResult" in r
        assert "5" in r

    def test_repr_empty(self) -> None:
        result = self._make([])
        r = repr(result)
        assert "empty" in r

    def test_repr_unlimited(self) -> None:
        result = PaginatedResult(items=["x"], limit=-1, offset=2, total=100)
        r = repr(result)
        assert "2:" in r  # slice repr for unlimited result

    def test_total_attribute(self) -> None:
        result = self._make(["a"], total=99)
        assert result.total == 99

    def test_offset_attribute(self) -> None:
        result = PaginatedResult(items=[], limit=10, offset=5, total=5)
        assert result.offset == 5


# ---------------------------------------------------------------------------
# Attempt
# ---------------------------------------------------------------------------


class TestAttempt:
    def test_default_status_is_preparing(self) -> None:
        a = Attempt(rollout_id="ro-1", attempt_id="at-1", sequence_id=1, start_time=time.time())
        assert a.status == "preparing"

    def test_all_fields(self) -> None:
        now = time.time()
        a = Attempt(
            rollout_id="ro-1",
            attempt_id="at-1",
            sequence_id=2,
            start_time=now,
            end_time=now + 5,
            status="succeeded",
            worker_id="w-1",
            last_heartbeat_time=now + 3,
            metadata={"score": 0.8},
        )
        assert a.sequence_id == 2
        assert a.status == "succeeded"
        assert a.worker_id == "w-1"
        assert a.metadata is not None and a.metadata["score"] == pytest.approx(0.8)

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValidationError):
            Attempt(
                rollout_id="ro-1",
                attempt_id="at-1",
                sequence_id=1,
                start_time=0.0,
                status="invalid_status",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


class TestRollout:
    def test_default_status_is_queuing(self) -> None:
        r = Rollout(rollout_id="ro-1", input={"x": 1}, start_time=0.0)
        assert r.status == "queuing"

    def test_default_config(self) -> None:
        r = Rollout(rollout_id="ro-1", input={}, start_time=0.0)
        assert r.config.max_attempts == 1
        assert r.config.retry_condition == []

    def test_rollout_accepts_arbitrary_input(self) -> None:
        r = Rollout(rollout_id="ro-x", input=[1, 2, 3], start_time=0.0)
        assert r.input == [1, 2, 3]

    def test_serialisation(self) -> None:
        r = Rollout(rollout_id="ro-1", input={"task": "test"}, start_time=1000.0, mode="train")
        data = r.model_dump()
        recovered = Rollout.model_validate(data)
        assert recovered.rollout_id == "ro-1"
        assert recovered.mode == "train"


# ---------------------------------------------------------------------------
# AttemptedRollout
# ---------------------------------------------------------------------------


class TestAttemptedRollout:
    def _make(self, rollout_id: str = "ro-1") -> AttemptedRollout:
        attempt = Attempt(
            rollout_id=rollout_id,
            attempt_id="at-1",
            sequence_id=1,
            start_time=0.0,
        )
        return AttemptedRollout(
            rollout_id=rollout_id,
            input={"x": 1},
            start_time=0.0,
            attempt=attempt,
        )

    def test_creation(self) -> None:
        ar = self._make()
        assert ar.rollout_id == "ro-1"
        assert ar.attempt.attempt_id == "at-1"

    def test_inconsistent_rollout_id_raises(self) -> None:
        attempt = Attempt(rollout_id="ro-999", attempt_id="at-1", sequence_id=1, start_time=0.0)
        with pytest.raises(ValidationError, match="Inconsistent rollout_id"):
            AttemptedRollout(
                rollout_id="ro-1",  # Different from attempt.rollout_id
                input={},
                start_time=0.0,
                attempt=attempt,
            )

    def test_is_subclass_of_rollout(self) -> None:
        ar = self._make()
        assert isinstance(ar, Rollout)


# ---------------------------------------------------------------------------
# RolloutConfig
# ---------------------------------------------------------------------------


class TestRolloutConfig:
    def test_defaults(self) -> None:
        cfg = RolloutConfig()
        assert cfg.max_attempts == 1
        assert cfg.retry_condition == []
        assert cfg.timeout_seconds is None
        assert cfg.unresponsive_seconds is None

    def test_max_attempts_must_be_at_least_one(self) -> None:
        with pytest.raises(ValidationError):
            RolloutConfig(max_attempts=0)

    def test_retry_condition_list(self) -> None:
        cfg = RolloutConfig(max_attempts=3, retry_condition=["timeout", "unresponsive"])
        assert "timeout" in cfg.retry_condition
        assert len(cfg.retry_condition) == 2


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class TestWorker:
    def test_default_status_is_unknown(self) -> None:
        w = Worker(worker_id="w-1")
        assert w.status == "unknown"

    def test_all_statuses(self) -> None:
        for status in ("idle", "busy", "unknown"):
            w = Worker(worker_id="w-1", status=status)  # type: ignore[arg-type]
            assert w.status == status

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValidationError):
            Worker(worker_id="w-1", status="napping")  # type: ignore[arg-type]

    def test_optional_fields_default_to_none(self) -> None:
        w = Worker(worker_id="w-1")
        assert w.heartbeat_stats is None
        assert w.last_heartbeat_time is None
        assert w.current_rollout_id is None
