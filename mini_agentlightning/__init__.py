# Copyright (c) Microsoft. All rights reserved.

"""mini-agent-lightning – a minimal reimplementation of agent-lightning's core training loop.

This package preserves the essential abstractions of the full library while
stripping away every subsystem that is not strictly required to run a
reinforcement-learning training loop:

- No MongoDB / SQLite backends (in-memory store only).
- No AgentOps / Weave tracers (OTEL only).
- No LLM proxy, CLI, instrumentation modules, or distributed execution.
- No complex memory-eviction or span-eviction logic.
- No retry/timeout configuration on rollouts.

The result is a self-contained package of roughly 1 500 lines that is easy to
read, easy to modify, and fast to import.

Quick-start example::

    import asyncio
    from mini_agentlightning import MiniAgent, Algorithm, MiniTrainer, emit_reward

    class GreetAgent(MiniAgent[str]):
        async def rollout_async(self, task: str) -> None:
            print(f"Hello, {task}!")
            emit_reward(1.0)

    class GreetAlgorithm(Algorithm):
        async def run_async(self) -> None:
            for name in ["Alice", "Bob"]:
                await self.enqueue(name)
            await self.wait_all()

    asyncio.run(MiniTrainer(GreetAgent(), GreetAlgorithm()).fit())
"""

from .adapter import spans_to_triplet, spans_to_triplets
from .agent import MiniAgent, MiniRunner
from .algorithm import Algorithm
from .emitter import emit_reward, find_final_reward, get_reward_value
from .store import InMemoryStore
from .tracer import MiniTracer
from .trainer import MiniTrainer
from .types import (
    Attempt,
    AttemptedRollout,
    AttemptStatus,
    NamedResources,
    Rollout,
    RolloutMode,
    RolloutStatus,
    Span,
    TaskInput,
    Triplet,
)

__version__ = "0.1.0"

__all__ = [
    # Trainer / orchestration
    "MiniTrainer",
    # Agent and runner
    "MiniAgent",
    "MiniRunner",
    # Algorithm
    "Algorithm",
    # Store
    "InMemoryStore",
    # Tracer
    "MiniTracer",
    # Emitters
    "emit_reward",
    "find_final_reward",
    "get_reward_value",
    # Adapter
    "spans_to_triplets",
    "spans_to_triplet",
    # Types
    "Span",
    "Triplet",
    "Rollout",
    "Attempt",
    "AttemptedRollout",
    "AttemptStatus",
    "RolloutStatus",
    "RolloutMode",
    "TaskInput",
    "NamedResources",
]
