# Copyright (c) Microsoft. All rights reserved.

"""End-to-end training loop tests for mini-agent-lightning."""

import asyncio
from typing import List

import pytest

from mini_agentlightning import (
    Algorithm,
    InMemoryStore,
    MiniAgent,
    MiniTrainer,
    Rollout,
    emit_reward,
    find_final_reward,
    spans_to_triplet,
    spans_to_triplets,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class EchoAgent(MiniAgent[str]):
    """Simple agent that emits a constant reward for any task."""

    collected_tasks: List[str] = []

    def __init__(self) -> None:
        super().__init__()
        self.collected_tasks = []

    async def rollout_async(self, task: str) -> None:
        self.collected_tasks.append(task)
        emit_reward(1.0)


class BatchAlgorithm(Algorithm):
    """Enqueues a fixed batch of tasks and waits for completion."""

    def __init__(self, tasks: List[str]) -> None:
        super().__init__()
        self.tasks = tasks
        self.finished_rollouts: List[Rollout] = []

    async def run_async(self) -> None:
        for task in self.tasks:
            await self.enqueue(task, mode="train")
        self.finished_rollouts = await self.wait_all(timeout=10.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_rollout_end_to_end():
    agent = EchoAgent()
    algorithm = BatchAlgorithm(tasks=["hello"])
    trainer = MiniTrainer(agent=agent, algorithm=algorithm, n_runners=1)

    await trainer.fit()

    assert "hello" in agent.collected_tasks
    assert len(algorithm.finished_rollouts) == 1
    rollout = algorithm.finished_rollouts[0]
    assert rollout.status == "succeeded"


@pytest.mark.asyncio
async def test_multiple_rollouts_end_to_end():
    tasks = ["task-0", "task-1", "task-2", "task-3", "task-4"]
    agent = EchoAgent()
    algorithm = BatchAlgorithm(tasks=tasks)
    trainer = MiniTrainer(agent=agent, algorithm=algorithm, n_runners=2)

    await trainer.fit()

    assert len(algorithm.finished_rollouts) == len(tasks)
    for rollout in algorithm.finished_rollouts:
        assert rollout.status == "succeeded"
    # Every task should have been processed (order is arbitrary with 2 runners).
    assert sorted(agent.collected_tasks) == sorted(tasks)


@pytest.mark.asyncio
async def test_rewards_captured_in_spans():
    agent = EchoAgent()
    algorithm = BatchAlgorithm(tasks=["reward-task"])
    trainer = MiniTrainer(agent=agent, algorithm=algorithm, n_runners=1)

    await trainer.fit()

    rollout = algorithm.finished_rollouts[0]
    spans = await trainer.store.query_spans(rollout.rollout_id)
    reward = find_final_reward(spans)
    assert reward == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_spans_to_triplet_from_rollout():
    agent = EchoAgent()
    algorithm = BatchAlgorithm(tasks=["triplet-task"])
    trainer = MiniTrainer(agent=agent, algorithm=algorithm, n_runners=1)

    await trainer.fit()

    rollout = algorithm.finished_rollouts[0]
    spans = await trainer.store.query_spans(rollout.rollout_id)
    triplet = spans_to_triplet(spans)
    # EchoAgent only emits a reward, no LLM calls → fallback triplet.
    assert triplet.reward == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_algorithm_can_query_spans():
    """Algorithm queries spans after waiting for rollouts to complete."""
    rewards_collected: List[float] = []

    class SpanQueryingAlgorithm(Algorithm):
        async def run_async(self) -> None:
            rollout = await self.enqueue("q-task", mode="train")
            finished = await self.wait_for_rollout(rollout.rollout_id, timeout=10.0)
            assert finished is not None
            spans = await self.query_spans(rollout.rollout_id)
            reward = find_final_reward(spans)
            if reward is not None:
                rewards_collected.append(reward)

    agent = EchoAgent()
    algorithm = SpanQueryingAlgorithm()
    trainer = MiniTrainer(agent=agent, algorithm=algorithm, n_runners=1)

    await trainer.fit()

    assert rewards_collected == [pytest.approx(1.0)]


@pytest.mark.asyncio
async def test_resources_visible_to_agent():
    """Resources published before fit() are available to the agent during rollout."""
    received_resources: List[dict] = []

    class ResourceAwareAgent(MiniAgent[str]):
        async def rollout_async(self, task: str) -> None:
            resources = self.get_resources()
            if resources:
                received_resources.append(dict(resources))
            emit_reward(1.0)

    agent = ResourceAwareAgent()
    algorithm = BatchAlgorithm(tasks=["res-task"])
    trainer = MiniTrainer(
        agent=agent,
        algorithm=algorithm,
        n_runners=1,
        initial_resources={"system_prompt": "Be concise."},
    )

    await trainer.fit()

    assert len(received_resources) == 1
    assert received_resources[0]["system_prompt"] == "Be concise."


@pytest.mark.asyncio
async def test_algorithm_updates_resources_mid_run():
    """Algorithm can update resources between batches; runners pick up the change."""
    received_prompts: List[str] = []

    class VersionedAgent(MiniAgent[str]):
        async def rollout_async(self, task: str) -> None:
            res = self.get_resources() or {}
            received_prompts.append(res.get("prompt", "none"))
            emit_reward(1.0)

    class TwoPhaseAlgorithm(Algorithm):
        async def run_async(self) -> None:
            # Phase 1: enqueue with initial resources
            r1 = await self.enqueue("phase-1")
            await self.wait_for_rollout(r1.rollout_id, timeout=10.0)

            # Update resources for phase 2
            await self.set_resources({"prompt": "phase-2-prompt"})

            r2 = await self.enqueue("phase-2")
            await self.wait_for_rollout(r2.rollout_id, timeout=10.0)

    agent = VersionedAgent()
    algorithm = TwoPhaseAlgorithm()
    trainer = MiniTrainer(
        agent=agent,
        algorithm=algorithm,
        n_runners=1,
        initial_resources={"prompt": "phase-1-prompt"},
    )

    await trainer.fit()

    assert len(received_prompts) == 2
    assert received_prompts[0] == "phase-1-prompt"
    assert received_prompts[1] == "phase-2-prompt"


@pytest.mark.asyncio
async def test_dev_mode_single_rollout():
    """MiniTrainer.dev() executes one rollout without an algorithm."""
    executed: List[str] = []

    class DevAgent(MiniAgent[str]):
        async def rollout_async(self, task: str) -> None:
            executed.append(task)
            emit_reward(0.5)

    agent = DevAgent()
    trainer = MiniTrainer(agent=agent, algorithm=BatchAlgorithm([]))

    spans = await trainer.dev("dev-task")
    assert "dev-task" in executed
    reward = find_final_reward(spans)
    assert reward == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_failed_rollout_recorded():
    """When the agent raises an exception the rollout is marked as failed."""

    class BrokenAgent(MiniAgent[str]):
        async def rollout_async(self, task: str) -> None:
            raise RuntimeError("agent error")

    class FailAlgorithm(Algorithm):
        async def run_async(self) -> None:
            await self.enqueue("fail-task")
            await self.wait_all(timeout=5.0)

    agent = BrokenAgent()
    algorithm = FailAlgorithm()
    trainer = MiniTrainer(agent=agent, algorithm=algorithm, n_runners=1)

    await trainer.fit()

    rollouts = await trainer.store.list_rollouts(status="failed")
    assert len(rollouts) == 1
