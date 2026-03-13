"""Minimal math-QA training loop using mini-agent-lightning.

This standalone example demonstrates the full mini-agent-lightning workflow
from end to end:

1. An **Algorithm** generates simple arithmetic problems, waits for the
   agent to solve them, and reads back the reward signal.
2. An **Agent** produces a (randomly correct) integer answer and emits a
   binary reward based on correctness.
3. A **MiniTrainer** wires everything together and runs the loop for two
   training epochs.

Usage::

    python examples/mini_agentlightning/math_qa.py

No LLM API key is required – the agent uses ``random.randint`` as a
stand-in for an LLM call so the example is fully self-contained.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Dict, List

from mini_agentlightning import (
    Algorithm,
    MiniAgent,
    MiniTrainer,
    emit_reward,
    find_final_reward,
    spans_to_triplet,
)
from mini_agentlightning.types import Rollout

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task schema
# ---------------------------------------------------------------------------

Task = Dict[str, Any]
"""A single arithmetic problem: ``{"a": int, "b": int, "answer": int}``."""


def make_task(a: int, b: int) -> Task:
    return {"a": a, "b": b, "answer": a + b}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class MathAgent(MiniAgent[Task]):
    """Agent that attempts to solve integer addition problems.

    Uses a biased coin flip (70 % correct) instead of a real LLM so the
    example requires no external dependencies.
    """

    async def rollout_async(self, task: Task) -> None:
        a, b, expected = task["a"], task["b"], task["answer"]

        # Simulate an LLM call with a 70 % chance of a correct answer.
        if random.random() < 0.70:
            predicted = a + b  # correct
        else:
            predicted = a + b + random.randint(1, 5)  # off by a small amount

        is_correct = predicted == expected
        logger.info("  %d + %d = %d  (expected %d)  → %s", a, b, predicted, expected, "✓" if is_correct else "✗")
        emit_reward(1.0 if is_correct else 0.0)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

PROBLEMS = [make_task(a, b) for a in range(1, 6) for b in range(1, 6)]  # 25 problems


class SimpleAlgorithm(Algorithm):
    """Enqueues all problems in one batch, waits, and logs the mean reward.

    Two epochs are run to show how the algorithm can loop.
    """

    def __init__(self, n_epochs: int = 2) -> None:
        super().__init__()
        self.n_epochs = n_epochs
        self.epoch_rewards: List[List[float]] = []

    async def run_async(self) -> None:
        for epoch in range(self.n_epochs):
            logger.info("=== Epoch %d / %d ===", epoch + 1, self.n_epochs)
            self._enqueued_ids.clear()

            # Shuffle the problem order each epoch for variety.
            batch = random.sample(PROBLEMS, len(PROBLEMS))
            for task in batch:
                await self.enqueue(task, mode="train")

            finished: List[Rollout] = await self.wait_all(timeout=30.0)

            # Collect rewards for this epoch.
            rewards: List[float] = []
            for rollout in finished:
                spans = await self.query_spans(rollout.rollout_id)
                reward = find_final_reward(spans)
                if reward is not None:
                    rewards.append(reward)

            mean_reward = sum(rewards) / len(rewards) if rewards else float("nan")
            self.epoch_rewards.append(rewards)
            logger.info(
                "Epoch %d finished: %d rollouts, mean reward = %.3f",
                epoch + 1,
                len(finished),
                mean_reward,
            )

            # In a real algorithm you would update model weights here.


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


async def main() -> None:
    agent = MathAgent()
    algorithm = SimpleAlgorithm(n_epochs=2)

    trainer = MiniTrainer(
        agent=agent,
        algorithm=algorithm,
        n_runners=4,  # 4 workers process problems in parallel
    )

    logger.info("Starting mini-agent-lightning math-QA example …")
    await trainer.fit()

    # Show a sample triplet from the last rollout in the last epoch.
    rollouts = await trainer.store.list_rollouts(status="succeeded")
    if rollouts:
        sample = rollouts[-1]
        spans = await trainer.store.query_spans(sample.rollout_id)
        triplet = spans_to_triplet(spans)
        logger.info("Sample triplet from last rollout: reward=%.1f", triplet.reward or 0.0)

    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
