# mini-agent-lightning Example: Math QA

A self-contained end-to-end demonstration of the
[mini-agent-lightning](../../mini_agentlightning/) training loop, inspired by
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and
[mini-sglang](https://github.com/zhaochenyang20/mini-sglang).

## What is mini-agent-lightning?

`mini_agentlightning` is a minimal reimplementation of the core
agent-lightning training loop.  It keeps the essential abstractions—store,
tracer, emitter, agent, algorithm, trainer—but strips away every subsystem
that is not strictly required to run a reinforcement-learning loop:

| Feature              | agent-lightning | mini-agent-lightning |
|----------------------|-----------------|----------------------|
| In-memory store      | ✅              | ✅                   |
| SQLite / MongoDB     | ✅              | ❌                   |
| OTEL tracer          | ✅              | ✅                   |
| AgentOps / Weave     | ✅              | ❌                   |
| LLM proxy            | ✅              | ❌                   |
| HTTP server bridge   | ✅              | ❌                   |
| Distributed runners  | ✅              | ❌                   |
| Core loop (~LOC)     | ~10 000         | ~700                 |

## Included Files

| File          | Role                                                                 |
|---------------|----------------------------------------------------------------------|
| `math_qa.py`  | End-to-end example: integer-addition QA with a fake stochastic agent |
| `README.md`   | This file                                                            |

## Smoke-test Instructions

```bash
# From the repository root
uv run --no-sync python examples/mini_agentlightning/math_qa.py
```

Expected output (values vary due to randomness):

```
...  Starting mini-agent-lightning math-QA example …
...  === Epoch 1 / 2 ===
...    1 + 3 = 4  (expected 4)  → ✓
       ...
...  Epoch 1 finished: 25 rollouts, mean reward = 0.720
...  === Epoch 2 / 2 ===
       ...
...  Epoch 2 finished: 25 rollouts, mean reward = 0.680
...  Sample triplet from last rollout: reward=1.0
...  Done!
```

The example requires no API keys or GPU – the agent uses `random.randint`
as a stand-in for an LLM.
