# Agent-lightning vs rLLM: A Comprehensive Architectural Comparison

This document offers a thorough, side-by-side comparison of
[Agent-lightning](https://github.com/microsoft/agent-lightning) and
[rLLM](https://github.com/rllm-org/rllm) — two open-source frameworks that
share the same north-star goal: train LLM-based agents with reinforcement
learning using near-zero code changes.

---

## Background

| | Agent-lightning | rLLM |
|---|---|---|
| **Origin** | Microsoft Research | Berkeley Sky Computing Lab |
| **License** | MIT | Apache-2.0 |
| **Python requirement** | 3.10+ | 3.10+ (3.11+ for Tinker backend) |
| **Primary paper** | [arXiv 2508.03680](https://arxiv.org/abs/2508.03680) | [rLLM blog](https://pretty-radio-b75.notion.site/rLLM-A-Framework-for-Post-Training-Language-Agents-21b81902c146819db63cd98a54ba5f31) |
| **Package name** | `agentlightning` | `rllm` |
| **Key claim** | Train **any** agent with RL, zero code change | Train AI agents with RL, any framework, minimal code changes |

---

## High-level Architecture Diagrams

### Agent-lightning

```
┌───────────┐  enqueue_rollout  ┌───────────────┐  dequeue_rollout  ┌─────────────┐
│ Algorithm │ ───────────────▶  │ LightningStore │ ◀─────────────── │   Runner    │
│           │ ◀─────────────── │   (central hub)│ ───────────────▶ │  (LitAgent) │
│           │  query + update   │               │   add_span(s)     │             │
└───────────┘                   └───────────────┘                   └─────────────┘
      │                                                                     │
      ▼                                                              ┌──────▼──────┐
 Algorithms                                                          │  LLM Proxy  │
 (VERL/GRPO,                                                         │ (LiteLLM)   │
  APO, SFT…)                                                         └─────────────┘
```

The `LightningStore` is the **center of gravity** — it owns the full lifecycle of
every rollout (queuing, preparing, running, succeeded, failed, requeuing) and acts
as the message bus between the Algorithm and the Runners.

### rLLM

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Your Agent  │─▶│    Traces    │─▶│   Rewards    │─▶│  RL Update   │
│ (@rollout)   │  │  (auto-logged│  │ (your logic) │  │ (GRPO etc.)  │
└──────────────┘  │  via proxy)  │  └──────────────┘  └──────────────┘
                  └──────────────┘
                        ▲
               ┌────────┴────────┐
               │  LiteLLM Proxy  │  ← intercepts every LLM call
               └─────────────────┘
```

rLLM follows a **linear pipeline**: the agent runs, the proxy intercepts every
LLM call and records a trace, a reward function scores the full episode, and a
training backend updates the model weights.

---

## Module Structure Comparison

### Agent-lightning (`agentlightning/`)

| Module | Responsibility |
|---|---|
| `litagent/` | `LitAgent` — the base class users subclass to implement `rollout()` |
| `runner/` | Dequeues rollouts, invokes `LitAgent.rollout()`, streams spans back |
| `store/` | `LightningStore` — central hub (memory, MongoDB, SQLite backends) |
| `algorithm/` | Algorithm base class; built-in: VERL/GRPO, APO, SFT |
| `tracer/` | OpenTelemetry-backed span collection; integrations with AgentOps, Weave |
| `adapter/` | `TraceAdapter` — converts raw OTEL spans into training-ready triplets |
| `emitter/` | `agl.emit_reward()` / `agl.emit_log()` helpers for explicit feedback |
| `trainer/` | `Trainer` — wires Store, Algorithm, and Runner together |
| `llm_proxy.py` | LiteLLM-based proxy capturing token IDs and logprobs |
| `execution/` | Inter-process communication (shared memory, client-server) |
| `types/` | Pydantic models: `Rollout`, `Span`, `Triplet`, `Resources`, `Worker` |
| `verl/` | VERL async training server and dataset utilities |
| `cli/` | `agl` CLI for launching training, server, and dashboards |
| `dashboard/` | React-based real-time monitoring UI |
| `instrumentation/` | OpenTelemetry, AgentOps, Weave, dummy tracers |

### rLLM (`rllm/`)

| Module | Responsibility |
|---|---|
| `sdk/` | Core SDK: `@rllm.rollout` decorator, proxy, session, tracers, shortcuts |
| `sdk/proxy/` | LiteLLM proxy server that transparently captures token IDs + logprobs |
| `sdk/tracers/` | Trace storage backends (memory, SQLite) |
| `workflows/` | Parallel rollout execution engines |
| `engine/` | `AgentExecutionEngine`, `AgentWorkflowEngine`, `AgentSDKEngine` |
| `trainer/` | `AgentTrainer` — dispatches to VERL, Tinker, or Fireworks backends |
| `agents/` | Built-in reference agents (math, code, SWE, AppWorld, WebArena…) |
| `environments/` | Built-in environments (FrozenLake, BrowserGym, AppWorld, code, SWE…) |
| `rewards/` | Built-in reward functions (math, code, search, countdown) |
| `data/` | Dataset wrappers for 50+ benchmarks |
| `integrations/` | Verifiers integration |
| `registry/` | Component registry for agents, environments, reward functions |
| `types.py` | Pydantic models: `Step`, `Trajectory`, `Episode` |
| `experimental/` | Unified trainer, eval protocol |
| `trajectory_visualizer.py` | HTML-based trajectory visualization tool |

---

## Core Data Models

### Agent-lightning

The fundamental data primitive is the **OpenTelemetry Span** (see [`Span`][agentlightning.Span] in the API reference):

```python
# A span maps naturally to a single LLM interaction
Span(
    span_id="...",
    rollout_id="...",
    attributes={
        "llm.messages": [...],      # prompt
        "llm.output": "...",        # response
        "agl.reward": 1.0,          # reward signal
    }
)
```

Spans are aggregated into `Rollout` objects that carry status and lifecycle
metadata. The `TraceAdapter` then converts span lists into `Triplet` objects
(prompt/response/reward) consumed by training algorithms.

```python
class Triplet(BaseModel):
    prompt: Any
    response: Any
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### rLLM

rLLM uses a three-level typed hierarchy:

```python
Step(input, output, reward)           # one LLM call
Trajectory(name, steps, reward)       # one agent run
Episode(trajectories, artifacts)      # one task (multiple rollouts)
```

The `Episode` → `Trajectory` → `Step` structure maps naturally to multi-agent
scenarios where each agent produces its own `Trajectory` within a shared
`Episode`.

### Comparison

| Aspect | Agent-lightning | rLLM |
|---|---|---|
| **Core primitive** | OpenTelemetry `Span` (OTEL-native) | `Step`/`Trajectory`/`Episode` (domain-specific) |
| **Reward attachment** | Span attribute `agl.reward` | `Step.reward` + `Trajectory.reward` |
| **Hierarchy** | Flat (all spans belong to a rollout) | Nested (Episode > Trajectory > Step) |
| **Observability interop** | Built-in OTEL export | Custom SQLite/memory storage |
| **Multi-agent** | Parallel spans under one rollout | One Trajectory per agent per Episode |

---

## The LLM Proxy

Both frameworks place a **LiteLLM-based proxy** between the agent and the model
server to transparently intercept every LLM call:

| Feature | Agent-lightning | rLLM |
|---|---|---|
| **Proxy implementation** | `agentlightning/llm_proxy.py` via `litellm.proxy` | `rllm/sdk/proxy/litellm_server.py` |
| **Token capture** | Yes — via OTEL spans written to the store | Yes — via LiteLLM callbacks → SQLite tracer |
| **Logprobs capture** | Yes (vLLM-compatible extensions) | Yes |
| **Span export** | OTLP/HTTP or in-process | In-process callback |
| **Proxy lifetime** | Managed by `LLMProxy` helper class | Managed by `ProxyManager` |
| **Hot reload** | Weights pushed as store resources | Weights swapped by training backend |

Both frameworks exploit the same insight published in the vLLM blog: returning
token IDs directly from the OpenAI-compatible API avoids retokenization drift
during training.

---

## Trace Collection & Instrumentation

### Agent-lightning

Agent-lightning embraces **OpenTelemetry as its canonical tracing protocol**.
The framework ships multiple `Tracer` implementations:

- `OtelTracer` — standard OTEL SDK tracer; can export to any OTLP-compatible backend.
- `AgentOpsTracer` — wraps AgentOps for hosted observability.
- `WeaveTracer` — wraps Weights & Biases Weave.
- `DummyTracer` — no-op, useful in tests.

Any existing OpenTelemetry instrumentation (LangChain, OpenAI SDK auto-instrumentation, etc.) is automatically picked up.

### rLLM

rLLM stores traces in SQLite or memory. Trace collection is done through LiteLLM callbacks: when the proxy receives a response it calls `litellm_callbacks.py`, which serializes the step and writes it to the current session's store.

Third-party observability is not yet first-class — users wanting dashboards must plug into rLLM's own `trajectory_visualizer.py` or build their own.

---

## Rollout Management & Worker Lifecycle

### Agent-lightning

The `LightningStore` maintains a **full rollout lifecycle**:

```
queuing → preparing → running → succeeded
                              ↘ failed → requeuing
                              ↘ cancelled
```

Each rollout has one or more `Attempt` records. The framework supports:

- **Timeout detection** — rollouts that run longer than `timeout_seconds` are cancelled.
- **Unresponsive detection** — rollouts with no new span for `unresponsive_seconds` are flagged.
- **Retry logic** — `RolloutConfig.max_attempts` and `retry_condition` enable automatic retries on failure.
- **Worker heartbeat** — every `Runner` reports heartbeat stats; the store can detect dead workers.

### rLLM

rLLM's rollout management is handled inside `AgentExecutionEngine` and `AgentWorkflowEngine`. Parallelism is achieved via `asyncio` tasks or Ray actors. There is no explicit attempt/retry model baked into the protocol — fault tolerance is delegated to Ray (for the VERL backend) or to the user.

---

## Training Backends

Both frameworks converge on **two backends**: a single-machine backend called
"Tinker" and a distributed GPU backend built on VERL + Ray.

| Feature | Agent-lightning | rLLM |
|---|---|---|
| **Single-machine** | Tinker | Tinker |
| **Distributed** | VERL (Ray + vLLM/SGLang) | VERL (Ray + vLLM/SGLang) |
| **Fireworks pipeline** | ✗ | ✓ (pipeline-optimized) |
| **Backend abstraction** | `Algorithm` class hierarchy | `AgentTrainer(backend="...")` |
| **Config format** | YAML / dataclass | Hydra / YAML / dataclass |
| **Async training** | ✓ (async Algorithm.run()) | ✓ (async workflows) |

---

## Supported Algorithms

| Algorithm | Agent-lightning | rLLM |
|---|---|---|
| **GRPO** | ✓ (via VERL) | ✓ (via VERL or Tinker) |
| **REINFORCE** | ✓ | ✓ |
| **RLOO** | ✓ | ✓ |
| **PPO** | ✓ (via VERL) | ✓ (via VERL) |
| **Rejection sampling / SFT** | ✓ | ✓ |
| **On-policy distillation** | ✗ | ✓ |
| **APO (Automatic Prompt Optimization)** | ✓ (unique feature) | ✗ |
| **Custom algorithm** | ✓ (subclass `Algorithm`) | ✓ (custom workflow) |

APO is Agent-lightning's most distinctive algorithm: it optimizes *prompt
templates* rather than model weights, enabling non-GPU training pipelines. rLLM's
`distillation_workflow` fills a different niche — it distills from a stronger
teacher model into a weaker student.

---

## Built-in Environments and Benchmarks

This is the **biggest practical difference** between the two frameworks.

| | Agent-lightning | rLLM |
|---|---|---|
| **Built-in envs** | ✗ (bring your own) | ✓ AppWorld, BrowserGym, FrozenLake, SWE, code |
| **Built-in reward functions** | ✗ (bring your own) | ✓ math, code, search, countdown |
| **Built-in benchmark datasets** | ✗ | ✓ 50+ via `rllm eval <benchmark>` |
| **Built-in reference agents** | ✗ | ✓ math, code, SWE, AppWorld, WebArena, tool |
| **CLI eval / train** | `agl train ...` | `rllm eval gsm8k` / `rllm train gsm8k` |

Agent-lightning positions itself as a *framework* — it provides the wiring but
expects users to supply environments and rewards. rLLM includes a rich
**benchmark battery** and lets users start training immediately with `rllm train
gsm8k` and no custom code.

---

## Multi-Agent Support

Both frameworks target multi-agent systems, but with different granularity.

**Agent-lightning** supports **selective optimization**: in a multi-agent
pipeline, you can choose to train only a subset of agents (e.g., the planner but
not the executor). Each trainable agent registers its own `LitAgent` and
participates in the store loop independently. The `LightningStore` serializes the
resources each agent needs.

**rLLM** models multi-agent systems through the `Episode` > `Trajectory`
hierarchy: each agent in the pipeline generates its own `Trajectory` inside a
shared `Episode`. Training signal can be applied at the trajectory or step level.
The `AgentSDKEngine` handles multi-agent SDK-style rollouts where multiple
decorated functions cooperate inside one episode.

---

## Observability & Debugging

| Feature | Agent-lightning | rLLM |
|---|---|---|
| **Real-time dashboard** | ✓ (React UI, `agl dashboard`) | ✓ (`rllm ui`) |
| **Trajectory visualizer** | ✓ (span timeline in dashboard) | ✓ (`trajectory_visualizer.py`) |
| **OTEL export** | ✓ native | ✗ (custom storage) |
| **Hosted platform support** | AgentOps, Weave | ✗ built-in |
| **Span-level debugging** | ✓ (OTEL attributes) | ✓ (SQLite queries) |
| **Worker health monitoring** | ✓ (heartbeat, unresponsive detection) | ✗ (delegated to Ray) |

---

## Developer Experience Summary

| | Agent-lightning | rLLM |
|---|---|---|
| **Minimal usage pattern** | Subclass `LitAgent`, implement `rollout()` | Decorate function with `@rllm.rollout` |
| **Reward injection** | `agl.emit_reward(value)` in agent code | Return `Episode` with rewards from `@rllm.evaluator` |
| **Config system** | YAML / Python dataclass | Hydra YAML / Python |
| **CLI** | `agl train`, `agl serve`, `agl dashboard` | `rllm eval`, `rllm train`, `rllm model setup` |
| **Zero-shot benchmark** | ✗ needs custom code | ✓ `rllm eval gsm8k` |
| **Production robustness** | ✓ (retry, timeout, heartbeat) | Research-grade |
| **Async first** | ✓ | ✓ |

---

## Similarities

Despite their differences, the two frameworks share a substantial common core:

1. **LiteLLM as the transparent proxy.** Both inject a LiteLLM-based proxy so
   that agent code requires no LLM-provider changes for training.
2. **VERL + Ray for distributed training.** Both delegate large-scale GPU
   training to [VERL](https://github.com/volcengine/verl) and its Ray-based
   actor model.
3. **Tinker for lightweight training.** Both ship a "Tinker" single-machine
   backend that works without GPUs or Ray, useful for debugging and CPU-scale
   experiments.
4. **SQLite-backed trace storage.** Both use SQLite as a persistence layer for
   traces (Agent-lightning via `SQLiteLightningStore`; rLLM via
   `rllm/sdk/tracers/sqlite.py`).
5. **Framework agnosticism.** Both work with LangChain, OpenAI Agents SDK,
   AutoGen, CrewAI, LangGraph, Strands, SmolAgent, or raw `openai.OpenAI`.
6. **Pydantic data models.** Both use Pydantic `BaseModel` for all core types.
7. **Python ≥ 3.10** and `pyproject.toml`-based packaging.
8. **MkDocs documentation** with the Material theme.
9. **GRPO, REINFORCE, RLOO, PPO, rejection sampling** — the same algorithm
   family covers both.
10. **Multi-agent support** for complex, cooperative agent pipelines.

---

## Differences

### 1. Architectural Paradigm

| | Agent-lightning | rLLM |
|---|---|---|
| **Pattern** | Hub-and-spoke (LightningStore at center) | Linear pipeline (Agent → Traces → Rewards → Update) |
| **Data bus** | LightningStore (queues, resources, spans) | LiteLLM proxy + SQLite tracer |
| **Control flow** | Algorithm enqueues tasks; Runners dequeue and execute | Engine launches parallel async agent coroutines |

### 2. Trace Protocol

Agent-lightning uses **OpenTelemetry** as its canonical wire format, meaning any
OTEL-instrumented library (database calls, HTTP calls, tool calls) is
automatically captured and can be streamed to any OTLP-compatible backend.

rLLM uses a **domain-specific Episode/Trajectory/Step model** that is easier to
reason about for RL developers but does not interoperate with the broader OTEL
ecosystem.

### 3. Resource Management

Agent-lightning introduces an explicit **Resources** abstraction: model weights,
prompt templates, or any other trainable asset are versioned in the store and
distributed to runners atomically. This enables hot model swapping during
training without restarting runners.

rLLM delegates resource management to the training backend (VERL/Tinker). There
is no equivalent "push new weights to running agents" mechanism in the framework
itself.

### 4. Production Readiness

Agent-lightning was designed with production deployments in mind:

- `RolloutConfig.max_attempts` + `retry_condition` for automatic retries.
- `timeout_seconds` / `unresponsive_seconds` for stuck-task detection.
- Worker heartbeat and status tracking across the entire worker fleet.
- Pluggable store backends including MongoDB for durable, high-throughput storage.

rLLM is primarily a research tool; fault tolerance is left to Ray or the user.

### 5. Built-in Content

rLLM ships with a substantial library of benchmarks, environments, reference
agents, and reward functions — everything needed to replicate published results
(DeepScaleR, DeepCoder, DeepSWE) out of the box.

Agent-lightning is a pure framework: it provides no benchmarks, environments, or
reward functions and expects users to supply them.

### 6. APO vs. Distillation

Agent-lightning's APO algorithm uniquely supports **gradient-free prompt
optimization**: it tunes prompt templates using RL without touching model weights,
making it accessible on CPU-only machines.

rLLM's `distillation_workflow` supports **on-policy distillation** from a
stronger teacher to a weaker student — a complementary training paradigm that
Agent-lightning currently lacks.

---

## Mutual Improvement Opportunities

### What Agent-lightning can borrow from rLLM

1. **Built-in benchmark and environment library.**
   An official `agentlightning-benchmarks` package (or `contrib/benchmarks/`)
   mirroring rLLM's 50+ datasets would remove the biggest friction point for new
   users who want a zero-code "train on GSM8K" experience.

2. **Episode/Trajectory/Step as an optional higher-level abstraction.**
   The raw OTEL span model is powerful but opaque for newcomers. Offering an
   optional typed wrapper (Episode → Trajectory → Step) as a view over raw spans
   would lower the learning curve without abandoning the OTEL core.

3. **On-policy distillation workflow.**
   rLLM's `distillation_workflow` is a well-tested approach for knowledge
   transfer. Adding a `DistillationAlgorithm` to Agent-lightning's algorithm zoo
   would broaden its training options.

4. **Registry pattern for components.**
   rLLM's `registry/` makes it easy to discover and compose environments, agents,
   and reward functions by name. Agent-lightning could adopt a similar registry
   for its algorithm zoo and adapter library.

5. **`@rollout` function decorator.**
   rLLM's decorator-based API (`@rllm.rollout`) is slightly more approachable
   than subclassing `LitAgent` for simple, single-function agents. Agent-lightning
   already has `@rollout` support but it can be more prominently featured in the
   quickstart.

### What rLLM can borrow from Agent-lightning

1. **LightningStore's rollout lifecycle and retry model.**
   rLLM currently has no built-in retry or timeout mechanism. Adopting a
   lightweight version of Agent-lightning's `RolloutConfig` (max_attempts,
   retry_condition, timeout_seconds) would make rLLM substantially more reliable
   for long-running or flaky environments.

2. **OpenTelemetry as the trace wire format.**
   Emitting OTEL spans from rLLM's proxy would unlock compatibility with the
   entire observability ecosystem (Jaeger, Grafana Tempo, AgentOps, Weave,
   Honeycomb) with no additional integration work.

3. **MongoDB and distributed store backend.**
   rLLM's SQLite/memory stores are single-machine. For large-scale, multi-machine
   rollout collection, a MongoDB-backed store (like Agent-lightning's) would
   enable truly decentralized worker pools.

4. **Explicit Resources versioning.**
   Providing a versioned "resources" concept (model checkpoint path + metadata)
   that runners can query would make it safe to hot-swap models during rLLM
   training without restarting the entire worker fleet.

5. **Selective per-agent optimization.**
   rLLM treats an Episode as a single training unit. Adding Agent-lightning-style
   selective optimization — where individual trajectories within an episode carry
   separate trainable resource identifiers — would allow fine-grained per-role
   fine-tuning in multi-agent systems.

6. **APO (Automatic Prompt Optimization) algorithm.**
   Incorporating Agent-lightning's APO into rLLM's algorithm suite would give
   researchers a gradient-free, CPU-accessible optimization path for prompt
   engineering, complementing the existing weight-update methods.

7. **Worker heartbeat and health monitoring.**
   rLLM's dashboard shows training curves but lacks per-worker health metrics.
   Adopting Agent-lightning's heartbeat mechanism (last_heartbeat_time,
   last_busy_time, worker status) would improve debuggability of large Ray
   clusters.

---

## Decision Guide: When to choose which

```
Is zero-code integration and battery-included benchmarks your top priority?
  → rLLM  (rllm eval gsm8k "just works")

Do you need production-grade fault tolerance, retries, and worker monitoring?
  → Agent-lightning

Do you want to optimize prompt templates instead of model weights (CPU-only)?
  → Agent-lightning  (APO algorithm)

Do you need on-policy distillation from a teacher model?
  → rLLM  (distillation_workflow)

Do you want OpenTelemetry-native traces compatible with Jaeger / Grafana Tempo?
  → Agent-lightning

Do you work with complex multi-agent architectures requiring selective per-agent optimization?
  → Agent-lightning

Do you want to reproduce published state-of-the-art results (DeepSWE, DeepCoder)?
  → rLLM  (built-in environments and reward functions)
```

---

## Conclusion

Agent-lightning and rLLM converge on the same core insight — intercept LLM calls
through a transparent proxy, collect traces, score with a reward function, and
update the model — but they diverge significantly in scope and design philosophy.

**Agent-lightning** is an infrastructure framework: it provides a robust,
production-ready backbone (the LightningStore hub, retry/timeout management, OTEL
traces, resource versioning) but ships no benchmarks or environments. It is the
right choice when you need a reliable, extensible backbone for a custom training
pipeline or when you want to plug RL training into an existing production agent
system.

**rLLM** is a batteries-included research platform: it ships with 50+ benchmarks,
a rich set of environments, reference agents, and reward functions that let
researchers get results quickly. It is the right choice when you want to reproduce
or extend a published result, run a new benchmark, or prototype a new RL
algorithm against standard tasks.

The two frameworks are highly complementary and, given their shared VERL/Tinker
backends and LiteLLM proxy design, there is a realistic path to deeper
interoperability — for example, running rLLM's benchmark environments through
Agent-lightning's fault-tolerant store, or adopting Agent-lightning's APO
algorithm as an rLLM workflow.
