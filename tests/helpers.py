"""Shared builders and assertions for policy tests.

The toy instance below is small enough to check answers by hand:

    model   cost    easy-0  medium-1  hard-2  hard-3
    cheap   1.0     0.80    0.50      0.10    0.20
    mid     2.0     0.85    0.75      0.40    0.45
    big     4.0     0.90    0.85      0.90    0.80

``make_agents()`` gives 4 agents with one task each (job id == agent id).
``make_multi_agents()`` gives the same 4 tasks to 2 agents: agent 0 owns easy-0 and
medium-1, agent 1 owns hard-2 and hard-3. Jobs are round-robin:

    job     0           1           2             3
    agent   0           1           0             1
    task    easy-0      hard-2      medium-1      hard-3
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Callable

import pytest

from masbudget.centralized import attempts_of
from masbudget.core import FALLBACK, AllocationState, Assignment, Task
from masbudget.registry import AgentRegistry, ModelRegistry

MODEL_COSTS = {"cheap": 1.0, "mid": 2.0, "big": 4.0}
TASKS = [
    ("easy-0", "easy", {"cheap": 0.80, "mid": 0.85, "big": 0.90}),
    ("medium-1", "medium", {"cheap": 0.50, "mid": 0.75, "big": 0.85}),
    ("hard-2", "hard", {"cheap": 0.10, "mid": 0.40, "big": 0.90}),
    ("hard-3", "hard", {"cheap": 0.20, "mid": 0.45, "big": 0.80}),
]
# Budget regimes for N = 4 agents (severe = N * c_min, generous = N * c_max).
BUDGETS = {"zero": 0.0, "scarce": 3.0, "severe": 4.0, "moderate": 8.0, "generous": 16.0}


def make_models(costs: dict[str, float] = MODEL_COSTS) -> ModelRegistry:
    models = ModelRegistry()
    for name, cost in costs.items():
        models.add(name, cost)
    return models


def make_agents(tasks: list[tuple[str, str, dict[str, float]]] = TASKS, tasks_per_agent: int = 1) -> AgentRegistry:
    """Consecutive chunks of ``tasks_per_agent`` tasks per agent."""
    built = [Task(task_id, difficulty, quality) for task_id, difficulty, quality in tasks]
    return AgentRegistry.from_task_lists(
        built[i:i + tasks_per_agent] for i in range(0, len(built), tasks_per_agent)
    )


def make_multi_agents() -> AgentRegistry:
    return make_agents(tasks_per_agent=2)


def make_state(
    agents: AgentRegistry,
    models: ModelRegistry,
    remaining_budget: float,
    total_budget: float | None = None,
    step: int = 0,
    history: list[Assignment] | None = None,
    seed: int = 0,
) -> AllocationState:
    """State seen by the job served at ``step`` with B_k = ``remaining_budget``."""
    return AllocationState(
        step=step,
        total_budget=remaining_budget if total_budget is None else total_budget,
        remaining_budget=remaining_budget,
        agents=agents,
        models=models,
        history=list(history or []),
        rng=random.Random(seed),
    )


def call_or_skip(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a policy method; skip the test while the policy is still a stub."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(f"not implemented yet: {exc}")


def final_model(mapping: dict[int, Any], job_id: int) -> str:
    """The model whose answer the job keeps (the last attempt)."""
    return attempts_of(mapping.get(job_id, FALLBACK))[-1]


def allocation_quality(mapping: dict[int, Any], agents: AgentRegistry, models: ModelRegistry) -> float:
    return sum(models.get(final_model(mapping, j.job_id)).quality(j.task) for j in agents.jobs)


def allocation_cost(mapping: dict[int, Any], agents: AgentRegistry, models: ModelRegistry) -> float:
    """Total paid, counting every attempt."""
    return sum(
        models.get(name).cost for j in agents.jobs for name in attempts_of(mapping.get(j.job_id, FALLBACK))
    )


def brute_force_optimum(agents: AgentRegistry, models: ModelRegistry, budget: float) -> float:
    """Exact MCKP optimum by enumeration (fine for the toy instance only)."""
    options = [models.fallback, *models]
    best = 0.0
    jobs = agents.jobs
    for choice in itertools.product(options, repeat=len(jobs)):
        if sum(m.cost for m in choice) <= budget + 1e-9:
            best = max(best, sum(m.quality(j.task) for m, j in zip(choice, jobs)))
    return best


def assert_valid_centralized_allocation(
    mapping: dict[int, str], agents: AgentRegistry, models: ModelRegistry, budget: float
) -> None:
    assert isinstance(mapping, dict), "allocate() must return {job_id: model_name or [attempts]}"
    assert set(mapping) <= set(agents.job_ids), "unknown job ids in allocation"
    for choice in mapping.values():
        for model_name in attempts_of(choice):
            assert model_name == FALLBACK or model_name in models.names, f"unknown model {model_name!r}"
    assert allocation_cost(mapping, agents, models) <= budget + 1e-9, "allocation exceeds the budget"
