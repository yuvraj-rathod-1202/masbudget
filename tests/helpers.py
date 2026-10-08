"""Shared builders and assertions for policy tests.

The toy instance below is small enough to check answers by hand:

    model   cost    easy-0  medium-1  hard-2  hard-3
    cheap   1.0     0.80    0.50      0.10    0.20
    mid     2.0     0.85    0.75      0.40    0.45
    big     4.0     0.90    0.85      0.90    0.80
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Callable

import pytest

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


def make_agents(tasks: list[tuple[str, str, dict[str, float]]] = TASKS) -> AgentRegistry:
    return AgentRegistry.from_tasks(Task(task_id, difficulty, quality) for task_id, difficulty, quality in tasks)


def make_state(
    agents: AgentRegistry,
    models: ModelRegistry,
    remaining_budget: float,
    total_budget: float | None = None,
    step: int = 0,
    history: list[Assignment] | None = None,
    seed: int = 0,
) -> AllocationState:
    """State seen by the agent acting at ``step`` with B_k = ``remaining_budget``."""
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


def allocation_quality(mapping: dict[int, str], agents: AgentRegistry, models: ModelRegistry) -> float:
    return sum(models.get(mapping.get(a.agent_id, FALLBACK)).quality(a.task) for a in agents)


def allocation_cost(mapping: dict[int, str], agents: AgentRegistry, models: ModelRegistry) -> float:
    return sum(models.get(mapping.get(a.agent_id, FALLBACK)).cost for a in agents)


def brute_force_optimum(agents: AgentRegistry, models: ModelRegistry, budget: float) -> float:
    """Exact MCKP optimum by enumeration (fine for the toy instance only)."""
    options = [models.fallback, *models]
    best = 0.0
    for choice in itertools.product(options, repeat=len(agents)):
        if sum(m.cost for m in choice) <= budget + 1e-9:
            best = max(best, sum(m.quality(a.task) for m, a in zip(choice, agents)))
    return best


def assert_valid_centralized_allocation(
    mapping: dict[int, str], agents: AgentRegistry, models: ModelRegistry, budget: float
) -> None:
    assert isinstance(mapping, dict), "allocate() must return {agent_id: model_name}"
    assert set(mapping) <= set(agents.ids), "unknown agent ids in allocation"
    for model_name in mapping.values():
        assert model_name == FALLBACK or model_name in models.names, f"unknown model {model_name!r}"
    assert allocation_cost(mapping, agents, models) <= budget + 1e-9, "allocation exceeds the budget"
