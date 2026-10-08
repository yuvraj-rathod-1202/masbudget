"""Core data types shared by policies, metrics and the experiment runner."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from masbudget.registry import AgentRegistry, ModelRegistry

FALLBACK = "fallback"  # name of the zero-quality, zero-cost fallback model
CENTRALIZED = "centralized"
DECENTRALIZED = "decentralized"
DIFFICULTY_LEVELS = ("easy", "medium", "hard")

EPS = 1e-9  # tolerance for floating-point budget comparisons


@dataclass(frozen=True)
class Task:
    """One benchmark task, as loaded from the dataset (see docs/dataset.md)."""

    task_id: str
    difficulty: str
    quality: dict[str, float]  # model name -> expected quality q_{i,m} in [0, 1]
    category: str | None = None
    features: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def difficulty_rank(self) -> int:
        """0 = easy, 1 = medium, 2 = hard."""
        return DIFFICULTY_LEVELS.index(self.difficulty)


@dataclass(frozen=True)
class Agent:
    """An autonomous agent i that owns exactly one task t_i."""

    agent_id: int
    task: Task


@dataclass(frozen=True)
class Assignment:
    """The model an agent ended up with in one run."""

    agent_id: int
    model: str
    quality: float
    cost: float
    step: int | None = None  # arrival position k (decentralized only)
    utility: float | None = None  # agent's own U_i(m) at decision time (decentralized only)
    remaining_budget: float | None = None  # B_k observed before acting (decentralized only)

    @property
    def starved(self) -> bool:
        return self.model == FALLBACK


@dataclass
class AllocationState:
    """What an agent / arrival policy observes at step k of the sequential game."""

    step: int
    total_budget: float  # B
    remaining_budget: float  # B_k
    agents: AgentRegistry
    models: ModelRegistry
    history: list[Assignment]  # decisions of agents that already acted
    rng: random.Random

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def budget_fraction(self) -> float:
        """B_k / B."""
        return self.remaining_budget / self.total_budget if self.total_budget > 0 else 0.0


@dataclass
class RunResult:
    """Outcome of a single allocation run (one paradigm, one seed)."""

    paradigm: str
    policy: str
    seed: int
    budget: float
    assignments: list[Assignment]  # arrival order for decentralized, agent order for centralized

    @property
    def qualities(self) -> list[float]:
        return [a.quality for a in self.assignments]

    @property
    def total_quality(self) -> float:
        return sum(self.qualities)

    @property
    def total_cost(self) -> float:
        return sum(a.cost for a in self.assignments)


@dataclass
class ExperimentResult:
    """All runs of one experiment, grouped by paradigm, plus computed metrics."""

    name: str
    budget: float
    num_agents: int
    runs: dict[str, list[RunResult]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


class BudgetExceededError(RuntimeError):
    pass


class BudgetLedger:
    """Shared budget B; every model call is charged against it."""

    def __init__(self, total: float):
        if total < 0:
            raise ValueError(f"Budget must be non-negative, got {total}")
        self.total = float(total)
        self.spent = 0.0

    @property
    def remaining(self) -> float:
        return self.total - self.spent

    def can_afford(self, cost: float) -> bool:
        return cost <= self.remaining + EPS

    def charge(self, cost: float) -> None:
        if not self.can_afford(cost):
            raise BudgetExceededError(
                f"Cannot charge {cost:.6g}: only {self.remaining:.6g} of {self.total:.6g} left"
            )
        self.spent += cost
