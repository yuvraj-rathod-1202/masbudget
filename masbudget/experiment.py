"""The Experiment: runs centralized and/or decentralized allocation on shared agents & models."""

from __future__ import annotations

import logging
import random
from typing import Any, Sequence

from masbudget.centralized import create_centralized_policy
from masbudget.core import CENTRALIZED, DECENTRALIZED, FALLBACK, Assignment, BudgetLedger, ExperimentResult, RunResult
from masbudget.decentralized import create_arrival_policy, create_decentralized_policy, create_utility_function
from masbudget.metrics.base import MetricSpec, compute_metrics
from masbudget.registry import AgentRegistry, ModelRegistry

logger = logging.getLogger(__name__)


class Experiment:
    """One experiment = one set of agents, models and budget, evaluated by up to two paradigms.

    The paradigms that run follow from the three factory names:
      * ``centralized_policy`` set  -> centralized run
      * ``utility`` set             -> decentralized run, ordered by ``arrival_policy``
        (``None`` = uncoordinated FCFS: agents call the model API directly)
    Both use the very same ``AgentRegistry`` and ``ModelRegistry``.
    """

    def __init__(
        self,
        agents: AgentRegistry,
        models: ModelRegistry,
        budget: float,
        *,
        centralized_policy: str | None = None,
        utility: str | None = None,
        arrival_policy: str | None = None,
        centralized_params: dict[str, Any] | None = None,
        utility_params: dict[str, Any] | None = None,
        arrival_params: dict[str, Any] | None = None,
        decentralized_policy: str = "sequential_best_response",
        decentralized_params: dict[str, Any] | None = None,
        name: str = "experiment",
        seed: int = 0,
        num_runs: int = 1,
    ):
        if centralized_policy is None and utility is None:
            raise ValueError("Nothing to run: give a centralized_policy and/or a decentralized utility")
        if not len(agents):
            raise ValueError("No agents registered")
        if not len(models):
            raise ValueError("No models registered")
        if budget < 0:
            raise ValueError(f"budget must be non-negative, got {budget}")
        if num_runs < 1:
            raise ValueError(f"num_runs must be >= 1, got {num_runs}")

        self.name = name
        self.agents = agents
        self.models = models
        self.budget = float(budget)
        self.seed = seed
        self.num_runs = num_runs
        self._check_quality_coverage()

        self.centralized = (
            create_centralized_policy(centralized_policy, **(centralized_params or {}))
            if centralized_policy is not None
            else None
        )
        self.decentralized = (
            create_decentralized_policy(
                decentralized_policy,
                create_arrival_policy(arrival_policy, **(arrival_params or {})),
                create_utility_function(utility, **(utility_params or {})),
                **(decentralized_params or {}),
            )
            if utility is not None
            else None
        )
        self.result: ExperimentResult | None = None

    def run(self, metrics: Sequence[MetricSpec] | None = None) -> ExperimentResult:
        """Run every paradigm ``num_runs`` times (run r uses seed + r). If ``metrics`` is given,
        they are computed and stored in ``result.metrics``."""
        logger.info(
            "Experiment '%s': %d agents, %d models, budget=%g, runs=%d, centralized=%s, decentralized=%s",
            self.name, len(self.agents), len(self.models), self.budget, self.num_runs,
            self.centralized and self.centralized.registry_name,
            self.decentralized and self._decentralized_label(),
        )
        result = ExperimentResult(self.name, self.budget, len(self.agents))
        for r in range(self.num_runs):
            run_seed = self.seed + r
            if self.centralized is not None:
                result.runs.setdefault(CENTRALIZED, []).append(self._run_centralized(run_seed))
            if self.decentralized is not None:
                result.runs.setdefault(DECENTRALIZED, []).append(self._run_decentralized(run_seed))
        self.result = result
        if metrics:
            result.metrics = self.evaluate(metrics)
        return result

    def evaluate(self, metrics: Sequence[MetricSpec]) -> dict[str, Any]:
        """Compute the named metrics on the last run's result."""
        if self.result is None:
            raise RuntimeError("Call run() before evaluate()")
        return compute_metrics(self.result, metrics)

    def _run_centralized(self, seed: int) -> RunResult:
        if not self.centralized:
            raise RuntimeError("No centralized policy configured")
        mapping = self.centralized.allocate(self.agents, self.models, self.budget, random.Random(seed))
        unknown = set(mapping) - set(self.agents.ids)
        if unknown:
            raise ValueError(f"{self.centralized.registry_name} assigned unknown agents {sorted(unknown)}")

        ledger = BudgetLedger(self.budget)
        assignments = []
        for agent in self.agents:
            model = self.models.get(mapping.get(agent.agent_id, FALLBACK))
            ledger.charge(model.cost)  # raises if the policy overspends
            assignments.append(Assignment(agent.agent_id, model.name, model.quality(agent.task), model.cost))
        run = RunResult(CENTRALIZED, self.centralized.registry_name, seed, self.budget, assignments)
        self._log_run(run)
        return run

    def _run_decentralized(self, seed: int) -> RunResult:
        if not self.decentralized:
            raise RuntimeError("No decentralized policy configured")
        assignments = self.decentralized.allocate(self.agents, self.models, self.budget, random.Random(seed))
        if sorted(a.agent_id for a in assignments) != sorted(self.agents.ids):
            raise ValueError(f"{self.decentralized.registry_name} must return exactly one assignment per agent")
        if sum(a.cost for a in assignments) > self.budget + 1e-9:
            raise ValueError(f"{self._decentralized_label()} exceeded the budget")
        run = RunResult(DECENTRALIZED, self._decentralized_label(), seed, self.budget, assignments)
        self._log_run(run)
        return run

    def _decentralized_label(self) -> str:
        if not self.decentralized:
            raise RuntimeError("No decentralized policy configured")
        d = self.decentralized
        return f"{d.registry_name}[arrival={d.arrival_policy.registry_name}, utility={d.utility.registry_name}]"

    def _check_quality_coverage(self) -> None:
        for agent in self.agents:
            missing = [m for m in self.models.names if m not in agent.task.quality]
            if missing:
                raise ValueError(f"Task '{agent.task.task_id}' has no quality for models {missing}")

    @staticmethod
    def _log_run(run: RunResult) -> None:
        logger.info(
            "  %s seed=%d: quality=%.4f cost=%.6g/%.6g starved=%d",
            run.policy, run.seed, run.total_quality, run.total_cost, run.budget,
            sum(a.starved for a in run.assignments),
        )
        for a in run.assignments:
            logger.debug("    agent=%d model=%s q=%.4f c=%.6g step=%s", a.agent_id, a.model, a.quality, a.cost, a.step)


def run_experiment(
    centralized_policy: str | None,
    utility: str | None,
    arrival_policy: str | None,
    *,
    agents: AgentRegistry,
    models: ModelRegistry,
    budget: float,
    metrics: Sequence[MetricSpec] | None = None,
    **kwargs: Any,
) -> ExperimentResult:
    """Main entry point: name the centralized policy, the decentralized utility and the
    arrival policy (factory names; ``None`` skips centralized / decentralized, and a
    ``None`` arrival policy means uncoordinated FCFS). Extra kwargs go to ``Experiment``."""
    experiment = Experiment(
        agents,
        models,
        budget,
        centralized_policy=centralized_policy,
        utility=utility,
        arrival_policy=arrival_policy,
        **kwargs,
    )
    return experiment.run(metrics)
