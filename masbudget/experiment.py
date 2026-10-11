"""The Experiment: runs centralized and/or decentralized allocation on shared agents & models."""

from __future__ import annotations

import logging
import random
from typing import Any, Sequence

from masbudget.centralized import attempts_of, create_centralized_policy
from masbudget.core import CENTRALIZED, DECENTRALIZED, FALLBACK, Assignment, BudgetLedger, ExperimentResult, RunResult
from masbudget.decentralized import create_arrival_policy, create_decentralized_policy, create_utility_function
from masbudget.metrics.base import MetricSpec, compute_metrics
from masbudget.registry import AgentRegistry, ModelRegistry

logger = logging.getLogger(__name__)


class Experiment:
    """One experiment = one set of agents (each owning several tasks), models and budget,
    evaluated by up to two paradigms. Every task is one job that gets its own model.

    The paradigms that run follow from the three factory names:
      * ``centralized_policy`` set  -> centralized run
      * ``utility`` set             -> decentralized run, ordered by ``arrival_policy``
        (``None`` = uncoordinated FCFS: agents call the model API directly, served
        round-robin over agents)
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
            "Experiment '%s': %d agents, %d jobs, %d models, budget=%g, runs=%d, centralized=%s, decentralized=%s",
            self.name, len(self.agents), len(self.agents.jobs), len(self.models), self.budget, self.num_runs,
            self.centralized and self.centralized.registry_name,
            self.decentralized and self._decentralized_label(),
        )
        result = ExperimentResult(self.name, self.budget, len(self.agents), len(self.agents.jobs))
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
        unknown = set(mapping) - set(self.agents.job_ids)
        if unknown:
            raise ValueError(f"{self.centralized.registry_name} assigned unknown jobs {sorted(unknown)}")

        ledger = BudgetLedger(self.budget)
        assignments = []
        for job in self.agents.jobs:
            attempts = [self.models.get(name) for name in attempts_of(mapping.get(job.job_id, FALLBACK))]
            for model in attempts:
                ledger.charge(model.cost)  # every call is paid; raises if the policy overspends
            final = attempts[-1]  # the job keeps the last answer
            assignments.append(
                Assignment(
                    job.job_id, job.agent_id, final.name, final.quality(job.task), sum(m.cost for m in attempts),
                    attempts=tuple(m.name for m in attempts) if len(attempts) > 1 else (),
                )
            )
        run = RunResult(CENTRALIZED, self.centralized.registry_name, seed, self.budget, assignments)
        self._log_run(run)
        return run

    def _run_decentralized(self, seed: int) -> RunResult:
        if not self.decentralized:
            raise RuntimeError("No decentralized policy configured")
        assignments = self.decentralized.allocate(self.agents, self.models, self.budget, random.Random(seed))
        if sorted(a.job_id for a in assignments) != self.agents.job_ids:
            raise ValueError(f"{self.decentralized.registry_name} must return exactly one assignment per job")
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
        for job in self.agents.jobs:
            missing = [m for m in self.models.names if m not in job.task.quality]
            if missing:
                raise ValueError(f"Task '{job.task.task_id}' has no quality for models {missing}")

    @staticmethod
    def _log_run(run: RunResult) -> None:
        logger.info(
            "  %s seed=%d: quality=%.4f cost=%.6g/%.6g starved=%d",
            run.policy, run.seed, run.total_quality, run.total_cost, run.budget,
            sum(a.starved for a in run.assignments),
        )
        for a in run.assignments:
            logger.debug(
                "    job=%d agent=%d model=%s q=%.4f c=%.6g step=%s attempts=%s",
                a.job_id, a.agent_id, a.model, a.quality, a.cost, a.step, ",".join(a.attempts) or "-",
            )


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
    ``None`` arrival policy means uncoordinated FCFS, i.e. round-robin over agents).
    Extra kwargs go to ``Experiment``."""
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
