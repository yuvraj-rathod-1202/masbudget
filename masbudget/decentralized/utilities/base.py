from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from masbudget.core import AllocationState, Job
from masbudget.factory import Registry
from masbudget.models import BaseModel


class BaseUtilityFunction(ABC):
    """Local payoff U_i(m). For each of its jobs, an agent best-responds by picking the
    affordable model with the highest value."""

    registry_name: str

    def __init__(self, **params: Any):
        self.params = params  # the `utility_params:` block from the YAML config

    @abstractmethod
    def compute(self, job: Job, model: BaseModel, state: AllocationState) -> float:
        """Utility, for the agent owning ``job``, of running the job's task on ``model``.

        Useful inputs: ``model.quality(job.task)`` (q_{i,m}), ``model.cost`` (c_m),
        ``job.agent_id`` / ``state.agents.get(job.agent_id)`` (the owner and all its tasks),
        ``state.remaining_budget`` (B_k), ``state.total_budget`` (B), ``state.step`` (k),
        ``state.models`` and ``state.agents``.
        """


UTILITY_FUNCTIONS: Registry[BaseUtilityFunction] = Registry("utility function")


def create_utility_function(name: str, **params: Any) -> BaseUtilityFunction:
    return UTILITY_FUNCTIONS.create(name, **params)
