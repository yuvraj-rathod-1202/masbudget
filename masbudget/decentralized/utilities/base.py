from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from masbudget.core import Agent, AllocationState
from masbudget.factory import Registry
from masbudget.models import BaseModel


class BaseUtilityFunction(ABC):
    """Local payoff U_i(m). An agent best-responds by picking the affordable model with the highest value."""

    registry_name: str

    def __init__(self, **params: Any):
        self.params = params  # the `utility_params:` block from the YAML config

    @abstractmethod
    def compute(self, agent: Agent, model: BaseModel, state: AllocationState) -> float:
        """Utility of ``agent`` choosing ``model`` given the current state.

        Useful inputs: ``model.quality(agent.task)`` (q_{i,m}), ``model.cost`` (c_m),
        ``state.remaining_budget`` (B_k), ``state.total_budget`` (B), ``state.step`` (k),
        ``state.models`` and ``state.agents``.
        """


UTILITY_FUNCTIONS: Registry[BaseUtilityFunction] = Registry("utility function")


def create_utility_function(name: str, **params: Any) -> BaseUtilityFunction:
    return UTILITY_FUNCTIONS.create(name, **params)
