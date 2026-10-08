from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from masbudget.core import Assignment
from masbudget.decentralized.arrival.base import BaseArrivalPolicy
from masbudget.decentralized.utilities.base import BaseUtilityFunction
from masbudget.factory import Registry
from masbudget.registry import AgentRegistry, ModelRegistry


class BaseDecentralizedPolicy(ABC):
    """Agents choose models themselves; ordering and payoffs come from the plugged-in policies."""

    registry_name: str

    def __init__(self, arrival_policy: BaseArrivalPolicy, utility: BaseUtilityFunction, **params: Any):
        self.arrival_policy = arrival_policy
        self.utility = utility
        self.params = params

    @abstractmethod
    def allocate(
        self, agents: AgentRegistry, models: ModelRegistry, budget: float, rng: random.Random
    ) -> list[Assignment]:
        """Return one ``Assignment`` per agent, in the order the agents acted."""


DECENTRALIZED_POLICIES: Registry[BaseDecentralizedPolicy] = Registry("decentralized policy")


def create_decentralized_policy(
    name: str, arrival_policy: BaseArrivalPolicy, utility: BaseUtilityFunction, **params: Any
) -> BaseDecentralizedPolicy:
    return DECENTRALIZED_POLICIES.create(name, arrival_policy, utility, **params)
