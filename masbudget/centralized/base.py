from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from masbudget.factory import Registry
from masbudget.registry import AgentRegistry, ModelRegistry


class BaseCentralizedPolicy(ABC):
    """A controller that sees all agents, tasks, model costs and qualities at once."""

    registry_name: str

    def __init__(self, **params: Any):
        self.params = params  # the `params:` block from the YAML config

    @abstractmethod
    def allocate(
        self, agents: AgentRegistry, models: ModelRegistry, budget: float, rng: random.Random
    ) -> dict[int, str]:
        """Return ``{agent_id: model_name}``.

        Map an agent to ``FALLBACK`` (or leave it out) to give it no model. The total cost
        must not exceed ``budget``; the experiment re-checks this with a ``BudgetLedger``.
        """


CENTRALIZED_POLICIES: Registry[BaseCentralizedPolicy] = Registry("centralized policy")


def create_centralized_policy(name: str, **params: Any) -> BaseCentralizedPolicy:
    return CENTRALIZED_POLICIES.create(name, **params)
