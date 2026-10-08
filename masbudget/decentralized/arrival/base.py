from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from masbudget.core import Agent, AllocationState
from masbudget.factory import Registry

UNCOORDINATED = "fcfs"  # used when the config says `arrival_policy: null`


class BaseArrivalPolicy(ABC):
    """Decides which waiting agent requests a model next (the permutation pi)."""

    registry_name: str

    def __init__(self, **params: Any):
        self.params = params  # the `arrival_params:` block from the YAML config

    @abstractmethod
    def select_next(self, remaining: list[Agent], state: AllocationState) -> Agent:
        """Return one agent from ``remaining`` (agents that have not acted yet).

        Use ``state.rng`` for any randomness so runs are reproducible.
        """


ARRIVAL_POLICIES: Registry[BaseArrivalPolicy] = Registry("arrival policy")


def create_arrival_policy(name: str | None, **params: Any) -> BaseArrivalPolicy:
    """``None`` means no coordinator: uncoordinated FCFS."""
    return ARRIVAL_POLICIES.create(name or UNCOORDINATED, **params)
