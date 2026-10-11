from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Sequence

from masbudget.factory import Registry
from masbudget.registry import AgentRegistry, ModelRegistry


class BaseCentralizedPolicy(ABC):
    """A controller that sees all agents, their tasks, model costs and qualities at once."""

    registry_name: str

    def __init__(self, **params: Any):
        self.params = params  # the `params:` block from the YAML config

    @abstractmethod
    def allocate(
        self, agents: AgentRegistry, models: ModelRegistry, budget: float, rng: random.Random
    ) -> dict[int, str]:
        """Return ``{job_id: model_name}``, one entry per job in ``agents.jobs``
        (``job.task`` is the task, ``job.agent_id`` its owner).

        A value can also be a list of model names tried in order (e.g. a cascade): every
        attempt is paid for and the job keeps the last model's answer.
        Map a job to ``FALLBACK`` (or leave it out) to give it no model. The total cost
        must not exceed ``budget``; the experiment re-checks this with a ``BudgetLedger``.
        """


CENTRALIZED_POLICIES: Registry[BaseCentralizedPolicy] = Registry("centralized policy")


def create_centralized_policy(name: str, **params: Any) -> BaseCentralizedPolicy:
    return CENTRALIZED_POLICIES.create(name, **params)


def attempts_of(choice: str | Sequence[str]) -> list[str]:
    """The models called for one job: ``allocate()`` gives a model name, or a list of names
    tried in order (all paid for, the last one's answer is kept)."""
    if isinstance(choice, str):
        return [choice]
    attempts = list(choice)
    if not attempts:
        raise ValueError("A list of attempts must name at least one model")
    return attempts
