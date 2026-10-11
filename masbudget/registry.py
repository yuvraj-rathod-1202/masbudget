"""Registries of the agents and models taking part in an experiment.

They are built once per experiment and shared by the centralized and decentralized runs,
so both paradigms see exactly the same agents, tasks, models and costs.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence

from masbudget.core import EPS, FALLBACK, Agent, Job, Task
from masbudget.models import BaseModel, FallbackModel, create_model


class AgentRegistry:
    """Agents in registration order, plus their jobs (one per task).

    Jobs are interleaved round-robin: agent 0's first task, agent 1's first task, ...,
    then everyone's second task, and so on. That is the FCFS arrival order of agents
    working at the same time.
    """

    def __init__(self) -> None:
        self._agents: dict[int, Agent] = {}
        self._jobs: list[Job] | None = None

    @classmethod
    def from_task_lists(cls, task_lists: Iterable[Sequence[Task]]) -> AgentRegistry:
        """One agent per list of tasks, with ids 0..N-1."""
        registry = cls()
        for agent_id, tasks in enumerate(task_lists):
            registry.register(Agent(agent_id, tuple(tasks)))
        return registry

    def register(self, agent: Agent) -> Agent:
        if agent.agent_id in self._agents:
            raise ValueError(f"Agent {agent.agent_id} is already registered")
        if not agent.tasks:
            raise ValueError(f"Agent {agent.agent_id} has no tasks")
        self._agents[agent.agent_id] = agent
        self._jobs = None
        return agent

    def get(self, agent_id: int) -> Agent:
        return self._agents[agent_id]

    @property
    def ids(self) -> list[int]:
        return list(self._agents)

    @property
    def jobs(self) -> list[Job]:
        """Every (agent, task) pair, round-robin over agents; job ids are 0..J-1 in this order."""
        if self._jobs is None:
            jobs: list[Job] = []
            agents = list(self._agents.values())
            for index in range(max(len(a.tasks) for a in agents) if agents else 0):
                for agent in agents:
                    if index < len(agent.tasks):
                        jobs.append(Job(len(jobs), agent.agent_id, agent.tasks[index]))
            self._jobs = jobs
        return list(self._jobs)

    @property
    def job_ids(self) -> list[int]:
        return [job.job_id for job in self.jobs]

    def jobs_of(self, agent_id: int) -> list[Job]:
        return [job for job in self.jobs if job.agent_id == agent_id]

    def __iter__(self) -> Iterator[Agent]:
        return iter(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)


class ModelRegistry:
    """Available models M with their fixed costs. The fallback model is kept apart:
    it is never part of iteration / ``affordable()``, only returned by ``get(FALLBACK)``."""

    def __init__(self) -> None:
        self._models: dict[str, BaseModel] = {}
        self.fallback: BaseModel = FallbackModel()

    def register(self, model: BaseModel) -> BaseModel:
        if model.name == FALLBACK:
            raise ValueError(f"'{FALLBACK}' is reserved for the fallback model")
        if model.name in self._models:
            raise ValueError(f"Model '{model.name}' is already registered")
        self._models[model.name] = model
        return model

    def add(self, name: str, cost: float, model_type: str = "dataset", **params: Any) -> BaseModel:
        """Create a model through the model factory and register it."""
        return self.register(create_model(name, cost, model_type, **params))

    def get(self, name: str) -> BaseModel:
        if name == FALLBACK:
            return self.fallback
        try:
            return self._models[name]
        except KeyError:
            raise KeyError(f"Unknown model '{name}'. Registered: {', '.join(self._models)}") from None

    def cost(self, name: str) -> float:
        return self.get(name).cost

    def cheapest(self) -> BaseModel:
        return min(self._models.values(), key=lambda m: m.cost)

    def affordable(self, budget: float) -> list[BaseModel]:
        """Feasible action set M(B_k) = {m : c_m <= B_k}, in registration order."""
        return [m for m in self._models.values() if m.cost <= budget + EPS]

    @property
    def names(self) -> list[str]:
        return list(self._models)

    def __iter__(self) -> Iterator[BaseModel]:
        return iter(self._models.values())

    def __len__(self) -> int:
        return len(self._models)
