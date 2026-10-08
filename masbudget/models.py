"""AI models (the actions agents choose between). Each model has a fixed cost per call."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from masbudget.core import FALLBACK, Task
from masbudget.factory import Registry


class BaseModel(ABC):
    """A model m with fixed invocation cost c_m."""

    registry_name: str

    def __init__(self, name: str, cost: float, **params: Any):
        if cost < 0:
            raise ValueError(f"Model '{name}' has negative cost {cost}")
        self.name = name
        self.cost = float(cost)
        self.params = params

    @abstractmethod
    def quality(self, task: Task) -> float:
        """Expected quality q_{i,m} of this model on ``task``, in [0, 1]."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, cost={self.cost})"


MODEL_TYPES: Registry[BaseModel] = Registry("model type")


@MODEL_TYPES.register("dataset")
class DatasetModel(BaseModel):
    """Offline model: q_{i,m} is read from the task's ``quality`` table in the dataset."""

    def quality(self, task: Task) -> float:
        try:
            return task.quality[self.name]
        except KeyError:
            raise KeyError(f"Task '{task.task_id}' has no quality entry for model '{self.name}'") from None


class FallbackModel(BaseModel):
    """Assigned to an agent that gets no model (e.g. budget exhausted): quality 0, cost 0."""

    def __init__(self) -> None:
        super().__init__(FALLBACK, 0.0)

    def quality(self, task: Task) -> float:
        return 0.0


def create_model(name: str, cost: float, model_type: str = "dataset", **params: Any) -> BaseModel:
    """Factory: build model ``name`` with cost ``cost`` using the registered ``model_type`` class."""
    return MODEL_TYPES.create(model_type, name, cost, **params)
