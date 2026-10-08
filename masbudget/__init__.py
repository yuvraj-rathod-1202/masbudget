"""Multi-agent model/API selection under a shared budget: experiment framework."""

from masbudget import centralized, decentralized, metrics  # noqa: F401  (registers built-in components)
from masbudget.centralized import BaseCentralizedPolicy, create_centralized_policy
from masbudget.core import FALLBACK, Agent, AllocationState, Assignment, ExperimentResult, RunResult, Task
from masbudget.dataset import load_dataset
from masbudget.decentralized import (
    BaseArrivalPolicy,
    BaseDecentralizedPolicy,
    BaseUtilityFunction,
    create_arrival_policy,
    create_decentralized_policy,
    create_utility_function,
)
from masbudget.experiment import Experiment, run_experiment
from masbudget.metrics import BaseMetrics, compute_metrics, get_metric
from masbudget.models import BaseModel, create_model
from masbudget.registry import AgentRegistry, ModelRegistry

__all__ = [
    "FALLBACK",
    "Agent",
    "AgentRegistry",
    "AllocationState",
    "Assignment",
    "BaseArrivalPolicy",
    "BaseCentralizedPolicy",
    "BaseDecentralizedPolicy",
    "BaseMetrics",
    "BaseModel",
    "BaseUtilityFunction",
    "Experiment",
    "ExperimentResult",
    "ModelRegistry",
    "RunResult",
    "Task",
    "compute_metrics",
    "create_arrival_policy",
    "create_centralized_policy",
    "create_decentralized_policy",
    "create_model",
    "create_utility_function",
    "get_metric",
    "load_dataset",
    "run_experiment",
]
