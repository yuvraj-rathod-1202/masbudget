"""YAML experiment configs (schema in docs/config.md)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

from masbudget.dataset import load_dataset, sample_tasks
from masbudget.experiment import Experiment
from masbudget.registry import AgentRegistry, ModelRegistry

MODES = ("centralized", "decentralized", "both")
EXPERIMENT_KEYS = {
    "name", "description", "mode", "dataset_path", "num_agents", "num_models", "budget",
    "seed", "num_runs", "models", "centralized", "decentralized", "metrics",
}


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> list[dict[str, Any]]:
    """Read a config file and return one fully resolved spec per experiment
    (``defaults`` merged in; experiment keys win)."""
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    defaults = raw.get("defaults") or {}
    experiments = raw.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ConfigError(f"{path}: 'experiments' must be a non-empty list")
    _check_keys(defaults, "defaults")

    specs, names = [], set()
    for index, experiment in enumerate(experiments):
        spec = {**defaults, **experiment}
        name = spec.get("name")
        if not name:
            raise ConfigError(f"experiments[{index}]: missing 'name'")
        if name in names:
            raise ConfigError(f"Duplicate experiment name '{name}'")
        names.add(name)
        _check_keys(experiment, name)
        spec["mode"] = _resolve_mode(spec)
        for key in ("dataset_path", "budget", "models"):
            if spec.get(key) is None:
                raise ConfigError(f"{name}: '{key}' is required (set it here or in defaults)")
        specs.append(spec)
    return specs


def build_experiment(spec: dict[str, Any]) -> Experiment:
    """Turn a resolved spec into an ``Experiment`` (loads the dataset, registers agents & models)."""
    seed = spec.get("seed", 0)
    models = build_models(spec["models"], spec.get("num_models"))
    tasks = sample_tasks(load_dataset(spec["dataset_path"]), spec.get("num_agents"), random.Random(seed))
    agents = AgentRegistry.from_tasks(tasks)

    kwargs: dict[str, Any] = {}
    if spec["mode"] in ("centralized", "both"):
        centralized = spec["centralized"]
        kwargs.update(centralized_policy=centralized["policy"], centralized_params=centralized.get("params"))
    if spec["mode"] in ("decentralized", "both"):
        decentralized = spec["decentralized"]
        kwargs.update(
            utility=decentralized["utility"],
            utility_params=decentralized.get("utility_params"),
            arrival_policy=decentralized.get("arrival_policy"),
            arrival_params=decentralized.get("arrival_params"),
            decentralized_policy=decentralized.get("policy", "sequential_best_response"),
            decentralized_params=decentralized.get("params"),
        )
    return Experiment(
        agents, models, spec["budget"], name=spec["name"], seed=seed, num_runs=spec.get("num_runs", 1), **kwargs
    )


def build_models(spec: dict[str, Any], num_models: int | None = None) -> ModelRegistry:
    """``{name: cost}`` or ``{name: {cost, type, params}}``; ``num_models`` keeps the first k."""
    if not isinstance(spec, dict) or not spec:
        raise ConfigError("'models' must be a non-empty mapping of model name -> cost")
    items = list(spec.items())
    if num_models is not None:
        if not 1 <= num_models <= len(items):
            raise ConfigError(f"num_models={num_models} but {len(items)} models are defined")
        items = items[:num_models]

    registry = ModelRegistry()
    for name, entry in items:
        if isinstance(entry, dict):
            registry.add(name, entry["cost"], entry.get("type", "dataset"), **(entry.get("params") or {}))
        else:
            registry.add(name, entry)
    return registry


def _resolve_mode(spec: dict[str, Any]) -> str:
    name = spec["name"]
    mode = spec.get("mode")
    if mode is None:
        has_c, has_d = bool(spec.get("centralized")), bool(spec.get("decentralized"))
        mode = "both" if has_c and has_d else "centralized" if has_c else "decentralized" if has_d else None
        if mode is None:
            raise ConfigError(f"{name}: define a 'centralized' and/or 'decentralized' section")
    if mode not in MODES:
        raise ConfigError(f"{name}: mode must be one of {MODES}, got {mode!r}")
    if mode in ("centralized", "both") and not (spec.get("centralized") or {}).get("policy"):
        raise ConfigError(f"{name}: mode '{mode}' needs centralized.policy")
    if mode in ("decentralized", "both") and not (spec.get("decentralized") or {}).get("utility"):
        raise ConfigError(f"{name}: mode '{mode}' needs decentralized.utility")
    return mode


def _check_keys(section: dict[str, Any], where: str) -> None:
    unknown = set(section) - EXPERIMENT_KEYS
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
