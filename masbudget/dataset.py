"""Loading and sampling of benchmark tasks. The JSON format is documented in docs/dataset.md."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from masbudget.core import DIFFICULTY_LEVELS, Task


class DatasetError(ValueError):
    pass


def load_dataset(path: str | Path) -> list[Task]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not isinstance(raw.get("tasks"), list):
        raise DatasetError(f"{path}: top level must be an object with a 'tasks' list")
    if not raw["tasks"]:
        raise DatasetError(f"{path}: 'tasks' is empty")

    tasks: list[Task] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw["tasks"]):
        task = _parse_task(entry, f"{path}: tasks[{index}]")
        if task.task_id in seen:
            raise DatasetError(f"{path}: duplicate task_id '{task.task_id}'")
        seen.add(task.task_id)
        tasks.append(task)
    return tasks


def sample_tasks(tasks: list[Task], num_agents: int | None, rng: random.Random) -> list[Task]:
    """Seeded random sample of ``num_agents`` tasks (all tasks, shuffled, if None)."""
    n = len(tasks) if num_agents is None else num_agents
    if not 1 <= n <= len(tasks):
        raise DatasetError(f"num_agents={n} but the dataset has {len(tasks)} tasks")
    return rng.sample(tasks, n)


def _parse_task(entry: Any, where: str) -> Task:
    if not isinstance(entry, dict):
        raise DatasetError(f"{where}: must be an object")
    for key in ("task_id", "difficulty", "quality"):
        if key not in entry:
            raise DatasetError(f"{where}: missing required field '{key}'")

    difficulty = str(entry["difficulty"]).lower()
    if difficulty not in DIFFICULTY_LEVELS:
        raise DatasetError(f"{where}: difficulty must be one of {DIFFICULTY_LEVELS}, got {entry['difficulty']!r}")

    quality = entry["quality"]
    if not isinstance(quality, dict) or not quality:
        raise DatasetError(f"{where}: 'quality' must be a non-empty object of model -> score")
    for model, score in quality.items():
        if not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
            raise DatasetError(f"{where}: quality['{model}'] must be a number in [0, 1], got {score!r}")

    features = entry.get("features")
    return Task(
        task_id=str(entry["task_id"]),
        difficulty=difficulty,
        quality={str(m): float(q) for m, q in quality.items()},
        category=entry.get("category"),
        features=tuple(float(x) for x in features) if features is not None else None,
        metadata=dict(entry.get("metadata") or {}),
    )
