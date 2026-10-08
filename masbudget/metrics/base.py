from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Mapping, Sequence

from masbudget.core import ExperimentResult, RunResult
from masbudget.factory import Registry


class BaseMetrics(ABC):
    """A metric evaluated on a finished experiment."""

    registry_name: str

    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def compute(self, result: ExperimentResult) -> dict[str, Any]:
        """Return ``{paradigm: value}`` for every paradigm the metric applies to.

        Per-run metrics report ``{"mean", "std", "n"}`` over the experiment's runs
        (see ``per_run``); a value is ``None`` when the metric is undefined.
        """


METRICS: Registry[BaseMetrics] = Registry("metric")

MetricSpec = str | Mapping[str, Mapping[str, Any] | None]


def get_metric(name: str, **params: Any) -> BaseMetrics:
    return METRICS.create(name, **params)


def compute_metrics(result: ExperimentResult, metrics: Sequence[MetricSpec]) -> dict[str, Any]:
    """Compute several metrics. Each spec is a name, or ``{name: {param: value}}``."""
    out: dict[str, Any] = {}
    for spec in metrics:
        name, params = parse_metric_spec(spec)
        out[name] = get_metric(name, **params).compute(result)
    return out


def parse_metric_spec(spec: MetricSpec) -> tuple[str, dict[str, Any]]:
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, Mapping) and len(spec) == 1:
        name, params = next(iter(spec.items()))
        return str(name), dict(params or {})
    raise ValueError(f"Invalid metric spec {spec!r}: use 'name' or {{name: {{param: value}}}}")


def summarize(values: Iterable[float | None]) -> dict[str, float] | None:
    """Mean / population std over runs, ignoring undefined (None) values."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) if math.isfinite(mean) else math.nan
    return {"mean": mean, "std": std, "n": len(vals)}


def per_run(
    result: ExperimentResult, fn: Callable[[RunResult], float | None], paradigms: Sequence[str] | None = None
) -> dict[str, Any]:
    """Apply ``fn`` to every run and summarize per paradigm."""
    return {
        paradigm: summarize(fn(run) for run in runs)
        for paradigm, runs in result.runs.items()
        if paradigms is None or paradigm in paradigms
    }
