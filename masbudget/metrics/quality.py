"""Quality & task performance metrics."""

from masbudget.metrics.base import METRICS, BaseMetrics, per_run


@METRICS.register("average_quality")
class AverageQuality(BaseMetrics):
    """Q = (1/N) * sum_i q_{i,m_i}."""

    def compute(self, result):
        return per_run(result, lambda run: run.total_quality / len(run.assignments))


@METRICS.register("success_rate")
class SuccessRate(BaseMetrics):
    """SR = (1/N) * sum_i 1[q_{i,m_i} >= tau]. Param: ``tau`` (default 0.5)."""

    def compute(self, result):
        tau = self.params.get("tau", 0.5)
        return per_run(result, lambda run: sum(q >= tau for q in run.qualities) / len(run.assignments))
