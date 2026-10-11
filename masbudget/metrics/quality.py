"""Quality & task performance metrics."""

from masbudget.metrics.base import METRICS, BaseMetrics, per_run


@METRICS.register("average_quality")
class AverageQuality(BaseMetrics):
    """Q = (1/J) * sum_j q_{j,m_j}, the mean over all J tasks (jobs) of all agents."""

    def compute(self, result):
        return per_run(result, lambda run: run.total_quality / len(run.assignments))


@METRICS.register("success_rate")
class SuccessRate(BaseMetrics):
    """SR = (1/J) * sum_j 1[q_{j,m_j} >= tau], over all tasks. Param: ``tau`` (default 0.5)."""

    def compute(self, result):
        tau = self.params.get("tau", 0.5)
        return per_run(result, lambda run: sum(q >= tau for q in run.qualities) / len(run.assignments))
