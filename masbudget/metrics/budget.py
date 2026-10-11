"""Budget & resource efficiency metrics."""

from masbudget.metrics.base import METRICS, BaseMetrics, per_run


@METRICS.register("total_cost")
class TotalCost(BaseMetrics):
    """C = sum_j c_{m_j}, over every task of every agent."""

    def compute(self, result):
        return per_run(result, lambda run: run.total_cost)


@METRICS.register("budget_utilization")
class BudgetUtilization(BaseMetrics):
    """BUR = C / B."""

    def compute(self, result):
        return per_run(result, lambda run: run.total_cost / run.budget if run.budget > 0 else None)


@METRICS.register("cost_effectiveness")
class CostEffectiveness(BaseMetrics):
    """CE = sum_j q_{j,m_j} / C (quality per unit cost)."""

    def compute(self, result):
        return per_run(result, lambda run: run.total_quality / run.total_cost if run.total_cost > 0 else None)
