"""Fairness & game-theoretic disparity metrics (per-agent payoff = realized quality)."""

import logging
import math
import statistics

from masbudget.core import DECENTRALIZED
from masbudget.metrics.base import METRICS, BaseMetrics, per_run

logger = logging.getLogger(__name__)


@METRICS.register("jains_fairness")
class JainsFairness(BaseMetrics):
    """J = (sum_i u_i)^2 / (N * sum_i u_i^2); 1 means perfectly equal payoffs."""

    def compute(self, result):
        def jain(run):
            squares = sum(q * q for q in run.qualities)
            return run.total_quality ** 2 / (len(run.assignments) * squares) if squares > 0 else None

        return per_run(result, jain)


@METRICS.register("starvation_rate")
class StarvationRate(BaseMetrics):
    """Fraction of agents that received the fallback model."""

    def compute(self, result):
        return per_run(result, lambda run: sum(a.starved for a in run.assignments) / len(run.assignments))


@METRICS.register("first_mover_advantage")
class FirstMoverAdvantage(BaseMetrics):
    """Mean payoff of the first arrival quartile / mean payoff of the last quartile (decentralized only)."""

    def compute(self, result):
        def ratio(run):
            ordered = sorted(run.assignments, key=lambda a: a.step)
            k = len(ordered) // 4
            if k == 0:
                return None
            first = statistics.fmean(a.quality for a in ordered[:k])
            last = statistics.fmean(a.quality for a in ordered[-k:])
            if last > 0:
                return first / last
            return math.inf if first > 0 else None

        return per_run(result, ratio, paradigms=[DECENTRALIZED])


@METRICS.register("order_sensitivity")
class OrderSensitivity(BaseMetrics):
    """Variance of social welfare across runs (arrival permutations). Needs ``num_runs >= 2``."""

    def compute(self, result):
        out = {}
        for paradigm, runs in result.runs.items():
            if len(runs) < 2:
                logger.warning("order_sensitivity needs num_runs >= 2 (%s has %d)", paradigm, len(runs))
                out[paradigm] = None
            else:
                out[paradigm] = statistics.pvariance(run.total_quality for run in runs)
        return out
