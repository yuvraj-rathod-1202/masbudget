"""Fairness & game-theoretic disparity metrics, measured across agents.

An agent's payoff u_i is the mean realized quality over its tasks (``RunResult.agent_payoffs``).
"""

import logging
import math
import statistics

from masbudget.core import DECENTRALIZED
from masbudget.metrics.base import METRICS, BaseMetrics, per_run

logger = logging.getLogger(__name__)


@METRICS.register("jains_fairness")
class JainsFairness(BaseMetrics):
    """J = (sum_i u_i)^2 / (N * sum_i u_i^2) over the N agents; 1 means perfectly equal payoffs."""

    def compute(self, result):
        def jain(run):
            payoffs = list(run.agent_payoffs.values())
            squares = sum(u * u for u in payoffs)
            return sum(payoffs) ** 2 / (len(payoffs) * squares) if squares > 0 else None

        return per_run(result, jain)


@METRICS.register("starvation_rate")
class StarvationRate(BaseMetrics):
    """Fraction of agents that received the fallback model on at least one of their tasks."""

    def compute(self, result):
        def rate(run):
            groups = run.by_agent().values()
            return sum(any(a.starved for a in rows) for rows in groups) / len(groups)

        return per_run(result, rate)


@METRICS.register("task_starvation_rate")
class TaskStarvationRate(BaseMetrics):
    """Fraction of tasks (jobs) that received the fallback model, ignoring who owns them."""

    def compute(self, result):
        return per_run(result, lambda run: sum(a.starved for a in run.assignments) / len(run.assignments))


@METRICS.register("first_mover_advantage")
class FirstMoverAdvantage(BaseMetrics):
    """Payoff of the earliest-acting agents / payoff of the latest-acting agents (decentralized only).

    Agents are ranked by the mean arrival step of their jobs; the first and last
    max(1, N // 4) agents are compared. Undefined with fewer than 2 agents.
    """

    def compute(self, result):
        def ratio(run):
            groups = run.by_agent()
            if len(groups) < 2:
                return None
            ordered = sorted(groups.values(), key=lambda rows: statistics.fmean(a.step for a in rows))
            payoffs = [statistics.fmean(a.quality for a in rows) for rows in ordered]
            k = max(1, len(payoffs) // 4)
            first = statistics.fmean(payoffs[:k])
            last = statistics.fmean(payoffs[-k:])
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
