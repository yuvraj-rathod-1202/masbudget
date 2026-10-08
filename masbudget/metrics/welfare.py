"""Social welfare & allocation efficiency metrics.

Welfare is measured in realized task quality (SW with lambda = 0) so that centralized and
decentralized runs are comparable regardless of the agents' private utility functions.
"""

import logging

from masbudget.core import CENTRALIZED, DECENTRALIZED
from masbudget.metrics.base import METRICS, BaseMetrics, per_run, summarize

logger = logging.getLogger(__name__)


@METRICS.register("social_welfare")
class SocialWelfare(BaseMetrics):
    """SW = sum_i q_{i,m_i}."""

    def compute(self, result):
        return per_run(result, lambda run: run.total_quality)


@METRICS.register("optimality_gap")
class OptimalityGap(BaseMetrics):
    """Gap = (SW_centralized - SW_decentralized) / SW_centralized, paired run by run.

    Needs both paradigms in the same experiment (``mode: both``).
    """

    def compute(self, result):
        centralized = result.runs.get(CENTRALIZED)
        decentralized = result.runs.get(DECENTRALIZED)
        if not centralized or not decentralized:
            logger.warning("optimality_gap needs both centralized and decentralized runs (mode: both); skipped")
            return {}
        gaps = [
            (c.total_quality - d.total_quality) / c.total_quality if c.total_quality > 0 else None
            for c, d in zip(centralized, decentralized)
        ]
        return {DECENTRALIZED: summarize(gaps)}
