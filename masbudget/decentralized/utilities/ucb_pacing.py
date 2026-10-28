from __future__ import annotations

import math

from masbudget.core import FALLBACK
from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS
from masbudget.decentralized.utilities.budget_pacing import BudgetPacingUtility


@UTILITY_FUNCTIONS.register("ucb_pacing")
class UCBPacingUtility(BudgetPacingUtility):
    """Learning agents (online bandit extension): q_{i,m} is unknown and learned from outcomes.

        U_i(m) = UCB_{d,m} - p_k * c_m,   UCB_{d,m} = (s + a) / (n + a + b) + c * sqrt(ln(1 + N_d) / (n + 1))

    Estimates are shared by all agents and kept per (difficulty label d of the task, model m):
    s successes in n calls observed so far, N_d calls on class d, Beta(a, b) prior. After a call
    the outcome ``task.metadata[feedback_key][model]`` (e.g. unit tests passing) is revealed to
    everyone (tasks without that metadata report q_{i,m} as the outcome). The price p_k is the
    ``budget_pacing`` one.

    Params: those of ``budget_pacing`` plus ``c`` (exploration bonus, default 0.5), ``prior_a``,
    ``prior_b`` (default 1, 1), ``prior`` ({difficulty: {model: mean}} or {model: mean}, optional)
    with ``prior_n`` pseudo-observations (default 5), ``feedback_key`` (default ``"realized"``).
    """

    def __init__(self, **params):
        super().__init__(**params)
        self._key = None
        self._seen = 0
        self._stats: dict[tuple[str, str], list[float]] = {}  # (d, m) -> [successes, calls]
        self._class_calls: dict[str, int] = {}
        self._task_of: dict[int, object] = {}

    def compute(self, job, model, state):
        self._update(state)
        return self.estimate(job.task.difficulty, model.name) - self.price(state) * model.cost

    def estimate(self, difficulty: str, model: str) -> float:
        a = float(self.params.get("prior_a", 1.0))
        b = float(self.params.get("prior_b", 1.0))
        prior = self.params.get("prior")
        if prior is not None:
            mean = prior[difficulty][model] if difficulty in prior else prior[model]
            n0 = float(self.params.get("prior_n", 5.0))
            a, b = mean * n0, (1.0 - mean) * n0
        s, n = self._stats.get((difficulty, model), (0.0, 0.0))
        bonus = float(self.params.get("c", 0.5)) * math.sqrt(math.log(1 + self._class_calls.get(difficulty, 0)) / (n + 1))
        return (s + a) / (n + a + b) + bonus

    def _update(self, state) -> None:
        key = id(state.agents)
        if key != self._key or len(state.history) < self._seen:  # new experiment or new run
            self._key, self._seen, self._stats, self._class_calls = key, 0, {}, {}
            self._task_of = {job.job_id: job.task for job in state.agents.jobs}
        feedback_key = self.params.get("feedback_key", "realized")
        for a in state.history[self._seen:]:
            if a.model == FALLBACK:
                continue
            task = self._task_of[a.job_id]
            outcomes = task.metadata.get(feedback_key)
            outcome = outcomes[a.model] if outcomes is not None else a.quality
            stats = self._stats.setdefault((task.difficulty, a.model), [0.0, 0.0])
            stats[0] += float(outcome)
            stats[1] += 1.0
            self._class_calls[task.difficulty] = self._class_calls.get(task.difficulty, 0) + 1
        self._seen = len(state.history)
