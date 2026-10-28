from __future__ import annotations

from typing import Iterable

from masbudget.core import Task
from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction
from masbudget.decentralized.utilities.budget_pacing import calibrate_price
from masbudget.registry import ModelRegistry


@UTILITY_FUNCTIONS.register("lottery_price")
class LotteryPriceUtility(BaseUtilityFunction):
    """Posted price with a coin flip per job (a mixed strategy, no state feedback needed).

        U_i(m) = q_{i,m} - p_i * c_m,   p_i = p_low with probability theta, else p_high

    At the optimal price lambda* the marginal task class is indifferent between two models, so a
    single posted price buys the upgrade for all of the class or none of it. Drawing the price
    just below lambda* (p_low) for a share theta of the jobs and just above it (p_high) for the
    rest buys the upgrade for about a share theta of the class. See ``calibrate_lottery``.

    Params: ``p_low``, ``p_high``, ``theta`` (all default 0). One draw per job and run, from the run's rng.
    """

    def __init__(self, **params):
        super().__init__(**params)
        self._state = None
        self._draws: dict[int, bool] = {}

    def compute(self, job, model, state):
        if state is not self._state:
            self._state, self._draws = state, {}
        if job.job_id not in self._draws:
            self._draws[job.job_id] = state.rng.random() < float(self.params.get("theta", 0.0))
        price = float(self.params.get("p_low" if self._draws[job.job_id] else "p_high", 0.0))
        return model.quality(job.task) - price * model.cost


def calibrate_lottery(
    tasks: Iterable[Task], models: ModelRegistry, budget: float, delta: float = 1e-3
) -> dict[str, float]:
    """``p_low``, ``p_high`` and ``theta`` for ``lottery_price`` from (historical) tasks and budget.

    p_high = lambda* (``calibrate_price``), the lowest price that keeps spending within budget;
    p_low = (1 - delta) * lambda*, which buys the next upgrade. theta mixes the two so that the
    expected spend equals the budget.
    """
    tasks = list(tasks)
    model_list = list(models)

    def spend(lam: float) -> float:
        return sum(max(model_list, key=lambda m: m.quality(t) - lam * m.cost).cost for t in tasks)

    p_high = calibrate_price(tasks, models, budget)
    p_low = (1.0 - delta) * p_high
    s_high, s_low = spend(p_high), spend(p_low)
    theta = (budget - s_high) / (s_low - s_high) if s_low > s_high else 0.0
    return {"p_low": p_low, "p_high": p_high, "theta": min(1.0, max(0.0, theta))}
