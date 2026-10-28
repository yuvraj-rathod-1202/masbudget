from __future__ import annotations

import math
from typing import Iterable

from masbudget.core import Task
from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction
from masbudget.registry import ModelRegistry


@UTILITY_FUNCTIONS.register("budget_pacing")
class BudgetPacingUtility(BaseUtilityFunction):
    """Pace-aware shadow price (dual mirror descent, cf. Balseiro & Gur 2019).

        U_i(m) = q_{i,m} - p_k * c_m,   p_k = p0 * exp(beta * (s_k - k / J))

    s_k = (B - B_k) / B is the share of the budget already spent and k / J the share of the
    jobs already served, both read off the public ledger. Spending ahead of schedule raises
    the price, spending behind schedule lowers it; ``beta = 0`` is a fixed posted price p0.

    Params: ``p0`` (reference price, quality per unit cost; see ``calibrate_price``),
    ``beta`` (default 5), ``horizon`` (J the agents assume; default: the experiment's job count),
    ``p_min`` (default 0): floor on the reference price, max(p0, p_min). A multiplicative
    price that starts at 0 never moves, so a positive floor lets the feedback act even when
    history suggests the budget is not binding.
    """

    def compute(self, job, model, state):
        return model.quality(job.task) - self.price(state) * model.cost

    def price(self, state) -> float:
        p0 = max(float(self.params.get("p0", 0.0)), float(self.params.get("p_min", 0.0)))
        beta = float(self.params.get("beta", 5.0))
        horizon = float(self.params.get("horizon") or state.num_jobs)
        spent = 1.0 - state.budget_fraction
        progress = min(1.0, state.step / horizon) if horizon > 0 else 1.0
        return p0 * math.exp(beta * (spent - progress))


def calibrate_price(tasks: Iterable[Task], models: ModelRegistry, budget: float, tol: float = 1e-6) -> float:
    """Lagrangian price lambda* of the MCKP relaxation: the smallest lambda >= 0 at which every
    task picking argmax_m q_{t,m} - lambda * c_m spends at most ``budget``.

    Computed on a sample of tasks (e.g. historical ones) with the same budget per task as the
    target run, it is the posted price p0 that makes independent best responses budget-feasible.
    """
    tasks = list(tasks)
    model_list = list(models)

    def spend(lam: float) -> float:
        return sum(max(model_list, key=lambda m: m.quality(t) - lam * m.cost).cost for t in tasks)

    if not tasks or spend(0.0) <= budget:
        return 0.0
    lo, hi = 0.0, 1.0
    while spend(hi) > budget:
        hi *= 2.0
        if hi > 1e12:
            return hi  # even the cheapest model everywhere overspends
    while hi - lo > tol * max(1.0, hi):
        mid = (lo + hi) / 2.0
        lo, hi = (lo, mid) if spend(mid) <= budget else (mid, hi)
    return hi
