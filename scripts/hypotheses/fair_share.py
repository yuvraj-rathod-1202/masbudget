"""Fair-Share Shadow Pricing (FSSP): a self-calibrating decentralized utility.

Agent pi_k, arriving at step k with remaining budget B_k and n - k jobs still to be served
(itself included), prices budget in units of the *remaining fair share*

    f_k = B_k / (n - k)          (dollars still available per job still waiting)
    b   = B / n                  (the initial fair share)

and best-responds with

    U(m) = q_hat(m) - p_k * c_m,     p_k = (kappa / b) * (b / f_k) ** alpha.

* kappa / b converts dollars into "fractions of one fair share", so the same kappa works for
  any budget scale (no per-budget tuning).
* alpha controls scarcity feedback: alpha = 0 is a budget-normalized fixed price
  (quasi-linear with lambda = kappa / b); alpha = 1 gives p_k = kappa / f_k, the price rises
  exactly when earlier agents spent more than their share and falls when they under-spent.

Information used by an agent: B, B_k, the total number of jobs n and the step k (how many
requests were already served). No other agent's task, quality or choice is observed.

Importing this module registers the utility as ``fair_share_price`` in the masbudget
utility registry.
"""

from __future__ import annotations

from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction


class FairSharePriceUtility(BaseUtilityFunction):
    registry_name = "fair_share_price"

    def compute(self, job, model, state):
        kappa = float(self.params.get("kappa", 0.07))
        alpha = float(self.params.get("alpha", 1.0))
        n = getattr(self, "_n", None)
        if n is None:
            n = self._n = len(state.agents.jobs)
        remaining_jobs = max(1, n - state.step)
        b = state.total_budget / n
        f = state.remaining_budget / remaining_jobs
        if b <= 1e-15:
            return model.quality(job.task)
        if f <= 1e-15:
            return model.quality(job.task) - float("inf") * model.cost
        price = (kappa / b) * (b / f) ** alpha
        return model.quality(job.task) - price * model.cost


try:
    UTILITY_FUNCTIONS.register("fair_share_price")(FairSharePriceUtility)
except ValueError:
    pass  # already registered in this process
