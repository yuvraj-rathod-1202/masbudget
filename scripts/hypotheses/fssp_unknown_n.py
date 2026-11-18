"""FSSP when the total number of requests n is NOT known in advance.

    python scripts/hypotheses/fssp_unknown_n.py

FSSP with alpha = 1 needs only the number of requests still to come, n - k (the initial fair
share B/n cancels out of p_k = kappa * (n - k) / B_k). In a real system this is a forecast of
the remaining demand. We test two practical ways to get it:

* forecast   The operator guesses the total demand wrongly, n_hat = rho * n, and agents use
             max(1, n_hat - k). rho in {0.5, 0.75, 1.25, 1.5, 2}.
* online     No total is known. The budget covers a period [0, 1] (e.g. one billing month),
             requests arrive at random times in it, and each agent sees only the elapsed time
             t and how many requests have arrived so far. It estimates the arrival rate
             r = (arrived + w * n0) / (t + w) (optionally starting from a prior guess n0 with
             weight w) and the remaining demand as 1 + r * (1 - t).
             Variants: no prior (w = 0), and a wrong prior n0 = 0.5 n or 2 n with w = 0.1.

Arrival times are i.i.d. uniform on [0, 1] (Poisson arrivals conditioned on the count), drawn
per instance from a seed derived from the instance's task ids; the k-th arriving request gets
the k-th smallest time. Everything else is the FSSP test setting (kappa, alpha from
fssp_params.json, random arrival order, expected-quality information, full model pool).

Output: results/hypotheses/fssp_unknown_n.csv.gz
"""

from __future__ import annotations

import json
import random
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction  # noqa: E402

import fair_share  # noqa: E402,F401  (registers "fair_share_price")
import fssp_stage as F  # noqa: E402
import harness as H  # noqa: E402
import run_all as R  # noqa: E402


class FairSharePriceUnknownN(BaseUtilityFunction):
    """FSSP (alpha = 1, p_k = kappa / f_k) with an *estimated* number of remaining requests."""

    registry_name = "fair_share_price_unknown_n"

    def _times(self, state):
        times = getattr(self, "_t", None)
        if times is None:
            jobs = state.agents.jobs
            key = ",".join(sorted(str(j.task.task_id) for j in jobs)).encode()
            rng = random.Random(zlib.crc32(key) ^ int(self.params.get("time_seed", 0)))
            times = self._t = sorted(rng.random() for _ in jobs)
        return times

    def remaining_estimate(self, state) -> float:
        n = len(state.agents.jobs)  # used only to build the scenario, never by the agent directly
        k = state.step  # requests served before this one
        mode = self.params["mode"]
        if mode == "known":
            return n - k
        if mode == "forecast":
            return max(1.0, float(self.params["rho"]) * n - k)
        if mode == "online":
            t = self._times(state)[k]  # arrival time of the current request
            w = float(self.params.get("w", 0.0))
            n0 = float(self.params.get("rho0", 1.0)) * n
            rate = (k + 1 + w * n0) / (t + w)
            return 1.0 + rate * (1.0 - t)
        raise ValueError(mode)

    def compute(self, job, model, state):
        kappa = float(self.params.get("kappa", 0.1))
        f = state.remaining_budget / self.remaining_estimate(state)
        q = model.quality(job.task)
        if f <= 1e-15:
            return q - float("inf") * model.cost
        return q - (kappa / f) * model.cost


try:
    UTILITY_FUNCTIONS.register("fair_share_price_unknown_n")(FairSharePriceUnknownN)
except ValueError:
    pass

VARIANTS = (
    [("known", {"mode": "known"})]
    + [(f"forecast_rho={r}", {"mode": "forecast", "rho": r}) for r in [0.5, 0.75, 1.25, 1.5, 2.0]]
    + [("online_noprior", {"mode": "online", "w": 0.0}),
       ("online_prior=0.5", {"mode": "online", "w": 0.1, "rho0": 0.5}),
       ("online_prior=2", {"mode": "online", "w": 0.1, "rho0": 2.0})]
)


def run(cfg):
    return H.run_instance(cfg)


F.run = run


def main():
    kappa = json.loads(F.PARAMS_PATH.read_text())["fssp"]["kappa"]
    specs = [F.spec("C:optimal_mckp", "central", central="optimal_mckp"),
             F.spec("D:random_fcfs|best_model", "decentral", arrival="random_fcfs", utility="best_model")]
    specs += [F.spec(f"D:random_fcfs|fssp_{name}", "decentral", arrival="random_fcfs",
                     utility="fair_share_price_unknown_n", params={"kappa": kappa, **p}) for name, p in VARIANTS]
    cfgs = [dict(info="expected", mix=mix, n_agents=10, tasks_per_agent=10, budget_mult=m, model_set="full",
                 seed=s, utility_params=H.DEFAULT_UTILITY_PARAMS, policies=specs, tag="fssp_unknown_n")
            for mix in H.MIXES for m in R.BUDGET_MULTS for s in R.TEST_SEEDS]
    F.pool_run(cfgs, "fssp_unknown_n")


if __name__ == "__main__":
    main()
