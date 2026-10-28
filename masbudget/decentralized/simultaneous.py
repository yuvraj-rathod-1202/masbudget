from __future__ import annotations

import logging

from masbudget.core import AllocationState, Assignment, BudgetLedger
from masbudget.decentralized.base import DECENTRALIZED_POLICIES, BaseDecentralizedPolicy

logger = logging.getLogger(__name__)


@DECENTRALIZED_POLICIES.register("simultaneous_best_response")
class SimultaneousBestResponse(BaseDecentralizedPolicy):
    """One-shot game without the sequencing layer (no arrival order, no B_k feedback).

    Every job decides at the same time from public static information only: the total budget B,
    the job count J and the model costs. Its agent sees B_k = B, step 0 and an empty history, so
    a state-dependent utility (e.g. ``budget_pacing``) reduces to its fixed reference price. The
    arrival policy is ignored.

    Params:

    * ``allowance`` (default ``None``): ``"job"`` restricts every job to models costing at most
      B / J, so the requests can never overspend.
    * ``settlement`` (default ``"ration"``): how the ledger settles the requests when together they
      cost more than B. The requests are filled in a random order the agents never see.
      ``"ration"``: a request that no longer fits is dropped (fallback model, the job starves).
      ``"downgrade"``: a request may not leave less than c_min for each request still unsettled
      (the budget reserve); a request that breaks this gets the agent's best model that keeps it,
      so nobody starves while B >= J * c_min.
    """

    def allocate(self, agents, models, budget, rng):
        jobs = agents.jobs
        allowance = self.params.get("allowance")
        settlement = self.params.get("settlement", "ration")
        if settlement not in ("ration", "downgrade"):
            raise ValueError(f"unknown settlement {settlement!r}")
        min_cost = models.cheapest().cost if len(models) else 0.0
        state = AllocationState(
            step=0, total_budget=budget, remaining_budget=budget, agents=agents, models=models, history=[], rng=rng,
        )

        cap = budget / len(jobs) if allowance == "job" and jobs else budget
        utilities = {}
        for job in jobs:
            utilities[job.job_id] = {m.name: self.utility.compute(job, m, state) for m in models.affordable(cap)}

        def best(job, feasible):
            if not feasible:
                return models.fallback, 0.0
            choice = max(feasible, key=lambda m: utilities[job.job_id][m.name])
            return choice, utilities[job.job_id][choice.name]

        requests = {job.job_id: best(job, models.affordable(cap)) for job in jobs}
        order = list(jobs)
        rng.shuffle(order)

        ledger = BudgetLedger(budget)
        history: list[Assignment] = []
        for step, job in enumerate(order):
            choice, utility = requests[job.job_id]
            remaining = ledger.remaining
            if settlement == "downgrade":
                floor = remaining - (len(order) - step - 1) * min_cost
                if choice.cost > floor + 1e-12:
                    keep = [m for m in models.affordable(min(cap, floor)) if m.name in utilities[job.job_id]]
                    affordable = models.affordable(remaining)
                    choice, utility = best(job, keep) if keep else (
                        (min(affordable, key=lambda m: m.cost), 0.0) if affordable else (models.fallback, 0.0)
                    )
            elif choice.cost > remaining + 1e-12:
                choice, utility = models.fallback, 0.0

            ledger.charge(choice.cost)
            history.append(
                Assignment(
                    job_id=job.job_id,
                    agent_id=job.agent_id,
                    model=choice.name,
                    quality=choice.quality(job.task),
                    cost=choice.cost,
                    step=step,
                    utility=utility,
                    remaining_budget=budget,
                )
            )
            logger.debug(
                "settle=%d job=%d agent=%d requested=%s -> %s (cost=%.6g)",
                step, job.job_id, job.agent_id, requests[job.job_id][0].name, choice.name, choice.cost,
            )
        return history
