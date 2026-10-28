from __future__ import annotations

import logging

from masbudget.core import AllocationState, Assignment, BudgetLedger
from masbudget.decentralized.base import DECENTRALIZED_POLICIES, BaseDecentralizedPolicy

logger = logging.getLogger(__name__)


@DECENTRALIZED_POLICIES.register("sequential_best_response")
class SequentialBestResponse(BaseDecentralizedPolicy):
    """Sequential resource-allocation game (proposal, Section 3.2).

    At each step k the arrival policy picks the next waiting job (one task of one agent).
    Its agent observes B_k, restricts itself to M(B_k) = {m : c_m <= B_k}, picks argmax U(m)
    and pays c_m immediately. If nothing is affordable the job gets the fallback model
    (quality 0, cost 0). An agent with several tasks therefore acts once per task.
    Ties are broken by model order in the config.

    Params: ``reserve`` (default ``False``). With ``reserve: true`` the ledger enforces a
    budget floor: a job may only pick a model that leaves at least c_min for every job
    still waiting, M_res(B_k) = {m : c_m <= B_k - (J - k - 1) * c_min}. Nobody is starved
    while B >= J * c_min. If the floor is already broken, the job gets the cheapest affordable
    model so that as many jobs as possible are served.

    ``abstain`` (default ``False``): the agent may also decline the call (the fallback model,
    quality 0 and cost 0), scored by the utility like any other model; it abstains when every
    affordable model has a lower utility, e.g. q_{i,m} - p * c_m < 0 under a price p.
    """

    def allocate(self, agents, models, budget, rng):
        ledger = BudgetLedger(budget)
        waiting = agents.jobs
        history: list[Assignment] = []
        reserve = bool(self.params.get("reserve", False))
        abstain = bool(self.params.get("abstain", False))
        min_cost = models.cheapest().cost if len(models) else 0.0

        for step in range(len(waiting)):
            state = AllocationState(
                step=step,
                total_budget=budget,
                remaining_budget=ledger.remaining,
                agents=agents,
                models=models,
                history=list(history),
                rng=rng,
            )
            job = self.arrival_policy.select_next(list(waiting), state)
            if all(j.job_id != job.job_id for j in waiting):
                raise RuntimeError(f"{self.arrival_policy.registry_name} selected job {job.job_id}, which is not waiting")
            waiting = [j for j in waiting if j.job_id != job.job_id]

            feasible = models.affordable(ledger.remaining)
            if reserve and feasible:
                floored = models.affordable(ledger.remaining - len(waiting) * min_cost)
                feasible = floored or [min(feasible, key=lambda m: m.cost)]
            if abstain:
                feasible = [*feasible, models.fallback]
            if feasible:
                utilities = {m.name: self.utility.compute(job, m, state) for m in feasible}
                choice = max(feasible, key=lambda m: utilities[m.name])
                utility = utilities[choice.name]
            else:
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
                    remaining_budget=state.remaining_budget,
                )
            )
            logger.debug(
                "step=%d job=%d agent=%d B_k=%.6g -> %s (cost=%.6g, U=%.6g)",
                step, job.job_id, job.agent_id, state.remaining_budget, choice.name, choice.cost, utility,
            )

        return history
