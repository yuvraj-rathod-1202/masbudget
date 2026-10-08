from __future__ import annotations

import logging

from masbudget.core import AllocationState, Assignment, BudgetLedger
from masbudget.decentralized.base import DECENTRALIZED_POLICIES, BaseDecentralizedPolicy

logger = logging.getLogger(__name__)


@DECENTRALIZED_POLICIES.register("sequential_best_response")
class SequentialBestResponse(BaseDecentralizedPolicy):
    """Sequential resource-allocation game (proposal, Section 3.2).

    At each step k the arrival policy picks the next agent, which observes B_k, restricts
    itself to M(B_k) = {m : c_m <= B_k}, picks argmax U(m) and pays c_m immediately.
    If nothing is affordable the agent gets the fallback model (quality 0, cost 0).
    Ties are broken by model order in the config.
    """

    def allocate(self, agents, models, budget, rng):
        ledger = BudgetLedger(budget)
        waiting = list(agents)
        history: list[Assignment] = []

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
            agent = self.arrival_policy.select_next(list(waiting), state)
            if all(a.agent_id != agent.agent_id for a in waiting):
                raise RuntimeError(
                    f"{self.arrival_policy.registry_name} selected agent {agent.agent_id}, which is not waiting"
                )
            waiting = [a for a in waiting if a.agent_id != agent.agent_id]

            feasible = models.affordable(ledger.remaining)
            if feasible:
                utilities = {m.name: self.utility.compute(agent, m, state) for m in feasible}
                choice = max(feasible, key=lambda m: utilities[m.name])
                utility = utilities[choice.name]
            else:
                choice, utility = models.fallback, 0.0

            ledger.charge(choice.cost)
            history.append(
                Assignment(
                    agent_id=agent.agent_id,
                    model=choice.name,
                    quality=choice.quality(agent.task),
                    cost=choice.cost,
                    step=step,
                    utility=utility,
                    remaining_budget=state.remaining_budget,
                )
            )
            logger.debug(
                "step=%d agent=%d B_k=%.6g -> %s (cost=%.6g, U=%.6g)",
                step, agent.agent_id, state.remaining_budget, choice.name, choice.cost, utility,
            )

        return history
