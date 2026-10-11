from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy
from masbudget.centralized.optimal_mckp import solve_mckp
from masbudget.core import FALLBACK


@CENTRALIZED_POLICIES.register("static_equal_quota")
class StaticEqualQuotaPolicy(BaseCentralizedPolicy):
    """Static equal quota (fair baseline).

    Split the budget upfront, b_i = B / N per agent, and let each agent spend its own slice
    b_i on its own tasks only: the best models for its tasks whose total cost fits in b_i
    (an exact knapsack per agent). Unused budget is not shared with other agents.
    """

    def allocate(self, agents, models, budget, rng):
        if not agents or budget <= 1e-9:
            return {j.job_id: FALLBACK for j in agents.jobs}

        slice_ = budget / len(agents)
        mapping = {}
        for agent in agents:
            mapping.update(solve_mckp(agents.jobs_of(agent.agent_id), models, slice_))
        return mapping
