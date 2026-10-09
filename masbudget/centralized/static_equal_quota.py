from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy


from masbudget.core import FALLBACK


@CENTRALIZED_POLICIES.register("static_equal_quota")
class StaticEqualQuotaPolicy(BaseCentralizedPolicy):
    """Static equal quota (fair baseline).

    Split the budget upfront, b_i = B / N, and let each agent pick the best model
    whose cost fits in its own slice b_i.
    """

    def allocate(self, agents, models, budget, rng):
        if not agents or budget <= 1e-9:
            return {a.agent_id: FALLBACK for a in agents}

        slice_ = budget / len(agents)
        affordable = models.affordable(slice_)
        mapping = {}
        for agent in agents:
            if affordable:
                best = max(affordable, key=lambda m: m.quality(agent.task))
                mapping[agent.agent_id] = best.name
            else:
                mapping[agent.agent_id] = FALLBACK
        return mapping

