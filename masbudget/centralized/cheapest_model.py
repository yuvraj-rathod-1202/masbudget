from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy


from masbudget.core import FALLBACK


@CENTRALIZED_POLICIES.register("cheapest_model")
class CheapestModelPolicy(BaseCentralizedPolicy):
    """Cheapest-model baseline. Every agent gets m_i = argmin_m c_m (lower bound on quality)."""

    def allocate(self, agents, models, budget, rng):
        if not agents or budget <= 1e-9:
            return {a.agent_id: FALLBACK for a in agents}

        cheapest = models.cheapest()
        mapping = {}
        rem = budget
        for agent in agents:
            if rem >= cheapest.cost - 1e-9:
                mapping[agent.agent_id] = cheapest.name
                rem -= cheapest.cost
            else:
                mapping[agent.agent_id] = FALLBACK
        return mapping

