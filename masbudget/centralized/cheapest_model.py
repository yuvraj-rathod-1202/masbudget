from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy


from masbudget.core import FALLBACK


@CENTRALIZED_POLICIES.register("cheapest_model")
class CheapestModelPolicy(BaseCentralizedPolicy):
    """Cheapest-model baseline. Every job gets m_j = argmin_m c_m (lower bound on quality)."""

    def allocate(self, agents, models, budget, rng):
        if not agents or budget <= 1e-9:
            return {j.job_id: FALLBACK for j in agents.jobs}

        cheapest = models.cheapest()
        mapping = {}
        rem = budget
        for job in agents.jobs:
            if rem >= cheapest.cost - 1e-9:
                mapping[job.job_id] = cheapest.name
                rem -= cheapest.cost
            else:
                mapping[job.job_id] = FALLBACK
        return mapping

