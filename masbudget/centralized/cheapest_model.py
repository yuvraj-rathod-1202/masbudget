from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy


@CENTRALIZED_POLICIES.register("cheapest_model")
class CheapestModelPolicy(BaseCentralizedPolicy):
    """Cheapest-model baseline. Every agent gets m_i = argmin_m c_m (lower bound on quality)."""

    def allocate(self, agents, models, budget, rng):
        raise NotImplementedError(f"{self.registry_name} is not implemented yet")
