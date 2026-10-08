from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy


@CENTRALIZED_POLICIES.register("static_equal_quota")
class StaticEqualQuotaPolicy(BaseCentralizedPolicy):
    """Static equal quota (fair baseline).

    Split the budget upfront, b_i = B / N, and let each agent pick the best model
    whose cost fits in its own slice b_i.
    """

    def allocate(self, agents, models, budget, rng):
        raise NotImplementedError(f"{self.registry_name} is not implemented yet")
