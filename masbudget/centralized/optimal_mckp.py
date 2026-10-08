from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy


@CENTRALIZED_POLICIES.register("optimal_mckp")
class OptimalMCKPPolicy(BaseCentralizedPolicy):
    """Exact Multiple-Choice Knapsack solver (upper bound), e.g. via PuLP / OR-Tools.

        max  sum_i sum_m q_{i,m} x_{i,m}
        s.t. sum_i sum_m c_m x_{i,m} <= B
             sum_m x_{i,m} <= 1          (agents with no model get the fallback)
             x_{i,m} in {0, 1}
    """

    def allocate(self, agents, models, budget, rng):
        raise NotImplementedError(f"{self.registry_name} is not implemented yet")
