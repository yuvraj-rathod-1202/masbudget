from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction


@UTILITY_FUNCTIONS.register("cost_sensitive_adaptive")
class CostSensitiveAdaptiveUtility(BaseUtilityFunction):
    """Cost-sensitive adaptive. U_i(m) = q_{i,m} - lambda(B_k / B) * c_m.

    lambda grows as the remaining budget fraction ``state.budget_fraction`` shrinks, so
    agents become more conservative as the shared pool depletes.
    """

    def compute(self, agent, model, state):
        raise NotImplementedError(f"{self.registry_name} is not implemented yet")
