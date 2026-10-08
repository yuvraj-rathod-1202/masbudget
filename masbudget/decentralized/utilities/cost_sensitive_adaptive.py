from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction


@UTILITY_FUNCTIONS.register("cost_sensitive_adaptive")
class CostSensitiveAdaptiveUtility(BaseUtilityFunction):
    """Cost-sensitive adaptive. U_i(m) = q_{i,m} - lambda(B_k / B) * c_m.

    lambda grows as the remaining budget fraction ``state.budget_fraction`` shrinks, so
    agents become more conservative as the shared pool depletes.
    """

    def compute(self, agent, model, state):
        lambda_0 = float(self.params.get("lambda_0", self.params.get("lambda_cost", 0.1)))
        alpha = float(self.params.get("alpha", 1.0))
        frac = state.budget_fraction
        if frac <= 1e-9:
            lam = float("inf")
        else:
            lam = lambda_0 / (frac ** alpha)
        return model.quality(agent.task) - lam * model.cost

