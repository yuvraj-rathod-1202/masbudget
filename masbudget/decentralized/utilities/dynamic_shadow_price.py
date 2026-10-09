from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction


@UTILITY_FUNCTIONS.register("dynamic_shadow_price")
class DynamicShadowPriceUtility(BaseUtilityFunction):
    """U_i(m) = q_{i,m} - p(B_k) * c_m, with p(B_k) = p0 * (B / B_k) ** alpha.

    Params: ``p0``, ``alpha``.
    """

    def compute(self, agent, model, state):
        p0 = float(self.params.get("p0", 0.1))
        alpha = float(self.params.get("alpha", 1.5))
        if state.remaining_budget <= 1e-9:
            price = float("inf")
        elif state.total_budget <= 1e-9:
            price = p0
        else:
            price = p0 * (state.total_budget / state.remaining_budget) ** alpha
        return model.quality(agent.task) - price * model.cost

