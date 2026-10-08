from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction


@UTILITY_FUNCTIONS.register("dynamic_shadow_price")
class DynamicShadowPriceUtility(BaseUtilityFunction):
    """U_i(m) = q_{i,m} - p(B_k) * c_m, with p(B_k) = p0 * (B / B_k) ** alpha.

    Params: ``p0``, ``alpha``.
    """

    def compute(self, agent, model, state):
        raise NotImplementedError(f"{self.registry_name} is not implemented yet")
