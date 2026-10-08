from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction


@UTILITY_FUNCTIONS.register("shapley_budget_share")
class ShapleyBudgetShareUtility(BaseUtilityFunction):
    """U_i(m) = q_{i,m} - lambda / (1 + gamma * phi_i) * c_m.

    phi_i is agent i's Shapley value / marginal contribution to collective welfare
    (computable from ``state.agents`` and ``state.models``). Params: ``lambda_cost``, ``gamma``.
    """

    def compute(self, agent, model, state):
        raise NotImplementedError(f"{self.registry_name} is not implemented yet")
