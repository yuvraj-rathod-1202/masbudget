from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction


@UTILITY_FUNCTIONS.register("quasi_linear")
class QuasiLinearUtility(BaseUtilityFunction):
    """U_i(m) = q_{i,m} - lambda * c_m, with fixed ``params["lambda_cost"]``."""

    def compute(self, agent, model, state):
        raise NotImplementedError(f"{self.registry_name} is not implemented yet")
