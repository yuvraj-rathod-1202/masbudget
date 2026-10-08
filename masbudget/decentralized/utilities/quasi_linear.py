from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction


@UTILITY_FUNCTIONS.register("quasi_linear")
class QuasiLinearUtility(BaseUtilityFunction):
    """U_i(m) = q_{i,m} - lambda * c_m, with fixed ``params["lambda_cost"]``."""

    def compute(self, agent, model, state):
        lambda_cost = float(self.params.get("lambda_cost", 0.1))
        return model.quality(agent.task) - lambda_cost * model.cost

