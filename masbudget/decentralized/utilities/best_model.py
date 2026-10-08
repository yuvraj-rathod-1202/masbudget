from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction


@UTILITY_FUNCTIONS.register("best_model")
class BestModelUtility(BaseUtilityFunction):
    """Greedy best-model baseline. U_i(m) = q_{i,m} (cost ignored until the budget runs out)."""

    def compute(self, agent, model, state):
        return model.quality(agent.task)

