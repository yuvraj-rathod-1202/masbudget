from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction


@UTILITY_FUNCTIONS.register("marginal_roi_ratio")
class MarginalROIRatioUtility(BaseUtilityFunction):
    """Submodular marginal utility per cost ratio.

        U_i(m) = (q_{i,m} - q_{i,base}) / (c_m - c_base)

    where ``base`` is the baseline (cheapest) model, ``state.models.cheapest()``.
    """

    def compute(self, agent, model, state):
        base = state.models.cheapest()
        delta_cost = model.cost - base.cost
        if abs(delta_cost) < 1e-9:
            return 0.0
        delta_quality = model.quality(agent.task) - base.quality(agent.task)
        return delta_quality / delta_cost

