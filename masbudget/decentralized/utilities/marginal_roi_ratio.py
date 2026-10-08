from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction


@UTILITY_FUNCTIONS.register("marginal_roi_ratio")
class MarginalROIRatioUtility(BaseUtilityFunction):
    """Submodular marginal utility per cost ratio.

        U_i(m) = (q_{i,m} - q_{i,base}) / (c_m - c_base)

    where ``base`` is the baseline (cheapest) model, ``state.models.cheapest()``.
    """

    def compute(self, agent, model, state):
        raise NotImplementedError(f"{self.registry_name} is not implemented yet")
