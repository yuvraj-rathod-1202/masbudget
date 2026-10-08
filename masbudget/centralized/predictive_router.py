from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy


@CENTRALIZED_POLICIES.register("predictive_router")
class PredictiveRouterPolicy(BaseCentralizedPolicy):
    """Predictive LLM router (RouteLLM / MasRouter style).

    Routes each task using a trained quality estimator over task features
    (``Task.features``), trading predicted quality gain against marginal cost,
    without solving an exact ILP.
    """

    def allocate(self, agents, models, budget, rng):
        raise NotImplementedError(f"{self.registry_name} is not implemented yet")
