from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy


@CENTRALIZED_POLICIES.register("confidence_cascade")
class ConfidenceCascadePolicy(BaseCentralizedPolicy):
    """Confidence-based cascade (FrugalGPT).

    Send each task to the cheapest model first and escalate to a more capable model
    only if verification fails (e.g. quality below ``params["threshold"]``).
    """

    def allocate(self, agents, models, budget, rng):
        raise NotImplementedError(f"{self.registry_name} is not implemented yet")
