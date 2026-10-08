from masbudget.decentralized.arrival.base import ARRIVAL_POLICIES, BaseArrivalPolicy


@ARRIVAL_POLICIES.register("marginal_roi_priority")
class MarginalROIPriorityArrival(BaseArrivalPolicy):
    """Pick the waiting agent with the highest expected quality gain per unit cost (dq / dc)."""

    def select_next(self, remaining, state):
        raise NotImplementedError(f"{self.registry_name} is not implemented yet")
