from masbudget.decentralized.arrival.base import ARRIVAL_POLICIES, BaseArrivalPolicy


@ARRIVAL_POLICIES.register("marginal_roi_priority")
class MarginalROIPriorityArrival(BaseArrivalPolicy):
    """Pick the waiting agent with the highest expected quality gain per unit cost (dq / dc)."""

    def select_next(self, remaining, state):
        base = state.models.cheapest()

        def agent_max_roi(agent):
            upgrades = [
                (m.quality(agent.task) - base.quality(agent.task)) / (m.cost - base.cost)
                for m in state.models
                if m.cost > base.cost + 1e-9
            ]
            return max(upgrades) if upgrades else 0.0

        return max(remaining, key=agent_max_roi)

