from masbudget.decentralized.arrival.base import ARRIVAL_POLICIES, BaseArrivalPolicy


@ARRIVAL_POLICIES.register("fcfs")
class FCFSArrival(BaseArrivalPolicy):
    """Uncoordinated first-come-first-served, used when ``arrival_policy`` is null.

    There is no coordinator: each agent calls the model API on its own and requests are
    served in the order the agents show up, i.e. their registration order (which follows
    the seeded task sampling).
    """

    def select_next(self, remaining, state):
        return remaining[0]
