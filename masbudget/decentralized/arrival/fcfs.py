from masbudget.decentralized.arrival.base import ARRIVAL_POLICIES, BaseArrivalPolicy


@ARRIVAL_POLICIES.register("fcfs")
class FCFSArrival(BaseArrivalPolicy):
    """Uncoordinated first-come-first-served, used when ``arrival_policy`` is null.

    There is no coordinator: all agents work at the same time and call the model API on
    their own, so requests are served round-robin: agent 0's first task, agent 1's first
    task, ..., then everyone's second task, and so on (``AgentRegistry.jobs`` order).
    """

    def select_next(self, remaining, state):
        return remaining[0]
