from masbudget.decentralized.arrival.base import ARRIVAL_POLICIES, BaseArrivalPolicy


@ARRIVAL_POLICIES.register("hardest_task_first")
class HardestTaskFirstArrival(BaseArrivalPolicy):
    """Pick the waiting agent with the hardest task (``agent.task.difficulty_rank``)."""

    def select_next(self, remaining, state):
        return max(remaining, key=lambda a: a.task.difficulty_rank)

