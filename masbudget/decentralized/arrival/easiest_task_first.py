from masbudget.decentralized.arrival.base import ARRIVAL_POLICIES, BaseArrivalPolicy


@ARRIVAL_POLICIES.register("easiest_task_first")
class EasiestTaskFirstArrival(BaseArrivalPolicy):
    """Pick the waiting agent with the easiest task (worst case: cheap tasks burn the budget)."""

    def select_next(self, remaining, state):
        return min(remaining, key=lambda a: a.task.difficulty_rank)

