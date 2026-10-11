from masbudget.decentralized.arrival.base import ARRIVAL_POLICIES, BaseArrivalPolicy


@ARRIVAL_POLICIES.register("hardest_task_first")
class HardestTaskFirstArrival(BaseArrivalPolicy):
    """Pick the waiting job with the hardest task (``job.task.difficulty_rank``)."""

    def select_next(self, remaining, state):
        return max(remaining, key=lambda job: job.task.difficulty_rank)

