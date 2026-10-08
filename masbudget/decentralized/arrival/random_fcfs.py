from masbudget.decentralized.arrival.base import ARRIVAL_POLICIES, BaseArrivalPolicy


@ARRIVAL_POLICIES.register("random_fcfs")
class RandomSequentialArrival(BaseArrivalPolicy):
    """Random sequential. At each step pick a waiting agent uniformly at random (``state.rng``)."""

    def select_next(self, remaining, state):
        raise NotImplementedError(f"{self.registry_name} is not implemented yet")
