from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction


@UTILITY_FUNCTIONS.register("shapley_budget_share")
class ShapleyBudgetShareUtility(BaseUtilityFunction):
    """U_i(m) = q_{i,m} - lambda / (1 + gamma * phi_i) * c_m.

    phi_i is agent i's Shapley value / marginal contribution to collective welfare
    (computable from ``state.agents`` and ``state.models``). Params: ``lambda_cost``, ``gamma``.
    """

    def compute_phi(self, agent, state) -> float:
        if not state.agents:
            return 0.0
        def agent_delta(a):
            qualities = [m.quality(a.task) for m in state.models]
            return max(qualities) - min(qualities) if qualities else 0.0

        deltas = [agent_delta(a) for a in state.agents]
        total_delta = sum(deltas)
        if total_delta > 1e-9:
            return agent_delta(agent) / total_delta
        return 1.0 / len(state.agents)

    def compute(self, agent, model, state):
        lambda_cost = float(self.params.get("lambda_cost", 0.1))
        gamma = float(self.params.get("gamma", 1.0))
        phi_i = self.compute_phi(agent, state)
        effective_lambda = lambda_cost / (1.0 + gamma * phi_i)
        return model.quality(agent.task) - effective_lambda * model.cost

