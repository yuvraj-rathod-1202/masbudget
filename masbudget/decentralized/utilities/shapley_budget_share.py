import random
from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction


@UTILITY_FUNCTIONS.register("shapley_budget_share")
class ShapleyBudgetShareUtility(BaseUtilityFunction):
    """U_i(m) = q_{i,m} - lambda / (1 + gamma * phi_i) * c_m.

    phi_i is the Shapley value / marginal contribution to collective welfare of agent i,
    the owner of the job (computable from ``state.agents`` and ``state.models``); a coalition
    of agents brings all of its members' tasks. Params: ``lambda_cost``, ``gamma``,
    ``num_samples``, ``seed``, ``characteristic``:

    * ``"best_model_welfare"`` (default): v(S) = max_m sum_{t in S} q_{t,m} - c_m.
    * ``"upgrade_gain"``: v(S) = sum_{t in S} max_m (q_{t,m} - q_{t,base}) / |tasks of the agent|,
      the quality the coalition gains from premium models over the cheapest one (task criticality).

    phi_i is fixed for a given set of agents, so it is computed once per agent and cached.
    """

    def __init__(self, **params):
        super().__init__(**params)
        self._phi_cache: dict[tuple[int, int], float] = {}

    def _get_coalition_welfare(self, coalition, models) -> float:
        """Characteristic function v(S): The maximum collective quality 

        minus cost achievable by a coalition of agents choosing the optimal model
        (quality summed over every task of every agent in the coalition).
        """
        if not coalition or not models:
            return 0.0

        if self.params.get("characteristic", "best_model_welfare") == "upgrade_gain":
            base = models.cheapest()
            return sum(
                max(m.quality(task) - base.quality(task) for m in models) / len(a.tasks)
                for a in coalition for task in a.tasks
            )

        max_welfare = float("-inf")
        for m in models:
            total_quality = sum(m.quality(task) for a in coalition for task in a.tasks)
            welfare = total_quality - m.cost
            if welfare > max_welfare:
                max_welfare = welfare
                
        return max_welfare

    def compute_phi(self, agent, state) -> float:
        agents = state.agents
        models = state.models
        
        if not agents or agent not in agents:
            return 0.0
        
        n = len(agents)
        if n == 1:
            return 1.0
        
        num_samples = int(self.params.get("num_samples", 100))

        rng = random.Random(self.params.get("seed", 0))

        total_marginal_contribution = 0.0
        agents_list = list(agents)
        
        for _ in range(num_samples):
            permutation = list(agents_list)
            rng.shuffle(permutation)
            
            idx = permutation.index(agent)
            
            coalition_before = permutation[:idx]
            coalition_with_agent = permutation[:idx + 1]
            
            v_before = self._get_coalition_welfare(coalition_before, models)
            v_with_agent = self._get_coalition_welfare(coalition_with_agent, models)
            
            total_marginal_contribution += (v_with_agent - v_before)
            
        phi_i = total_marginal_contribution / num_samples
        return phi_i
            

    def compute(self, job, model, state):
        lambda_cost = float(self.params.get("lambda_cost", 0.1))
        gamma = float(self.params.get("gamma", 1.0))
        
        key = (id(state.agents), job.agent_id)
        if key not in self._phi_cache:
            self._phi_cache[key] = self.compute_phi(state.agents.get(job.agent_id), state)
        phi_i = self._phi_cache[key]
        
        denominator = 1.0 + gamma * phi_i
        if denominator < 1e-3:
            denominator = 1e-3
        
        effective_lambda = lambda_cost / denominator
        return model.quality(job.task) - effective_lambda * model.cost