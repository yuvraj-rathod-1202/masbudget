import random
from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction


@UTILITY_FUNCTIONS.register("shapley_budget_share")
class ShapleyBudgetShareUtility(BaseUtilityFunction):
    """U_i(m) = q_{i,m} - lambda / (1 + gamma * phi_i) * c_m.

    phi_i is agent i's Shapley value / marginal contribution to collective welfare
    (computable from ``state.agents`` and ``state.models``). Params: ``lambda_cost``, ``gamma``.
    """
    
    def _get_coalition_welfare(self, coalition, models) -> float:
        """Characteristic function v(S): The maximum collective quality 

        minus cost achievable by a coalition of agents choosing the optimal model.
        """
        if not coalition or not models:
            return 0.0

        max_welfare = float("-inf")
        for m in models:
            total_quality = sum(m.quality(a.task) for a in coalition)
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
            

    def compute(self, agent, model, state):
        lambda_cost = float(self.params.get("lambda_cost", 0.1))
        gamma = float(self.params.get("gamma", 1.0))
        
        phi_i = self.compute_phi(agent, state)
        
        denominator = 1.0 + gamma * phi_i
        if denominator < 1e-3:
            denominator = 1e-3
        
        effective_lambda = lambda_cost / denominator
        return model.quality(agent.task) - effective_lambda * model.cost