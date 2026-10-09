from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy


from masbudget.core import FALLBACK


@CENTRALIZED_POLICIES.register("optimal_mckp")
class OptimalMCKPPolicy(BaseCentralizedPolicy):
    """Exact Multiple-Choice Knapsack solver (upper bound), e.g. via PuLP / OR-Tools / SciPy MILP.

        max  sum_i sum_m q_{i,m} x_{i,m}
        s.t. sum_i sum_m c_m x_{i,m} <= B
             sum_m x_{i,m} <= 1          (agents with no model get the fallback)
             x_{i,m} in {0, 1}
    """

    def allocate(self, agents, models, budget, rng):
        if not agents or budget <= 1e-9:
            return {a.agent_id: FALLBACK for a in agents}

        agent_list = list(agents)
        model_list = list(models)
        N = len(agent_list)
        M = len(model_list)
        if N == 0 or M == 0:
            return {a.agent_id: FALLBACK for a in agents}

        try:
            import numpy as np
            from scipy.optimize import LinearConstraint, milp

            c = np.zeros(N * M)
            for i, a in enumerate(agent_list):
                for m_idx, m in enumerate(model_list):
                    c[i * M + m_idx] = -m.quality(a.task)

            A = np.zeros((1 + N, N * M))
            for i in range(N):
                for m_idx, m in enumerate(model_list):
                    A[0, i * M + m_idx] = m.cost
                    A[1 + i, i * M + m_idx] = 1.0

            lhs = np.zeros(1 + N)
            lhs[0] = -np.inf
            rhs = np.zeros(1 + N)
            rhs[0] = budget
            rhs[1:] = 1.0

            res = milp(c=c, integrality=np.ones(N * M), constraints=LinearConstraint(A, lhs, rhs))
            if res.success:
                x = res.x.reshape((N, M))
                mapping = {}
                for i, a in enumerate(agent_list):
                    chosen = int(np.argmax(x[i]))
                    if x[i, chosen] > 0.5:
                        mapping[a.agent_id] = model_list[chosen].name
                    else:
                        mapping[a.agent_id] = FALLBACK
                return mapping
        except Exception:
            pass

        # Fallback exact branch-and-bound search
        options = [models.fallback, *model_list]
        best_val = -1.0
        best_mapping: dict[int, str] = {}

        def search(idx, current_cost, current_quality, current_mapping):
            nonlocal best_val, best_mapping
            if current_cost > budget + 1e-9:
                return
            if idx == len(agent_list):
                if current_quality > best_val:
                    best_val = current_quality
                    best_mapping = dict(current_mapping)
                return
            agent = agent_list[idx]
            for m in options:
                new_cost = current_cost + m.cost
                if new_cost <= budget + 1e-9:
                    current_mapping[agent.agent_id] = m.name
                    search(idx + 1, new_cost, current_quality + m.quality(agent.task), current_mapping)

        search(0, 0.0, 0.0, {})
        return best_mapping

