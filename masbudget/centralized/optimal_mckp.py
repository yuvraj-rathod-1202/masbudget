from __future__ import annotations

import logging

from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy
from masbudget.core import FALLBACK

logger = logging.getLogger(__name__)


@CENTRALIZED_POLICIES.register("optimal_mckp")
class OptimalMCKPPolicy(BaseCentralizedPolicy):
    """Exact Multiple-Choice Knapsack solver (upper bound), using Google OR-Tools / PuLP.

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
        if not agent_list or not model_list:
            return {a.agent_id: FALLBACK for a in agents}

        # 1. Primary: Google OR-Tools (SCIP / CBC integer solver)
        try:
            from ortools.linear_solver import pywraplp

            solver = pywraplp.Solver.CreateSolver("SCIP")
            if not solver:
                solver = pywraplp.Solver.CreateSolver("CBC")
            if solver:
                x = {}
                for a in agent_list:
                    for m in model_list:
                        x[a.agent_id, m.name] = solver.BoolVar(f"x_{a.agent_id}_{m.name}")

                # Budget constraint: sum_i sum_m (c_m * x_{i,m}) <= B
                solver.Add(
                    sum(m.cost * x[a.agent_id, m.name] for a in agent_list for m in model_list) <= budget + 1e-9
                )

                # Each agent chooses at most one model: sum_m x_{i,m} <= 1
                for a in agent_list:
                    solver.Add(sum(x[a.agent_id, m.name] for m in model_list) <= 1)

                # Objective: maximize total quality sum_i sum_m (q_{i,m} * x_{i,m})
                objective = solver.Objective()
                for a in agent_list:
                    for m in model_list:
                        objective.SetCoefficient(x[a.agent_id, m.name], m.quality(a.task))
                objective.SetMaximization()

                status = solver.Solve()
                if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                    mapping = {}
                    for a in agent_list:
                        chosen = FALLBACK
                        for m in model_list:
                            if x[a.agent_id, m.name].solution_value() > 0.5:
                                chosen = m.name
                                break
                        mapping[a.agent_id] = chosen
                    return mapping
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("OR-Tools solver failed (%s); trying fallback solver", exc)

        # 2. Secondary: SciPy MILP
        try:
            import numpy as np
            from scipy.optimize import LinearConstraint, milp

            N, M = len(agent_list), len(model_list)
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
                x_res = res.x.reshape((N, M))
                mapping = {}
                for i, a in enumerate(agent_list):
                    chosen_idx = int(np.argmax(x_res[i]))
                    if x_res[i, chosen_idx] > 0.5:
                        mapping[a.agent_id] = model_list[chosen_idx].name
                    else:
                        mapping[a.agent_id] = FALLBACK
                return mapping
        except Exception:
            pass

        # 3. Exact branch-and-bound search fallback
        options = [models.fallback, *model_list]
        best_val = -1.0
        best_mapping = {a.agent_id: FALLBACK for a in agent_list}

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


