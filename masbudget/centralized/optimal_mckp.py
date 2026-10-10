from __future__ import annotations

import logging

from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy
from masbudget.core import FALLBACK

logger = logging.getLogger(__name__)


@CENTRALIZED_POLICIES.register("optimal_mckp")
class OptimalMCKPPolicy(BaseCentralizedPolicy):
    """Exact Multiple-Choice Knapsack solver (upper bound), using Google OR-Tools.

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
        