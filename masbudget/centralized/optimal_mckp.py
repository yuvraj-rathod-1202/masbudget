from __future__ import annotations

from typing import Iterable

from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy
from masbudget.core import FALLBACK, Job
from masbudget.registry import ModelRegistry


@CENTRALIZED_POLICIES.register("optimal_mckp")
class OptimalMCKPPolicy(BaseCentralizedPolicy):
    """Exact Multiple-Choice Knapsack solver (upper bound), using Google OR-Tools.

        max  sum_j sum_m q_{j,m} x_{j,m}
        s.t. sum_j sum_m c_m x_{j,m} <= B
             sum_m x_{j,m} <= 1          (jobs with no model get the fallback)
             x_{j,m} in {0, 1}

    over every job j (each task of each agent).
    """

    def allocate(self, agents, models, budget, rng):
        return solve_mckp(agents.jobs, models, budget)


def solve_mckp(jobs: Iterable[Job], models: ModelRegistry, budget: float) -> dict[int, str]:
    """Optimal ``{job_id: model_name}`` for ``jobs`` within ``budget`` (OR-Tools SCIP, else CBC)."""
    job_list = list(jobs)
    model_list = list(models)
    if not job_list or not model_list or budget <= 1e-9:
        return {j.job_id: FALLBACK for j in job_list}

    from ortools.linear_solver import pywraplp

    solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("CBC")
    if not solver:
        raise RuntimeError("OR-Tools has neither the SCIP nor the CBC solver")

    x = {}
    for j in job_list:
        for m in model_list:
            x[j.job_id, m.name] = solver.BoolVar(f"x_{j.job_id}_{m.name}")

    # Budget constraint: sum_j sum_m (c_m * x_{j,m}) <= B
    solver.Add(sum(m.cost * x[j.job_id, m.name] for j in job_list for m in model_list) <= budget + 1e-9)

    # Each job gets at most one model: sum_m x_{j,m} <= 1
    for j in job_list:
        solver.Add(sum(x[j.job_id, m.name] for m in model_list) <= 1)

    # Objective: maximize total quality sum_j sum_m (q_{j,m} * x_{j,m})
    objective = solver.Objective()
    for j in job_list:
        for m in model_list:
            objective.SetCoefficient(x[j.job_id, m.name], m.quality(j.task))
    objective.SetMaximization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"OR-Tools found no solution (status {status})")
    mapping = {}
    for j in job_list:
        chosen = FALLBACK
        for m in model_list:
            if x[j.job_id, m.name].solution_value() > 0.5:
                chosen = m.name
                break
        mapping[j.job_id] = chosen
    return mapping
