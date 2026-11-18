"""Experiment harness for the hypothesis evaluation (results.md).

It drives the existing masbudget policy classes (centralized controllers, arrival policies,
utility functions and the sequential best-response game) and adds two things the YAML runner
does not have:

* Information models.
  - ``oracle``:   policies see the realized pass/fail of every (task, model) pair.
  - ``expected``: policies see q_hat[i,m] = pass rate of model m on the task's difficulty tier,
    estimated by K-fold cross-fitting (a task's own outcome is never used for its estimate).
  In both cases the *realized* binary pass@1 is used for scoring.
* Re-scoring: every chosen model is scored against the realized outcome, and per-run metrics
  (quality, cost, starvation, fairness, per-tier quality, per-position quality, spend by tier)
  are returned as flat dicts.
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from masbudget.centralized.base import attempts_of, create_centralized_policy
from masbudget.core import DIFFICULTY_LEVELS, FALLBACK, Agent, Task
from masbudget.decentralized.arrival.base import create_arrival_policy
from masbudget.decentralized.sequential import SequentialBestResponse
from masbudget.decentralized.utilities.base import create_utility_function
from masbudget.decentralized.utilities.shapley_budget_share import ShapleyBudgetShareUtility
from masbudget.registry import AgentRegistry, ModelRegistry

import masbudget.centralized.optimal_mckp as _om  # noqa: E402
import masbudget.centralized.static_equal_quota as _sq  # noqa: E402


def solve_mckp_grouped(jobs, models, budget):
    """Exact MCKP, solved over integer counts of interchangeable jobs.

    Jobs whose quality vectors are identical are interchangeable, so the 0/1 MCKP
        max sum q_{j,m} x_{j,m}  s.t.  sum c_m x_{j,m} <= B,  sum_m x_{j,m} <= 1
    has the same optimum as the integer program over n_{g,m} = #jobs of group g on model m:
        max sum q_{g,m} n_{g,m}  s.t.  sum c_m n_{g,m} <= B,  sum_m n_{g,m} <= |g|.
    This removes the symmetry that makes SCIP stall when many jobs share one vector (every
    task of a tier in the expected-information setting). Jobs inside a group are filled in
    job-id order. Same optimum as ``masbudget.centralized.optimal_mckp.solve_mckp``.
    """
    from ortools.linear_solver import pywraplp

    job_list = list(jobs)
    model_list = list(models)
    if not job_list or not model_list or budget <= 1e-9:
        return {j.job_id: FALLBACK for j in job_list}
    groups: dict[tuple, list] = {}
    for j in sorted(job_list, key=lambda j: j.job_id):
        groups.setdefault(tuple(round(m.quality(j.task), 12) for m in model_list), []).append(j)
    solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("CBC")
    n = {}
    for g, (vec, members) in enumerate(groups.items()):
        for k, m in enumerate(model_list):
            n[g, k] = solver.IntVar(0, len(members), f"n_{g}_{k}")
        solver.Add(sum(n[g, k] for k in range(len(model_list))) <= len(members))
    solver.Add(sum(model_list[k].cost * n[g, k] for (g, k) in n) <= budget + 1e-9)
    obj = solver.Objective()
    for g, (vec, _) in enumerate(groups.items()):
        for k in range(len(model_list)):
            obj.SetCoefficient(n[g, k], vec[k])
    obj.SetMaximization()
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"grouped MCKP: no solution (status {status})")
    mapping = {}
    for g, (vec, members) in enumerate(groups.items()):
        it = iter(members)
        for k, m in enumerate(model_list):
            for _ in range(int(round(n[g, k].solution_value()))):
                mapping[next(it).job_id] = m.name
        for j in it:
            mapping[j.job_id] = FALLBACK
    return mapping


_om.solve_mckp = solve_mckp_grouped
_sq.solve_mckp = solve_mckp_grouped

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "llmrouterbench_livecodebench.json"
COST_PATH = ROOT / "data" / "llmrouterbench_livecodebench_model_costs.json"

TIERS = DIFFICULTY_LEVELS  # ("easy", "medium", "hard")

MIXES = {
    "uniform": (1 / 3, 1 / 3, 1 / 3),
    "easy_heavy": (0.8, 0.1, 0.1),
    "hard_heavy": (0.1, 0.1, 0.8),
}

MODEL_SETS = {
    "full": [
        "deepseek-v3.1-terminus",
        "deepseek-v3-0324",
        "gemini-2.5-flash",
        "claude-sonnet-4",
        "deepseek-r1-0528",
        "gemini-2.5-pro",
    ],
    # small cost spread (1x - 5.2x), small quality spread
    "flat": ["deepseek-v3.1-terminus", "deepseek-v3-0324", "gemini-2.5-flash"],
    # large cost spread (1x - 120x), large quality spread on hard tasks
    "steep": ["deepseek-v3.1-terminus", "deepseek-r1-0528", "gemini-2.5-pro"],
}

CENTRAL_POLICIES = ["optimal_mckp", "predictive_router", "confidence_cascade", "static_equal_quota", "cheapest_model"]
ARRIVALS = ["fcfs", "random_fcfs", "hardest_task_first", "easiest_task_first", "marginal_roi_priority"]
UTILITIES = ["best_model", "quasi_linear", "dynamic_shadow_price", "shapley_budget_share", "marginal_roi_ratio"]

# Proposal / YAML defaults (reported for reference in the tuning section).
DEFAULT_UTILITY_PARAMS = {
    "best_model": {},
    "quasi_linear": {"lambda_cost": 0.5},
    "dynamic_shadow_price": {"p0": 0.1, "alpha": 1.5},
    "shapley_budget_share": {"lambda_cost": 0.5, "gamma": 1.0, "num_samples": 100, "seed": 42},
    "marginal_roi_ratio": {},
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Bench:
    task_ids: list[str]
    tiers: np.ndarray  # int tier index per task
    scores: dict[str, np.ndarray]  # model -> realized pass@1 per task
    costs: dict[str, float]

    @property
    def n(self) -> int:
        return len(self.task_ids)


def load_bench() -> Bench:
    raw = json.loads(DATA_PATH.read_text())["tasks"]
    costs_raw = json.loads(COST_PATH.read_text())
    models = MODEL_SETS["full"]
    return Bench(
        task_ids=[t["task_id"] for t in raw],
        tiers=np.array([TIERS.index(t["difficulty"]) for t in raw]),
        scores={m: np.array([float(t["quality"][m]) for t in raw]) for m in models},
        costs={m: float(costs_raw[m]["mean_cost_usd"]) for m in models},
    )


def cross_fitted_qhat(bench: Bench, k: int = 5, seed: int = 0) -> np.ndarray:
    """q_hat[task, tier, model]: tier pass rates estimated on the other K-1 folds.

    Returns array of shape (n_tasks, 3, n_models) so an agent with a (possibly noisy)
    observed tier label t reads q_hat[i, t, :].
    """
    rng = np.random.default_rng(seed)
    folds = rng.permutation(bench.n) % k
    models = MODEL_SETS["full"]
    out = np.zeros((bench.n, 3, len(models)))
    for f in range(k):
        train = folds != f
        for t in range(3):
            mask = train & (bench.tiers == t)
            rates = [bench.scores[m][mask].mean() for m in models]
            out[folds == f, t, :] = rates
    return out


# ---------------------------------------------------------------------------
# Instance construction
# ---------------------------------------------------------------------------


@dataclass
class Instance:
    """One sampled instance: N agents x T tasks, with the oracle and expected views."""

    task_idx: np.ndarray  # dataset index per slot (len N*T)
    true_tier: np.ndarray
    obs_tier: np.ndarray  # tier label the agents / policies observe
    n_agents: int
    tasks_per_agent: int
    resampled: bool  # True when a tier pool had to be sampled with replacement


def sample_instance(bench: Bench, mix: str, n_agents: int, tasks_per_agent: int, seed: int, label_noise: float = 0.0) -> Instance:
    rng = np.random.default_rng(10_000 + seed)
    n = n_agents * tasks_per_agent
    props = MIXES[mix]
    counts = [int(round(p * n)) for p in props]
    counts[int(np.argmax(props))] += n - sum(counts)
    picks = []
    resampled = False
    for t, c in enumerate(counts):
        pool = np.flatnonzero(bench.tiers == t)
        replace = c > len(pool)
        resampled |= replace
        picks.append(rng.choice(pool, size=c, replace=replace))
    idx = rng.permutation(np.concatenate(picks))
    true_tier = bench.tiers[idx]
    obs = true_tier.copy()
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        shift = rng.integers(1, 3, size=n)  # move to one of the two other tiers
        obs = np.where(flip, (true_tier + shift) % 3, true_tier)
    return Instance(idx, true_tier, obs, n_agents, tasks_per_agent, resampled)


def build_agents(bench: Bench, inst: Instance, info: str, qhat: np.ndarray, models: list[str]) -> AgentRegistry:
    """Agent registry whose Task.quality is what the policies are allowed to see."""
    full = MODEL_SETS["full"]
    col = [full.index(m) for m in models]
    task_lists = []
    for a in range(inst.n_agents):
        tasks = []
        for s in range(a * inst.tasks_per_agent, (a + 1) * inst.tasks_per_agent):
            i = int(inst.task_idx[s])
            if info == "oracle":
                q = {m: float(bench.scores[m][i]) for m in models}
                tier = TIERS[int(inst.true_tier[s])]
            else:
                ot = int(inst.obs_tier[s])
                q = {m: float(qhat[i, ot, c]) for m, c in zip(models, col)}
                tier = TIERS[ot]
            tasks.append(Task(task_id=f"{bench.task_ids[i]}@{s}", difficulty=tier, quality=q, metadata={"slot": s}))
        task_lists.append(tasks)
    return AgentRegistry.from_task_lists(task_lists)


def build_models(bench: Bench, models: list[str]) -> ModelRegistry:
    reg = ModelRegistry()
    for m in sorted(models, key=lambda m: bench.costs[m]):
        reg.add(m, bench.costs[m])
    return reg


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class CachedShapley(ShapleyBudgetShareUtility):
    """Same utility as ``shapley_budget_share``; phi_i is computed once per (run, agent).

    phi_i depends only on the agents, models and params (not on B_k or the step), so caching
    is exact. ``shared_cache`` lets several arrival policies on one instance share it.
    """

    registry_name = "shapley_budget_share"

    def __init__(self, shared_cache: dict | None = None, **params):
        super().__init__(**params)
        self._cache = shared_cache if shared_cache is not None else {}

    def compute(self, job, model, state):
        key = job.agent_id
        if key not in self._cache:
            self._cache[key] = self.compute_phi(state.agents.get(job.agent_id), state)
        phi_i = self._cache[key]
        lambda_cost = float(self.params.get("lambda_cost", 0.1))
        gamma = float(self.params.get("gamma", 1.0))
        denominator = max(1.0 + gamma * phi_i, 1e-3)
        return model.quality(job.task) - lambda_cost / denominator * model.cost


@dataclass
class PolicySpec:
    name: str  # label in results
    paradigm: str  # "central" | "decentral"
    central: str | None = None
    arrival: str | None = None
    utility: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


def all_policy_specs(utility_params: dict[str, dict]) -> list[PolicySpec]:
    specs = [PolicySpec(f"C:{c}", "central", central=c, params={"threshold": 0.8} if c == "confidence_cascade" else {}) for c in CENTRAL_POLICIES]
    for a in ARRIVALS:
        for u in UTILITIES:
            specs.append(PolicySpec(f"D:{a}|{u}", "decentral", arrival=a, utility=u, params=dict(utility_params[u])))
    return specs


# ---------------------------------------------------------------------------
# Running one instance
# ---------------------------------------------------------------------------


def _decile(step: int, n: int) -> int:
    return min(9, int(10 * step / n))


def evaluate(bench, inst, models_reg, chosen: dict[int, list[str]], steps: dict[int, int] | None, qhat_view: AgentRegistry, budget: float) -> dict[str, float]:
    """Score a full assignment {job_id: [attempted models]} against realized outcomes."""
    jobs = qhat_view.jobs
    n = len(jobs)
    q_real = np.zeros(n)
    q_exp = np.zeros(n)
    cost = np.zeros(n)
    starved = np.zeros(n, dtype=bool)
    agent = np.zeros(n, dtype=int)
    tier = np.zeros(n, dtype=int)
    final_model = []
    for j in jobs:
        att = chosen.get(j.job_id, [FALLBACK])
        last = att[-1]
        s = j.task.metadata["slot"]
        i = int(inst.task_idx[s])
        agent[j.job_id] = j.agent_id
        tier[j.job_id] = int(inst.true_tier[s])
        cost[j.job_id] = sum(models_reg.cost(m) for m in att if m != FALLBACK)
        if last == FALLBACK:
            starved[j.job_id] = True
        else:
            q_real[j.job_id] = bench.scores[last][i]
            q_exp[j.job_id] = j.task.quality[last]
        final_model.append(last)
    total_cost = cost.sum()
    if total_cost > budget * (1 + 1e-6):  # policies use 1e-9 per-step tolerances that can accumulate
        raise RuntimeError(f"budget violated: {total_cost} > {budget}")
    payoffs = np.array([q_real[agent == a].mean() for a in range(inst.n_agents)])
    jain = payoffs.sum() ** 2 / (len(payoffs) * (payoffs**2).sum()) if (payoffs**2).sum() > 0 else np.nan
    out = {
        "quality": q_real.mean(),
        "exp_quality": q_exp.mean(),
        "total_cost": total_cost,
        "budget_util": total_cost / budget,
        "task_starve": starved.mean(),
        "agent_starve": float(np.mean([starved[agent == a].any() for a in range(inst.n_agents)])),
        "jain": jain,
        "min_agent_q": payoffs.min(),
    }
    for t, name in enumerate(TIERS):
        m = tier == t
        out[f"q_{name}"] = q_real[m].mean() if m.any() else np.nan
        out[f"spend_{name}"] = cost[m].sum() / total_cost if total_cost > 0 else np.nan
    for mname in models_reg.names:
        out[f"use_{mname}"] = float(np.mean([fm == mname for fm in final_model]))
    if steps is not None:
        order = np.zeros(n, dtype=int)
        for jid, st in steps.items():
            order[jid] = st
        dec = np.array([_decile(o, n) for o in order])
        for d in range(10):
            m = dec == d
            out[f"pos_q_{d}"] = q_real[m].mean()
            out[f"pos_starve_{d}"] = starved[m].mean()
        # first-mover advantage ratio (agent level: agents ranked by mean arrival step)
        mean_step = np.array([order[agent == a].mean() for a in range(inst.n_agents)])
        ranked = payoffs[np.argsort(mean_step)]
        k = max(1, inst.n_agents // 4)
        first, last = ranked[:k].mean(), ranked[-k:].mean()
        out["fmr"] = first / last if last > 0 else (np.inf if first > 0 else np.nan)
        out["fmr_diff"] = first - last
    return out


def run_instance(cfg: dict) -> list[dict]:
    """Run every requested policy on one sampled instance. ``cfg`` keys:
    info, mix, n_agents, tasks_per_agent, budget_mult, model_set, seed, label_noise,
    utility_params, policies (list of PolicySpec dicts or None = all), tag
    """
    bench = _BENCH_CACHE.get("bench") or _BENCH_CACHE.setdefault("bench", load_bench())
    qhat = _BENCH_CACHE.get("qhat")
    if qhat is None:
        qhat = _BENCH_CACHE.setdefault("qhat", cross_fitted_qhat(bench))
    models = MODEL_SETS[cfg["model_set"]]
    inst = sample_instance(bench, cfg["mix"], cfg["n_agents"], cfg["tasks_per_agent"], cfg["seed"], cfg.get("label_noise", 0.0))
    reg = build_models(bench, models)
    c_min = min(bench.costs[m] for m in models)
    budget = cfg["budget_mult"] * c_min * cfg["n_agents"] * cfg["tasks_per_agent"]
    view = build_agents(bench, inst, cfg["info"], qhat, models)
    oracle_view = view if cfg["info"] == "oracle" else build_agents(bench, inst, "oracle", qhat, models)

    specs = cfg.get("policies") or all_policy_specs(cfg["utility_params"])
    specs = [PolicySpec(**s) if isinstance(s, dict) else s for s in specs]
    shap_cache: dict = {}
    rows = []
    for sp in specs:
        rng = random.Random(cfg["seed"] * 7919 + 17)
        steps = None
        if sp.paradigm == "central":
            pol = create_centralized_policy(sp.central, **sp.params)
            # The cascade verifies each answer by running the unit tests after paying for it, so
            # its escalation decisions use realized outcomes in both information settings.
            src = oracle_view if sp.central == "confidence_cascade" else view
            mapping = pol.allocate(src, reg, budget, rng)
            chosen = {jid: attempts_of(v) for jid, v in mapping.items()}
        else:
            arrival = create_arrival_policy(sp.arrival)
            if sp.utility == "shapley_budget_share":
                util = CachedShapley(shared_cache=shap_cache, **sp.params)
            else:
                util = create_utility_function(sp.utility, **sp.params)
            hist = SequentialBestResponse(arrival, util).allocate(view, reg, budget, rng)
            chosen = {a.job_id: [a.model] for a in hist}
            steps = {a.job_id: a.step for a in hist}
        res = evaluate(bench, inst, reg, chosen, steps, view, budget)
        res.update(
            policy=sp.name,
            paradigm=sp.paradigm,
            arrival=sp.arrival,
            utility=sp.utility,
            info=cfg["info"],
            mix=cfg["mix"],
            n_agents=cfg["n_agents"],
            tasks_per_agent=cfg["tasks_per_agent"],
            budget_mult=cfg["budget_mult"],
            budget=budget,
            model_set=cfg["model_set"],
            seed=cfg["seed"],
            label_noise=cfg.get("label_noise", 0.0),
            resampled=inst.resampled,
            tag=cfg.get("tag", ""),
            params=json.dumps(sp.params, sort_keys=True),
        )
        rows.append(res)
    return rows


_BENCH_CACHE: dict[str, Any] = {}


def mean_or_nan(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return statistics.fmean(xs) if xs else float("nan")
