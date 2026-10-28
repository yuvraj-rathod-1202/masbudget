"""Shared infrastructure for the experiment study (see experiments/README.md).

Everything runs through the ``masbudget`` framework (``Experiment``, registries, policies);
this module only builds instances, runs many of them in parallel and scores the runs.

Information regimes (what policies see as q_{i,m}; outcomes are always scored on the
realized LiveCodeBench pass@1 result, kept in ``task.metadata["realized"]``):

* ``oracle``     - q = the realized pass/fail of model m on task i (the dataset as converted).
* ``difficulty`` - q = pass rate of model m on calibration tasks with the same difficulty label.
* ``blind``      - q = overall pass rate of model m on the calibration tasks.

The 1,055 tasks are split once (stratified by difficulty, seed ``SPLIT_SEED``) into a
calibration split (estimates q-hat, tunes hyper-parameters, calibrates prices) and an
evaluation split (every reported number).
"""

from __future__ import annotations

import functools
import logging
import math
import os
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from masbudget.core import CENTRALIZED, DECENTRALIZED, DIFFICULTY_LEVELS, FALLBACK, RunResult, Task  # noqa: E402
from masbudget.dataset import load_dataset  # noqa: E402
from masbudget.decentralized.utilities.budget_pacing import calibrate_price  # noqa: E402
from masbudget.experiment import Experiment  # noqa: E402
from masbudget.registry import AgentRegistry, ModelRegistry  # noqa: E402

logging.getLogger("masbudget").setLevel(logging.WARNING)

DATASET = ROOT / "data" / "llmrouterbench_livecodebench.json"
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "experiments" / "figures"
SPLIT_SEED = 2026
CALIB_FRACTION = 0.3

# Mean measured cost per call (data/llmrouterbench_livecodebench_model_costs.json), cheapest first.
MEASURED_COSTS = {
    "deepseek-v3.1-terminus": 0.0011249379,
    "deepseek-v3-0324": 0.0014471013,
    "gemini-2.5-flash": 0.0058342806,
    "deepseek-r1-0528": 0.0288723124,
}
SHORT = {"deepseek-v3.1-terminus": "V3.1", "deepseek-v3-0324": "V3", "gemini-2.5-flash": "Flash", "deepseek-r1-0528": "R1"}


def scaled_costs(ratio: float) -> dict[str, float]:
    """Cost-performance disparity: c'_m = c_min * (c_m / c_min) ** g with g chosen so that
    the frontier model costs ``ratio`` x the cheapest. Keeps the cost order; ratio 25.67 = measured."""
    c_min, c_max = min(MEASURED_COSTS.values()), max(MEASURED_COSTS.values())
    g = math.log(ratio) / math.log(c_max / c_min)
    return {m: c_min * (c / c_min) ** g for m, c in MEASURED_COSTS.items()}


PRICINGS = {"measured": MEASURED_COSTS, "steep10x": scaled_costs(10.0), "flat2x": scaled_costs(2.0)}
MIXES = {"uniform": (1 / 3, 1 / 3, 1 / 3), "easy_heavy": (0.8, 0.1, 0.1), "hard_heavy": (0.1, 0.1, 0.8)}
REGIMES = ("oracle", "difficulty", "blind")


# ---------------------------------------------------------------- data, split and views


@functools.lru_cache(maxsize=1)
def base_tasks() -> tuple[Task, ...]:
    return tuple(load_dataset(DATASET))


@functools.lru_cache(maxsize=1)
def split() -> tuple[tuple[Task, ...], tuple[Task, ...]]:
    """(calibration, evaluation), stratified by difficulty."""
    rng = random.Random(SPLIT_SEED)
    calib, evaluation = [], []
    for level in DIFFICULTY_LEVELS:
        group = [t for t in base_tasks() if t.difficulty == level]
        rng.shuffle(group)
        k = round(len(group) * CALIB_FRACTION)
        calib += group[:k]
        evaluation += group[k:]
    return tuple(calib), tuple(evaluation)


@functools.lru_cache(maxsize=1)
def quality_tables() -> dict[str, Any]:
    """Calibration pass rates: per difficulty label, and overall ("blind")."""
    calib, _ = split()
    models = list(MEASURED_COSTS)
    by_level = {
        level: {m: statistics.fmean(t.quality[m] for t in calib if t.difficulty == level) for m in models}
        for level in DIFFICULTY_LEVELS
    }
    blind = {m: statistics.fmean(t.quality[m] for t in calib) for m in models}
    return {"difficulty": by_level, "blind": blind}


def view(task: Task, regime: str, label: str | None = None) -> Task:
    """The task as policies see it under ``regime``; ``label`` overrides the difficulty label."""
    label = label or task.difficulty
    tables = quality_tables()
    if regime == "oracle":
        q = dict(task.quality)
    elif regime == "difficulty":
        q = dict(tables["difficulty"][label])
    elif regime == "blind":
        q = dict(tables["blind"])
    else:
        raise ValueError(f"unknown regime {regime!r}")
    metadata = {**task.metadata, "realized": dict(task.quality), "true_difficulty": task.difficulty}
    return Task(task.task_id, label, q, task.category, task.features, metadata)


def pools(which: str = "eval") -> dict[str, list[Task]]:
    tasks = split()[0 if which == "calib" else 1]
    return {level: [t for t in tasks if t.difficulty == level] for level in DIFFICULTY_LEVELS}


# ---------------------------------------------------------------- scenarios and instances


@dataclass(frozen=True)
class Scenario:
    """One cell of the experiment grid; ``seed`` picks the task sample (the instance)."""

    regime: str = "difficulty"
    mix: str = "uniform"
    num_agents: int = 20
    tasks_per_agent: int = 10
    pricing: str = "measured"
    tau: float = 0.2  # budget tightness: B = J*c_min + tau * J * (c_max - c_min)
    label_noise: float = 0.0  # probability a task's difficulty label is replaced by another one
    split: str = "eval"
    seed: int = 0

    @property
    def num_jobs(self) -> int:
        return self.num_agents * self.tasks_per_agent

    def costs(self) -> dict[str, float]:
        return PRICINGS[self.pricing]

    def budget(self) -> float:
        costs = self.costs()
        c_min, c_max = min(costs.values()), max(costs.values())
        return self.num_jobs * (c_min + self.tau * (c_max - c_min))


@dataclass
class Instance:
    scenario: Scenario
    tasks: list[Task]  # viewed tasks, in dealing order
    agents: AgentRegistry
    models: ModelRegistry
    budget: float

    @functools.cached_property
    def calib_tasks(self) -> list[Task]:
        """Calibration split under the same regime: the 'historical' tasks agents can learn prices from."""
        return [view(t, self.scenario.regime) for t in split()[0]]


def mix_counts(mix: str, n: int) -> list[int]:
    shares = MIXES[mix]
    counts = [int(math.floor(s * n)) for s in shares]
    order = sorted(range(3), key=lambda i: -(shares[i] * n - counts[i]))
    for i in order[: n - sum(counts)]:
        counts[i] += 1
    return counts


def build_instance(sc: Scenario) -> Instance:
    rng = random.Random(1000 + sc.seed)
    by_level = pools(sc.split)
    sample: list[Task] = []
    for level, count in zip(DIFFICULTY_LEVELS, mix_counts(sc.mix, sc.num_jobs)):
        if count > len(by_level[level]):
            raise ValueError(f"{sc}: needs {count} {level} tasks, pool has {len(by_level[level])}")
        sample += rng.sample(by_level[level], count)
    rng.shuffle(sample)

    noise_rng = random.Random(5000 + sc.seed)
    viewed = []
    for t in sample:
        label = t.difficulty
        if sc.label_noise and noise_rng.random() < sc.label_noise:
            label = noise_rng.choice([lv for lv in DIFFICULTY_LEVELS if lv != t.difficulty])
        viewed.append(view(t, sc.regime, label))

    tpa = sc.tasks_per_agent
    agents = AgentRegistry.from_task_lists(viewed[i * tpa:(i + 1) * tpa] for i in range(sc.num_agents))
    models = ModelRegistry()
    for name, cost in sc.costs().items():
        models.add(name, cost)
    return Instance(sc, viewed, agents, models, sc.budget())


@functools.lru_cache(maxsize=4096)
def _calibrated_price(regime: str, pricing: str, budget_per_job: float, which: str) -> float:
    if which == "calib":
        tasks = [view(t, regime) for t in split()[0]]
    else:
        raise ValueError(which)
    models = ModelRegistry()
    for name, cost in PRICINGS[pricing].items():
        models.add(name, cost)
    return calibrate_price(tasks, models, budget_per_job * len(tasks))


def historical_price(inst: Instance) -> float:
    """lambda* of the calibration split at the instance's budget per job (what agents could
    learn from past batches; does not know this batch's difficulty mix)."""
    sc = inst.scenario
    return _calibrated_price(sc.regime, sc.pricing, round(inst.budget / sc.num_jobs, 12), "calib")


def in_batch_price(inst: Instance) -> float:
    """lambda* of this very batch (perfect knowledge of the batch; a reference, not decentralized)."""
    return calibrate_price(inst.tasks, inst.models, inst.budget)


# ---------------------------------------------------------------- policies


@dataclass(frozen=True)
class Policy:
    """A centralized policy or a decentralized (utility, arrival, reserve) combination.

    ``params_fn(instance) -> dict`` adds instance-dependent params (e.g. a calibrated price).
    """

    label: str
    paradigm: str
    name: str  # centralized policy or utility factory name
    params: tuple = ()
    arrival: str | None = None
    reserve: bool = False
    params_fn: Callable[[Instance], dict] | None = None
    group: str = ""
    abstain: bool = False
    engine: str = "sequential_best_response"  # decentralized game; "simultaneous_best_response" = no sequencing
    engine_params: tuple = ()

    @property
    def stochastic(self) -> bool:
        return self.paradigm == DECENTRALIZED and (self.arrival == "random_fcfs" or self.engine != "sequential_best_response")

    def resolved_params(self, inst: Instance) -> dict:
        params = dict(self.params)
        if self.params_fn is not None:
            params.update(self.params_fn(inst))
        return params


def central(label: str, name: str, group: str = "centralized", **params) -> Policy:
    return Policy(label, CENTRALIZED, name, tuple(params.items()), group=group)


def decentral(
    label: str, utility: str, arrival: str | None = "random_fcfs", reserve: bool = False,
    params_fn: Callable[[Instance], dict] | None = None, group: str = "decentralized", abstain: bool = False,
    **params,
) -> Policy:
    return Policy(label, DECENTRALIZED, utility, tuple(params.items()), arrival, reserve, params_fn, group, abstain)


@dataclass(frozen=True)
class pacing_params:
    """params_fn for ``budget_pacing``: p0 = scale * lambda* (historical or in-batch)."""

    scale: float = 1.0
    beta: float = 5.0
    source: str = "historical"
    horizon_scale: float = 1.0
    floor: float = 0.0  # p_min = floor / (c_max - c_min): the price at which a full quality point
    #                     is worth exactly the premium over the cheapest model

    def __call__(self, inst: Instance) -> dict:
        lam = historical_price(inst) if self.source == "historical" else in_batch_price(inst)
        costs = [m.cost for m in inst.models]
        out = {"p0": self.scale * lam, "beta": self.beta, "p_min": self.floor / (max(costs) - min(costs))}
        if self.horizon_scale != 1.0:
            out["horizon"] = self.horizon_scale * inst.scenario.num_jobs
        return out


CENTRAL_POLICIES = [
    central("MCKP (optimal)", "optimal_mckp"),
    central("Predictive router", "predictive_router"),
    central("Cascade (verified)", "confidence_cascade", threshold=0.5, verify_key="realized",
            skip_dominated=True, escalation="jump"),
    central("Static equal quota", "static_equal_quota"),
    central("Cheapest model", "cheapest_model", group="baseline"),
]
REFERENCE = "MCKP (optimal)"
CHEAPEST = "Cheapest model"


# ---------------------------------------------------------------- running and scoring


def score(run: RunResult, inst: Instance) -> dict[str, Any]:
    """Metrics of one run. ``q_exp`` is quality as the policies saw it (q-hat), ``q_real`` the
    realized pass@1. Agent payoff = mean quality over the agent's tasks."""
    task_of = {job.job_id: job.task for job in inst.agents.jobs}
    J = len(run.assignments)

    def realized(a) -> float:
        return 0.0 if a.model == FALLBACK else task_of[a.job_id].metadata["realized"][a.model]

    real = [realized(a) for a in run.assignments]
    exp = [a.quality for a in run.assignments]
    out: dict[str, Any] = {
        "q_exp": sum(exp) / J,
        "q_real": sum(real) / J,
        "cost": run.total_cost,
        "bur": run.total_cost / run.budget if run.budget > 0 else math.nan,
        "task_starve": sum(a.starved for a in run.assignments) / J,
    }
    groups: dict[int, list[int]] = {}
    for idx, a in enumerate(run.assignments):
        groups.setdefault(a.agent_id, []).append(idx)
    out["agent_starve"] = sum(any(run.assignments[i].starved for i in rows) for rows in groups.values()) / len(groups)
    for key, vals in (("exp", exp), ("real", real)):
        payoffs = [statistics.fmean(vals[i] for i in rows) for rows in groups.values()]
        squares = sum(u * u for u in payoffs)
        out[f"jain_{key}"] = sum(payoffs) ** 2 / (len(payoffs) * squares) if squares > 0 else math.nan

    for level in DIFFICULTY_LEVELS:
        rows = [i for i, a in enumerate(run.assignments) if task_of[a.job_id].metadata["true_difficulty"] == level]
        out[f"q_real_{level}"] = statistics.fmean(real[i] for i in rows) if rows else math.nan
        out[f"premium_{level}"] = (
            statistics.fmean(run.assignments[i].model == "deepseek-r1-0528" for i in rows) if rows else math.nan
        )

    if run.paradigm == DECENTRALIZED:
        # Position privilege: quality by arrival position (the assignment list is in arrival order).
        q = len(exp) // 4 or 1
        first, last = statistics.fmean(exp[:q]), statistics.fmean(exp[-q:])
        out["fma_jobs"] = first / last if last > 0 else (math.inf if first > 0 else math.nan)
        ordered = sorted(groups.values(), key=lambda rows: statistics.fmean(rows))
        k = max(1, len(ordered) // 4)
        pf = statistics.fmean(statistics.fmean(exp[i] for i in rows) for rows in ordered[:k])
        pl = statistics.fmean(statistics.fmean(exp[i] for i in rows) for rows in ordered[-k:])
        out["fma_agents"] = pf / pl if pl > 0 else (math.inf if pf > 0 else math.nan)
        for d in range(10):
            lo, hi = d * J // 10, (d + 1) * J // 10
            out[f"pos{d}"] = statistics.fmean(exp[lo:hi]) if hi > lo else math.nan
    return out


def run_policy(policy: Policy, inst: Instance, num_runs: int, seed: int) -> list[RunResult]:
    params = policy.resolved_params(inst)
    if policy.paradigm == CENTRALIZED:
        exp = Experiment(inst.agents, inst.models, inst.budget, centralized_policy=policy.name,
                         centralized_params=params, seed=seed)
        return exp.run().runs[CENTRALIZED]
    engine_params = (
        {"reserve": policy.reserve, "abstain": policy.abstain} if policy.engine == "sequential_best_response"
        else dict(policy.engine_params)
    )
    exp = Experiment(
        inst.agents, inst.models, inst.budget, utility=policy.name, utility_params=params,
        arrival_policy=policy.arrival, decentralized_policy=policy.engine, decentralized_params=engine_params,
        seed=seed, num_runs=num_runs if policy.stochastic else 1,
    )
    return exp.run().runs[DECENTRALIZED]


def run_unit(args: tuple[Scenario, Sequence[Policy], int, dict]) -> list[dict[str, Any]]:
    sc, policies, num_runs, extra = args
    inst = build_instance(sc)
    rows = []
    for policy in policies:
        for r, run in enumerate(run_policy(policy, inst, num_runs, seed=10_000 * sc.seed)):
            rows.append({
                **scenario_columns(sc), **extra, "budget": inst.budget,
                "policy": policy.label, "paradigm": policy.paradigm, "group": policy.group, "run": r,
                **score(run, inst),
            })
    return rows


def scenario_columns(sc: Scenario) -> dict[str, Any]:
    return {
        "regime": sc.regime, "mix": sc.mix, "N": sc.num_agents, "tpa": sc.tasks_per_agent, "J": sc.num_jobs,
        "pricing": sc.pricing, "tau": sc.tau, "label_noise": sc.label_noise, "split": sc.split, "instance": sc.seed,
    }


def run_grid(
    scenarios: Iterable[Scenario], policies: Sequence[Policy], num_runs: int = 10,
    extra: Callable[[Scenario], dict] | None = None, workers: int | None = None,
) -> pd.DataFrame:
    units = [(sc, list(policies), num_runs, extra(sc) if extra else {}) for sc in scenarios]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rows: list[dict] = []
    if workers == 1:
        for u in units:
            rows += run_unit(u)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for out in pool.map(run_unit, units, chunksize=1):
                rows += out
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- analysis


SCENARIO_KEYS = ["regime", "mix", "N", "tpa", "J", "pricing", "tau", "label_noise", "split"]


def instance_means(df: pd.DataFrame, extra_keys: Sequence[str] = ()) -> pd.DataFrame:
    """Average the arrival-seed runs of each (scenario, instance, policy)."""
    keys = [k for k in SCENARIO_KEYS if k in df.columns] + list(extra_keys) + ["instance", "policy", "paradigm", "group"]
    num = df.select_dtypes("number").columns.difference(keys + ["run"])
    return df.groupby(keys, dropna=False)[list(num)].mean().reset_index()


def add_gaps(df: pd.DataFrame, extra_keys: Sequence[str] = (), reference: str = REFERENCE, floor: str = CHEAPEST):
    """Per-instance gaps vs. the reference (optimal MCKP) and headroom captured over the
    cheapest-model floor: (SW - SW_floor) / (SW_ref - SW_floor). Needs both policies in df."""
    m = instance_means(df, extra_keys)
    keys = [k for k in SCENARIO_KEYS if k in m.columns] + list(extra_keys) + ["instance"]
    ref = m[m.policy == reference][keys + ["q_exp", "q_real"]].rename(columns={"q_exp": "ref_exp", "q_real": "ref_real"})
    flo = m[m.policy == floor][keys + ["q_exp", "q_real"]].rename(columns={"q_exp": "floor_exp", "q_real": "floor_real"})
    m = m.merge(ref, on=keys, how="left").merge(flo, on=keys, how="left")
    for k in ("exp", "real"):
        m[f"gap_{k}"] = (m[f"ref_{k}"] - m[f"q_{k}"]) / m[f"ref_{k}"]
        span = m[f"ref_{k}"] - m[f"floor_{k}"]
        m[f"headroom_{k}"] = (m[f"q_{k}"] - m[f"floor_{k}"]) / span.where(span.abs() > 1e-12)
    return m


def summarize(m: pd.DataFrame, by: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Mean and 95% CI half-width over instances."""
    g = m.groupby(list(by), dropna=False, sort=False)
    out = g[list(metrics)].mean()
    n = g.size()
    for col in metrics:
        sd = g[col].std(ddof=1)
        out[f"{col}_ci"] = 1.96 * sd / n.pow(0.5)
    out["n"] = n
    return out.reset_index()


def save(df: pd.DataFrame, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{name}.csv")


def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(format(v, floatfmt) if isinstance(v, float) and math.isfinite(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# The mechanism selected in E5-D (on the calibration split): pace-aware price + budget reserve.
FINAL_LABEL = "Pacing + reserve (ours)"


def final_policy(label: str = FINAL_LABEL, abstain: bool = False, **overrides) -> Policy:
    params = {"beta": 10.0, "floor": 0.1, **overrides}
    return decentral(label, "budget_pacing", reserve=True, abstain=abstain, params_fn=pacing_params(**params), group="ours")
