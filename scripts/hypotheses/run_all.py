"""Run every experiment stage for the hypothesis evaluation.

    python scripts/hypotheses/run_all.py tune     # hyper-parameter tuning on tuning seeds
    python scripts/hypotheses/run_all.py main     # budget x mix x info x all policies (H1, H3-H6)
    python scripts/hypotheses/run_all.py scale    # number of agents (H2)
    python scripts/hypotheses/run_all.py disparity  # flat vs steep model pools (H7)
    python scripts/hypotheses/run_all.py noise    # robustness to noisy difficulty labels
    python scripts/hypotheses/run_all.py all

Outputs: results/hypotheses/<stage>.csv.gz (one row per instance x policy).
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

import harness as H  # noqa: E402

OUT = ROOT / "results" / "hypotheses"
TUNED_PATH = OUT / "tuned_params.json"

BUDGET_MULTS = [1, 1.5, 3, 6, 12, 25, 50, 120]  # B = mult * c_min * (#tasks); 120*c_min = c_max
TEST_SEEDS = list(range(50))
TUNE_SEEDS = list(range(1000, 1020))
N_DEFAULT, T_DEFAULT = 10, 10


def _pool_run(cfgs, label):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(H.run_instance, cfgs, chunksize=2)):
            rows.extend(r)
            if (i + 1) % max(1, len(cfgs) // 20) == 0:
                print(f"[{label}] {i + 1}/{len(cfgs)} instances  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{label}.csv.gz", index=False)
    print(f"[{label}] done: {len(df)} rows in {time.time() - t0:.0f}s", flush=True)
    return df


def tuned_params(model_set: str, mult: float) -> dict:
    """Per-budget calibrated utility params (selected on TUNE_SEEDS, uniform mix, random arrival)."""
    t = json.loads(TUNED_PATH.read_text())
    return t["per_budget"][model_set][str(float(mult))]


# ---------------------------------------------------------------------------


def tune_grid():
    grid = []
    for lam in [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        grid.append(("quasi_linear", {"lambda_cost": lam}))
    for p0, alpha in itertools.product([0.1, 0.3, 1, 3, 10, 30, 100], [0.5, 1, 1.5, 2, 3]):
        grid.append(("dynamic_shadow_price", {"p0": p0, "alpha": alpha}))
    for lam in [0.5, 5, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        grid.append(("shapley_budget_share", {"lambda_cost": lam, "gamma": 1.0, "num_samples": 100, "seed": 42}))
    return grid


def stage_tune():
    specs = [
        H.PolicySpec(f"D:random_fcfs|{u}|{json.dumps(p, sort_keys=True)}", "decentral", arrival="random_fcfs", utility=u, params=p).__dict__
        for u, p in tune_grid()
    ]
    specs.append(H.PolicySpec("C:optimal_mckp", "central", central="optimal_mckp").__dict__)
    cfgs = [
        dict(info="expected", mix="uniform", n_agents=N_DEFAULT, tasks_per_agent=T_DEFAULT, budget_mult=m, model_set=ms,
             seed=s, utility_params=H.DEFAULT_UTILITY_PARAMS, policies=specs, tag="tune")
        for ms in ["full", "flat", "steep"] for m in BUDGET_MULTS for s in TUNE_SEEDS
    ]
    df = _pool_run(cfgs, "tune")
    d = df[df.paradigm == "decentral"]
    out = {"global": {}, "per_budget": {}}
    # (a) one setting for all budgets (full pool) -- reported as a finding
    g = d[d.model_set == "full"].groupby(["utility", "params"]).quality.mean().reset_index()
    for u in H.UTILITIES:
        gu = g[g.utility == u]
        out["global"][u] = json.loads(gu.sort_values("quality", ascending=False).iloc[0].params) if len(gu) else H.DEFAULT_UTILITY_PARAMS[u]
    # (b) calibrated per (model pool, budget level) -- used in all test stages
    s = d.groupby(["model_set", "budget_mult", "utility", "params"]).quality.mean().reset_index()
    for ms in ["full", "flat", "steep"]:
        out["per_budget"][ms] = {}
        for m in BUDGET_MULTS:
            sel = {}
            for u in H.UTILITIES:
                su = s[(s.model_set == ms) & (s.budget_mult == m) & (s.utility == u)]
                sel[u] = json.loads(su.sort_values("quality", ascending=False).iloc[0].params) if len(su) else dict(H.DEFAULT_UTILITY_PARAMS[u])
            out["per_budget"][ms][str(float(m))] = sel
    TUNED_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["global"], indent=1))
    for m in BUDGET_MULTS:
        print(m, {u: out["per_budget"]["full"][str(float(m))][u] for u in ["quasi_linear", "dynamic_shadow_price"]})


def stage_main():
    cfgs = [
        dict(info=info, mix=mix, n_agents=N_DEFAULT, tasks_per_agent=T_DEFAULT, budget_mult=m, model_set="full",
             seed=s, utility_params=tuned_params("full", m), tag="main")
        for info in ["expected", "oracle"] for mix in H.MIXES for m in BUDGET_MULTS for s in TEST_SEEDS
    ]
    _pool_run(cfgs, "main")


def stage_scale():
    cfgs = [
        dict(info="expected", mix=mix, n_agents=n, tasks_per_agent=T_DEFAULT, budget_mult=m, model_set="full",
             seed=s, utility_params=tuned_params("full", m), tag="scale")
        for mix in H.MIXES for n in [5, 10, 20, 50] for m in [3, 12] for s in TEST_SEEDS
    ]
    _pool_run(cfgs, "scale")


def stage_disparity():
    cfgs = [
        dict(info="expected", mix=mix, n_agents=N_DEFAULT, tasks_per_agent=T_DEFAULT, budget_mult=m, model_set=ms,
             seed=s, utility_params=tuned_params(ms, m), tag="disparity")
        for ms in ["flat", "steep"] for mix in H.MIXES for m in BUDGET_MULTS for s in TEST_SEEDS
    ]
    _pool_run(cfgs, "disparity")


def stage_noise():
    cfgs = [
        dict(info="expected", mix="uniform", n_agents=N_DEFAULT, tasks_per_agent=T_DEFAULT, budget_mult=m, model_set="full",
             seed=s, label_noise=eps, utility_params=tuned_params("full", m), tag="noise")
        for eps in [0.0, 0.2, 0.4] for m in [3, 12, 50] for s in TEST_SEEDS
    ]
    _pool_run(cfgs, "noise")


STAGES = {"tune": stage_tune, "main": stage_main, "scale": stage_scale, "disparity": stage_disparity, "noise": stage_noise}

if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    if which == ["all"]:
        which = list(STAGES)
    for w in which:
        STAGES[w]()
