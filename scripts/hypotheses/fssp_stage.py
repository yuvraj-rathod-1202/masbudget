"""Experiments for the proposed algorithm, Fair-Share Shadow Pricing (FSSP).

    python scripts/hypotheses/fssp_stage.py tune    # pick ONE (kappa, alpha) on tuning seeds
    python scripts/hypotheses/fssp_stage.py test    # held-out seeds: mixes x budgets x pools x info
    python scripts/hypotheses/fssp_stage.py scale   # number of agents N in {5, 10, 20, 50}

Outputs: results/hypotheses/fssp_{tune,test,scale}.csv.gz and fssp_params.json.
The single (kappa, alpha) is selected on the uniform mix / full pool / expected information
only, then reused unchanged for every other setting.
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

import fair_share  # noqa: E402,F401  (registers the "fair_share_price" utility)
import harness as H  # noqa: E402
import run_all as R  # noqa: E402

PARAMS_PATH = R.OUT / "fssp_params.json"
KAPPAS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
ALPHAS = [0, 0.5, 1, 1.5, 2, 3]


def run(cfg):
    return H.run_instance(cfg)


def pool_run(cfgs, label):
    t0 = time.time()
    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(run, cfgs, chunksize=2)):
            rows.extend(r)
            if (i + 1) % max(1, len(cfgs) // 10) == 0:
                print(f"[{label}] {i + 1}/{len(cfgs)}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(R.OUT / f"{label}.csv.gz", index=False)
    print(f"[{label}] done: {len(df)} rows in {time.time() - t0:.0f}s", flush=True)
    return df


def spec(name, paradigm, **kw):
    return H.PolicySpec(name, paradigm, **kw).__dict__


def stage_tune():
    specs = [spec(f"D:random_fcfs|fair_share_price|k={k}|a={a}", "decentral", arrival="random_fcfs",
                  utility="fair_share_price", params={"kappa": k, "alpha": a})
             for k, a in itertools.product(KAPPAS, ALPHAS)]
    specs.append(spec("C:optimal_mckp", "central", central="optimal_mckp"))
    cfgs = [dict(info="expected", mix="uniform", n_agents=10, tasks_per_agent=10, budget_mult=m, model_set="full",
                 seed=s, utility_params=H.DEFAULT_UTILITY_PARAMS, policies=specs, tag="fssp_tune")
            for m in R.BUDGET_MULTS for s in R.TUNE_SEEDS]
    df = pool_run(cfgs, "fssp_tune")
    d = df[df.paradigm == "decentral"]
    score = d.groupby("params").quality.mean().sort_values(ascending=False)
    best = json.loads(score.index[0])
    PARAMS_PATH.write_text(json.dumps({"fssp": best, "top10": {k: float(v) for k, v in score.head(10).items()}}, indent=2))
    print("selected", best, "mean Q", round(score.iloc[0], 4))
    print(score.head(10))


def test_specs(model_set, mult):
    fs = json.loads(PARAMS_PATH.read_text())["fssp"]
    cal = R.tuned_params(model_set, mult)  # per-budget calibrated baselines (tuning seeds)
    out = [spec("C:optimal_mckp", "central", central="optimal_mckp"),
           spec("C:cheapest_model", "central", central="cheapest_model"),
           spec("C:predictive_router", "central", central="predictive_router"),
           spec("C:static_equal_quota", "central", central="static_equal_quota")]
    for a in H.ARRIVALS:
        out.append(spec(f"D:{a}|fssp", "decentral", arrival=a, utility="fair_share_price", params=fs))
    out += [
        spec("D:random_fcfs|fssp_alpha0", "decentral", arrival="random_fcfs", utility="fair_share_price",
             params={"kappa": fs["kappa"], "alpha": 0.0}),
        spec("D:random_fcfs|best_model", "decentral", arrival="random_fcfs", utility="best_model"),
        spec("D:random_fcfs|dsp_default", "decentral", arrival="random_fcfs", utility="dynamic_shadow_price",
             params=H.DEFAULT_UTILITY_PARAMS["dynamic_shadow_price"]),
        spec("D:random_fcfs|dsp_calibrated", "decentral", arrival="random_fcfs", utility="dynamic_shadow_price",
             params=cal["dynamic_shadow_price"]),
        spec("D:random_fcfs|ql_calibrated", "decentral", arrival="random_fcfs", utility="quasi_linear",
             params=cal["quasi_linear"]),
        spec("D:random_fcfs|marginal_roi_ratio", "decentral", arrival="random_fcfs", utility="marginal_roi_ratio"),
        spec("D:random_fcfs|shapley_calibrated", "decentral", arrival="random_fcfs", utility="shapley_budget_share",
             params=cal["shapley_budget_share"]),
    ]
    return out


def stage_test():
    cfgs = [dict(info=info, mix=mix, n_agents=10, tasks_per_agent=10, budget_mult=m, model_set=ms, seed=s,
                 utility_params=H.DEFAULT_UTILITY_PARAMS, policies=test_specs(ms, m), tag="fssp_test")
            for info in ["expected", "oracle"] for ms in ["full", "steep"] for mix in H.MIXES
            for m in R.BUDGET_MULTS for s in R.TEST_SEEDS]
    pool_run(cfgs, "fssp_test")


def stage_scale():
    cfgs = [dict(info="expected", mix=mix, n_agents=n, tasks_per_agent=10, budget_mult=m, model_set="full", seed=s,
                 utility_params=H.DEFAULT_UTILITY_PARAMS, policies=test_specs("full", m), tag="fssp_scale")
            for mix in H.MIXES for n in [5, 10, 20, 50] for m in [3, 12] for s in R.TEST_SEEDS]
    pool_run(cfgs, "fssp_scale")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["tune", "test", "scale"]:
        {"tune": stage_tune, "test": stage_test, "scale": stage_scale}[w]()
