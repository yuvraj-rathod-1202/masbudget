"""Extra robustness runs for FSSP (uses the single setting in fssp_params.json).

    python scripts/hypotheses/fssp_extra.py noise   # noisy difficulty labels
    python scripts/hypotheses/fssp_extra.py burst   # agents arrive in random order, all tasks at once

Outputs: results/hypotheses/fssp_noise.csv.gz, fssp_burst.csv.gz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import burst_stage  # noqa: E402,F401  (registers "random_agent_burst")
import fair_share  # noqa: E402,F401  (registers "fair_share_price")
import fssp_stage as F  # noqa: E402
import harness as H  # noqa: E402
import run_all as R  # noqa: E402


def run(cfg):
    return H.run_instance(cfg)


F.run = run  # make the pool use this module's picklable entry point


def stage_noise():
    cfgs = [dict(info="expected", mix="uniform", n_agents=10, tasks_per_agent=10, budget_mult=m, model_set="full",
                 seed=s, label_noise=eps, utility_params=H.DEFAULT_UTILITY_PARAMS, policies=F.test_specs("full", m), tag="fssp_noise")
            for eps in [0.0, 0.2, 0.4] for m in [3, 12, 50] for s in R.TEST_SEEDS]
    F.pool_run(cfgs, "fssp_noise")


def stage_burst():
    fs = json.loads(F.PARAMS_PATH.read_text())["fssp"]
    cfgs = []
    for m in R.BUDGET_MULTS:
        cal = R.tuned_params("full", m)
        specs = [F.spec("C:optimal_mckp", "central", central="optimal_mckp"),
                 F.spec("D:random_agent_burst|fssp", "decentral", arrival="random_agent_burst", utility="fair_share_price", params=fs),
                 F.spec("D:random_agent_burst|best_model", "decentral", arrival="random_agent_burst", utility="best_model"),
                 F.spec("D:random_agent_burst|dsp_calibrated", "decentral", arrival="random_agent_burst",
                        utility="dynamic_shadow_price", params=cal["dynamic_shadow_price"]),
                 F.spec("D:random_agent_burst|ql_calibrated", "decentral", arrival="random_agent_burst",
                        utility="quasi_linear", params=cal["quasi_linear"])]
        for s in R.TEST_SEEDS:
            cfgs.append(dict(info="expected", mix="uniform", n_agents=10, tasks_per_agent=10, budget_mult=m, model_set="full",
                             seed=s, utility_params=H.DEFAULT_UTILITY_PARAMS, policies=specs, tag="fssp_burst"))
    F.pool_run(cfgs, "fssp_burst")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["noise", "burst"]:
        {"noise": stage_noise, "burst": stage_burst}[w]()
