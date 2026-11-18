"""H6 extra stage: agent-level random arrival ("burst" ordering).

``random_fcfs`` draws a random *job* at every step, so each agent's tasks are spread evenly
over the sequence and agent-level first-mover effects average out. Here agents arrive in a
uniformly random order and each submits all of its tasks back-to-back, which is the literal
"agents arrive in a random permutation pi" of the proposal.

    python scripts/hypotheses/burst_stage.py   ->  results/hypotheses/burst.csv.gz
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from masbudget.decentralized.arrival.base import ARRIVAL_POLICIES, BaseArrivalPolicy  # noqa: E402

import harness as H  # noqa: E402


if "random_agent_burst" not in getattr(ARRIVAL_POLICIES, "_items", getattr(ARRIVAL_POLICIES, "_registry", {})):
    try:
        @ARRIVAL_POLICIES.register("random_agent_burst")
        class RandomAgentBurstArrival(BaseArrivalPolicy):
            """Agents arrive in a random order; each one submits all its tasks consecutively."""

            def select_next(self, remaining, state):
                if not hasattr(self, "_order"):
                    ids = sorted({j.agent_id for j in remaining})
                    state.rng.shuffle(ids)
                    self._order = {a: k for k, a in enumerate(ids)}
                return min(remaining, key=lambda j: (self._order[j.agent_id], j.job_id))
    except ValueError:
        pass  # already registered in this process


def run(cfg):
    return H.run_instance(cfg)


if __name__ == "__main__":
    import run_all as R

    specs = [H.PolicySpec(f"D:random_agent_burst|{u}", "decentral", arrival="random_agent_burst", utility=u).__dict__ for u in H.UTILITIES]
    specs.append(H.PolicySpec("C:optimal_mckp", "central", central="optimal_mckp").__dict__)
    cfgs = []
    for m in R.BUDGET_MULTS:
        up = R.tuned_params("full", m)
        sp = [dict(s, params=dict(up[s["utility"]])) if s["utility"] else s for s in specs]
        for s in R.TEST_SEEDS:
            cfgs.append(dict(info="expected", mix="uniform", n_agents=10, tasks_per_agent=10, budget_mult=m, model_set="full",
                             seed=s, utility_params=up, policies=sp, tag="burst"))
    import multiprocessing as mp
    import time

    import pandas as pd

    t0 = time.time()
    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for r in pool.imap_unordered(run, cfgs, chunksize=2):
            rows.extend(r)
    df = pd.DataFrame(rows)
    df.to_csv(R.OUT / "burst.csv.gz", index=False)
    print(f"[burst] {len(df)} rows in {time.time() - t0:.0f}s")
