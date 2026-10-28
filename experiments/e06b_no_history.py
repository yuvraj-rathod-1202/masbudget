"""E6-B - Does the mechanism need the historical price lambda*_hist? The factorial of E6 for the
pacing agents with p0 from history vs. p0 = 0 (start from the floor price only).

    python experiments/e06b_no_history.py
"""

from __future__ import annotations

import itertools
import time

from e06_factorial import MIXES, NS, PRICINGS, SERVE_ALL, TAUS
from lib import Scenario, add_gaps, central, final_policy, md_table, run_grid, save, summarize

POLICIES = [
    central("MCKP (optimal)", "optimal_mckp"),
    central(SERVE_ALL, "optimal_mckp", serve_all=True),
    central("Cheapest model", "cheapest_model", group="baseline"),
    final_policy(),
    final_policy("Pacing + reserve (no price history)", scale=0.0),
    final_policy("Pacing + reserve + abstain (ours)", abstain=True),
    final_policy("Pacing + reserve + abstain (no price history)", abstain=True, scale=0.0),
]

if __name__ == "__main__":
    t0 = time.time()
    scen = [Scenario(num_agents=n, tasks_per_agent=5, mix=m, tau=t, pricing=p, seed=s)
            for n, m, t, p in itertools.product(NS, MIXES, TAUS, PRICINGS) for s in range(10)]
    df = run_grid(scen, POLICIES, num_runs=5)
    save(df, "e06b_raw")
    g = add_gaps(df)
    s = summarize(g, ["N", "mix", "tau", "pricing", "policy"], ["gap_exp", "task_starve"])
    save(s, "e06b_summary")
    overall = s.groupby("policy").agg(mean_gap=("gap_exp", "mean"), worst_gap=("gap_exp", "max"),
                                      cells_over_5pct=("gap_exp", lambda x: (x > 0.05).mean()),
                                      mean_starve=("task_starve", "mean"))
    print(md_table(overall.sort_values("mean_gap").reset_index(), ".4f"))
    print(f"done in {time.time() - t0:.0f}s")
