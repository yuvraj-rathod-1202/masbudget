"""E5-D - Price floor for the pacing utility (fixes p0 = 0 when history says the budget is slack).

p_min = floor / (c_max - c_min); floors x beta, with and without reserve, every mix and budget.

    python experiments/e05d_price_floor.py          # evaluation split, N = 20 x 10
    python experiments/e05d_price_floor.py calib    # calibration split, N = 10 x 10 (selection)
"""

from __future__ import annotations

import sys
import time

from lib import Scenario, add_gaps, central, decentral, md_table, pacing_params, run_grid, save, summarize

FLOORS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
BETAS = (5.0, 10.0)
TAUS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)

if __name__ == "__main__":
    t0 = time.time()
    which = sys.argv[1] if len(sys.argv) > 1 else "eval"
    n_agents = 10 if which == "calib" else 20
    pols = [central("MCKP (optimal)", "optimal_mckp"), central("Cheapest model", "cheapest_model", group="baseline")]
    for f in FLOORS:
        for b in BETAS:
            for res in (False, True):
                pols.append(decentral(f"floor={f}|beta={b}|res={res}", "budget_pacing", reserve=res,
                                      params_fn=pacing_params(beta=b, floor=f), group="D"))
    scen = [Scenario(mix=m, tau=t, seed=s, split=which, num_agents=n_agents) for m in ("easy_heavy", "uniform", "hard_heavy") for t in TAUS for s in range(10)]
    df = run_grid(scen, pols, num_runs=5)
    suffix = "" if which == "eval" else "_calib"
    save(df, f"e05d_raw{suffix}")
    s = summarize(add_gaps(df), ["mix", "tau", "policy"], ["gap_exp", "task_starve"])
    s = s[s.policy.str.startswith("floor=")]
    save(s, f"e05d_summary{suffix}")
    worst = s.groupby(["policy", "mix"]).gap_exp.max().unstack()
    worst["worst"] = worst.max(axis=1)
    worst["mean_all"] = s.groupby("policy").gap_exp.mean()
    print("### worst-case gap over budgets, per mix\n" + md_table(worst.sort_values("worst").reset_index(), ".4f"))
    print(f"done in {time.time() - t0:.0f}s")
