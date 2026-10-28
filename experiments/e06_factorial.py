"""E6 - Full factorial (RQ5): agents x difficulty mix x budget x cost-performance disparity.

N in {5, 10, 20, 50} agents x 5 tasks each; mixes uniform / easy-heavy (80/10/10) / hard-heavy
(10/10/80); tau in {0, 0.05, 0.2, 0.5, 1}; pricing flat (R1 = 2x cheapest), steep (10x), measured
(25.7x). Difficulty regime, random arrival. Fixed-parameter utilities keep the values tuned in E4
under measured pricing (a transfer test).

    python experiments/e06_factorial.py
"""

from __future__ import annotations

import itertools
import sys
import time

import numpy as np

from lib import (
    CENTRAL_POLICIES, FIGURES, Scenario, add_gaps, central, decentral, final_policy, load, md_table, run_grid, save,
    summarize,
)
import plots

SERVE_ALL = "MCKP (serve all)"
POLICIES = CENTRAL_POLICIES + [
    central(SERVE_ALL, "optimal_mckp", serve_all=True),
    decentral("Best model", "best_model", group="naive"),
    decentral("Best model + reserve", "best_model", reserve=True, group="reserve"),
    decentral("U1 quasi-linear (lambda=30)", "quasi_linear", lambda_cost=30, group="tuned"),
    decentral("U1 quasi-linear (lambda=10) + reserve", "quasi_linear", lambda_cost=10, reserve=True, group="tuned"),
    decentral("U2 shadow price (p0=3, a=0.5) + reserve", "dynamic_shadow_price", p0=3, alpha=0.5, reserve=True, group="tuned"),
    final_policy(),
    final_policy("Pacing + reserve + abstain (ours)", abstain=True),
]
NS, MIXES, TAUS, PRICINGS = (5, 10, 20, 50), ("uniform", "easy_heavy", "hard_heavy"), (0.0, 0.05, 0.2, 0.5, 1.0), ("flat2x", "steep10x", "measured")

HEAT = ["Best model", "Best model + reserve", "Pacing + reserve (ours)", "Pacing + reserve + abstain (ours)"]


def figure(s):
    """Gap heatmaps: rows = mix x pricing, columns = budget, averaged over N."""
    from matplotlib.colors import LinearSegmentedColormap

    plt = plots.plt
    plots.style()
    cmap = LinearSegmentedColormap.from_list("seq", plots.SEQ)
    rows = [(m, p) for m in MIXES for p in PRICINGS]
    fig, axes = plt.subplots(1, len(HEAT), figsize=(15, 4.6), layout="constrained", sharey=True)
    for ax, pol in zip(axes, HEAT):
        d = s[s.policy == pol].groupby(["mix", "pricing", "tau"]).gap_exp.mean()
        grid = np.array([[100 * d.get((m, p, t), np.nan) for t in TAUS] for m, p in rows])
        im = ax.imshow(np.clip(grid, 0, None), cmap=cmap, vmin=0, vmax=60, aspect="auto")
        for i in range(len(rows)):
            for j in range(len(TAUS)):
                v = grid[i, j]
                ax.text(j, i, f"{0.0 if abs(v) < 0.05 else v:.1f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if v > 30 else plots.TEXT)
        ax.set_xticks(range(len(TAUS)), [f"{t:g}" for t in TAUS])
        ax.set_xlabel("budget tightness tau")
        ax.set_title(pol, fontsize=10)
        ax.grid(False)
    axes[0].set_yticks(range(len(rows)), [f"{m.replace('_', '-')} / {p}" for m, p in rows])
    fig.colorbar(im, ax=axes, shrink=0.8, label="optimality gap vs MCKP (%)")
    plots.save(fig, FIGURES / "e06_factorial_heatmap.png")


if __name__ == "__main__" and sys.argv[1:] == ["fig"]:
    figure(load("e06_summary"))
elif __name__ == "__main__":
    t0 = time.time()
    scen = [Scenario(num_agents=n, tasks_per_agent=5, mix=m, tau=t, pricing=p, seed=s)
            for n, m, t, p in itertools.product(NS, MIXES, TAUS, PRICINGS) for s in range(10)]
    df = run_grid(scen, POLICIES, num_runs=5)
    save(df, "e06_raw")
    g = add_gaps(df)
    keys = ["regime", "mix", "N", "tpa", "J", "pricing", "tau", "label_noise", "split", "instance", "policy"]
    g2 = add_gaps(df, reference=SERVE_ALL)[keys + ["gap_exp", "gap_real"]].rename(
        columns={"gap_exp": "gap_sa_exp", "gap_real": "gap_sa_real"})
    g = g.merge(g2, on=keys)
    s = summarize(g, ["N", "mix", "tau", "pricing", "policy"],
                  ["q_exp", "q_real", "gap_exp", "gap_real", "gap_sa_exp", "gap_sa_real", "headroom_exp",
                   "task_starve", "jain_real", "jain_exp", "bur"])
    save(s, "e06_summary")
    overall = s.groupby("policy").agg(mean_gap=("gap_exp", "mean"), worst_gap=("gap_exp", "max"),
                                      cells_gap_over_5pct=("gap_exp", lambda x: (x > 0.05).mean()),
                                      mean_gap_vs_serve_all=("gap_sa_exp", "mean"),
                                      worst_gap_vs_serve_all=("gap_sa_exp", "max"),
                                      mean_starve=("task_starve", "mean"), mean_jain=("jain_exp", "mean"))
    print("### all 180 cells\n" + md_table(overall.sort_values("mean_gap").reset_index(), ".4f"))
    for key in ("N", "mix", "tau", "pricing"):
        piv = s.pivot_table(index="policy", columns=key, values="gap_exp", aggfunc="mean")
        print(f"\n### mean gap by {key}\n" + md_table(piv.reset_index(), ".4f"))
    figure(s)
    print(f"done in {time.time() - t0:.0f}s")
