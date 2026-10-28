"""E2 - Price of anarchy: centralized controllers vs. naive decentralized agents over a budget sweep.

Decentralized utilities use the parameters of the repository configs (lambda = 0.5, p0 = 0.1,
alpha = 1.5, ...) under random sequential arrival, no reserve.

    python experiments/e02_price_of_anarchy.py
"""

from __future__ import annotations

import time

from lib import CENTRAL_POLICIES, FIGURES, Scenario, add_gaps, decentral, md_table, run_grid, save, summarize
import plots

TAUS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
INSTANCES = 20
RUNS = 10

NAIVE = [
    decentral("Best model", "best_model"),
    decentral("Best model (round-robin)", "best_model", arrival=None),
    decentral("Quasi-linear (repo lambda=0.5)", "quasi_linear", lambda_cost=0.5),
    decentral("Shadow price (repo p0=0.1, a=1.5)", "dynamic_shadow_price", p0=0.1, alpha=1.5),
    decentral("Shapley share (repo lambda=0.5, g=1)", "shapley_budget_share", lambda_cost=0.5, gamma=1.0, num_samples=100, seed=42),
    decentral("Marginal ROI ratio", "marginal_roi_ratio"),
]
POLICIES = CENTRAL_POLICIES + NAIVE
METRICS = ["q_exp", "q_real", "gap_exp", "gap_real", "headroom_exp", "task_starve", "agent_starve", "bur",
           "jain_real", "fma_jobs", "q_real_easy", "q_real_medium", "q_real_hard"]


def figure(s):
    plt = plots.plt
    plots.style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), layout="constrained", sharex=True)
    central = ["Predictive router", "Cascade (verified)", "Static equal quota", "Cheapest model"]
    naive = ["Best model", "Quasi-linear (repo lambda=0.5)", "Shadow price (repo p0=0.1, a=1.5)",
             "Shapley share (repo lambda=0.5, g=1)", "Marginal ROI ratio", "Best model (round-robin)"]
    for row, regime in enumerate(["difficulty", "oracle"]):
        for col, (title, labels) in enumerate([("centralized controllers", central), ("naive decentralized agents", naive)]):
            ax = axes[row, col]
            colors = plots.colors(labels)
            for lab in labels:
                d = s[(s.regime == regime) & (s.policy == lab)].sort_values("tau")
                ax.plot(d.tau, 100 * d.gap_exp, "-o", color=colors[lab], label=lab, markersize=4)
            ax.set_title(f"{title} - {regime} regime")
            ax.set_ylabel("optimality gap vs MCKP (%)")
            ax.set_ylim(-5, 100)
            if row == 1:
                ax.set_xlabel("budget tightness tau")
            if row == 0:
                ax.legend(fontsize=8, loc="upper right")
    plots.save(fig, FIGURES / "e02_price_of_anarchy.png")


if __name__ == "__main__":
    t0 = time.time()
    scenarios = [Scenario(regime=r, tau=t, seed=s) for r in ("difficulty", "oracle") for t in TAUS for s in range(INSTANCES)]
    df = run_grid(scenarios, POLICIES, num_runs=RUNS)
    save(df, "e02_raw")
    g = add_gaps(df)
    s = summarize(g, ["regime", "tau", "policy"], METRICS)
    save(s, "e02_summary")
    figure(s)
    print(f"{len(df)} runs in {time.time() - t0:.0f}s")
    for regime in ("difficulty", "oracle"):
        for tau in (0.0, 0.05, 0.2, 0.5, 1.0):
            sub = s[(s.regime == regime) & (s.tau == tau)][
                ["policy", "q_exp", "q_real", "gap_exp", "gap_exp_ci", "task_starve", "bur", "jain_real", "fma_jobs"]]
            print(f"\n### {regime}, tau={tau}\n" + md_table(sub.sort_values("gap_exp"), ".3f"))
