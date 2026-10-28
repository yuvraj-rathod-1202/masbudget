"""E3 - Arrival ordering and position privilege (RQ2, RQ3).

Five arrival policies x four agent behaviours, difficulty regime, uniform mix.
Main grid: N = 20 agents x 10 tasks. Proposal-faithful grid: N = 50 agents x 1 task.

    python experiments/e03_arrival_ordering.py
"""

from __future__ import annotations

import time

import pandas as pd

from lib import (
    FIGURES, Scenario, add_gaps, central, decentral, md_table, pacing_params, run_grid, save, summarize,
)
import plots

ARRIVALS = {"Round-robin FCFS": None, "Random": "random_fcfs", "Hardest first": "hardest_task_first",
            "Easiest first": "easiest_task_first", "Marginal-ROI priority": "marginal_roi_priority"}
BEHAVIOURS = {
    "best model": dict(utility="best_model"),
    "best model + reserve": dict(utility="best_model", reserve=True),
    "quasi-linear (lambda=30)": dict(utility="quasi_linear", lambda_cost=30),
    "pacing + reserve": dict(utility="budget_pacing", reserve=True, params_fn=pacing_params(beta=5.0)),
}
TAUS = [0.0, 0.05, 0.2, 0.5]


def policies():
    out = [central("MCKP (optimal)", "optimal_mckp"), central("Cheapest model", "cheapest_model", group="baseline")]
    for a_label, arrival in ARRIVALS.items():
        for b_label, kw in BEHAVIOURS.items():
            kw = dict(kw)
            utility = kw.pop("utility")
            out.append(decentral(f"{a_label} | {b_label}", utility, arrival=arrival, group=a_label, **kw))
    return out


def order_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """SD of social welfare (sum of q_exp) across arrival permutations, per instance, then averaged."""
    d = df[df.paradigm == "decentralized"].assign(sw=lambda x: x.q_exp * x.J)
    sd = d.groupby(["tpa", "tau", "policy", "instance"]).sw.std(ddof=0).reset_index()
    return sd.groupby(["tpa", "tau", "policy"]).sw.mean().rename("sw_sd").reset_index()


def figure(df: pd.DataFrame, s: pd.DataFrame) -> None:
    plt = plots.plt
    plots.style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), layout="constrained")
    sub = s[(s.tpa == 10) & (s.tau == 0.2) & (s.paradigm == "decentralized")].copy()
    sub[["arrival", "behaviour"]] = sub.policy.str.split(" \\| ", expand=True)
    col = plots.colors(list(BEHAVIOURS))
    width = 0.2
    for i, b in enumerate(BEHAVIOURS):
        d = sub[sub.behaviour == b].set_index("arrival").reindex(list(ARRIVALS))
        x = [k + (i - 1.5) * width for k in range(len(ARRIVALS))]
        axes[0].bar(x, 100 * d.gap_exp, width=width - 0.03, color=col[b], label=b)
    axes[0].set_xticks(range(len(ARRIVALS)), [a.replace(" ", "\n", 1) for a in ARRIVALS], fontsize=8)
    axes[0].set_ylabel("optimality gap vs MCKP (%)")
    axes[0].set_title("Gap by arrival order (tau = 0.2)")
    axes[0].legend(fontsize=7.5, title="agent behaviour", title_fontsize=8)

    # Position privilege: q-hat by arrival decile, random arrival.
    raw = df[(df.tpa == 10) & (df.paradigm == "decentralized")]
    for ax, tau in zip(axes[1:], (0.05, 0.2)):
        for b in BEHAVIOURS:
            d = raw[(raw.tau == tau) & (raw.policy == f"Random | {b}")]
            means = [d[f"pos{k}"].mean() for k in range(10)]
            ax.plot(range(1, 11), means, "-o", color=col[b], label=b, markersize=4)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("arrival decile (1 = first 10% of jobs)")
        ax.set_ylabel("mean expected quality q-hat")
        ax.set_title(f"Position privilege, random arrival (tau = {tau})")
    plots.save(fig, FIGURES / "e03_arrival_ordering.png")


if __name__ == "__main__":
    t0 = time.time()
    scen = [Scenario(tau=t, seed=s) for t in TAUS for s in range(20)]
    scen += [Scenario(tau=t, num_agents=50, tasks_per_agent=1, seed=s) for t in (0.05, 0.2) for s in range(20)]
    df = run_grid(scen, policies(), num_runs=10)
    save(df, "e03_raw")
    g = add_gaps(df)
    s = summarize(g, ["tpa", "tau", "policy", "paradigm", "group"],
                  ["q_exp", "q_real", "gap_exp", "gap_real", "task_starve", "agent_starve", "jain_exp", "jain_real",
                   "fma_jobs", "fma_agents", "q_real_easy", "q_real_medium", "q_real_hard", "bur"])
    s = s.merge(order_sensitivity(df), on=["tpa", "tau", "policy"], how="left")
    save(s, "e03_summary")
    figure(df, s)
    for tpa, tau in [(10, 0.0), (10, 0.05), (10, 0.2), (10, 0.5), (1, 0.05), (1, 0.2)]:
        sub = s[(s.tpa == tpa) & (s.tau == tau)]
        print(f"\n### tasks/agent={tpa}, tau={tau}\n" + md_table(sub[[
            "policy", "gap_exp", "gap_exp_ci", "task_starve", "agent_starve", "jain_exp", "fma_jobs", "fma_agents",
            "sw_sd", "q_real_easy", "q_real_hard"]], ".3f"))
    print(f"done in {time.time() - t0:.0f}s")
