"""E7 - Information regimes and noisy difficulty labels.

What policies know about q_{i,m}: oracle (per-task outcome), difficulty (calibrated class pass
rates), blind (overall pass rates), plus difficulty labels flipped with probability eps.
Scored on the realized pass@1; the oracle MCKP of the same instance is the absolute upper bound.

    python experiments/e07_information.py
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from lib import (
    FIGURES, SCENARIO_KEYS, Scenario, add_gaps, central, decentral, final_policy, instance_means, load, md_table,
    run_grid, save, summarize,
)
import plots

POLICIES = [
    central("MCKP (optimal)", "optimal_mckp"),
    central("Cascade (verified)", "confidence_cascade", threshold=0.5, verify_key="realized", skip_dominated=True,
            escalation="jump"),
    central("Cheapest model", "cheapest_model", group="baseline"),
    decentral("Best model", "best_model"),
    decentral("Best model + reserve", "best_model", reserve=True),
    final_policy(),
    final_policy("Pacing + reserve + abstain (ours)", abstain=True),
]
TAUS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0)
CONDITIONS = [("oracle", 0.0), ("difficulty", 0.0), ("difficulty", 0.1), ("difficulty", 0.25), ("difficulty", 0.5), ("blind", 0.0)]


def condition(row) -> str:
    return row.regime if row.label_noise == 0 else f"{row.regime} (noise {row.label_noise:g})"


def analyse(df: pd.DataFrame) -> pd.DataFrame:
    g = add_gaps(df)  # gaps within each regime
    g["condition"] = g.apply(condition, axis=1)
    oracle = instance_means(df[(df.regime == "oracle") & (df.policy == "MCKP (optimal)")])
    oracle = oracle[["tau", "instance", "q_real"]].rename(columns={"q_real": "oracle_q_real"})
    g = g.merge(oracle, on=["tau", "instance"], how="left")
    g["gap_to_oracle"] = (g.oracle_q_real - g.q_real) / g.oracle_q_real
    return g


def figure(s: pd.DataFrame) -> None:
    plt = plots.plt
    plots.style()
    conds = ["oracle", "difficulty", "difficulty (noise 0.25)", "blind"]
    pols = ["MCKP (optimal)", "Cascade (verified)", "Best model", "Pacing + reserve (ours)", "Pacing + reserve + abstain (ours)"]
    col = plots.colors(pols)
    fig, axes = plt.subplots(1, len(conds), figsize=(15, 4), layout="constrained", sharey=True)
    for ax, c in zip(axes, conds):
        for p in pols:
            d = s[(s.condition == c) & (s.policy == p)].sort_values("tau")
            ax.plot(d.tau, d.q_real, "-o", color=col[p], label=p, markersize=4)
        o = s[(s.condition == "oracle") & (s.policy == "MCKP (optimal)")].sort_values("tau")
        ax.plot(o.tau, o.q_real, ":", color=plots.MUTED, linewidth=1.5, label="oracle MCKP (bound)")
        ax.set_title(f"policies know: {c}")
        ax.set_xlabel("budget tightness tau")
    axes[0].set_ylabel("realized pass@1")
    axes[-1].legend(fontsize=7.5, loc="lower right")
    plots.save(fig, FIGURES / "e07_information.png")


if __name__ == "__main__" and sys.argv[1:] == ["fig"]:
    figure(load("e07_summary"))
elif __name__ == "__main__":
    t0 = time.time()
    scen = [Scenario(regime=r, label_noise=e, tau=t, seed=s) for r, e in CONDITIONS for t in TAUS for s in range(20)]
    df = run_grid(scen, POLICIES, num_runs=5)
    save(df, "e07_raw")
    g = analyse(df)
    s = summarize(g, ["condition", "tau", "policy"],
                  ["q_real", "gap_exp", "gap_real", "gap_to_oracle", "task_starve", "bur", "jain_real"])
    save(s, "e07_summary")
    figure(s)
    for value in ("q_real", "gap_to_oracle", "gap_exp"):
        for tau in (0.05, 0.2, 0.5):
            piv = s[s.tau == tau].pivot(index="policy", columns="condition", values=value)
            print(f"\n### {value}, tau={tau}\n" + md_table(piv.reset_index(), ".4f"))
    piv = s.pivot_table(index="policy", columns="condition", values="gap_exp", aggfunc="max")
    print("\n### worst gap_exp (within regime) over budgets\n" + md_table(piv.reset_index(), ".4f"))
    print(f"done in {time.time() - t0:.0f}s")
