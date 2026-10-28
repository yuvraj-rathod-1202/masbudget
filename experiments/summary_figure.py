"""Headline figure for results.md, from the factorial summaries (E6, E6-B).

    python experiments/summary_figure.py
"""

from __future__ import annotations

import pandas as pd

from lib import FIGURES, load
import plots

LINES = ["Best model", "U1 quasi-linear (lambda=30)", "Best model + reserve", "Predictive router",
         "Pacing + reserve (ours)", "Pacing + reserve + abstain (ours)"]
BARS = ["Best model", "U1 quasi-linear (lambda=30)", "Cheapest model", "Cascade (verified)", "Best model + reserve",
        "U2 shadow price (p0=3, a=0.5) + reserve", "U1 quasi-linear (lambda=10) + reserve", "Pacing + reserve (ours)",
        "Predictive router", "MCKP (serve all)", "Static equal quota", "Pacing + reserve + abstain (ours)"]
CENTRAL = {"Cheapest model", "Cascade (verified)", "Predictive router", "MCKP (serve all)", "Static equal quota"}

if __name__ == "__main__":
    s = load("e06_summary")
    plt = plots.plt
    plots.style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.15]})

    ax = axes[0]
    col = plots.colors(LINES)
    by_tau = s.groupby(["policy", "tau"]).gap_exp.mean().reset_index()
    for lab in LINES:
        d = by_tau[by_tau.policy == lab].sort_values("tau")
        ax.plot(d.tau, 100 * d.gap_exp, marker="o", color=col[lab], label=lab, markersize=4,
                linestyle="--" if lab in CENTRAL else "-")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("budget tightness tau  (0: B = J*c_min,  1: B = J*c_max)")
    ax.set_ylabel("optimality gap vs MCKP (%, symlog)")
    ax.set_title("Gap vs budget (mean over N, mix, pricing)")
    ax.legend(fontsize=7.8, loc="upper right")

    ax = axes[1]
    agg = s.groupby("policy").gap_exp.agg(["mean", "max"]).reindex(BARS)
    y = range(len(agg))
    ax.barh(list(y), 100 * agg["mean"], height=0.62,
            color=[plots.SERIES[0] if p in CENTRAL else plots.SERIES[1] for p in agg.index])
    ax.plot(100 * agg["max"], list(y), "|", color=plots.TEXT, markersize=12, markeredgewidth=2, label="worst cell")
    for i, (m, w) in enumerate(zip(agg["mean"], agg["max"])):
        ax.text(100 * w + 1.2, i, f"{100 * m:.1f}% (worst {100 * w:.0f}%)", va="center", fontsize=8, color=plots.MUTED)
    ax.set_yticks(list(y), agg.index, fontsize=8.5)
    ax.set_xlim(0, 125)
    ax.set_xlabel("optimality gap vs MCKP (%): bar = mean of 180 cells, tick = worst cell")
    ax.set_title("All 180 factorial cells")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=plots.SERIES[0], label="centralized"), Patch(color=plots.SERIES[1], label="decentralized"),
                       plt.Line2D([], [], marker="|", linestyle="", color=plots.TEXT, markersize=10, label="worst cell")],
              loc="upper right", fontsize=8)
    ax.grid(axis="y", visible=False)
    plots.save(fig, FIGURES / "headline.png")
    print("ok")
