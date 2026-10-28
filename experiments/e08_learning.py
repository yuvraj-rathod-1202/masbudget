"""E8 - Learning agents (online UCB extension, D6): qualities unknown, learned from call outcomes.

Learners share per-(difficulty, model) pass-rate estimates updated from the revealed outcome of
every call, and price budget with the pacing feedback starting from the floor price only (no
historical lambda*). Regret is measured in expected successes (q-hat of the chosen models)
against the same pacing + reserve agents that know q-hat.

    python experiments/e08_learning.py
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from lib import (
    FIGURES, Scenario, add_gaps, central, decentral, final_policy, instance_means, load, md_table, pacing_params,
    quality_tables, run_grid, save, summarize,
)
import plots

KNOWN = "Pacing + reserve (ours, knows q-hat)"
FLOOR_ONLY = pacing_params(scale=0.0, beta=10.0, floor=0.1)


def policies():
    blind = dict(quality_tables()["blind"])
    out = [
        central("MCKP (optimal)", "optimal_mckp"),
        central("Cheapest model", "cheapest_model", group="baseline"),
        final_policy(KNOWN),
        final_policy("Pacing + reserve (knows q-hat, no price history)", scale=0.0),
    ]
    for c in (0.0, 0.25, 0.5, 1.0):
        out.append(decentral(f"UCB + pacing + reserve (c={c:g})", "ucb_pacing", reserve=True, params_fn=FLOOR_ONLY,
                             group="learning", c=c))
    out.append(decentral("UCB + pacing + reserve (c=0.25, blind prior)", "ucb_pacing", reserve=True,
                         params_fn=FLOOR_ONLY, group="learning", c=0.25, prior=blind, prior_n=5))
    return out


def figure(df: pd.DataFrame) -> None:
    plt = plots.plt
    plots.style()
    labels = [KNOWN, "UCB + pacing + reserve (c=0)", "UCB + pacing + reserve (c=0.25)", "UCB + pacing + reserve (c=1)",
              "UCB + pacing + reserve (c=0.25, blind prior)"]
    col = plots.colors(labels)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
    for ax, J in zip(axes, (200, 500)):
        for lab in labels:
            d = df[(df.J == J) & (df.tau == 0.2) & (df.policy == lab)]
            ax.plot(range(1, 11), [d[f"pos{k}"].mean() for k in range(10)], "-o", color=col[lab], label=lab, markersize=4)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("arrival decile")
        ax.set_title(f"Learning curve, J = {J} jobs (tau = 0.2)")
    axes[0].set_ylabel("mean expected quality q-hat of chosen model")
    axes[1].legend(fontsize=7.5, loc="lower right")
    plots.save(fig, FIGURES / "e08_learning.png")


if __name__ == "__main__" and sys.argv[1:] == ["fig"]:
    figure(load("e08_raw"))
elif __name__ == "__main__":
    t0 = time.time()
    scen = [Scenario(num_agents=n, tau=t, seed=s) for n in (20, 50) for t in (0.05, 0.2, 0.5) for s in range(20)]
    df = run_grid(scen, policies(), num_runs=5)
    save(df, "e08_raw")
    g = add_gaps(df)
    known = instance_means(df[df.policy == KNOWN])[["J", "tau", "instance", "q_exp"]].rename(columns={"q_exp": "known_q"})
    g = g.merge(known, on=["J", "tau", "instance"])
    g["regret_successes"] = (g.known_q - g.q_exp) * g.J
    s = summarize(g, ["J", "tau", "policy"], ["q_exp", "q_real", "gap_exp", "gap_real", "regret_successes", "task_starve", "bur"])
    save(s, "e08_summary")
    figure(df)
    for J in (200, 500):
        for tau in (0.05, 0.2, 0.5):
            sub = s[(s.J == J) & (s.tau == tau)][["policy", "q_exp", "q_real", "gap_exp", "gap_exp_ci",
                                                   "regret_successes", "task_starve", "bur"]]
            print(f"\n### J={J}, tau={tau}\n" + md_table(sub, ".4f"))
    print(f"done in {time.time() - t0:.0f}s")
