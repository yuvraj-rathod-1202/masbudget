"""E5 - Robustness of the pacing mechanism (and of fixed prices) to what agents may get wrong.

  A. Misspecified reference price: p0 = scale * lambda*_hist, scale in [0.01, 100], beta in {0, 2, 5, 10}.
  B. Difficulty-mix shift: prices learned on the (uniform) calibration history, deployed on
     easy-heavy / hard-heavy batches; vs. fixed lambda and an in-batch posted price.
  C. Misestimated horizon: agents assume J' = s * J jobs.

Random arrival, difficulty regime, N = 20 x 10 tasks.

    python experiments/e05_pacing_robustness.py
"""

from __future__ import annotations

import sys
import time

from lib import FIGURES, Scenario, add_gaps, central, decentral, load, md_table, pacing_params, run_grid, save, summarize
import plots

REFS = [central("MCKP (optimal)", "optimal_mckp"), central("Cheapest model", "cheapest_model", group="baseline")]
SCALES = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)
BETAS = (0.0, 2.0, 5.0, 10.0)


def part_a():
    pols = list(REFS)
    for scale in SCALES:
        for beta in BETAS:
            for res in (False, True):
                pols.append(decentral(f"scale={scale}|beta={beta}|res={res}", "budget_pacing", reserve=res,
                                      params_fn=pacing_params(scale=scale, beta=beta), group="A"))
    scen = [Scenario(tau=t, seed=s) for t in (0.05, 0.2, 0.5) for s in range(10)]
    df = run_grid(scen, pols, num_runs=5)
    save(df, "e05a_raw")
    s = summarize(add_gaps(df), ["tau", "policy"], ["gap_exp", "task_starve", "bur"])
    s = s[s.policy.str.startswith("scale=")]
    s[["scale", "beta", "res"]] = s.policy.str.extract(r"scale=([\d.]+)\|beta=([\d.]+)\|res=(\w+)")
    s[["scale", "beta"]] = s[["scale", "beta"]].astype(float)
    save(s, "e05a_summary")
    return s


B_POLICIES = REFS + [
    central("Static equal quota", "static_equal_quota"),
    decentral("Best model + reserve", "best_model", reserve=True),
    decentral("Fixed lambda=30 (tuned, no reserve)", "quasi_linear", lambda_cost=30),
    decentral("Fixed lambda=10 + reserve (tuned)", "quasi_linear", lambda_cost=10, reserve=True),
    decentral("Posted price lambda*_hist (beta=0)", "budget_pacing", params_fn=pacing_params(beta=0.0)),
    decentral("Posted price lambda*_hist + reserve", "budget_pacing", reserve=True, params_fn=pacing_params(beta=0.0)),
    decentral("Posted price lambda*_batch + reserve", "budget_pacing", reserve=True, params_fn=pacing_params(beta=0.0, source="batch")),
    decentral("Pacing (beta=5)", "budget_pacing", params_fn=pacing_params(beta=5.0)),
    decentral("Pacing (beta=5) + reserve", "budget_pacing", reserve=True, params_fn=pacing_params(beta=5.0)),
]


def part_b():
    scen = [Scenario(mix=m, tau=t, seed=s) for m in ("easy_heavy", "uniform", "hard_heavy")
            for t in (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0) for s in range(20)]
    df = run_grid(scen, B_POLICIES, num_runs=10)
    save(df, "e05b_raw")
    s = summarize(add_gaps(df), ["mix", "tau", "policy"], ["gap_exp", "gap_real", "headroom_exp", "task_starve", "bur"])
    save(s, "e05b_summary")
    return s


def part_c():
    pols = list(REFS)
    for hs in (0.5, 0.8, 1.0, 1.25, 2.0):
        for res in (False, True):
            pols.append(decentral(f"horizon x{hs}|res={res}", "budget_pacing", reserve=res,
                                  params_fn=pacing_params(beta=5.0, horizon_scale=hs), group="C"))
    scen = [Scenario(tau=t, seed=s) for t in (0.05, 0.2, 0.5) for s in range(10)]
    df = run_grid(scen, pols, num_runs=5)
    save(df, "e05c_raw")
    s = summarize(add_gaps(df), ["tau", "policy"], ["gap_exp", "task_starve", "bur"])
    save(s, "e05c_summary")
    return s


def figure(sa, sb):
    plt = plots.plt
    plots.style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    ax = axes[0]
    col = plots.colors([f"beta = {b:g}" for b in BETAS])
    for b in BETAS:
        d = sa[(sa.tau == 0.2) & (sa.beta == b) & (sa.res == "False")].sort_values("scale")
        ax.plot(d.scale, 100 * d.gap_exp, "-o", color=col[f"beta = {b:g}"], label=f"beta = {b:g}", markersize=4)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("p0 / lambda*_hist (misspecification, log)")
    ax.set_ylabel("optimality gap vs MCKP (%, symlog)")
    ax.set_title("A. Misspecified price (tau = 0.2, no reserve)")
    ax.legend(fontsize=8)

    labels = ["Fixed lambda=10 + reserve (tuned)", "Posted price lambda*_hist + reserve",
              "Posted price lambda*_batch + reserve", "Best model + reserve", "Pacing (beta=5) + reserve"]
    col = plots.colors(labels)
    for ax, mix in zip(axes[1:], ("easy_heavy", "hard_heavy")):
        for lab in labels:
            d = sb[(sb.mix == mix) & (sb.policy == lab)].sort_values("tau")
            ax.plot(d.tau, 100 * d.gap_exp, "-o", color=col[lab], label=lab, markersize=4)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("budget tightness tau")
        ax.set_ylabel("optimality gap vs MCKP (%, symlog)")
        ax.set_title(f"B. Mix shift: history uniform, batch {mix.replace('_', '-')}")
    handles, labs = axes[2].get_legend_handles_labels()
    fig.legend(handles, labs, loc="outside lower center", ncols=3, fontsize=8)
    plots.save(fig, FIGURES / "e05_pacing_robustness.png")


if __name__ == "__main__" and sys.argv[1:] == ["fig"]:
    sa = load("e05a_summary")
    sa["res"] = sa.res.astype(str)
    figure(sa, load("e05b_summary"))
elif __name__ == "__main__":
    t0 = time.time()
    sa = part_a()
    for res in ("False", "True"):
        for tau in (0.05, 0.2, 0.5):
            piv = sa[(sa.tau == tau) & (sa.res == res)].pivot(index="scale", columns="beta", values="gap_exp")
            print(f"\n### A. gap, tau={tau}, reserve={res} (rows p0 scale, cols beta)\n" + md_table(piv.reset_index(), ".4f"))
    sb = part_b()
    for mix in ("easy_heavy", "uniform", "hard_heavy"):
        piv = sb[sb.mix == mix].pivot(index="policy", columns="tau", values="gap_exp")
        piv["max"] = piv.max(axis=1)
        print(f"\n### B. gap, mix={mix}\n" + md_table(piv.sort_values("max").reset_index(), ".4f"))
    sc = part_c()
    piv = sc[sc.policy.str.startswith("horizon")].pivot(index="policy", columns="tau", values="gap_exp")
    print("\n### C. horizon misestimation\n" + md_table(piv.reset_index(), ".4f"))
    figure(sa, sb)
    print(f"done in {time.time() - t0:.0f}s")
