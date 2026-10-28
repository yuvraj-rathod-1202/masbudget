"""E4 - Utility design (U1-U4 + budget pacing), tuned on the calibration split, tested on the evaluation split.

Phase A (calibration split, tau = 0.2, random arrival): grid search of every utility's parameters,
with and without the budget reserve; pick the best setting per (utility, reserve) by mean q_exp.
Phase B (evaluation split, all budgets): the tuned settings, the repo defaults and the references.
Phase C (evaluation split): fixed-lambda sweep at every budget (is there one lambda for all budgets?).

    python experiments/e04_utility_design.py
"""

from __future__ import annotations

import itertools
import time

import pandas as pd

from lib import (
    CENTRAL_POLICIES, FIGURES, Scenario, add_gaps, central, decentral, instance_means, md_table, pacing_params,
    run_grid, save, summarize,
)
import plots

TAUS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
TUNE_TAU = 0.2

GRIDS = {
    "quasi_linear": [{"lambda_cost": v} for v in (0.5, 2, 5, 8, 10, 11, 12, 15, 20, 30, 50, 100, 300)],
    "dynamic_shadow_price": [{"p0": p, "alpha": a} for p, a in itertools.product((0.1, 1, 3, 5, 8, 11.5, 15), (0.5, 1, 1.5, 2, 3))],
    "shapley_budget_share": (
        [{"lambda_cost": l, "gamma": g, "num_samples": 100, "seed": 42} for l, g in itertools.product((5, 10, 20, 50, 100, 200), (0.1, 1.0))]
        + [{"lambda_cost": l, "gamma": g, "num_samples": 100, "seed": 42, "characteristic": "upgrade_gain"}
           for l, g in itertools.product((10, 15, 20, 30, 50), (1.0, 5.0, 20.0))]
    ),
    "marginal_roi_ratio": [{}],
    "budget_pacing": [{"beta": b} for b in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)],
}
SHORT_UTIL = {"quasi_linear": "U1 quasi-linear", "dynamic_shadow_price": "U2 shadow price",
              "shapley_budget_share": "U3 Shapley share", "marginal_roi_ratio": "U4 marginal ROI",
              "budget_pacing": "Pacing (ours)"}


def make_policy(utility: str, params: dict, reserve: bool, tag: str = "") -> object:
    label = f"{utility}|{'res' if reserve else 'nores'}|{tag or params_key(params)}"
    if utility == "budget_pacing":
        return decentral(label, utility, reserve=reserve, params_fn=pacing_params(beta=params["beta"]), group=utility)
    return decentral(label, utility, reserve=reserve, group=utility, **params)


def params_key(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in params.items() if k not in ("num_samples", "seed"))


def phase_a() -> pd.DataFrame:
    policies = [central("MCKP (optimal)", "optimal_mckp"), central("Cheapest model", "cheapest_model", group="baseline")]
    for utility, grid in GRIDS.items():
        for params, reserve in itertools.product(grid, (False, True)):
            policies.append(make_policy(utility, params, reserve))
    scenarios = [Scenario(split="calib", tau=TUNE_TAU, seed=s) for s in range(10)]
    df = run_grid(scenarios, policies, num_runs=5)
    save(df, "e04a_tuning_raw")
    g = add_gaps(df)
    s = summarize(g, ["policy", "group"], ["q_exp", "gap_exp", "headroom_exp", "task_starve"])
    s = s[s.group.isin(GRIDS)]
    s[["utility", "reserve", "params"]] = s.policy.str.split("|", expand=True)
    save(s, "e04a_tuning_summary")
    return s


def tuned_choices(s: pd.DataFrame) -> dict[tuple[str, str], str]:
    best = s.sort_values("q_exp", ascending=False).groupby(["utility", "reserve"]).head(1)
    return {(r.utility, r.reserve): r.params for r in best.itertuples()}


def parse(params: str) -> dict:
    out = {}
    for kv in filter(None, params.split(",")):
        k, v = kv.split("=")
        out[k] = v if k == "characteristic" else float(v)
    return out


def phase_b(choices) -> pd.DataFrame:
    policies = list(CENTRAL_POLICIES)
    policies += [
        decentral("Best model", "best_model", group="baseline"),
        decentral("Best model + reserve", "best_model", reserve=True, group="baseline"),
        decentral("U1 quasi-linear (repo)", "quasi_linear", lambda_cost=0.5, group="repo"),
        decentral("U2 shadow price (repo)", "dynamic_shadow_price", p0=0.1, alpha=1.5, group="repo"),
        decentral("U3 Shapley share (repo)", "shapley_budget_share", lambda_cost=0.5, gamma=1.0, num_samples=100, seed=42, group="repo"),
    ]
    for (utility, reserve), params in sorted(choices.items()):
        p = parse(params)
        if utility == "shapley_budget_share":
            p.update(num_samples=100, seed=42)
        label = f"{SHORT_UTIL[utility]} tuned{' + reserve' if reserve == 'res' else ''}"
        pol = make_policy(utility, p, reserve == "res")
        policies.append(type(pol)(**{**pol.__dict__, "label": label, "group": "tuned"}))
    scenarios = [Scenario(tau=t, seed=s) for t in TAUS for s in range(20)]
    df = run_grid(scenarios, policies, num_runs=10)
    save(df, "e04b_eval_raw")
    return df


def phase_c() -> pd.DataFrame:
    lambdas = (2, 5, 8, 11, 12, 15, 20, 30, 50, 100, 300, 1000)
    policies = [central("MCKP (optimal)", "optimal_mckp"), central("Cheapest model", "cheapest_model", group="baseline")]
    policies += [decentral(f"lambda={v}|{r}", "quasi_linear", reserve=r, lambda_cost=v, group="sweep")
                 for v in lambdas for r in (False, True)]
    policies += [decentral(f"pacing|{r}", "budget_pacing", reserve=r, params_fn=pacing_params(beta=5.0), group="sweep")
                 for r in (False, True)]
    scenarios = [Scenario(tau=t, seed=s) for t in TAUS for s in range(10)]
    df = run_grid(scenarios, policies, num_runs=5)
    save(df, "e04c_lambda_sweep_raw")
    return df


def figure(sb: pd.DataFrame, sc: pd.DataFrame) -> None:
    plt = plots.plt
    plots.style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained")
    ax = axes[0]
    labels = ["Best model", "U1 quasi-linear tuned + reserve", "U2 shadow price tuned + reserve",
              "U3 Shapley share tuned + reserve", "U4 marginal ROI tuned + reserve", "Pacing (ours) tuned + reserve"]
    col = plots.colors(labels)
    for lab in labels:
        d = sb[sb.policy == lab].sort_values("tau")
        ax.plot(d.tau, 100 * d.gap_exp, "-o", color=col[lab], label=lab, markersize=4)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("budget tightness tau")
    ax.set_ylabel("optimality gap vs MCKP (%, symlog)")
    ax.set_title("Tuned utilities (tuned at tau = 0.2, calibration split)")
    ax.legend(fontsize=7.8, loc="upper right")

    ax = axes[1]
    taus = [0.02, 0.1, 0.2, 0.35, 0.5, 0.75]
    seq = plots.colors([f"tau={t}" for t in taus])
    for t in taus:
        d = sc[(sc.tau == t) & sc.policy.str.startswith("lambda=") & sc.policy.str.endswith("|True")].copy()
        d["lam"] = d.policy.str.extract(r"lambda=([\d.]+)").astype(float)
        d = d.sort_values("lam")
        ax.plot(d.lam, 100 * d.gap_exp, "-o", color=seq[f"tau={t}"], label=f"tau = {t}", markersize=4)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("fixed cost weight lambda (quality per USD, log)")
    ax.set_ylabel("optimality gap vs MCKP (%, symlog)")
    ax.set_title("U1 + reserve: gap vs fixed lambda, per budget")
    ax.legend(fontsize=8, ncols=2)
    plots.save(fig, FIGURES / "e04_utility_design.png")


if __name__ == "__main__":
    t0 = time.time()
    sa = phase_a()
    choices = tuned_choices(sa)
    print("## Phase A: best settings at tau=0.2 (calibration split)")
    best = sa.sort_values("q_exp", ascending=False).groupby(["utility", "reserve"]).head(1)
    print(md_table(best[["utility", "reserve", "params", "q_exp", "gap_exp", "headroom_exp", "task_starve"]], ".4f"))
    print("\n## Phase A: full grid of quasi-linear and pacing")
    print(md_table(sa[sa.utility.isin(["quasi_linear", "budget_pacing"])][["utility", "reserve", "params", "q_exp", "gap_exp", "task_starve"]], ".4f"))

    db = phase_b(choices)
    sb = summarize(add_gaps(db), ["tau", "policy", "group"],
                   ["q_exp", "q_real", "gap_exp", "gap_real", "headroom_exp", "task_starve", "bur", "jain_real"])
    save(sb, "e04b_eval_summary")
    print("\n## Phase B: evaluation split")
    for tau in TAUS:
        print(f"\n### tau={tau}\n" + md_table(sb[sb.tau == tau].sort_values("gap_exp")[
            ["policy", "q_exp", "q_real", "gap_exp", "gap_exp_ci", "headroom_exp", "task_starve", "bur"]], ".4f"))

    dc = phase_c()
    sc = summarize(add_gaps(dc), ["tau", "policy"], ["q_exp", "gap_exp", "task_starve", "bur"])
    save(sc, "e04c_lambda_sweep_summary")
    piv = sc[sc.policy.str.endswith("|True")].pivot(index="policy", columns="tau", values="gap_exp")
    print("\n## Phase C: gap of fixed lambda (with reserve) per budget\n" + md_table(piv.reset_index(), ".4f"))
    piv = sc[sc.policy.str.endswith("|False")].pivot(index="policy", columns="tau", values="gap_exp")
    print("\n## Phase C: gap of fixed lambda (no reserve) per budget\n" + md_table(piv.reset_index(), ".4f"))
    figure(sb, sc)
    print(f"done in {time.time() - t0:.0f}s")
