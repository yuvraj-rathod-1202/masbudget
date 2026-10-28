"""E9 - Without the sequencing layer: how well can agents do if they all decide at once?

The decentralized game is sequential: jobs arrive one at a time and each agent sees the budget
B_k its predecessors left. Here that layer is removed (``simultaneous_best_response``): every job
picks its model at the same time, knowing only B, J and the model costs. The ledger then settles
the requests in a random order the agents never see, either dropping requests that no longer fit
("drop") or downgrading them so that every job keeps c_min ("downgrade", the reserve).

Policies without sequencing: greedy (drop / downgrade), per-job allowance B/J, per-agent quota
(each agent spends B * n_i / J on its own tasks), posted price lambda*_hist and a lottery price
(lambda*_hist with a calibrated coin flip for the marginal class). References: MCKP, the cheapest
model, and the sequential game with greedy agents and with the final mechanism (pacing + reserve).

    python experiments/e09_no_sequencing.py
"""

from __future__ import annotations

import dataclasses
import functools
import time

import pandas as pd

from lib import (
    DECENTRALIZED, FIGURES, PRICINGS, RESULTS, Instance, Policy, Scenario, add_gaps, central, decentral, final_policy,
    historical_price, md_table, run_grid, save, split, summarize, view,
)
from masbudget.decentralized.utilities.lottery_price import calibrate_lottery
from masbudget.registry import ModelRegistry
import plots

TAUS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0]
MAIN_TAU = 0.2
SIMULTANEOUS = "simultaneous_best_response"
WITH, WITHOUT = "with sequencing", "without sequencing"


@functools.lru_cache(maxsize=256)
def _lottery(regime: str, pricing: str, budget_per_job: float) -> tuple:
    tasks = [view(t, regime) for t in split()[0]]
    models = ModelRegistry()
    for name, cost in PRICINGS[pricing].items():
        models.add(name, cost)
    return tuple(calibrate_lottery(tasks, models, budget_per_job * len(tasks)).items())


def lottery_params(inst: Instance) -> dict:
    sc = inst.scenario
    return dict(_lottery(sc.regime, sc.pricing, round(inst.budget / sc.num_jobs, 12)))


def posted_price(inst: Instance) -> dict:
    return {"lambda_cost": historical_price(inst)}


def simultaneous(label: str, utility: str, params_fn=None, **engine) -> Policy:
    return Policy(label, DECENTRALIZED, utility, params_fn=params_fn, group=WITHOUT, engine=SIMULTANEOUS,
                  engine_params=tuple(engine.items()))


POLICIES = [
    central("MCKP (optimal)", "optimal_mckp", group="reference"),
    central("Cheapest model", "cheapest_model", group="baseline"),
    decentral("Sequential: greedy", "best_model", group=WITH),
    dataclasses.replace(final_policy("Sequential: pacing + reserve"), group=WITH),
    simultaneous("Greedy, drop overflow", "best_model", settlement="ration"),
    simultaneous("Greedy, downgrade overflow", "best_model", settlement="downgrade"),
    simultaneous("Per-job allowance B/J", "best_model", allowance="job"),
    central("Per-agent quota", "static_equal_quota", group=WITHOUT),
    simultaneous("Posted price", "quasi_linear", params_fn=posted_price, settlement="downgrade"),
    simultaneous("Lottery price", "lottery_price", params_fn=lottery_params, settlement="downgrade"),
]
GROUP = {p.label: p.group for p in POLICIES}
ORDER = [p.label for p in POLICIES[2:]]
METRICS = ["gap_exp", "gap_real", "q_exp", "q_real", "bur", "task_starve", "jain_exp"]

# Robustness at the moderate budget: other task mixes and one task per agent.
SETTINGS = {
    "uniform, 20x10": dict(),
    "easy-heavy, 20x10": dict(mix="easy_heavy"),
    "hard-heavy, 20x10": dict(mix="hard_heavy"),
    "uniform, 50x1": dict(num_agents=50, tasks_per_agent=1),
}


def setting_of(sc: Scenario) -> dict:
    defaults = Scenario()
    for name, kw in SETTINGS.items():
        if all(getattr(sc, f) == kw.get(f, getattr(defaults, f)) for f in ("mix", "num_agents", "tasks_per_agent")):
            return {"setting": name}
    raise ValueError(sc)


def run() -> pd.DataFrame:
    scenarios = [Scenario(tau=tau, seed=s) for tau in TAUS for s in range(20)]
    scenarios += [Scenario(tau=MAIN_TAU, seed=s, **kw) for kw in list(SETTINGS.values())[1:] for s in range(20)]
    return run_grid(scenarios, POLICIES, num_runs=10, extra=setting_of)


# ---------------------------------------------------------------- figures


def _achieved(s: pd.DataFrame) -> pd.Series:
    return 100 * (1 - s.gap_exp)


def fig_budget(s: pd.DataFrame) -> None:
    """One line per policy: % of the optimal welfare achieved, by budget."""
    plt = plots.plt
    plots.style()
    shown = ["Sequential: pacing + reserve", "Lottery price", "Per-agent quota", "Posted price", "Greedy, drop overflow"]
    col = plots.colors(shown)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    d = s[s.setting == "uniform, 20x10"]
    for label in shown:
        row = d[d.policy == label].sort_values("tau")
        style = "--" if GROUP[label] == WITH else "-"
        ax.plot(row.tau, _achieved(row), style, marker="o", color=col[label], label=label)
    ax.set_ylim(0, 105)
    ax.set_xlabel("budget tightness tau (0 = severe, 1 = generous)")
    ax.set_ylabel("% of optimal welfare achieved")
    ax.set_title("Without sequencing, a lottery price is as good as the sequential mechanism")
    ax.legend(loc="lower right", fontsize=8.5, title="dashed = with sequencing", title_fontsize=8)
    plots.save(fig, FIGURES / "e09_gap_vs_budget.png")


def fig_moderate(s: pd.DataFrame) -> None:
    """All policies at the moderate budget, one bar each, coloured by with/without sequencing."""
    plt = plots.plt
    plots.style()
    col = {WITH: plots.SERIES[0], WITHOUT: plots.SERIES[1]}
    d = s[(s.setting == "uniform, 20x10") & (s.tau == MAIN_TAU) & s.policy.isin(ORDER)].copy()
    d["achieved"] = _achieved(d)
    d = d.sort_values("achieved")
    fig, ax = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
    ax.barh(d.policy, d.achieved, color=[col[GROUP[p]] for p in d.policy], height=0.62)
    for y, v in enumerate(d.achieved):
        ax.text(v + 1, y, f"{v:.1f}%", va="center", fontsize=8.5, color=plots.TEXT)
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of optimal welfare achieved (tau = 0.2)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in col.values()]
    ax.legend(handles, list(col), loc="lower right", fontsize=8.5)
    ax.set_title("Moderate budget: every policy against the optimum")
    plots.save(fig, FIGURES / "e09_policies_moderate.png")


def fig_robustness(s: pd.DataFrame) -> None:
    """Best policies with and without sequencing, across task mixes and agent counts."""
    plt = plots.plt
    plots.style()
    shown = ["Sequential: pacing + reserve", "Lottery price", "Per-agent quota"]
    col = plots.colors(shown)
    d = s[(s.tau == MAIN_TAU) & s.policy.isin(shown)]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    width = 0.26
    for i, label in enumerate(shown):
        row = d[d.policy == label].set_index("setting").reindex(list(SETTINGS))
        x = [k + (i - 1) * width for k in range(len(SETTINGS))]
        ax.bar(x, 100 * row.gap_exp, width=width - 0.03, color=col[label], label=label)
        for xi, v in zip(x, 100 * row.gap_exp):
            ax.text(xi, max(v, 0) + 0.1, f"{max(v, 0):.1f}", ha="center", fontsize=7.5, color=plots.TEXT)
    ax.set_xticks(range(len(SETTINGS)), list(SETTINGS))
    ax.set_ylabel("gap to optimum (%, lower is better)")
    ax.set_title("Robustness at tau = 0.2: task mix and agents x tasks")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.5)
    plots.save(fig, FIGURES / "e09_robustness.png")


if __name__ == "__main__":
    t0 = time.time()
    raw = run()
    save(raw, "e09_raw")
    m = add_gaps(raw, extra_keys=["setting"])
    s = summarize(m, ["setting", "tau", "policy", "group"], METRICS)
    save(s, "e09_summary")

    lines = [f"E9 - no sequencing layer ({time.time() - t0:.0f}s)", ""]
    main = s[(s.setting == "uniform, 20x10") & s.policy.isin(ORDER)]
    gap = main.pivot(index="policy", columns="tau", values="gap_exp").reindex(ORDER) * 100
    lines += ["Gap to MCKP (%), uniform mix, N = 20 x 10, by tau:", md_table(gap.reset_index(), ".2f"), ""]
    mod = main[main.tau == MAIN_TAU].set_index("policy").reindex(ORDER).reset_index()
    mod = mod.assign(gap_exp=100 * mod.gap_exp, gap_real=100 * mod.gap_real, task_starve=100 * mod.task_starve)
    lines += [f"tau = {MAIN_TAU}:", md_table(mod[["policy", "group", "gap_exp", "gap_real", "q_real", "bur",
                                                  "task_starve", "jain_exp"]], ".3f"), ""]
    rob = s[(s.tau == MAIN_TAU) & s.policy.isin(ORDER)]
    rob = rob.pivot(index="policy", columns="setting", values="gap_exp").reindex(ORDER)[list(SETTINGS)] * 100
    lines += [f"Gap to MCKP (%) at tau = {MAIN_TAU} by setting:", md_table(rob.reset_index(), ".2f")]
    text = "\n".join(lines)
    print(text)
    (RESULTS / "e09_stdout.txt").write_text(text, encoding="utf-8")

    fig_budget(s)
    fig_moderate(s)
    fig_robustness(s)
