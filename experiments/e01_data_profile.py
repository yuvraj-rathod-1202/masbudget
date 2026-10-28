"""E1 - Benchmark profile: model pool, cost-quality frontier, outcome structure, price of budget.

    python experiments/e01_data_profile.py
"""

from __future__ import annotations

import collections

import pandas as pd

from lib import (
    FIGURES, MEASURED_COSTS, PRICINGS, SHORT, Scenario, base_tasks, build_instance, calibrate_price,
    md_table, quality_tables, save, split, view,
)
from masbudget.registry import ModelRegistry
import plots


def model_table() -> pd.DataFrame:
    tasks = base_tasks()
    c_min = min(MEASURED_COSTS.values())
    rows = []
    for m, c in MEASURED_COSTS.items():
        row = {"model": m, "cost_usd": c, "cost_x_cheapest": c / c_min,
               "pass@1_all": sum(t.quality[m] for t in tasks) / len(tasks)}
        for lv in ("easy", "medium", "hard"):
            sub = [t for t in tasks if t.difficulty == lv]
            row[f"pass@1_{lv}"] = sum(t.quality[m] for t in sub) / len(sub)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["dominated"] = [
        any(o.cost_usd <= r.cost_usd and o["pass@1_all"] > r["pass@1_all"] for _, o in df.iterrows() if o.model != r.model)
        for _, r in df.iterrows()
    ]
    return df


def patterns() -> pd.DataFrame:
    models = list(MEASURED_COSTS)
    counter = collections.Counter(
        (t.difficulty, "".join(SHORT[m][0] if t.quality[m] else "." for m in models) or "") for t in base_tasks()
    )
    rows = [{"difficulty": d, "solved_by(V3.1,V3,Flash,R1)": "".join("1" if c != "." else "0" for c in p), "tasks": n}
            for (d, p), n in sorted(counter.items(), key=lambda kv: (kv[0][0], -kv[1]))]
    return pd.DataFrame(rows)


def oracle_value() -> dict:
    """What per-task knowledge is worth: cheapest model that passes vs. best model by q-hat."""
    tasks = base_tasks()
    models = sorted(MEASURED_COSTS, key=MEASURED_COSTS.get)
    any_pass = sum(any(t.quality[m] for m in models) for t in tasks) / len(tasks)
    cost_oracle = sum(next((MEASURED_COSTS[m] for m in models if t.quality[m]), MEASURED_COSTS[models[0]]) for t in tasks) / len(tasks)
    return {"oracle_pass_rate(any model)": any_pass, "oracle_mean_cost_per_task": cost_oracle,
            "r1_everywhere_pass_rate": sum(t.quality[models[-1]] for t in tasks) / len(tasks),
            "r1_everywhere_cost_per_task": MEASURED_COSTS[models[-1]]}


def price_curve() -> pd.DataFrame:
    """lambda*(tau): the shadow price of budget (quality per USD) on the calibration split."""
    rows = []
    for pricing, costs in PRICINGS.items():
        models = ModelRegistry()
        for n, c in costs.items():
            models.add(n, c)
        c_min, c_max = min(costs.values()), max(costs.values())
        for regime in ("difficulty", "blind", "oracle"):
            tasks = [view(t, regime) for t in split()[0]]
            for tau in (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.75, 1.0):
                budget = len(tasks) * (c_min + tau * (c_max - c_min))
                rows.append({"pricing": pricing, "regime": regime, "tau": tau,
                             "lambda_star": calibrate_price(tasks, models, budget)})
    return pd.DataFrame(rows)


def figures(mt: pd.DataFrame, pc: pd.DataFrame) -> None:
    plt = plots.plt
    plots.style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    ax = axes[0]
    col = plots.colors(["easy", "medium", "hard"])
    for lv in ("hard", "medium", "easy"):
        ax.plot(mt.cost_usd * 1000, mt[f"pass@1_{lv}"], "o", color=col[lv], label=lv, markersize=8,
                markeredgecolor=plots.SURFACE, markeredgewidth=1.5)
    for _, r in mt.iterrows():
        ax.annotate(SHORT[r.model], (r.cost_usd * 1000, r["pass@1_hard"]), textcoords="offset points",
                    xytext=(6, -12), color=plots.MUTED, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("mean cost per call (USD x 1e-3, log scale)")
    ax.set_ylabel("pass@1")
    ax.set_title("Model pool: cost vs pass@1 by difficulty")
    ax.legend(title="difficulty", loc="center left")

    ax = axes[1]
    sub = pc[(pc.pricing == "measured")]
    col = plots.colors(["difficulty", "oracle", "blind"])
    for regime in ("difficulty", "oracle", "blind"):
        s = sub[sub.regime == regime]
        ax.plot(s.tau, s.lambda_star, "-o", color=col[regime], label=regime, drawstyle="steps-post")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("budget tightness tau  (0 = J*c_min, 1 = J*c_max)")
    ax.set_ylabel("lambda* (pass@1 per USD)")
    ax.set_title("Shadow price of budget (calibration split)")
    ax.legend(title="information regime")
    plots.save(fig, FIGURES / "e01_profile.png")


if __name__ == "__main__":
    mt = model_table()
    pt = patterns()
    pc = price_curve()
    calib, evaluation = split()
    qt = quality_tables()
    qhat = pd.DataFrame(qt["difficulty"]).T.rename(columns=SHORT)
    qhat.loc["blind (all)"] = pd.Series(qt["blind"]).rename(SHORT)
    save(mt, "e01_models")
    save(pt, "e01_patterns")
    save(pc, "e01_price_curve")
    figures(mt, pc)
    print(f"calibration tasks: {len(calib)}, evaluation tasks: {len(evaluation)}")
    print(md_table(mt.assign(model=mt.model.map(SHORT))))
    print(md_table(pt, ".0f"))
    print(md_table(qhat.reset_index().rename(columns={"index": "label"}), ".3f"))
    print(oracle_value())
    print(md_table(pc[pc.pricing == "measured"].pivot(index="tau", columns="regime", values="lambda_star").reset_index(), ".2f"))
    inst = build_instance(Scenario(tau=0.2))
    print("example J=200 budgets (USD):", {t: round(Scenario(tau=t).budget(), 4) for t in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0)})
