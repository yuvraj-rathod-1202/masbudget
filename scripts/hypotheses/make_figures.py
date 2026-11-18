"""Figures + summary tables for results.md, from results/hypotheses/*.csv.gz.

    python scripts/hypotheses/make_figures.py

Writes figures/*.png and results/hypotheses/summary_*.csv, and prints markdown tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

RES = ROOT / "results" / "hypotheses"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- style (reference palette, light mode) ---------------------------------
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})
MIX_LABEL = {"uniform": "Uniform (⅓/⅓/⅓)", "easy_heavy": "Easy-heavy (80/10/10)", "hard_heavy": "Hard-heavy (10/10/80)"}
ARR_LABEL = {"fcfs": "Round-robin FCFS", "random_fcfs": "Random", "hardest_task_first": "Hardest-first",
             "easiest_task_first": "Easiest-first", "marginal_roi_priority": "Marginal-ROI priority"}
UTIL_LABEL = {"best_model": "Best-model", "quasi_linear": "Quasi-linear", "dynamic_shadow_price": "Shadow price",
              "shapley_budget_share": "Shapley share", "marginal_roi_ratio": "ΔU/ΔC ratio"}
CEN_LABEL = {"optimal_mckp": "MCKP (optimal)", "predictive_router": "Predictive router", "confidence_cascade": "Cascade†",
             "static_equal_quota": "Equal quota", "cheapest_model": "Cheapest model"}
MULTS = [1, 1.5, 3, 6, 12, 25, 50, 120]
RNG = np.random.default_rng(0)


def label(p):
    if p.startswith("C:"):
        return CEN_LABEL[p[2:]]
    a, u = p[2:].split("|")[:2]
    return f"{ARR_LABEL[a]} + {UTIL_LABEL[u]}"


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    bs = RNG.choice(x, size=(n, len(x)), replace=True).mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def signflip_p(d, n=10000):
    """Two-sided paired sign-flip permutation p-value for mean(d) != 0."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean())
    flips = RNG.choice([-1, 1], size=(n, len(d)))
    return float(((np.abs((flips * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n + 1))


def add_gap(df, keys):
    """gap = (Q_mckp - Q_policy) / Q_mckp, paired on the same instance (same info setting)."""
    ref = df[df.policy == "C:optimal_mckp"][keys + ["quality"]].rename(columns={"quality": "q_ref"})
    out = df.merge(ref, on=keys, how="left")
    out["gap"] = (out.q_ref - out.quality) / out.q_ref
    return out


KEYS = ["info", "mix", "n_agents", "tasks_per_agent", "budget_mult", "model_set", "seed", "label_noise"]


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def md_table(df, floatfmt=3):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join([":---"] + [":---:"] * (len(cols) - 1)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ============================================================================
def main():
    main_df = add_gap(pd.read_csv(RES / "main.csv.gz"), KEYS)
    exp = main_df[main_df["info"] == "expected"]
    report = {}

    # ---------------- T1: policy league table (expected info, all budgets, per mix) ---------
    agg = (exp.groupby(["mix", "policy"]).agg(quality=("quality", "mean"), gap=("gap", "mean"),
                                              starve=("task_starve", "mean"), jain=("jain", "mean"))
           .reset_index())
    agg.to_csv(RES / "summary_policy_by_mix.csv", index=False)

    # best decentralized config per (mix, budget) and overall
    dec = exp[exp.paradigm == "decentral"]
    cell = dec.groupby(["mix", "budget_mult", "policy"]).agg(quality=("quality", "mean"), gap=("gap", "mean")).reset_index()

    # ---------------- Fig 1: budget sweep (H1) ----------------------------------------------
    focus = [("C:optimal_mckp", "MCKP (centralized optimum)", S[0], "-"),
             ("D:random_fcfs|best_model", "Random + Best-model (naive decentralized)", S[1], "-"),
             ("D:random_fcfs|quasi_linear", "Random + Quasi-linear (calibrated λ)", S[2], "-"),
             ("D:easiest_task_first|dynamic_shadow_price", "Easiest-first + Shadow price", S[6], "-"),
             ("C:cheapest_model", "Cheapest model", INK2, ":")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, mix in zip(axes, H.MIXES):
        for pol, lab, col, ls in focus:
            g = exp[(exp.mix == mix) & (exp.policy == pol)].groupby("budget_mult").quality
            m, lo, hi = zip(*[boot_ci(g.get_group(b)) for b in MULTS])
            ax.plot(MULTS, m, color=col, ls=ls, label=lab, marker="o", ms=4)
            ax.fill_between(MULTS, lo, hi, color=col, alpha=0.15, lw=0)
        orc = main_df[(main_df["info"] == "oracle") & (main_df.mix == mix) & (main_df.policy == "C:optimal_mckp")].groupby("budget_mult").quality.mean()
        ax.plot(MULTS, orc.loc[MULTS].values, color=INK2, ls="--", lw=1.2, label="MCKP with oracle (hindsight bound)")
        ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(MULTS, [f"{m:g}" for m in MULTS])
        ax.set_title(MIX_LABEL[mix])
        ax.set_xlabel("Budget per task  (× cheapest-model cost)")
    axes[0].set_ylabel("Mean realized pass@1")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("H1 - Quality vs shared budget (expected-quality information, N=10 × 10 tasks, 50 seeds, 95% CI)", x=0.01, ha="left")
    save(fig, "fig1_budget_sweep.png")

    # ---------------- Fig 2: gap of every utility under random arrival vs budget (H1/H4) ------
    for info, fname in [("expected", "fig2_gap_by_utility.png"), ("oracle", "fig2b_gap_by_utility_oracle.png")]:
        src_df = main_df[main_df["info"] == info]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
        for ax, mix in zip(axes, H.MIXES):
            for k, u in enumerate(H.UTILITIES):
                g = src_df[(src_df.mix == mix) & (src_df.policy == f"D:random_fcfs|{u}")].groupby("budget_mult").gap.mean()
                ax.plot(MULTS, 100 * g.loc[MULTS].values, color=S[k], marker="o", ms=4, label=f"Random + {UTIL_LABEL[u]}")
            best = src_df[(src_df.mix == mix) & (src_df.paradigm == "decentral")].groupby(["budget_mult", "policy"]).gap.mean().groupby("budget_mult").min()
            ax.plot(MULTS, 100 * best.loc[MULTS].values, color=INK2, ls="--", lw=1.2, label="best of all 25 decentralized configs")
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xticks(MULTS, [f"{m:g}" for m in MULTS])
            ax.set_title(MIX_LABEL[mix])
            ax.set_xlabel("Budget per task  (× cheapest-model cost)")
        axes[0].set_ylabel("Optimality gap vs MCKP (%)")
        axes[0].legend(fontsize=8)
        ttl = "expected-quality information" if info == "expected" else "ORACLE information"
        fig.suptitle(f"H1/H4 - Price of anarchy under random arrival, by utility ({ttl})", x=0.01, ha="left")
        save(fig, fname)

    # ---------------- Fig 3: arrival x utility heatmap (H3/H4) --------------------------------
    for regime, mults in {"tight": [1.5, 3, 6], "all": MULTS}.items():
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
        for ax, mix in zip(axes, H.MIXES):
            sub = exp[(exp.mix == mix) & (exp.paradigm == "decentral") & (exp.budget_mult.isin(mults))]
            mat = sub.groupby(["arrival", "utility"]).gap.mean().unstack().loc[H.ARRIVALS, H.UTILITIES] * 100
            im = ax.imshow(mat.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=max(80, mat.values.max()))
            ax.grid(False)
            ax.set_xticks(range(5), [UTIL_LABEL[u] for u in H.UTILITIES], rotation=30, ha="right")
            ax.set_yticks(range(5), [ARR_LABEL[a] for a in H.ARRIVALS] if mix == "uniform" else [""] * 5)
            for i in range(5):
                for j in range(5):
                    v = mat.values[i, j]
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > 45 else INK)
            ax.set_title(MIX_LABEL[mix])
        cb = fig.colorbar(im, ax=axes, shrink=0.85)
        cb.set_label("Optimality gap (%)")
        span = "budget ×1.5–6 (tight)" if regime == "tight" else "all 8 budget levels"
        fig.suptitle(f"H3/H4 - Gap to MCKP by arrival policy × utility, mean over {span}", x=0.01, ha="left")
        save(fig, f"fig3_arrival_utility_heatmap_{regime}.png")

    # ---------------- Table: centralized vs best decentralized per mix & budget -------------
    rows = []
    for mix in H.MIXES:
        for b in MULTS:
            c = cell[(cell.mix == mix) & (cell.budget_mult == b)].sort_values("quality", ascending=False)
            best = c.iloc[0]
            e = exp[(exp.mix == mix) & (exp.budget_mult == b)]
            mck = e[e.policy == "C:optimal_mckp"].quality.mean()
            naive = e[e.policy == "D:random_fcfs|best_model"]
            rq = e[e.policy == "D:random_fcfs|quasi_linear"]
            bestrows = e[e.policy == best.policy]
            mck_exp = e[e.policy == "C:optimal_mckp"].exp_quality.mean()
            best_obj = e[e.paradigm == "decentral"].groupby("policy").exp_quality.mean().max()
            rows.append({
                "Mix": mix, "Budget ×c_min": b, "MCKP Q": mck,
                "Naive (Random+Best) Q": naive.quality.mean(), "Naive gap %": 100 * naive.gap.mean(),
                "Random+QL gap %": 100 * rq.gap.mean(),
                "Best decentralized": label(best.policy), "Best dec. Q": best.quality,
                "Best dec. gap % [95% CI]": "{:.1f} [{:.1f}, {:.1f}]".format(*[100 * v for v in boot_ci(bestrows.gap)]),
                "Best dec. gap on q-hat objective %": 100 * (1 - best_obj / mck_exp),
            })
    t_main = pd.DataFrame(rows)
    t_main.to_csv(RES / "summary_central_vs_decentral.csv", index=False)
    report["t_main"] = md_table(t_main)

    # ---------------- H3: arrival policies (averaged over utilities), paired tests ----------
    rows = []
    for mix in H.MIXES:
        for tight in [True, False]:
            mults = [1.5, 3, 6] if tight else [12, 25, 50]
            sub = exp[(exp.mix == mix) & (exp.paradigm == "decentral") & exp.budget_mult.isin(mults)]
            per = sub.groupby(["arrival", "seed", "budget_mult"]).gap.mean().unstack(0)
            r = {"Mix": mix, "Budget": "tight (×1.5–6)" if tight else "loose (×12–50)"}
            for a in H.ARRIVALS:
                r[ARR_LABEL[a]] = 100 * per[a].mean()
            d = per["hardest_task_first"] - per["random_fcfs"]
            r["HTF − Random (pp), p"] = f"{100 * d.mean():+.1f}, p={signflip_p(d.values):.3g}"
            d = per["easiest_task_first"] - per["random_fcfs"]
            r["ETF − Random (pp), p"] = f"{100 * d.mean():+.1f}, p={signflip_p(d.values):.3g}"
            rows.append(r)
    t_arr = pd.DataFrame(rows)
    t_arr.to_csv(RES / "summary_arrival.csv", index=False)
    report["t_arrival"] = md_table(t_arr, 1)

    # ---------------- H4: utilities (random arrival), paired vs best-model ------------------
    rows = []
    for mix in H.MIXES:
        for tight in [True, False]:
            mults = [1.5, 3, 6] if tight else [12, 25, 50]
            sub = exp[(exp.mix == mix) & (exp.arrival == "random_fcfs") & exp.budget_mult.isin(mults)]
            per = sub.pivot_table(index=["seed", "budget_mult"], columns="utility", values="gap")
            r = {"Mix": mix, "Budget": "tight (×1.5–6)" if tight else "loose (×12–50)"}
            for u in H.UTILITIES:
                d = per[u] - per["best_model"]
                r[UTIL_LABEL[u]] = f"{100 * per[u].mean():.1f}" + ("" if u == "best_model" else f" ({100 * d.mean():+.1f}{'*' if signflip_p(d.values) < 0.01 else ''})")
            rows.append(r)
    t_util = pd.DataFrame(rows)
    t_util.to_csv(RES / "summary_utility.csv", index=False)
    report["t_utility"] = md_table(t_util)

    # ---------------- H5: per-tier quality and budget share (MCKP & cheapest) ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    e = exp[(exp.mix == "uniform") & (exp.policy == "C:optimal_mckp")].groupby("budget_mult")
    ch = exp[(exp.mix == "uniform") & (exp.policy == "C:cheapest_model")]
    for k, t in enumerate(H.TIERS):
        axes[0].plot(MULTS, e[f"q_{t}"].mean().loc[MULTS].values, color=S[k], marker="o", ms=4, label=f"{t.capitalize()} tasks")
        axes[0].axhline(ch[f"q_{t}"].mean(), color=S[k], ls=":", lw=1.2)
    axes[0].set_xscale("log"); axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_xticks(MULTS, [f"{m:g}" for m in MULTS])
    axes[0].set_xlabel("Budget per task  (× cheapest-model cost)")
    axes[0].set_ylabel("Mean realized pass@1 (MCKP)")
    axes[0].set_title("MCKP quality by tier (dotted = cheapest model)")
    axes[0].legend(fontsize=8)
    bottom = np.zeros(len(MULTS))
    x = np.arange(len(MULTS))
    for k, t in enumerate(H.TIERS):
        v = e[f"spend_{t}"].mean().loc[MULTS].values * 100
        axes[1].bar(x, v, bottom=bottom, color=S[k], width=0.7, label=f"{t.capitalize()} tasks", edgecolor=SURF, linewidth=2)
        bottom += v
    axes[1].set_xticks(x, [f"{m:g}" for m in MULTS])
    axes[1].set_xlabel("Budget per task  (× cheapest-model cost)")
    axes[1].set_ylabel("Share of MCKP spend (%)")
    axes[1].set_title("Share of MCKP spend by tier (uniform mix)")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("H5 - Value of expensive models by task difficulty", x=0.01, ha="left")
    save(fig, "fig5_difficulty_value.png")

    bench = H.load_bench()
    rows = []
    for m in H.MODEL_SETS["full"]:
        r = {"Model": m, "Cost $/call": bench.costs[m]}
        for t, name in enumerate(H.TIERS):
            r[f"pass@1 {name}"] = bench.scores[m][bench.tiers == t].mean()
        base = "deepseek-v3.1-terminus"
        dc = bench.costs[m] - bench.costs[base]
        r["Δpass hard vs cheapest"] = bench.scores[m][bench.tiers == 2].mean() - bench.scores[base][bench.tiers == 2].mean()
        r["Δpass medium vs cheapest"] = bench.scores[m][bench.tiers == 1].mean() - bench.scores[base][bench.tiers == 1].mean()
        r["Δpass(hard)/Δ$"] = r["Δpass hard vs cheapest"] / dc if dc > 0 else np.nan
        rows.append(r)
    t_models = pd.DataFrame(rows)
    report["t_models"] = md_table(t_models, 4)

    # ---------------- H6: arrival position under random arrival ------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    sub = exp[(exp.mix == "uniform") & (exp.arrival == "random_fcfs")]
    for k, (u, b) in enumerate([("best_model", 6), ("quasi_linear", 6), ("dynamic_shadow_price", 6), ("best_model", 25)]):
        g = sub[(sub.utility == u) & (sub.budget_mult == b)]
        q = [g[f"pos_q_{d}"].mean() for d in range(10)]
        st = [100 * g[f"pos_starve_{d}"].mean() for d in range(10)]
        lab = f"{UTIL_LABEL[u]}, budget ×{b:g}"
        axes[0].plot(range(1, 11), q, color=S[k], marker="o", ms=4, label=lab)
        axes[1].plot(range(1, 11), st, color=S[k], marker="o", ms=4, label=lab)
    axes[0].set_ylabel("Mean realized pass@1")
    axes[1].set_ylabel("Tasks starved (%)")
    for ax in axes:
        ax.set_xlabel("Arrival position decile (1 = first 10% of requests)")
        ax.set_xticks(range(1, 11))
    axes[0].legend(fontsize=8)
    axes[0].set_title("Quality by arrival position")
    axes[1].set_title("Starvation by arrival position")
    fig.suptitle("H6 - First-mover advantage under random arrival (uniform mix)", x=0.01, ha="left")
    save(fig, "fig6_first_mover.png")

    rows = []
    for u in H.UTILITIES:
        for b in [3, 6, 25]:
            g = sub[(sub.utility == u) & (sub.budget_mult == b)]
            rows.append({"Utility": UTIL_LABEL[u], "Budget ×c_min": b,
                         "Q first decile": g.pos_q_0.mean(), "Q last decile": g.pos_q_9.mean(),
                         "Starved last decile %": 100 * g.pos_starve_9.mean(),
                         "Agent FMR (diff, pp)": 100 * g.fmr_diff.mean(),
                         "Agent starvation %": 100 * g.agent_starve.mean(), "Jain": g.jain.mean()})
    report["t_fmr"] = md_table(pd.DataFrame(rows))
    # agent-level FMR under an agent-sequential order (each agent submits all of its tasks in a burst)
    report["fmr_note"] = ""

    # ---------------- Value of information: oracle vs expected --------------------------------
    rows = []
    for mix in H.MIXES:
        for b in [3, 12, 50]:
            r = {"Mix": mix, "Budget ×c_min": b}
            for info in ["expected", "oracle"]:
                g = main_df[(main_df["info"] == info) & (main_df.mix == mix) & (main_df.budget_mult == b)]
                r[f"MCKP Q ({info})"] = g[g.policy == "C:optimal_mckp"].quality.mean()
                bd = g[g.paradigm == "decentral"].groupby("policy").quality.mean()
                r[f"Best dec. Q ({info})"] = bd.max()
                r[f"Best dec. gap % ({info})"] = 100 * (1 - bd.max() / r[f"MCKP Q ({info})"])
            rows.append(r)
    t_info = pd.DataFrame(rows)
    report["t_info"] = md_table(t_info)

    # ---------------- H2: scale -----------------------------------------------------------
    if (RES / "scale.csv.gz").exists():
        sc = add_gap(pd.read_csv(RES / "scale.csv.gz"), KEYS)
        sc.to_csv(RES / "_scale_gap.csv.gz", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        pols = [("D:random_fcfs|best_model", "Random + Best-model"), ("D:random_fcfs|quasi_linear", "Random + Quasi-linear"),
                ("D:random_fcfs|dynamic_shadow_price", "Random + Shadow price"), ("D:easiest_task_first|dynamic_shadow_price", "Easiest-first + Shadow price")]
        Ns = [5, 10, 20, 50]
        for ax, b in zip(axes, [3, 12]):
            for k, (p, lab) in enumerate(pols):
                g = sc[(sc.policy == p) & (sc.budget_mult == b) & (sc.mix == "uniform")].groupby("n_agents").gap
                m, lo, hi = zip(*[boot_ci(g.get_group(n)) for n in Ns])
                ax.plot(Ns, 100 * np.array(m), color=S[k], marker="o", ms=4, label=lab)
                ax.fill_between(Ns, 100 * np.array(lo), 100 * np.array(hi), color=S[k], alpha=0.15, lw=0)
            ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xticks(Ns, [str(n) for n in Ns])
            ax.set_xlabel("Number of agents N (10 tasks each; budget ∝ #tasks)")
            ax.set_title(f"Budget ×{b} c_min per task (uniform mix)")
        axes[0].set_ylabel("Optimality gap vs MCKP (%)")
        axes[0].legend(fontsize=8)
        fig.suptitle("H2 - Does adding agents widen the gap?", x=0.01, ha="left")
        save(fig, "fig4_scale_agents.png")
        rows = []
        for mix in H.MIXES:
            for b in [3, 12]:
                r = {"Mix": mix, "Budget ×c_min": b}
                for n in Ns:
                    g = sc[(sc.mix == mix) & (sc.budget_mult == b) & (sc.n_agents == n) & (sc.paradigm == "decentral")]
                    r[f"N={n}: mean dec. gap %"] = 100 * g.gap.mean()
                    r[f"N={n}: agent starve %"] = 100 * g[g.policy == "D:random_fcfs|best_model"].agent_starve.mean()
                # slope test: gap at N=50 minus N=5, paired on seed, averaged over decentralized policies
                a = sc[(sc.mix == mix) & (sc.budget_mult == b) & (sc.paradigm == "decentral")].groupby(["n_agents", "seed"]).gap.mean().unstack(0)
                d = a[50] - a[5]
                r["Δgap N50−N5 (pp), p"] = f"{100 * d.mean():+.1f}, p={signflip_p(d.values):.3g}"
                rows.append(r)
        report["t_scale"] = md_table(pd.DataFrame(rows), 1)

    # ---------------- H7: disparity ----------------------------------------------------------
    if (RES / "disparity.csv.gz").exists():
        dp = add_gap(pd.read_csv(RES / "disparity.csv.gz"), KEYS)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
        for ax, mix in zip(axes, H.MIXES):
            for k, ms in enumerate(["flat", "steep"]):
                g = dp[(dp.model_set == ms) & (dp.mix == mix) & (dp.paradigm == "decentral")]
                naive = g[g.policy == "D:random_fcfs|best_model"].groupby("budget_mult").gap.mean()
                best = g.groupby(["budget_mult", "policy"]).gap.mean().groupby("budget_mult").min()
                ax.plot(MULTS, 100 * naive.loc[MULTS].values, color=S[k], marker="o", ms=4, label=f"{ms} pool: Random + Best-model")
                ax.plot(MULTS, 100 * best.loc[MULTS].values, color=S[k], ls="--", marker="s", ms=4, label=f"{ms} pool: best decentralized config")
            ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xticks(MULTS, [f"{m:g}" for m in MULTS])
            ax.set_xlabel("Budget per task  (× cheapest-model cost)")
            ax.set_title(MIX_LABEL[mix])
            ax.axhline(0, color=INK2, lw=0.8)
        axes[0].set_ylabel("Optimality gap vs MCKP (%)")
        axes[0].legend(fontsize=8)
        fig.suptitle("H7 - Flat (1–5×, cheap models) vs steep (1–120×, frontier) cost/quality pools", x=0.01, ha="left")
        save(fig, "fig7_disparity.png")
        rows = []
        for mix in H.MIXES:
            for ms in ["flat", "steep"]:
                g = dp[(dp.model_set == ms) & (dp.mix == mix)]
                dec_ = g[g.paradigm == "decentral"]
                r = {"Mix": mix, "Pool": ms,
                     "MCKP Q (tight ×1.5–6)": g[(g.policy == "C:optimal_mckp") & g.budget_mult.isin([1.5, 3, 6])].quality.mean(),
                     "MCKP Q (loose ×12–50)": g[(g.policy == "C:optimal_mckp") & g.budget_mult.isin([12, 25, 50])].quality.mean(),
                     "Naive gap % tight": 100 * dec_[(dec_.policy == "D:random_fcfs|best_model") & dec_.budget_mult.isin([1.5, 3, 6])].gap.mean(),
                     "Mean dec. gap % tight": 100 * dec_[dec_.budget_mult.isin([1.5, 3, 6])].gap.mean(),
                     "Mean dec. gap % loose": 100 * dec_[dec_.budget_mult.isin([12, 25, 50])].gap.mean(),
                     "Best dec. gap % tight": 100 * dec_[dec_.budget_mult.isin([1.5, 3, 6])].groupby(["budget_mult", "policy"]).gap.mean().groupby("budget_mult").min().mean()}
                rows.append(r)
        report["t_disparity"] = md_table(pd.DataFrame(rows), 3)

    # ---------------- Robustness: label noise ------------------------------------------------
    if (RES / "noise.csv.gz").exists():
        nz = add_gap(pd.read_csv(RES / "noise.csv.gz"), KEYS)
        rows = []
        for eps in [0.0, 0.2, 0.4]:
            for b in [3, 12, 50]:
                g = nz[(nz.label_noise == eps) & (nz.budget_mult == b)]
                bd = g[g.paradigm == "decentral"].groupby("policy").quality.mean()
                rows.append({"Label noise ε": eps, "Budget ×c_min": b,
                             "MCKP Q": g[g.policy == "C:optimal_mckp"].quality.mean(),
                             "Random+Best Q": g[g.policy == "D:random_fcfs|best_model"].quality.mean(),
                             "ETF+Shadow Q": g[g.policy == "D:easiest_task_first|dynamic_shadow_price"].quality.mean(),
                             "HTF+Shadow Q": g[g.policy == "D:hardest_task_first|dynamic_shadow_price"].quality.mean(),
                             "Best dec.": label(bd.idxmax()), "Best dec. Q": bd.max()})
        report["t_noise"] = md_table(pd.DataFrame(rows))

    # ---------------- Tuning curve -----------------------------------------------------------
    if (RES / "tune.csv.gz").exists():
        tu = pd.read_csv(RES / "tune.csv.gz")
        tu = tu[(tu.model_set == "full")]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        ql = tu[tu.utility == "quasi_linear"].copy()
        ql["lam"] = ql.params.map(lambda s: json.loads(s)["lambda_cost"])
        piv = ql.groupby(["budget_mult", "lam"]).quality.mean().unstack(0)
        for k, b in enumerate([1.5, 6, 25, 120]):
            axes[0].plot(piv.index, piv[b], color=S[k], marker="o", ms=4, label=f"budget ×{b:g}")
        avg = piv.mean(axis=1)
        axes[0].plot(avg.index, avg.values, color=INK, lw=1.2, ls="--", label="mean over all 8 budgets")
        axes[0].axvline(0.5, color=INK2, lw=0.8, ls=":")
        axes[0].text(0.55, 0.12, "proposal default λ=0.5", color=INK2, fontsize=8, rotation=90, va="bottom")
        axes[0].set_xscale("log"); axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[0].set_xlabel("Quasi-linear cost sensitivity λ")
        axes[0].set_ylabel("Mean realized pass@1 (random arrival)")
        axes[0].set_title("A single λ cannot serve every budget")
        axes[0].legend(fontsize=8)
        tuned = json.loads((RES / "tuned_params.json").read_text())["per_budget"]["full"]
        lam = [tuned[str(float(b))]["quasi_linear"]["lambda_cost"] for b in MULTS]
        bpt = [b * H.load_bench().costs["deepseek-v3.1-terminus"] for b in MULTS]
        axes[1].plot(bpt, lam, color=S[0], marker="o", ms=4, label="calibrated λ*")
        axes[1].plot(bpt, [1 / x for x in bpt], color=INK2, ls="--", lw=1.2, label="1 / (budget per task)")
        axes[1].set_xscale("log"); axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Budget per task ($)")
        axes[1].set_ylabel("λ*")
        axes[1].set_title("Calibrated λ* tracks the inverse per-task budget")
        axes[1].legend(fontsize=8)
        fig.suptitle("Utility calibration (tuning seeds 1000–1019; never used for test numbers)", x=0.01, ha="left")
        save(fig, "fig8_lambda_calibration.png")
        report["tuned"] = json.dumps(json.loads((RES / "tuned_params.json").read_text())["global"])

    # ---------------- Calibration robustness (tuning seeds) -----------------------------------
    if (RES / "tune.csv.gz").exists():
        tu = pd.read_csv(RES / "tune.csv.gz")
        tu = tu[(tu.model_set == "full") & (tu.paradigm == "decentral")]
        perf = tu.groupby(["utility", "params", "budget_mult"]).quality.mean().reset_index()
        rows = []
        for u in ["quasi_linear", "dynamic_shadow_price", "shapley_budget_share"]:
            pu = perf[perf.utility == u]
            best_at = pu.loc[pu.groupby("budget_mult").quality.idxmax()].set_index("budget_mult")
            reg_matrix = []
            for b_train in MULTS:
                prm = best_at.loc[b_train, "params"]
                for b_test in MULTS:
                    q = pu[(pu.params == prm) & (pu.budget_mult == b_test)].quality.iloc[0]
                    reg_matrix.append((b_train, b_test, best_at.loc[b_test, "quality"] - q))
            rm = pd.DataFrame(reg_matrix, columns=["train", "test", "regret"])
            off = rm[rm.train != rm.test]
            near = rm[(rm.train != rm.test) & (np.abs(np.log(rm.train / rm.test)) <= np.log(2.1))]
            glob = pu.groupby("params").quality.mean().max()
            rows.append({"Utility": UTIL_LABEL[u],
                         "Mean Q, calibrated per budget": best_at.quality.mean(),
                         "Mean Q, best single setting": glob,
                         "Mean regret, param from adjacent budget (x2)": near.regret.mean(),
                         "Mean regret, param from any other budget": off.regret.mean(),
                         "Worst-case regret": off.regret.max()})
        report["t_calibration"] = md_table(pd.DataFrame(rows))

    # ---------------- H6 (agent-level burst arrival) -----------------------------------------
    if (RES / "burst.csv.gz").exists():
        bu = add_gap(pd.read_csv(RES / "burst.csv.gz"), KEYS)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for k, u in enumerate(H.UTILITIES):
            g = bu[bu.utility == u].groupby("budget_mult")
            axes[0].plot(MULTS, 100 * g.fmr_diff.mean().loc[MULTS].values, color=S[k], marker="o", ms=4, label=UTIL_LABEL[u])
            axes[1].plot(MULTS, g.jain.mean().loc[MULTS].values, color=S[k], marker="o", ms=4, label=UTIL_LABEL[u])
        mk = bu[bu.policy == "C:optimal_mckp"].groupby("budget_mult").jain.mean()
        axes[1].plot(MULTS, mk.loc[MULTS].values, color=INK2, ls="--", lw=1.2, label="MCKP (centralized)")
        for ax in axes:
            ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xticks(MULTS, [f"{m:g}" for m in MULTS])
            ax.set_xlabel("Budget per task  (x cheapest-model cost)")
        axes[0].axhline(0, color=INK2, lw=0.8)
        axes[0].set_ylabel("First-quartile minus last-quartile\nagent payoff (pp)")
        axes[0].set_title("First-mover advantage")
        axes[1].set_ylabel("Jain's fairness index (agent payoffs)")
        axes[1].set_title("Fairness across agents")
        axes[1].legend(fontsize=8)
        fig.suptitle("H6 - Agents arrive in random order, each submits its 10 tasks back-to-back (uniform mix, 50 seeds)", x=0.01, ha="left")
        save(fig, "fig6b_first_mover_agents.png")
        rows = []
        for u in H.UTILITIES:
            for b in [3, 6, 25]:
                g = bu[(bu.utility == u) & (bu.budget_mult == b)]
                rows.append({"Utility": UTIL_LABEL[u], "Budget x c_min": b,
                             "First-quartile Q minus last-quartile Q (pp)": 100 * g.fmr_diff.mean(),
                             "Runs where last quartile gets 0": f"{100 * (g.fmr == np.inf).mean():.0f}%",
                             "Agent starve %": 100 * g.agent_starve.mean(), "Jain": g.jain.mean(), "Gap %": 100 * g.gap.mean()})
        report["t_burst"] = md_table(pd.DataFrame(rows))

    # ---------------- League table (overall) ---------------------------------------------------
    lt = exp.groupby("policy").agg(Q=("quality", "mean"), gap=("gap", "mean"), starve=("task_starve", "mean"),
                                   agent_starve=("agent_starve", "mean"), jain=("jain", "mean"), util=("budget_util", "mean")).reset_index()
    lt["Policy"] = lt.policy.map(label)
    lt["Paradigm"] = np.where(lt.policy.str.startswith("C:"), "Centralized", "Decentralized")
    lt = lt.sort_values("Q", ascending=False)
    lt["gap"] *= 100
    lt["starve"] *= 100
    lt["agent_starve"] *= 100
    lt["util"] *= 100
    lt = lt[["Paradigm", "Policy", "Q", "gap", "starve", "agent_starve", "jain", "util"]].rename(columns={
        "Q": "Mean Q", "gap": "Gap %", "starve": "Task starve %", "agent_starve": "Agent starve %", "jain": "Jain", "util": "Budget used %"})
    report["t_league"] = md_table(lt)

    (RES / "report_tables.md").write_text("\n\n".join(f"## {k}\n\n{v}" for k, v in report.items()), encoding="utf-8")
    print("wrote", RES / "report_tables.md")


if __name__ == "__main__":
    main()
