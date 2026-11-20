"""Figures and LaTeX tables for report/report.tex, generated from results/hypotheses/*.csv.gz.

    python scripts/hypotheses/paper_figures.py

Writes  report/figures/*.pdf      (figures used in the paper)
        report/generated/*.tex    (tables and numbers \\input by the paper)
        figures/paper/*.png       (same figures for the README)
Every number in the paper's results section comes from this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

RES = ROOT / "results" / "hypotheses"
PDF = ROOT / "report" / "figures"
GEN = ROOT / "report" / "generated"
PNG = ROOT / "figures" / "paper"
for d in (PDF, GEN, PNG):
    d.mkdir(parents=True, exist_ok=True)

S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.8,
    "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold", "legend.frameon": False,
    "font.family": "serif", "mathtext.fontset": "stix",
})
MULTS = [1, 1.5, 3, 6, 12, 25, 50, 120]
TIGHT = [1.5, 3, 6]
MIXES = ["uniform", "easy_heavy", "hard_heavy"]
MIX_LABEL = {"uniform": "Uniform mix", "easy_heavy": "Easy-heavy mix (80% easy)", "hard_heavy": "Hard-heavy mix (80% hard)"}
MIX_SHORT = {"uniform": "Uniform", "easy_heavy": "Easy-heavy", "hard_heavy": "Hard-heavy"}
KEYS = ["info", "mix", "n_agents", "tasks_per_agent", "budget_mult", "model_set", "seed", "label_noise"]
RNG = np.random.default_rng(0)


def add_gap(df):
    ref = df[df.policy == "C:optimal_mckp"][KEYS + ["quality"]].rename(columns={"quality": "q_ref"})
    out = df.merge(ref, on=KEYS, how="left")
    out["gap"] = 100 * (out.q_ref - out.quality) / out.q_ref
    return out


def boot(x, n=4000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    bs = RNG.choice(x, size=(n, len(x)), replace=True).mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def signflip_p(d, n=20000):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    flips = RNG.choice([-1, 1], size=(n, len(d)))
    return float(((np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12).sum() + 1) / (n + 1))


def nozero(tex):
    """Print values that round to zero as 0.0 instead of -0.0."""
    import re
    return re.sub(r"(?<![\d.])-0\.(0+)(?![\d])", r"0.\1", tex)


def ptex(p):
    """p-value as LaTeX math, e.g. 1.5\\times10^{-4}."""
    m, e = f"{p:.1e}".split("e")
    return f"{float(m):g}\\times10^{{{int(e)}}}" if p < 0.001 else f"{p:.3f}"


def logx(ax, ticks=(1, 3, 6, 12, 25, 50, 120)):
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])


def save(fig, name):
    fig.savefig(PDF / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(PNG / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fmt(v, d=1):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


NUM: dict[str, str] = {}


def num(key, value, d=1):
    NUM[key] = fmt(value, d) if isinstance(value, (float, np.floating, int)) else str(value)


# ============================================================================
ft = add_gap(pd.read_csv(RES / "fssp_test.csv.gz"))
fs_params = json.loads((RES / "fssp_params.json").read_text())["fssp"]
num("kappa", f"{fs_params['kappa']:g}")
num("alpha", f"{fs_params['alpha']:g}")
E = ft[(ft["info"] == "expected") & (ft.model_set == "full")]

# ---------------- Fig: budget sweep ----------------------------------------------------------
lines = [("C:optimal_mckp", "Centralized optimum (MCKP)", S[0], "-", "o"),
         ("D:random_fcfs|fssp", "FSSP (ours), one setting", S[5], "-", "s"),
         ("D:random_fcfs|dsp_calibrated", "Shadow price, tuned per budget", S[3], "--", "^"),
         ("D:random_fcfs|best_model", "Greedy best-model agents", S[1], "-", "v"),
         ("C:cheapest_model", "Cheapest model for all", INK2, ":", None)]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=False)
for ax, mix in zip(axes, MIXES):
    for pol, lab, col, ls, mk in lines:
        g = E[(E.mix == mix) & (E.policy == pol)].groupby("budget_mult").quality
        m, lo, hi = zip(*[boot(g.get_group(b)) for b in MULTS])
        ax.plot(MULTS, m, color=col, ls=ls, marker=mk, ms=3, label=lab, lw=1.6 if pol != "D:random_fcfs|fssp" else 2.2)
        if pol in ("D:random_fcfs|fssp", "D:random_fcfs|best_model"):
            ax.fill_between(MULTS, lo, hi, color=col, alpha=0.15, lw=0)
    o = ft[(ft["info"] == "oracle") & (ft.model_set == "full") & (ft.mix == mix) & (ft.policy == "C:optimal_mckp")].groupby("budget_mult").quality.mean()
    ax.plot(MULTS, o.loc[MULTS].values, color=INK2, lw=0.9, ls=(0, (4, 2)), label="MCKP with oracle knowledge")
    logx(ax)
    ax.set_title(MIX_LABEL[mix])
    ax.set_xlabel(r"Budget per task ($\times$ cheapest call)")
axes[0].set_ylabel("Mean pass@1 (realized)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17), fontsize=7.5)
fig.tight_layout()
save(fig, "fig_budget_sweep")

# ---------------- Fig: design space heatmap (arrival x utility, tight budgets) -------------
main = add_gap(pd.read_csv(RES / "main.csv.gz"))
mE = main[(main["info"] == "expected")]
ARR = ["fcfs", "random_fcfs", "hardest_task_first", "easiest_task_first", "marginal_roi_priority"]
ARR_L = ["Round-robin", "Random", "Hardest-first", "Easiest-first", "ROI priority"]
UT = ["best_model", "marginal_roi_ratio", "quasi_linear", "shapley_budget_share", "dynamic_shadow_price", "fssp"]
UT_L = ["Greedy", "Marg. ratio","Quasi-lin.*", "Shapley*", "Shadow*", "FSSP"]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
for ax, mix in zip(axes, MIXES):
    mat = np.zeros((5, 6))
    for i, a in enumerate(ARR):
        for j, u in enumerate(UT):
            if u == "fssp":
                g = E[(E.mix == mix) & (E.policy == f"D:{a}|fssp") & E.budget_mult.isin(TIGHT)]
            else:
                g = mE[(mE.mix == mix) & (mE.policy == f"D:{a}|{u}") & mE.budget_mult.isin(TIGHT)]
            mat[i, j] = g.gap.mean()
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=90, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(6), UT_L, fontsize=6.5, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(5), ARR_L if mix == "uniform" else [""] * 5, fontsize=7)
    for i in range(5):
        for j in range(6):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="white" if mat[i, j] > 45 else INK)
    ax.add_patch(plt.Rectangle((4.5, -0.5), 1, 5, fill=False, ec=S[5], lw=1.6))
    ax.set_title(MIX_SHORT[mix])
cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.015)
cb.set_label("Gap to optimum (%)", fontsize=7)
save(fig, "fig_design_space")

# ---------------- Fig: calibration (tuning seeds) -------------------------------------------
tu = pd.read_csv(RES / "tune.csv.gz")
tu = tu[(tu.model_set == "full")]
fst = pd.read_csv(RES / "fssp_tune.csv.gz")
mck_t = tu[tu.policy == "C:optimal_mckp"].groupby("budget_mult").quality.mean()
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
ax.plot(MULTS, mck_t.loc[MULTS].values, color=S[0], marker="o", ms=3, label="Centralized optimum (MCKP)")
singles = []
for u, col, lab in [("quasi_linear", S[2], "Quasi-linear"), ("dynamic_shadow_price", S[3], "Shadow price")]:
    pu = tu[(tu.utility == u)].groupby(["params", "budget_mult"]).quality.mean().unstack()
    per_budget = pu.max(axis=0)
    singles.append(pu.loc[pu.mean(axis=1).idxmax()])
    ax.plot(MULTS, per_budget.loc[MULTS].values, color=col, ls="--", marker="^", ms=3, label=f"{lab}, tuned per budget")
# both best single settings collapse to "always cheapest"; check and draw them as one line
assert np.allclose(singles[0].loc[MULTS].values, singles[1].loc[MULTS].values)
ax.plot(MULTS, singles[0].loc[MULTS].values, color=INK2, ls=":", lw=1.4, label="Quasi-linear or shadow price, best single setting")
fsel = fst[(fst.paradigm == "decentral") & (fst.params == json.dumps(fs_params, sort_keys=True))].groupby("budget_mult").quality.mean()
ax.plot(MULTS, fsel.loc[MULTS].values, color=S[5], marker="s", ms=3, lw=2.2, label="FSSP (ours), one setting")
logx(ax)
ax.set_xlabel(r"Budget per task ($\times$ cheapest call)")
ax.set_ylabel("Mean pass@1 (tuning seeds)")
ax.set_title("One FSSP setting matches per-budget tuning")
h, l = ax.get_legend_handles_labels()
ax = axes[1]
fd = fst[fst.paradigm == "decentral"].copy()
fd["kappa"] = fd.params.map(lambda s: json.loads(s)["kappa"])
fd["alpha"] = fd.params.map(lambda s: json.loads(s)["alpha"])
for k, a in enumerate([0, 1, 2]):
    g = fd[fd.alpha == a].groupby("kappa").quality.mean()
    ax.plot(g.index, g.values, color=[S[1], S[5], S[6]][k], marker="o", ms=3, label=fr"$\alpha={a}$" + (" (fixed price)" if a == 0 else ""))
ax.axhline(mck_t.mean(), color=S[0], lw=0.9, ls=(0, (4, 2)), label="MCKP")
ax.set_xscale("log")
ax.set_xlabel(r"Price scale $\eta$")
ax.set_ylabel("Mean pass@1 over all budgets")
ax.set_title(r"Sensitivity to $\eta$ and $\alpha$")
ax.set_ylim(0.6, 0.76)
ax.legend(fontsize=6.5, loc="lower right")
fig.tight_layout()
fig.legend(h, l, loc="lower left", ncol=3, bbox_to_anchor=(0.02, -0.1), fontsize=6.5)
save(fig, "fig_calibration")

# ---------------- Fig: fairness (agent-level random arrival) --------------------------------
bu = add_gap(pd.read_csv(RES / "fssp_burst.csv.gz"))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.9))
for pol, lab, col, mk in [("D:random_agent_burst|best_model", "Greedy best-model", S[1], "v"),
                          ("D:random_agent_burst|ql_calibrated", "Quasi-linear*", S[2], "^"),
                          ("D:random_agent_burst|dsp_calibrated", "Shadow price*", S[3], "D"),
                          ("D:random_agent_burst|fssp", "FSSP (ours)", S[5], "s")]:
    g = bu[bu.policy == pol].groupby("budget_mult")
    axes[0].plot(MULTS, 100 * g.fmr_diff.mean().loc[MULTS].values, color=col, marker=mk, ms=3, label=lab, lw=2.2 if "fssp" in pol else 1.6)
    axes[1].plot(MULTS, g.jain.mean().loc[MULTS].values, color=col, marker=mk, ms=3, label=lab, lw=2.2 if "fssp" in pol else 1.6)
mk = bu[bu.policy == "C:optimal_mckp"].groupby("budget_mult").jain.mean()
axes[1].plot(MULTS, mk.loc[MULTS].values, color=S[0], ls=(0, (4, 2)), lw=1.1, label="MCKP")
for ax in axes:
    logx(ax)
    ax.set_xlabel(r"Budget per task ($\times$ cheapest call)")
axes[0].axhline(0, color=INK2, lw=0.7)
axes[0].set_ylabel("First minus last quartile (pp)")
axes[0].set_title("First-mover advantage")
axes[1].set_ylabel("Jain's index")
axes[1].set_title("Fairness of agent payoffs")
axes[1].legend(fontsize=6.5, loc="lower right")
fig.tight_layout()
save(fig, "fig_fairness")

# ============================================================================ tables
POL_L = {
    "C:optimal_mckp": ("Centralized", "Optimal MCKP (exact ILP)"),
    "C:predictive_router": ("Centralized", "Predictive router"),
    "C:static_equal_quota": ("Centralized", "Static equal quota"),
    "C:cheapest_model": ("Centralized", "Cheapest model"),
    "D:random_fcfs|fssp": ("Decentralized", r"\textbf{FSSP (ours)}, random arrival"),
    "D:easiest_task_first|fssp": ("Decentralized", r"\textbf{FSSP (ours)}, easiest-first"),
    "D:random_fcfs|fssp_alpha0": ("Decentralized", r"FSSP with $\alpha=0$ (fixed price)"),
    "D:random_fcfs|dsp_calibrated": ("Decentralized", r"Shadow price, tuned per budget$^*$"),
    "D:random_fcfs|ql_calibrated": ("Decentralized", r"Quasi-linear, tuned per budget$^*$"),
    "D:random_fcfs|shapley_calibrated": ("Decentralized", r"Shapley share, tuned per budget$^*$"),
    "D:random_fcfs|marginal_roi_ratio": ("Decentralized", "Marginal ratio"),
    "D:random_fcfs|best_model": ("Decentralized", "Greedy best-model"),
}
rows = []
for pol, (par, lab) in POL_L.items():
    g = E[E.policy == pol]
    t = g[g.budget_mult.isin(TIGHT)]
    rows.append((par, lab, g.quality.mean(), g.gap.mean(), t.gap.mean(), 100 * g.task_starve.mean(), g.jain.mean(), 100 * g.budget_util.mean()))
tex = [r"\begin{tabular}{llcccccc}", r"\toprule",
       r" & Method & Mean pass@1 & Gap (\%) & Gap, tight (\%) & Starved tasks (\%) & Jain & Budget used (\%) \\", r"\midrule"]
last = None
for par, lab, q, gp, gt, st, j, bu_ in rows:
    if last and par != last:
        tex.append(r"\midrule")
    tex.append(f"{par if par != last else ''} & {lab} & {q:.3f} & {gp:.1f} & {gt:.1f} & {st:.1f} & {j:.3f} & {bu_:.0f} \\\\")
    last = par
tex += [r"\bottomrule", r"\end{tabular}"]
(GEN / "table_main.tex").write_text(nozero("\n".join(tex)), encoding="utf-8")

fs_all = E[E.policy == "D:random_fcfs|fssp"]
num("fsspQ", fs_all.quality.mean(), 3)
num("mckpQ", E[E.policy == "C:optimal_mckp"].quality.mean(), 3)
num("fsspGap", fs_all.gap.mean(), 1)
m, lo, hi = boot(fs_all.groupby(["mix", "budget_mult", "seed"]).gap.mean())
num("fsspGapLo", lo, 1)
num("fsspGapHi", hi, 1)
num("fsspGapTight", fs_all[fs_all.budget_mult.isin(TIGHT)].gap.mean(), 1)
naive = E[E.policy == "D:random_fcfs|best_model"]
num("naiveGap", naive.gap.mean(), 1)
num("naiveGapTight", naive[naive.budget_mult.isin(TIGHT)].gap.mean(), 1)
num("naiveStarve", 100 * naive.task_starve.mean(), 0)
num("alphaZeroGap", E[E.policy == "D:random_fcfs|fssp_alpha0"].gap.mean(), 1)
num("dspDefaultGap", E[E.policy == "D:random_fcfs|dsp_default"].gap.mean(), 1)
cell = E[E.policy == "D:random_fcfs|fssp"].groupby(["mix", "budget_mult"]).gap.mean()
num("fsspWorstCell", cell.max(), 1)
wm, wb = cell.idxmax()
num("fsspWorstCellWhere", f"{MIX_SHORT[wm].lower()}, budget $\\times{wb:g}$")
dsp_cell = E[E.policy == "D:random_fcfs|dsp_calibrated"].groupby(["mix", "budget_mult"]).gap.mean()
num("dspWorstCell", dsp_cell.max(), 1)
# paired FSSP vs per-budget-tuned shadow price
pv = E[E.policy.isin(["D:random_fcfs|fssp", "D:random_fcfs|dsp_calibrated"])].pivot_table(index=["mix", "budget_mult", "seed"], columns="policy", values="quality")
d = 100 * (pv["D:random_fcfs|fssp"] - pv["D:random_fcfs|dsp_calibrated"])
num("fsspVsDsp", d.mean(), 2)
num("fsspVsDspP", ptex(signflip_p(d.values)))
pv = E[E.policy.isin(["D:random_fcfs|fssp", "D:random_fcfs|best_model"])].pivot_table(index=["mix", "budget_mult", "seed"], columns="policy", values="quality")
d = 100 * (pv["D:random_fcfs|fssp"] - pv["D:random_fcfs|best_model"])
num("fsspVsNaive", d.mean(), 1)
num("fsspVsNaiveP", ptex(signflip_p(d.values)))
# arrival order effect under FSSP
arr_gap = {a: E[E.policy == f"D:{a}|fssp"].gap.mean() for a in ARR}
num("fsspArrMin", min(arr_gap.values()), 1)
num("fsspArrMax", max(arr_gap.values()), 1)
for a in ARR:
    num(f"fsspArr{a.replace('_', '')}", arr_gap[a], 1)
# design-space numbers from the old grid (tight budgets, all mixes)
mt = mE[mE.budget_mult.isin(TIGHT) & (mE.paradigm == "decentral")]
for a in ARR:
    num(f"arr{a.replace('_', '')}", mt[mt.arrival == a].gap.mean(), 1)
for u in ["best_model", "marginal_roi_ratio", "quasi_linear", "shapley_budget_share", "dynamic_shadow_price"]:
    num(f"ut{u.replace('_', '')}", mt[(mt.utility == u) & (mt.arrival == "random_fcfs")].gap.mean(), 1)
cas = mE[mE.policy == "C:confidence_cascade"]
num("cascadeQ", cas.quality.mean(), 3)
num("cascadeGap", -cas.gap.mean(), 1)
num("mainMckpQ", mE[mE.policy == "C:optimal_mckp"].quality.mean(), 3)

# ---------------- robustness table ----------------------------------------------------------
def rob_row(label, df, fssp="D:random_fcfs|fssp"):
    def gp(p):
        return df[df.policy == p].gap.mean()  # NaN (printed as --) if the policy was not run
    return (label, df[df.policy == "C:optimal_mckp"].quality.mean(), df[df.policy == fssp].quality.mean(),
            gp(fssp), gp("D:easiest_task_first|fssp"), gp("D:random_fcfs|dsp_calibrated"), gp("D:random_fcfs|best_model"))


rob = []
for mix in MIXES:
    rob.append(rob_row(f"{MIX_SHORT[mix]} mix (main setting)", E[E.mix == mix]))
st_ = ft[(ft["info"] == "expected") & (ft.model_set == "steep")]
rob.append(rob_row(r"Steep pool \{v3.1, r1, gemini-pro\}", st_))
orc = ft[(ft["info"] == "oracle") & (ft.model_set == "full")]
rob.append(rob_row("Oracle information (true pass/fail known)", orc))
sc = add_gap(pd.read_csv(RES / "fssp_scale.csv.gz"))
for n in [5, 20, 50]:
    rob.append(rob_row(f"$N={n}$ agents (budgets $\\times$3, $\\times$12)", sc[sc.n_agents == n]))
nz = add_gap(pd.read_csv(RES / "fssp_noise.csv.gz"))
for eps in [0.0, 0.2, 0.4]:
    lab = "Correct labels, budgets $\\times$3, 12, 50" if eps == 0 else f"Noisy difficulty labels, $\\epsilon={eps}$"
    rob.append(rob_row(lab, nz[nz.label_noise == eps]))
noise_drop = [100 * (nz[(nz.label_noise == 0) & (nz.policy == p)].quality.mean() - nz[(nz.label_noise == 0.4) & (nz.policy == p)].quality.mean())
              for p in ["C:optimal_mckp", "D:random_fcfs|fssp"]]
num("noiseDropMckp", noise_drop[0], 1)
num("noiseDropFssp", noise_drop[1], 1)
# total number of requests n unknown to the agents (fssp_unknown_n.py)
un = add_gap(pd.read_csv(RES / "fssp_unknown_n.csv.gz"))
UN = "D:random_fcfs|fssp_"
for lab, v in [(r"$n$ unknown: online estimate of $R_k$, no prior", "online_noprior"),
               (r"$n$ unknown: fixed forecast $2n$ (too high)", "forecast_rho=2.0"),
               (r"$n$ unknown: fixed forecast $0.5n$ (too low)", "forecast_rho=0.5")]:
    rob.append(rob_row(lab, un, UN + v))


def un_pol(v):
    return un[un.policy == UN + v]


num("unKnownGap", un_pol("known").gap.mean(), 1)
for v, key in [("online_noprior", "Online"), ("online_prior=0.5", "OnlineLow"), ("online_prior=2", "OnlineHigh"),
               ("forecast_rho=0.5", "Half"), ("forecast_rho=0.75", "ThreeQ"), ("forecast_rho=1.25", "FiveQ"),
               ("forecast_rho=2.0", "Double")]:
    g = un_pol(v)
    num(f"un{key}Gap", g.gap.mean(), 1)
    num(f"un{key}Worst", g.groupby(["mix", "budget_mult"]).gap.mean().max(), 1)
    num(f"un{key}Starve", 100 * g.task_starve.mean(), 1)
num("unOnlineMaxGap", max(un_pol(v).gap.mean() for v in ["online_noprior", "online_prior=0.5", "online_prior=2"]), 1)
tex = [r"\begin{tabular}{lcccccc}", r"\toprule",
       r" & \multicolumn{2}{c}{Mean pass@1} & \multicolumn{4}{c}{Gap to MCKP (\%)} \\",
       r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}",
       r"Setting & MCKP & FSSP & FSSP & FSSP+ETF & Shadow price$^*$ & Greedy \\", r"\midrule"]
for i, (lab, qm, qf, g1, g2, g3, g4) in enumerate(rob):
    if i in (3, 5, 8, 11):
        tex.append(r"\midrule")
    tex.append(f"{lab} & {qm:.3f} & {qf:.3f} & {fmt(g1)} & {fmt(g2)} & {fmt(g3)} & {fmt(g4)} \\\\")
tex += [r"\bottomrule", r"\end{tabular}"]
(GEN / "table_robustness.tex").write_text(nozero("\n".join(tex)), encoding="utf-8")
oracle_cell = orc[orc.policy == "D:random_fcfs|fssp"].groupby(["mix", "budget_mult"]).gap.mean()
num("oracleWorst", oracle_cell.max(), 1)
om, ob = oracle_cell.idxmax()
num("oracleWorstWhere", f"{MIX_SHORT[om].lower()}, budget $\\times{ob:g}$")
num("oracleGap", orc[orc.policy == "D:random_fcfs|fssp"].gap.mean(), 1)
num("steepNaive", st_[st_.policy == "D:random_fcfs|best_model"].gap.mean(), 1)
num("steepFssp", st_[st_.policy == "D:random_fcfs|fssp"].gap.mean(), 1)
info_gain = (orc[orc.policy == "C:optimal_mckp"].groupby("mix").quality.mean() - E[E.policy == "C:optimal_mckp"].groupby("mix").quality.mean()) * 100
num("infoGainMin", info_gain.min(), 0)
num("infoGainMax", info_gain.max(), 0)

# fairness numbers
for pol, key in [("D:random_agent_burst|best_model", "naive"), ("D:random_agent_burst|fssp", "fssp"), ("D:random_agent_burst|dsp_calibrated", "dsp")]:
    g = bu[(bu.policy == pol) & bu.budget_mult.isin(TIGHT)]
    num(f"fm{key}", 100 * g.fmr_diff.mean(), 1)
    num(f"jain{key}", g.jain.mean(), 2)
num("jainmckp", bu[(bu.policy == "C:optimal_mckp") & bu.budget_mult.isin(TIGHT)].jain.mean(), 2)

# ---------------- model table ----------------------------------------------------------------
bench = H.load_bench()
tex = [r"\begin{tabular}{lcccc}", r"\toprule", r"Model & Cost per call (\$) & pass@1 easy & pass@1 medium & pass@1 hard \\", r"\midrule"]
for m in H.MODEL_SETS["full"]:
    p = [bench.scores[m][bench.tiers == t].mean() for t in range(3)]
    tex.append(f"\\texttt{{{m}}} & {bench.costs[m]:.4f} & {p[0]:.2f} & {p[1]:.2f} & {p[2]:.2f} \\\\")
tex += [r"\bottomrule", r"\end{tabular}"]
(GEN / "table_models.tex").write_text("\n".join(tex), encoding="utf-8")

# ---------------- run counts -----------------------------------------------------------------
total = 0
for f in ["tune", "main", "scale", "disparity", "noise", "burst", "fssp_tune", "fssp_test", "fssp_scale", "fssp_noise", "fssp_burst", "fssp_unknown_n"]:
    total += len(pd.read_csv(RES / f"{f}.csv.gz", usecols=["policy"]))
num("totalRuns", f"{total:,}".replace(",", "{,}"))

(GEN / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\n{k}}}{{{v}}}" for k, v in NUM.items()) + "\n", encoding="utf-8")
print(json.dumps(NUM, indent=1))
