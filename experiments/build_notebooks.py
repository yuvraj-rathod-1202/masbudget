"""Build and execute the analysis notebooks in notebooks/ from the saved summaries.

    python experiments/build_notebooks.py
"""

from __future__ import annotations

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from lib import ROOT

SETUP = """import os, sys
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "masbudget").is_dir() else Path.cwd().parent
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
import pandas as pd
from IPython.display import Image, display
from lib import load, md_table
import plots
plots.style()
pd.set_option("display.width", 200, "display.max_columns", 30, "display.precision", 4)
%matplotlib inline"""


def nb(cells):
    book = new_notebook()
    book.cells = [new_code_cell(SETUP)] + [new_markdown_cell(c[3:]) if c.startswith("md:") else new_code_cell(c) for c in cells]
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return book


NOTEBOOKS = {
    "01_benchmark_profile.ipynb": [
        "md:# E1 - Benchmark profile (LiveCodeBench via LLMRouterBench)\n\nWhat the allocation problem looks like before any policy runs: the model pool, "
        "the per-task outcome structure, the calibrated expected qualities and the shadow price of budget. "
        "Produced by `experiments/e01_data_profile.py`; notes in `experiments/E01_data_profile.md`.",
        "md:## Model pool\nGemini-2.5-Flash and DeepSeek-V3-0324 are dominated on average (more expensive than V3.1, not better). "
        "R1 costs 25.7x the cheapest model for +13 points of pass@1.",
        "models = load('e01_models'); models",
        "md:## Which models solve which tasks\nPattern = solved by (V3.1, V3, Flash, R1). Easy tasks are solved by everyone; "
        "178 hard tasks by no one. In the **oracle** regime an agent therefore knows in advance whether a call will pass.",
        "load('e01_patterns')",
        "md:## Calibrated expected quality (difficulty regime)\nPass rates on the 317-task calibration split, per difficulty label. "
        "These are what agents and controllers see as q in the realistic regime.",
        "import lib\nqt = lib.quality_tables()\nq = pd.DataFrame(qt['difficulty']).T.rename(columns=lib.SHORT)\nq.loc['blind'] = pd.Series(qt['blind']).rename(lib.SHORT)\nq",
        "md:### Upgrade value per dollar (the quantities that decide every allocation)",
        "c = lib.MEASURED_COSTS\nrows = []\nfor lv, row in qt['difficulty'].items():\n    for m in ('deepseek-v3-0324', 'deepseek-r1-0528'):\n"
        "        dq = row[m] - row['deepseek-v3.1-terminus']; dc = c[m] - c['deepseek-v3.1-terminus']\n"
        "        rows.append({'difficulty': lv, 'upgrade': 'V3.1 -> ' + lib.SHORT[m], 'dq': dq, 'dc_usd': dc, 'dq_per_usd': dq / dc})\npd.DataFrame(rows)",
        "md:## Shadow price of budget lambda*(tau)\nThe Lagrangian price of the MCKP relaxation. In the difficulty regime it is a *step function*: "
        "flat at 11.56 pass@1/$ (= the ROI of V3.1 -> R1 on hard tasks) for tau in [0.02, 0.33]. At that price every hard task is "
        "indifferent, so a static posted price cannot implement the optimum - a price that moves with spending can.",
        "pc = load('e01_price_curve'); pc[pc.pricing == 'measured'].pivot(index='tau', columns='regime', values='lambda_star')",
        "display(Image('experiments/figures/e01_profile.png'))",
    ],
    "02_price_of_anarchy_and_ordering.ipynb": [
        "md:# E2 + E3 - Price of anarchy and arrival ordering\n\nE2: centralized controllers vs. naive decentralized agents over a budget sweep "
        "(difficulty and oracle regimes). E3: five arrival policies x four agent behaviours. Notes in `experiments/E02_*.md`, `E03_*.md`.",
        "s = load('e02_summary')\ngap = s.pivot_table(index='policy', columns=['regime', 'tau'], values='gap_exp')\n(100 * gap['difficulty']).round(1)",
        "md:**Tragedy of the commons.** At moderate budget (tau = 0.2) agents that pick their best affordable model lose 60% of the optimal "
        "welfare; the repo-default utilities (lambda = 0.5, p0 = 0.1) are numerically identical to best-model because costs are ~1e-3 USD.",
        "cols = ['policy', 'q_exp', 'q_real', 'gap_exp', 'task_starve', 'jain_real', 'fma_jobs', 'q_real_easy', 'q_real_medium', 'q_real_hard']\n"
        "s[(s.regime == 'difficulty') & (s.tau == 0.2)][cols].sort_values('gap_exp')",
        "md:Stratified by difficulty: naive agents starve **easy** tasks (q_real_easy = 0.37 vs 1.0 for MCKP) - the budget is gone before "
        "trivial tasks arrive. The verified cascade has a *higher realized* pass rate than MCKP (0.78 vs 0.73) because unit-test "
        "verification reveals information that q-hat does not have.",
        "display(Image('experiments/figures/e02_price_of_anarchy.png'))",
        "md:## E3 - Arrival ordering",
        "a = load('e03_summary')\nd = a[(a.tpa == 10) & (a.paradigm == 'decentralized')].copy()\n"
        "d[['arrival', 'behaviour']] = d.policy.str.split(' \\\\| ', expand=True)\n"
        "(100 * d.pivot_table(index=['tau', 'arrival'], columns='behaviour', values='gap_exp')).round(1)",
        "md:Hardest-first is the *worst* order for greedy agents (86% gap: hard tasks drain the budget, easy tasks starve) and the "
        "*best* order with a reserve (1.1%). Pacing assumes spending can be spread over time, so it is best under uncoordinated "
        "(random / round-robin) arrival and loses a few points under sorted queues.",
        "d[d.tau == 0.2].pivot_table(index='arrival', columns='behaviour', values='sw_sd').round(3)",
        "md:Order sensitivity (SD of social welfare across permutations, random arrival) falls from 2.76 to 0.14 with pacing + reserve.",
        "display(Image('experiments/figures/e03_arrival_ordering.png'))",
    ],
    "03_mechanism_design.ipynb": [
        "md:# E4-E6 - Utility design, the pacing mechanism and the full factorial\n\nNotes in `experiments/E04_*.md` - `E06_*.md`.",
        "md:## E4 - Tuned utilities (tuned at tau = 0.2 on the calibration split, evaluated on the evaluation split)",
        "b = load('e04b_eval_summary')\np = b.pivot(index='policy', columns='tau', values='gap_exp')\np['worst'] = p.max(axis=1)\n(100 * p.sort_values('worst')).round(2)",
        "md:Fixed-parameter utilities tuned at one budget fail at others unless the ledger reserve stops starvation; "
        "the pacing utility is within 0.5% of optimal at every budget (1.2% without reserve).",
        "c = load('e04c_lambda_sweep_summary')\npv = c.pivot(index='policy', columns='tau', values='gap_exp')\n(100 * pv).round(1)",
        "display(Image('experiments/figures/e04_utility_design.png'))",
        "md:## E5 - Robustness of the pacing price\nA: misspecified reference price; B: difficulty-mix shift; C: wrong horizon; D: price floor.",
        "a = load('e05a_summary')\n(100 * a[(a.tau == 0.2)].pivot_table(index='scale', columns=['res', 'beta'], values='gap_exp')).round(2)",
        "bb = load('e05b_summary')\np = bb.pivot_table(index='policy', columns='mix', values='gap_exp', aggfunc='max')\n(100 * p.sort_values('hard_heavy')).round(2)",
        "dd = load('e05d_summary_calib')\nw = dd.groupby(['policy', 'mix']).gap_exp.max().unstack()\nw['worst'] = w.max(axis=1)\n(100 * w.sort_values('worst')).round(2).head(10)",
        "md:Selected on the calibration split: beta = 10, floor 0.1 / (c_max - c_min), reserve on.",
        "display(Image('experiments/figures/e05_pacing_robustness.png'))",
        "md:## E6 - Factorial: 4 agent counts x 3 mixes x 5 budgets x 3 cost disparities (180 cells)",
        "f = load('e06_summary')\nagg = f.groupby('policy').agg(mean_gap=('gap_exp', 'mean'), worst_gap=('gap_exp', 'max'),\n"
        "    gap_vs_serve_all=('gap_sa_exp', 'mean'), starve=('task_starve', 'mean'), jain=('jain_exp', 'mean'))\n(agg.sort_values('mean_gap') * [100, 100, 100, 100, 1]).round(2)",
        "md:**Decomposition of the residual gap.** Against the best allocation that serves every agent (`MCKP (serve all)`), pacing + reserve "
        "is within 0.23% on average; the rest of its 2.1% gap to MCKP is the *sacrifice gap*: under flat pricing the optimum refuses "
        "service to low-value hard tasks to fund R1 elsewhere. Letting agents abstain when their surplus at the current price is "
        "negative recovers most of it (0.7% mean gap).",
        "for key in ('N', 'mix', 'tau', 'pricing'):\n    display((100 * f[f.policy.isin(['Best model', 'Best model + reserve', 'Pacing + reserve (ours)', "
        "'Pacing + reserve + abstain (ours)', 'Predictive router', 'MCKP (serve all)'])].pivot_table(index='policy', columns=key, values='gap_exp')).round(2))",
        "display(Image('experiments/figures/e06_factorial_heatmap.png'))",
        "nh = load('e06b_summary')\n(100 * nh.groupby('policy').gap_exp.agg(['mean', 'max'])).round(2)",
        "md:The historical price only matters with abstention: plain pacing + reserve does as well starting from the floor price alone.",
        "display(Image('experiments/figures/headline.png'))",
    ],
    "04_information_and_learning.ipynb": [
        "md:# E7 + E8 - Information and learning\n\nNotes in `experiments/E07_*.md`, `E08_*.md`.",
        "s = load('e07_summary')\n(s[s.tau == 0.2].pivot(index='policy', columns='condition', values='q_real')).round(3)",
        "md:Realized pass@1 at tau = 0.2. Per-task (oracle) knowledge is worth ~11 points; coordination with difficulty-level knowledge "
        "is worth less than that. Within every information regime the pacing mechanism stays within 1% of that regime's MCKP:",
        "(100 * s.pivot_table(index='policy', columns='condition', values='gap_exp', aggfunc='max')).round(2)",
        "display(Image('experiments/figures/e07_information.png'))",
        "md:## E8 - Learning agents (qualities unknown, learned online from call outcomes)",
        "l = load('e08_summary')\nl[l.tau == 0.2][['J', 'policy', 'q_exp', 'q_real', 'gap_exp', 'regret_successes']]",
        "md:Regret per job falls as the horizon grows (J = 200 -> 500); a light exploration bonus is best. On realized pass@1 the learners "
        "match agents given calibrated q-hat at tight budgets, since they learn from the current batch.",
        "pos = load('e08_position')\npos[(pos.tau == 0.2)].set_index(['J', 'policy']).round(3)",
        "display(Image('experiments/figures/e08_learning.png'))",
    ],
}

if __name__ == "__main__":
    out_dir = ROOT / "notebooks"
    for name, cells in NOTEBOOKS.items():
        book = nb(cells)
        NotebookClient(book, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(out_dir)}}).execute()
        nbformat.write(book, out_dir / name)
        print("wrote", name)
