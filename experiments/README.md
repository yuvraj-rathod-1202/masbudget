# Experiment study

Experiments for the project proposal, run through the `masbudget` framework on the real LLMRouterBench /
LiveCodeBench data. The overall findings are in [../results.md](../results.md).

| # | Note | Question | Script | Notebook |
|---|---|---|---|---|
| E1 | [E01_data_profile.md](E01_data_profile.md) | What does the problem look like on real data? | `e01_data_profile.py` | `01_benchmark_profile` |
| E2 | [E02_price_of_anarchy.md](E02_price_of_anarchy.md) | Gap of naive decentralized agents vs. MCKP (RQ1) | `e02_price_of_anarchy.py` | `02_price_of_anarchy_and_ordering` |
| E3 | [E03_arrival_ordering.md](E03_arrival_ordering.md) | Arrival order, first-mover advantage, queue policies (RQ2-3) | `e03_arrival_ordering.py` | `02_...` |
| E4 | [E04_utility_design.md](E04_utility_design.md) | Utility design U1-U4 vs. pacing price (RQ4) | `e04_utility_design.py` | `03_mechanism_design` |
| E5 | [E05_pacing_robustness.md](E05_pacing_robustness.md) | Robustness of the mechanism; final setting | `e05_pacing_robustness.py`, `e05d_price_floor.py` | `03_...` |
| E6 | [E06_factorial.md](E06_factorial.md) | Agents × mix × budget × cost disparity (RQ5) | `e06_factorial.py`, `e06b_no_history.py` | `03_...` |
| E7 | [E07_information.md](E07_information.md) | Information regimes, noisy difficulty labels | `e07_information.py` | `04_information_and_learning` |
| E8 | [E08_learning.md](E08_learning.md) | Learning agents with unknown qualities (UCB) | `e08_learning.py` | `04_...` |
| E9 | [E09_no_sequencing.md](E09_no_sequencing.md) | How well can agents do without the sequencing layer (all decide at once)? | `e09_no_sequencing.py` | - |

## Method in one paragraph

The 1,055 tasks are split once (stratified by difficulty, seed 2026) into a **calibration split** (317 tasks:
estimates q̂, tunes parameters, calibrates prices) and an **evaluation split** (738 tasks: every reported
number). An *instance* is a seeded sample of J tasks from a split, in a given difficulty mix, dealt to N agents.
Budgets are set by tightness τ: B = J·c_min + τ·J·(c_max − c_min) (τ = 0 is the proposal's "severe",
τ = 1 its "generous"). Policies see q_{i,m} under an **information regime**: `oracle` (realized outcome),
`difficulty` (calibrated per-label pass rate; the default) or `blind`. Outcomes are always also scored on the
realized pass@1. The gap is (SW_MCKP − SW)/SW_MCKP on expected quality, paired per instance. Tables report means over
instances (random arrivals averaged within an instance first) with 95% CIs in the summary CSVs.

## Framework additions used by the study

| Where | What |
|---|---|
| `decentralized/sequential.py` | `params: {reserve: true}`: a call may not leave less than c_min per waiting job. `params: {abstain: true}`: an agent may decline its call. |
| `decentralized/utilities/budget_pacing.py` | `budget_pacing` utility, U = q − p_k·c with p_k = max(p0, p_min)·exp(β(spent share − elapsed share)), plus `calibrate_price()` (λ\* of the MCKP relaxation). |
| `decentralized/utilities/ucb_pacing.py` | `ucb_pacing`: learning agents (shared UCB pass-rate estimates + pacing price). |
| `decentralized/utilities/shapley_budget_share.py` | φ cached per agent (was recomputed on every call); `characteristic: upgrade_gain` option. |
| `centralized/confidence_cascade.py` | `verify_key` (verify on realized outcomes), `skip_dominated`, `escalation: jump`. |
| `centralized/optimal_mckp.py` | `serve_all: true`: optimum among allocations that serve every job. |
| `decentralized/simultaneous.py` | `simultaneous_best_response`: the game without sequencing. All jobs decide at once from B, J and costs; `settlement: ration \| downgrade`, `allowance: job`. |
| `decentralized/utilities/lottery_price.py` | `lottery_price`: posted price with a per-job coin flip, plus `calibrate_lottery()`. |

Tests: `tests/test_study_extensions.py`, `tests/test_no_sequencing.py` (all 297 tests pass).

## Reproduce

From the repo root, with the project venv (about 35 minutes on 12 cores in total):

```bash
cd experiments
python e01_data_profile.py
python e02_price_of_anarchy.py
python e04_utility_design.py          # E3 uses the E4 tuning results (lambda = 30)
python e03_arrival_ordering.py
python e05_pacing_robustness.py
python e05d_price_floor.py calib && python e05d_price_floor.py
python e06_factorial.py && python e06b_no_history.py
python e07_information.py
python e08_learning.py
python e09_no_sequencing.py
python derive.py && python summary_figure.py && python build_notebooks.py
```

Each script writes `results/<exp>_raw.csv` (every run; git-ignored, regenerate as above),
`results/<exp>_summary.csv` (committed) and `figures/<exp>.png`, and prints its tables (`results/*_stdout.txt`).
`python export_datasets.py` writes `data/livecodebench_difficulty_eval.json` for
[`configs/final_mechanism.yaml`](../configs/final_mechanism.yaml), which reruns the headline comparison
with `python -m masbudget`.
