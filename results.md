# Empirical Evaluation of Multi-Agent Budgeted Model Selection on LiveCodeBench

This report details the experimental evaluation of multi-agent LLM routing under hard shared budget constraints using the genuinely measured LiveCodeBench dataset from LLMRouterBench across all 6 flagship models.

---

## 1. Experimental Setup

- **Benchmark Dataset:** `data/llmrouterbench_livecodebench.json` (and associated difficulty mixes), derived from the official `NPULH/LLMRouterBench` repository (`livecodebench/test`, 1,055 tasks).
- **Environment:**
  - $N = 10$ autonomous agents sharing a fixed API budget $B$.
  - 20 tasks per agent ($200$ total tasks per run, drawn without replacement).
  - 5 seeded runs per configuration (seeds 42–46).
- **Model Pool & Observed Token-Based Pricing:**
  1. `deepseek-v3.1-terminus`: Mean cost **\$0.0011249/call** (Pass@1: 67.30%)
  2. `deepseek-v3-0324`: Mean cost **\$0.0014471/call** (Pass@1: 66.64%)
  3. `gemini-2.5-flash`: Mean cost **\$0.0058343/call** (Pass@1: 60.00%)
  4. `claude-sonnet-4`: Mean cost **\$0.0134617/call** (Pass@1: 58.10%)
  5. `deepseek-r1-0528`: Mean cost **\$0.0288723/call** (Pass@1: 80.09%)
  6. `gemini-2.5-pro`: Mean cost **\$0.1351539/call** (Pass@1: 79.15%)

---

## 2. Experimental Results

The results are divided into two distinct tables representing the centralized allocation benchmark and the decentralized sequential game mechanisms.

### Table 2.1: Centralized Allocation Benchmark (Optimal MCKP ILP Solver)

The central controller has complete global visibility of all 200 tasks simultaneously and solves an exact Multiple-Choice Knapsack Problem (ILP) to globally maximize collective quality subject to $\sum c_m \le B$.

| Task Difficulty Mix | Budget Limit ($B$) | Tasks Solved / 200 | Average Quality ($\bar{q}$) | Total Cost (USD) | Budget Utilization | Jain's Fairness | Agent Starvation Rate | Unassigned Tasks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uniform Random** *(33% E / 33% M / 33% H)* | \$0.28942 | 151 / 200 | **0.7550** | \$0.2683 | 92.71% | **0.9798** | 90% | 49 |
| **Easy-Heavy (80/20)** *(80% E / 10% M / 10% H)* | \$0.28942 | 190 / 200 | **0.9500** | \$0.2702 | 93.36% | **0.9983** | 70% | 10 |
| **Hard-Heavy (20/80)** *(10% E / 10% M / 80% H)* | \$1.16686 | 123 / 200 | **0.6150** | \$1.1350 | 97.27% | **0.9655** | 100% | 77 |

---

### Table 2.2: Decentralized Allocation (Sequential Game Mechanisms)

Agents arrive sequentially and observe only the remaining shared budget $B_k$, selecting affordable models based on their local utility function. Metrics represent the mean across 5 seeded arrival permutations ($\pm$ standard deviation).

| Task Difficulty Mix | Arrival Policy | Utility Function | Budget Limit ($B$) | Tasks Solved / 200 | Average Quality ($\bar{q}$) | Total Cost (USD) | Budget Utilization | Jain's Fairness | Agent Starvation Rate | Optimality Gap (vs. MCKP) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uniform Random** | Random FCFS | Dynamic Shadow Price | \$0.28942 | ~67 / 200 | **0.3340** $\pm$ 0.088 | \$0.2889 $\pm$ 0.0003 | 99.83% | 0.8944 $\pm$ 0.036 | 100% (94–147 starved) | **55.76%** $\pm$ 11.66% *(Price of Anarchy)* |
| **Easy-Heavy (80/20)** | **Easiest-Task-First (ETF)** | **Quasi-Linear** ($\lambda=0.5$) | \$0.28942 | 190 / 200 | **0.9500** $\pm$ 0.000 | \$0.2814 $\pm$ 0.0000 | 97.25% | **0.9978** $\pm$ 0.000 | **0%** (0 starved) | **0.00%** *(Zero Welfare Loss)* |
| **Hard-Heavy (20/80)** | Hardest-Task-First (HTF) | Dynamic Shadow Price | \$1.16686 | 48 / 200 | **0.2400** $\pm$ 0.000 | \$1.1661 $\pm$ 0.0000 | 99.93% | 0.9216 $\pm$ 0.000 | 100% (129 starved) | **60.98%** *(Reasoning Deficit)* |

---

## 3. Comparative Analysis & Key Findings

### Finding 1: The Price of Anarchy is Real (55.76% Quality Loss)
Under the **Uniform difficulty mix** with moderate budget pressure (\$0.289 for 200 tasks), uncoordinated decentralized agents arriving randomly suffer an **optimality gap of 55.76%**. Early-arriving agents selfishly select mid-tier models, exhausting the budget prematurely and causing **94 to 147 out of 200 tasks to starve completely**. The centralized MCKP solver, seeing all tasks globally, achieves more than double the quality (**0.7550 vs. 0.3340**).

### Finding 2: Decentralized Mechanism Design Can Match Centralized Optimum (0.00% Gap)
On the **Easy-Heavy mix** (80% easy, 10% medium, 10% hard—typical of real-world software engineering where routine tasks outnumber complex logic), the decentralized **Easiest-Task-First (ETF) arrival policy with Quasi-Linear utility** achieves:
- **0.00% Optimality Gap:** Identical average quality to the global ILP optimum (**0.9500**, or 190/200 tasks solved).
- **Superior Fairness & Zero Starvation:** Starves **0% of agents** (0 out of 10) compared to **70% of agents** (7 out of 10) experiencing at least one starved job under the centralized solver.
- **Near-Perfect Fairness:** Jain's fairness index of **0.9978**.

This demonstrates that simple decentralized queue ordering and cost sensitivity can eliminate welfare loss without requiring an omniscient central controller.

### Finding 3: Centralized Coordination is Essential for Frontier Reasoning Bottlenecks (60.98% Gap)
On the **Hard-Heavy mix** (80% hard problems), only the frontier reasoning models (`deepseek-r1-0528`, `gemini-2.5-pro`) possess sufficient capability on complex unit tests. Under decentralized execution, agents burn budget rapidly on early hard tasks, causing an optimality gap of **60.98%** (quality **0.2400** vs. **0.6150**). Centralized MCKP knapsack optimization strategically allocates expensive models to the tasks with the highest marginal return on investment, solving 123 tasks vs. only 48 decentralized.

---

## 4. Verification & Audit Suite

1. **Automated Unit & Integration Tests:**
   `pytest` passes **230 out of 230 tests** cleanly, validating MCKP solvers, budget ledger conservation, arrival policies, and dataset loaders.
2. **Exhaustive Source-Trace Audit:**
   An exhaustive audit of **6,330 out of 6,330 task-model pairs** verified that all scores and costs match the official raw Hugging Face archive records identically with zero mismatches or synthetic interpolations.

---

## 5. Reproduction

To reproduce these experiment results from the repository root:

```bash
# Run all LiveCodeBench benchmark experiments
python -m masbudget configs/llmrouterbench_livecodebench.yaml

# View side-by-side policy comparison table in Python
python -c "
from masbudget.analysis import PolicyComparison
cmp = PolicyComparison.from_config('configs/llmrouterbench_livecodebench.yaml')
print(cmp.table(['average_quality', 'total_cost', 'budget_utilization', 'jains_fairness', 'optimality_gap', 'starvation_rate']))
"
```
