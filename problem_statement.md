# Problem Statement: Multi-Agent Model/API Selection Under a Shared Budget

---

## 1. Executive Summary

Modern AI applications increasingly rely on ensembles of autonomous agents to handle complex workflows (e.g., automated software engineering, customer support pipelines, and data analysis). These agents have access to a diverse pool of Large Language Models (LLMs) and APIs with starkly differing reasoning capabilities and financial/token costs (from lightweight models like GPT-4o-mini and Claude 3.5 Haiku to premium reasoning engines like Claude 3.5 Sonnet and OpenAI o1).

In practical deployments, multi-agent systems operate under a **shared, finite financial or token budget ($B$)**. When autonomous agents make independent, uncoordinated model choices to maximize their individual task outcomes, a fundamental **"Tragedy of the Commons"** emerges:
* **Early-arriving agents** exploit the abundant budget to select the most expensive models, even for simple tasks where cheaper models would suffice.
* **Late-arriving agents** face an exhausted budget and are forced to use inadequate fallback models or fail completely, even when tackling mission-critical, high-difficulty tasks.

This research project formalizes and investigates this dilemma as a **multi-agent resource allocation problem**. We study the fundamental efficiency gap (**Price of Anarchy**) between **centralized optimal allocation** and **decentralized sequential games**, and investigate how intelligent queuing policies and game-theoretic utility designs can close this gap.

---

## 2. Core Problem & Real-World Motivation

### The Dilemma: Who Gets to Claim the Shared Budget?
Consider a multi-agent continuous integration (CI/CD) pipeline with a daily budget $B = \$10$:
* **Scenario A (Naive Decentralized):** A documentation-formatting agent arrives first at 9:00 AM. Seeing $\$10$ available, it greedily calls a frontier model costing $\$8$. At 2:00 PM, a critical concurrency bug-fixing agent arrives, but finds only $\$2$ remaining. It is forced to use a cheap model that fails to diagnose the bug.
* **Scenario B (Socially Coordinated):** A coordinator assigns the $\$0.50$ model to documentation (achieving 98% quality) and reserves the $\$8.00$ model for the difficult bug fix (achieving 95% quality), maximizing overall system reliability within the exact same $\$10$ budget.

While centralized coordination maximizes total quality, pure centralization introduces major practical drawbacks:
1. **Asynchronous Reality:** In real-world systems, tasks arrive dynamically and unpredictably over time rather than in a single synchronous batch.
2. **Bottleneck & Overhead:** Centralized integer programming requires complete upfront knowledge of all tasks, creating latency and a single point of failure.

**Core Research Challenge:**  
*How can we design decentralized decision-making rules, queue policies, and local agent utility functions such that autonomous, uncoordinated agents achieve aggregate system quality that closely approaches the centralized theoretical optimum?*

---

## 3. Formal Mathematical Formulation

### 3.1 System Components
* **Agents:** A set of $N$ autonomous agents $\mathcal{N} = \{1, 2, \dots, N\}$.
* **Tasks:** Each agent $i \in \mathcal{N}$ is assigned a task $t_i$ characterized by difficulty level $d_i \in \{\text{Easy}, \text{Medium}, \text{Hard}\}$.
* **Model Pool:** A set of $M$ AI models $\mathcal{M} = \{1, 2, \dots, M\}$.
* **Costs:** Each model $m \in \mathcal{M}$ incurs an inference cost $c_m > 0$.
* **Expected Quality:** $q_{i,m} \in [0, 1]$ denotes the expected quality or task success rate when agent $i$ executes task $t_i$ using model $m$.
* **Shared Budget:** A fixed global budget constraint $B > 0$.

An allocation vector is denoted as $\mathbf{m} = (m_1, m_2, \dots, m_N)$, where $m_i \in \mathcal{M}$.  
An allocation is **feasible** if and only if the total expenditure does not exceed the shared budget:
$$\sum_{i=1}^{N} c_{m_i} \leq B$$

---

### 3.2 The Two Allocation Paradigms

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Shared Budget Pool (B)                          │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
     [ Centralized Paradigm ]        [ Decentralized Sequential Game ]
                    │                                │
                    ▼                                ▼
       Global Knapsack Controller           Asynchronous Arrival Sequence
           (Solves MCKP ILP)                    π = (π₁, π₂, ..., πN)
                    │                                │
       Optimal Social Allocation         Agents check remaining budget Bk
         (m₁*, m₂*, ..., mN*)            and pick local best-response m*
```

#### Paradigm 1: Centralized Optimal Allocation (Global Knapsack Benchmark)
The centralized controller has omniscient visibility over all tasks $\{t_1, \dots, t_N\}$, model costs, and expected performance profiles. It solves the **Multiple-Choice Knapsack Problem (MCKP)**:

$$\max_{\mathbf{m}} \sum_{i=1}^{N} q_{i, m_i} \quad \text{subject to} \quad \sum_{i=1}^{N} c_{m_i} \leq B, \quad m_i \in \mathcal{M}$$

This provides the **theoretical upper bound** of achievable social welfare under the budget constraint.

#### Paradigm 2: Decentralized Sequential Resource-Allocation Game
Agents arrive asynchronously according to an arrival ordering $\pi = (\pi_1, \pi_2, \dots, \pi_N)$, where $\pi_k$ denotes the agent making a decision at step $k \in \{1, \dots, N\}$:

1. **Remaining Budget State:** At step $k$, the available budget is:
   $$B_k = B - \sum_{j=1}^{k-1} c_{m_{\pi_j}}, \quad \text{with } B_1 = B$$
2. **Feasible Model Set:** Agent $\pi_k$ can only select models it can afford:
   $$\mathcal{M}(B_k) = \{ m \in \mathcal{M} \mid c_m \leq B_k \}$$
   If $B_k < \min_{m \in \mathcal{M}} c_m$, the agent is starved and receives a zero-quality fallback action.
3. **Autonomous Best Response:** Agent $\pi_k$ selects the model maximizing its individual utility function:
   $$m_{\pi_k} = \arg\max_{m \in \mathcal{M}(B_k)} U_{\pi_k}(m)$$
4. **Immediate Deduction:** The selected model's cost is deducted immediately ($B_{k+1} = B_k - c_{m_{\pi_k}}$), shaping the constraints of all subsequent agents.

---

## 4. Key Research Questions

1. **Quantifying the Efficiency Gap (Price of Anarchy):**  
   How large is the gap between decentralized sequential selection and the centralized optimal knapsack solution across varying budget pressures?
   $$\text{Gap} = \frac{SW_{\text{centralized}} - SW_{\text{decentralized}}}{SW_{\text{centralized}}}$$
2. **Arrival Ordering & Position Privilege:**  
   How severely does arrival ordering ($\pi$) distort outcomes? To what degree does a random arrival order cause positional inequality (first-mover advantage) and late-agent starvation?
3. **Queue Scheduling Policies:**  
   Can simple, lightweight queue re-ordering (e.g., Hardest-Task-First or Marginal ROI Priority) bridge the gap to centralized performance while preserving agent autonomy?
4. **Utility Function Engineering:**  
   Can replacing naive quasi-linear utility ($Q - \lambda C$) with scarcity-adaptive shadow pricing, Shapley-value budget shares, or submodular marginal yield ($\frac{\Delta U}{\Delta C}$) induce agents to self-regulate and approach centralized optimality?
5. **Task Heterogeneity & Budget Tightness:**  
   Under what combinations of task difficulty mix and budget constraint does decentralized decision-making experience catastrophic failure versus near-optimal efficiency?

---

## 5. Project Tasks & Deliverables

To answer the research questions, the project is structured into five concrete implementation tasks:

| Task ID | Task Title | Description & Deliverables |
| :--- | :--- | :--- |
| **Task 1** | **Simulation Environment & Benchmark Construction** | Build a controlled simulation framework modeling $N$ agents, $M$ model tiers, and software engineering tasks (code generation, bug fixing, test generation) with realistic performance and cost parameters (e.g., HumanEval/MBPP performance distributions). |
| **Task 2** | **Centralized Baselines Implementation** | Implement the exact MCKP solver (using Integer Linear Programming via Google OR-Tools) as the optimal upper bound, along with heuristic central baselines (equal quota, predictive routing). |
| **Task 3** | **Decentralized Sequential Game Simulator** | Implement the sequential arrival engine supporting arbitrary agent permutations $\pi$, tracking residual budget $B_k$, enforcing feasibility, and logging individual payoffs. |
| **Task 4** | **Mechanism & Utility Function Design** | Implement and test alternative decentralized decision rules: (1) Baseline Linear Utility, (2) Scarcity-Adaptive Shadow Pricing ($p(B_k)$), (3) Shapley Value Budget Entitlements, and (4) Submodular Marginal ROI ($\frac{\Delta U}{\Delta C}$). |
| **Task 5** | **Empirical Evaluation & Comparative Analysis** | Run full factorial experiments across budget levels, agent counts, and difficulty mixtures. Compute quality, cost, social welfare, Jain's fairness index, starvation rates, and price of anarchy. |
