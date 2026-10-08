# Experimental Plan: Centralized vs. Decentralized Multi-Agent Model Selection

This document outlines the experimental design, combinations, and configurations for comparing centralized controllers and decentralized mechanisms under a shared API/model budget constraint.

---

## 1. Problem Setup & Environment

* **Agents ($N$):** A set of $N$ autonomous agents $\{1, \dots, N\}$. Each agent $i$ is assigned a specific software engineering task $t_i$.
* **Model Pool ($\mathcal{M}$):** A set of $M$ AI models with diverse capabilities and costs:
  * *Low-tier (Cheap/Fast):* e.g., GPT-4o-mini, Claude 3.5 Haiku, Llama 3 8B.
  * *Mid-tier:* e.g., GPT-4o, Claude 3.5 Sonnet, Llama 3 70B.
  * *High-tier (Frontier/Reasoning):* e.g., OpenAI o1 / o3-mini, Claude Opus.
* **Shared Budget ($B$):** A hard global spending limit for the entire batch/run ($\sum_{i=1}^N c_{m_i} \leq B$).
* **Task Benchmark:** Software-engineering tasks with defined difficulty levels (Easy, Medium, Hard):
  * Code Generation (e.g., HumanEval, MBPP)
  * Bug Fixing / Debugging
  * Unit Test Generation
  * Code Explanation & Optimization

---

## 2. Experimental Combinations

### A. Centralized Approaches (Global Information)

The central controller observes all tasks $\{t_1, \dots, t_N\}$ and model costs, deciding the assignment vector $\mathbf{m} = (m_1, \dots, m_N)$.

| Controller ID | Controller Name | Description | Mechanics |
| :--- | :--- | :--- | :--- |
| **C1** | **Optimal MCKP Solver** *(Upper Bound)* | Solves the exact Multiple-Choice Knapsack Problem. | Uses Integer Linear Programming (Google OR-Tools / PuLP) to globally maximize $\sum q_{i, m_i}$ subject to $\sum c_{m_i} \leq B$. |
| **C2** | **Predictive LLM Router** *(RouteLLM / MasRouter)* | Routes tasks to models based on task feature embeddings and predicted quality gains. | Balances estimated quality vs. marginal cost using a trained quality estimator without solving an exact ILP. |
| **C3** | **Confidence-Based Cascade** *(FrugalGPT)* | Sends tasks to cheap models first; escalates only if test cases or self-consistency checks fail. | Dynamically consumes budget only when cheap models fail on specific verification criteria. |
| **C4** | **Static Equal Quota** *(Fair Baseline)* | Splits budget evenly upfront: $b_i = B / N$ for every agent. | Simple baseline without cross-agent coordination; each agent chooses the best model within its individual slice $b_i$. |

---

### B. Decentralized Approaches (Sequential Game)

Agents arrive asynchronously in an arrival sequence $\pi = (\pi_1, \pi_2, \dots, \pi_N)$. Each agent observes the remaining budget $B_k$, picks an affordable model maximizing its individual utility $u_i(m) = q_{i,m} - \lambda c_m$, and deducts its cost.

| Approach ID | Approach Name | Arrival / Decision Policy | Research Focus |
| :--- | :--- | :--- | :--- |
| **D1** | **Random Sequential (FCFS)** | Agents arrive in uniformly random order. Early agents act with no awareness of future agents. | Measures baseline first-mover advantage, resource starvation, and natural price of anarchy. |
| **D2** | **Hardest-Task-First (HTF)** | Agents with higher task difficulty arrive earlier in the queue. | Tests whether prioritizing tasks with steep quality curves under expensive models bridges the gap to centralized optimality. |
| **D3** | **Easiest-Task-First (ETF)** | Agents with simple tasks arrive first. | Represents the worst-case scenario where trivial tasks burn the budget early. |
| **D4** | **Marginal ROI Priority** | Agents are prioritized by expected quality gain per unit cost $(\Delta q / \Delta c)$. | Tests whether greedy heuristic queuing approximates the global knapsack solution without central model assignment. |
| **D5** | **Cost-Sensitive Adaptive Utility** | Agents adjust their cost sensitivity parameter $\lambda$ based on the remaining budget fraction $B_k / B$. | Evaluates decentralized thriftiness: agents become more conservative as the shared pool depletes. |
| **D6** | **Online Bandit / UCB Extension** | Model performance $q_{i,m}$ is unknown upfront; agents balance exploration vs. exploitation using UCB. | Evaluates the impact of performance uncertainty in decentralized resource allocation. |

---

### C. Decentralized Utility Function Designs (Local Payoff Engineering)

In addition to arrival ordering policies, we investigate **utility function design**—systematically altering how individual agents evaluate model choices so that decentralized best-response choices approach the centralized knapsack optimum. We compare four distinct utility formulations:

#### U1. Baseline Quasi-Linear Utility
* **Formulation:**
  $$U_i(m) = Q_{i,m} - \lambda C_m$$
* **Concept:** Standard selfish baseline where each agent trades off task quality against model cost with a fixed global parameter $\lambda \geq 0$.
* **Reference:** Classical mechanism design and quasi-linear game theory.

#### U2. Dynamic Shadow Price / Scarcity-Adaptive Utility
* **Formulation:**
  $$U_i(m) = Q_{i,m} - p(B_k) \cdot C_m$$
  where $p(B_k)$ is the endogenous shadow price of the remaining budget $B_k$:
  $$p(B_k) = p_0 \cdot \left( \frac{B}{B_k} \right)^\alpha \quad \text{or} \quad p(B_k) = \lambda_0 \cdot \frac{(B - B_k) / B}{k / N}$$
* **Concept:** Dynamically reflects budget scarcity:
  * When budget is plentiful ($B_k \approx B$), $p(B_k) \downarrow$, allowing agents to select stronger models.
  * When budget becomes scarce ($B_k \downarrow$), $p(B_k) \uparrow$, automatically making expensive models less attractive and conserving tokens.
* **Paper References:**
  * Balseiro, S. R., & Gur, Y. (2019). *"Learning in Repeated Auctions with Budgets: Regret Minimization and Equilibrium."* *Management Science*, 65(9), 3952-3968. [DOI: 10.1287/mnsc.2018.3101](https://doi.org/10.1287/mnsc.2018.3101)
  * Balseiro, S. R., Besbes, O., & Castro, G. Y. (2015). *"Repeated Auctions with Budgets in Ad Exchanges: Approximations and Design."* *Management Science*, 61(4), 864-884.
  * Kelly, F. P., Maulloo, A. K., & Tan, D. K. (1998). *"Rate control for communication networks: shadow prices, proportional fairness and stability."* *Journal of the Operational Research Society*, 49(3), 237-252.

#### U3. Shapley Value / Marginal Contribution-Based Budget Share
* **Formulation:**
  Agents receive an entitlement or modified cost penalty based on their Shapley value / marginal contribution $\phi_i$ to the collective mission quality:
  $$U_i(m) = Q_{i,m} - \left( \frac{\lambda}{1 + \gamma \cdot \phi_i} \right) C_m$$
  Alternatively, an agent's accessible budget slice is scaled by its relative Shapley weight: $B_i^{\text{max}} = B_k \cdot \frac{\phi_i}{\sum_{j \geq k} \phi_j}$.
* **Concept:** Agents with high task criticality or steeper difficulty curves have higher marginal contributions to the collective welfare, granting them preferential access to premium models while restricting agents with low-impact tasks.
* **Paper References:**
  * Chandan, R., Paccagnan, D., & Marden, J. R. (2019). *"Optimal Price of Anarchy in Cost-Sharing Games."* *American Control Conference (ACC) / IEEE*, [arXiv:1903.06288](https://arxiv.org/abs/1903.06288).
  * Marden, J. R., & Wierman, A. (2013). *"Distributed Welfare Games."* *Operations Research*, 61(1), 155-168. [DOI: 10.1287/opre.1120.1118](https://doi.org/10.1287/opre.1120.1118)

#### U4. Submodular Marginal Utility per Cost Ratio ($\frac{\Delta U}{\Delta C}$ / Incremental ROI)
* **Formulation:**
  Rather than evaluating total quality, the agent computes the marginal improvement over a baseline/cheapest model:
  $$\Delta U_i(m) = Q_{i,m} - Q_{i, m_{\text{baseline}}}$$
  $$\Delta C_i(m) = C_m - C_{m_{\text{baseline}}}$$
  The agent selects the model that maximizes the marginal return on investment:
  $$m^* = \arg\max_{m \in \mathcal{M}(B_k)} \frac{\Delta U_i(m)}{\Delta C_i(m)} \quad \text{or} \quad \arg\max_{m \in \mathcal{M}(B_k)} \left[ \Delta U_i(m) - p(B_k) \Delta C_i(m) \right]$$
* **Concept:** Directly asks: *"How much additional task quality do I get for the additional budget I consume?"* This reflects the incremental benefit of resource consumption. Under monotone submodular utility structures, decentralized methods achieve provable constant-factor approximation guarantees (such as $1 - 1/e \approx 63.2\%$).
* **Paper References:**
  * Williams, R. K., et al. (2017). *"Decentralised Submodular Multi-Robot Task Allocation."* *IEEE Transactions on Robotics / Robotics: Science and Systems*.
  * Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). *"An analysis of approximations for maximizing submodular set functions—I."* *Mathematical Programming*, 14(1), 265-294.

---

### D. Reference Baselines

* **B1 (Cheapest-Model Baseline):** Every agent always selects the lowest-cost model available ($\min_{m} c_m$).
* **B2 (Greedy Best-Model Baseline):** Every agent selects the highest-performing model until the budget runs out, leaving all remaining agents starved.

---

## 3. Experimental Variables & Test Matrix

Experiments will be conducted by systematically varying five key parameters:

| Variable | Values / Range | Purpose |
| :--- | :--- | :--- |
| **Budget Constraint ($B$)** | Severe ($B = N \cdot c_{\text{min}}$), Moderate, Generous ($B \ge N \cdot c_{\text{max}}$) | Observe when budget pressure triggers catastrophic decentralized failure. |
| **Agent Count ($N$)** | $N \in \{5, 10, 20, 50\}$ | Study scalability and contention as agent density increases. |
| **Task Difficulty Mix** | (1) Uniform random, (2) Easy-heavy (80/20), (3) Hard-heavy (20/80) | Evaluate how task composition affects routing and prioritization value. |
| **Cost-Performance Disparity** | Flat gap (small difference between models) vs. Steep gap (frontier model is $10\times$ cost with $2\times$ quality) | Study how pricing tiers influence agent incentives. |
| **Arrival Permutations ($\pi$)** | 50–100 random seeds per decentralized configuration | Quantify variance and stability across different execution runs. |

---

## 4. Execution Workflow

1. **Step 1: Benchmark Initialization:** Generate task instances with assigned ground-truth difficulty levels and model performance profiles.
2. **Step 2: Centralized Runs:** Execute controllers C1, C2, C3, and C4 across all parameter combinations to establish optimal benchmarks.
3. **Step 3: Decentralized Runs:** Simulate sequential games D1 through D6 under identical seeds and parameter sets.
4. **Step 4: Metric Aggregation:** Compute comparative metrics (efficiency gap, starvation rate, Jain's fairness index, etc.).
