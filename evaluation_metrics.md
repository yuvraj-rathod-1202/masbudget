# Evaluation Metrics Specification

This document details all performance, efficiency, fairness, and game-theoretic metrics captured during the multi-agent model selection experiments.

---

## 1. Metric Overview Taxonomy

```
                                  Evaluation Metrics
                                          │
        ┌───────────────────┬─────────────┴───────┬───────────────────┐
Quality & Task       Budget & Resource    Social Welfare &    Fairness & Game-
Performance          Efficiency           Optimality Gap      Theoretic Disparity
```

---

## 2. Detailed Metric Formulations

### A. Quality & Task Performance Metrics

#### 1. Average Task Quality ($\bar{Q}$)
* **Definition:** The arithmetic mean of expected or measured task quality scores across all $N$ agents.
* **Formula:**
  $$\bar{Q} = \frac{1}{N} \sum_{i=1}^{N} q_{i, m_i}$$
* **Range:** $[0.0, 1.0]$ or percentage $[0\%, 100\%]$.
* **Significance:** Measures overall task execution efficacy across the multi-agent system.

#### 2. Stratified Quality by Difficulty ($\bar{Q}_{\text{easy}}, \bar{Q}_{\text{med}}, \bar{Q}_{\text{hard}}$)
* **Definition:** Average quality evaluated independently for each difficulty tier.
* **Significance:** Unveils whether budget savings come at the cost of high-difficulty task failures.

#### 3. Task Success Rate ($SR$)
* **Definition:** The proportion of tasks whose quality meets or exceeds an acceptable functional threshold $\tau$ (e.g., passing unit tests in HumanEval with $\tau = 1.0$).
* **Formula:**
  $$SR = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(q_{i, m_i} \geq \tau)$$

---

### B. Budget & Resource Efficiency Metrics

#### 4. Total Cost ($C$)
* **Definition:** The sum of API costs incurred across all selected models.
* **Formula:**
  $$C = \sum_{i=1}^{N} c_{m_i} \quad \text{such that } C \leq B$$

#### 5. Budget Utilization Rate ($BUR$)
* **Definition:** The fraction of the available budget that was actually spent.
* **Formula:**
  $$BUR = \frac{C}{B} = \frac{\sum_{i=1}^N c_{m_i}}{B}$$
* **Significance:** Identifies "under-spending" (excessive agent conservatism) vs. efficient budget exhaustion.

#### 6. Cost-Effectiveness / ROI ($CE$)
* **Definition:** Quality delivered per dollar spent.
* **Formula:**
  $$CE = \frac{\sum_{i=1}^N q_{i, m_i}}{C}$$

---

### C. Social Welfare & Allocation Efficiency (Price of Anarchy)

#### 7. Social Welfare ($SW$)
* **Definition:** The aggregate utility of all agents in the system, balancing quality gains against resource expenditure.
* **Formula:**
  $$SW(\mathbf{m}) = \sum_{i=1}^{N} u_i(m_i) = \sum_{i=1}^{N} \left( q_{i, m_i} - \lambda c_{m_i} \right)$$
* **Note:** When $\lambda = 0$, social welfare reduces directly to total task quality $\sum q_i$.

#### 8. Optimality Efficiency Gap ($\text{Gap}$)
* **Definition:** The relative efficiency loss (Price of Anarchy) of a decentralized approach relative to the centralized optimal knapsack solver ($C1$).
* **Formula:**
  $$\text{Gap} = \frac{SW_{\text{centralized}} - SW_{\text{decentralized}}}{SW_{\text{centralized}}}$$
* **Significance:** Core benchmark metric comparing how close heuristic or sequential games get to global coordination.

---

### D. Fairness & Game-Theoretic Disparity Metrics

#### 9. Jain's Fairness Index ($J$)
* **Definition:** A standard metric measuring whether resources/utilities are equitably distributed across agents.
* **Formula:**
  $$J(u_1, u_2, \dots, u_N) = \frac{\left( \sum_{i=1}^N u_i \right)^2}{N \cdot \sum_{i=1}^N u_i^2}$$
* **Range:** $[\frac{1}{N}, 1.0]$. A value of $1.0$ represents perfectly equal utility across all agents.

#### 10. Starvation Rate ($SR_{\text{starve}}$)
* **Definition:** The percentage of agents that receive a zero-utility fallback action because previous agents exhausted the budget ($B_k < \min c_m$).
* **Formula:**
  $$SR_{\text{starve}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(B_k < \min_{m \in \mathcal{M}} c_m)$$
* **Significance:** Measures the severity of the tragedy of the commons in decentralized setups.

#### 11. First-Mover Advantage Ratio ($FMR$)
* **Definition:** The ratio between the average utility received by the earliest-arriving quartile (top 25%) and the latest-arriving quartile (bottom 25%) in arrival order $\pi$.
* **Formula:**
  $$FMR = \frac{\text{Mean Utility of Agents } \pi_1 \dots \pi_{\lfloor N/4 \rfloor}}{\text{Mean Utility of Agents } \pi_{\lceil 3N/4 \rceil} \dots \pi_N}$$
* **Significance:** Directly quantifies arrival inequality and positional privilege.

#### 12. Order Sensitivity ($\sigma^2_{\pi}$)
* **Definition:** The variance in Social Welfare across different random arrival permutations $\pi$ under the same tasks and budget.
* **Formula:**
  $$\sigma^2_{\pi}(SW) = \frac{1}{|\Pi|} \sum_{\pi \in \Pi} \left( SW(\pi) - \mathbb{E}[SW] \right)^2$$
* **Significance:** Measures system stability. High variance means system outcomes depend heavily on arrival luck.

---

### E. Learning Metrics (Bandit / Uncertainty Extension)

#### 13. Cumulative Regret ($R_T$)
* **Definition:** The difference between the cumulative utility of an oracle with full model performance knowledge and the agent's exploratory choices over $T$ iterations.
* **Formula:**
  $$R_T = \sum_{t=1}^{T} \left( u^*(t) - u_{\text{chosen}}(t) \right)$$

---

### F. Utility-Specific Diagnostic Metrics

#### 14. Marginal ROI Efficiency ($\overline{\text{ROI}}_{\Delta}$)
* **Definition:** The average realized quality improvement per incremental dollar spent beyond the baseline model:
  $$\overline{\text{ROI}}_{\Delta} = \frac{1}{N} \sum_{i=1}^{N} \frac{q_{i, m_i} - q_{i, \text{baseline}}}{c_{m_i} - c_{\text{baseline}}}$$
* **Significance:** Directly validates the efficacy of Utility Formulation U4 ($\frac{\Delta U}{\Delta C}$), confirming whether agents only buy expensive models when the return on investment is high.

#### 15. Shapley-Budget Alignment Index ($SBA$)
* **Definition:** The rank correlation (Spearman's $\rho$) between each agent's marginal contribution / Shapley value $\phi_i$ and the budget consumed $c_{m_i}$:
  $$SBA = \mathrm{Corr}\left( \mathrm{Rank}(\phi), \, \mathrm{Rank}(c_m) \right)$$
* **Significance:** Measures whether Utility Formulation U3 successfully channels higher budget shares to the most impactful tasks.

#### 16. Shadow Price Trajectory & Volatility ($\sigma_p, \bar{p}$)
* **Definition:** The mean and standard deviation of the endogenous shadow price $p(B_k)$ across the arrival sequence $k = 1, \dots, N$:
  $$\bar{p} = \frac{1}{N} \sum_{k=1}^{N} p(B_k), \qquad \sigma_p = \sqrt{\frac{1}{N} \sum_{k=1}^N \left( p(B_k) - \bar{p} \right)^2}$$
* **Significance:** Tracks whether Utility Formulation U2 smooths out budget depletion or causes erratic pricing spikes.

#### 17. Submodular Approximation Ratio ($\alpha_{\text{sub}}$)
* **Definition:** The ratio of the realized decentralized social welfare to the theoretical $(1 - 1/e \approx 63.2\%)$ submodular approximation guarantee and to the centralized optimum:
  $$\alpha_{\text{sub}} = \frac{SW_{\text{decentralized}}}{SW_{\text{centralized}}}$$
* **Significance:** Empirically tests whether the decentralized marginal utility approach matches or exceeds the theoretical $1 - 1/e$ lower bound.
