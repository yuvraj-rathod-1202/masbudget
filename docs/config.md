# Experiment config (YAML)

A config file has optional `defaults` and a required list of `experiments`. Each experiment
is `defaults` merged with its own keys (top-level keys only; the experiment's value wins, so
a `centralized:` block in an experiment replaces the whole default block).
Unknown keys are rejected to catch typos. See [configs/example.yaml](../configs/example.yaml).

```yaml
defaults:            # any experiment key below can go here
  dataset_path: data/example_tasks.json
  num_agents: 5      # 50 tasks -> floor(50 / 5) = 10 tasks per agent
  models:
    gpt-4o-mini: 0.01
    o1-reasoning: 0.25
  metrics:
    - average_quality
    - total_cost

experiments:
  - name: my_run
    mode: both
    budget: 0.5
    centralized: {policy: optimal_mckp}
    decentralized: {arrival_policy: null, utility: quasi_linear, utility_params: {lambda_cost: 0.5}}
```

## Agents and tasks

1. **Tasks per agent:** `tasks_per_agent` = floor(number of tasks in the dataset / `num_agents`),
   unless you set it.
2. **Dealing:** `num_agents × tasks_per_agent` distinct tasks are drawn without replacement
   (seeded by `seed`). Agent 0 gets the first `tasks_per_agent` of them, agent 1 the next, and
   so on. Leftover tasks (dataset size mod `num_agents`) are not used.
3. **One call per task:** every task is one model call (a *job*) and gets its own model, paid
   from the shared budget B. An agent with 10 tasks makes 10 choices.
4. **Order:** with `arrival_policy: null` (no coordinator) the agents ask round-robin: agent 0's
   1st task, agent 1's 1st task, ..., agent N-1's 1st task, then agent 0's 2nd task, and so on.
   Another arrival policy can reorder any waiting task, whichever agent owns it. A centralized
   policy sees all tasks at once and has no order.

## Experiment keys

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `name` | yes | | Unique name; used for result/log file names. |
| `description` | no | | Free text. |
| `mode` | no | inferred from the sections present | `centralized`, `decentralized` or `both`. |
| `dataset_path` | yes | | JSON dataset ([dataset.md](dataset.md)), relative to the repo root. |
| `budget` | yes | | Shared budget B (same unit as model costs). |
| `models` | yes | | `name: cost`, or `name: {cost: 0.01, type: dataset, params: {}}` for a custom model class. Order matters for `num_models` and tie-breaking. |
| `num_models` | no | all | Use only the first k models of `models`. |
| `num_agents` | yes | | N, the number of agents sharing the budget. |
| `tasks_per_agent` | no | floor(dataset size / N) | Tasks each agent owns (see above). N × `tasks_per_agent` must not exceed the dataset size. |
| `seed` | no | `0` | Seeds which tasks are drawn and which agent gets them; run r also uses `seed + r` for policy randomness. |
| `num_runs` | no | `1` | Repetitions on the **same agents and tasks**; only policy randomness changes (e.g. `random_fcfs` arrival permutations). Metrics report mean/std over runs, so a deterministic setup (centralized, or null arrival) gives std 0. To vary the tasks, change `seed`. |
| `centralized.policy` | if mode uses it | | Centralized factory name. |
| `centralized.params` | no | `{}` | Passed to the policy constructor (`self.params`). |
| `decentralized.utility` | if mode uses it | | Utility factory name. |
| `decentralized.utility_params` | no | `{}` | Utility parameters. |
| `decentralized.arrival_policy` | no | `null` | Arrival factory name. `null` = no coordinator: uncoordinated FCFS (`fcfs`), round-robin over agents. |
| `decentralized.arrival_params` | no | `{}` | Arrival policy parameters. |
| `decentralized.policy` | no | `sequential_best_response` | Decentralized engine factory name. |
| `decentralized.params` | no | `{}` | Engine parameters. |
| `metrics` | no | none | Metric names, or `{name: {param: value}}`, e.g. `- success_rate: {tau: 0.8}`. |

## Metrics

Quality and cost metrics are over all tasks. Fairness metrics compare **agents**: an agent's
payoff u_i is the mean quality over its tasks.

| Metric | Over | Meaning |
| --- | --- | --- |
| `average_quality` | tasks | Mean quality over all tasks. |
| `success_rate` | tasks | Share of tasks with quality >= `tau` (param, default 0.5). |
| `social_welfare` | tasks | Sum of quality over all tasks. |
| `total_cost` | tasks | Total spent, including failed attempts (`confidence_cascade` pays for every model it tries). |
| `budget_utilization` | tasks | Total spent / B. |
| `cost_effectiveness` | tasks | Total quality / total spent. |
| `optimality_gap` | tasks | (SW_centralized - SW_decentralized) / SW_centralized, paired run by run. Only in `mode: both`. |
| `jains_fairness` | agents | Jain's index of the agent payoffs; 1 = all agents equally well off. |
| `starvation_rate` | agents | Share of agents with at least one task that got no model. |
| `task_starvation_rate` | tasks | Share of tasks that got no model. |
| `first_mover_advantage` | agents | Payoff of the earliest-acting agents / payoff of the latest-acting (first and last max(1, N // 4) agents, ranked by mean arrival step). Decentralized only. |
| `order_sensitivity` | runs | Variance of social welfare across runs. Needs `num_runs >= 2`. |

## Output

`results/<name>_<timestamp>.json` holds the resolved config, `num_agents`, `num_jobs` (total
tasks) and, per metric, one entry per paradigm, usually `{"mean", "std", "n"}` over runs
(`null` when undefined; `"inf"` / `"nan"` are written as strings).
`logs/<name>_<timestamp>.log` lists every task's model choice (DEBUG level).
