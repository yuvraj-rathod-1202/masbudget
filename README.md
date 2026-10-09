# masbudget

Experiment framework for **multi-agent model/API selection under a shared budget**
(see [project_proposal/project_proposal.tex](project_proposal/project_proposal.tex)).
It compares centralized allocation with the decentralized sequential resource-allocation game
on the same agents, models, costs and budget.

## Setup

```bash
pip install -r requirements.txt
```

## Run experiments

```bash
python -m masbudget configs/example.yaml                 # all experiments in the file
python -m masbudget configs/example.yaml --only NAME ... # a subset
python -m masbudget --list                               # registered factory names
```

Run from the repository root (dataset paths are resolved relative to it).

- `results/<experiment>_<timestamp>.json`: summary metrics + resolved config
- `logs/<experiment>_<timestamp>.log`: full log, including every agent's decision (DEBUG)

A failing experiment (e.g. one that uses a policy that is not implemented yet) is logged and
skipped; the rest of the batch still runs.

## Tests

```bash
python -m pytest            # from the repo root
python -m pytest -rs        # also show why tests were skipped
```

Policy tests skip while a policy is still a stub (`NotImplementedError`) and start running as
soon as it is implemented. See [docs/extending.md](docs/extending.md#testing-your-policy).

## Use from Python

```python
from masbudget import AgentRegistry, ModelRegistry, load_dataset, run_experiment, get_metric

agents = AgentRegistry.from_tasks(load_dataset("data/example_tasks.json")[:10])
models = ModelRegistry()
models.add("gpt-4o-mini", 0.01)
models.add("o1-reasoning", 0.25)

result = run_experiment(
    "optimal_mckp",        # centralized policy  (None = skip centralized)
    "quasi_linear",        # decentralized utility (None = skip decentralized)
    None,                  # arrival policy (None = uncoordinated FCFS)
    agents=agents, models=models, budget=0.5,
    utility_params={"lambda_cost": 0.5},
    metrics=["average_quality", "optimality_gap"],
)
print(result.metrics)
print(get_metric("jains_fairness").compute(result))   # any metric, after the fact
```

`Experiment(...)` gives the same thing as a class: `exp.run(metrics=[...])`, then `exp.evaluate([...])`.

## Layout

```
configs/                  experiment YAML files
data/                     datasets (JSON)
docs/                     config schema, dataset format, extension guide
logs/  results/           outputs
masbudget/
  core.py                 Task, Agent, Assignment, RunResult, BudgetLedger, ...
  models.py               BaseModel + model factory (fallback: quality 0, cost 0)
  registry.py             AgentRegistry, ModelRegistry (shared by both paradigms)
  dataset.py              JSON loading / validation / sampling
  experiment.py           Experiment, run_experiment()
  config.py  runner.py    YAML loading, batch runs, results/logs
  centralized/            BaseCentralizedPolicy + one file per policy
  decentralized/
    sequential.py         the sequential best-response game (implemented)
    arrival/              BaseArrivalPolicy + one file per ordering policy
    utilities/            BaseUtilityFunction + one file per utility
  metrics/                BaseMetrics + metric implementations
tests/
  helpers.py              toy instance (4 agents, 3 models), builders, brute-force optimum
  test_centralized.py     test_arrival.py  test_utilities.py   policy tests
  test_metrics.py         test_framework.py                    framework tests
```

## Status

| Component | Factory names | State |
| --- | --- | --- |
| Centralized | `optimal_mckp`, `predictive_router`, `confidence_cascade`, `static_equal_quota`, `cheapest_model` | stubs |
| Arrival | `random_fcfs`, `hardest_task_first`, `easiest_task_first`, `marginal_roi_priority` | stubs |
| Arrival | `fcfs` (used for `null`) | implemented |
| Utilities | `quasi_linear`, `dynamic_shadow_price`, `shapley_budget_share`, `marginal_roi_ratio`, `best_model` | stubs |
| Decentralized engine | `sequential_best_response` | implemented |
| Metrics | `average_quality`, `success_rate`, `total_cost`, `budget_utilization`, `cost_effectiveness`, `social_welfare`, `optimality_gap`, `jains_fairness`, `starvation_rate`, `first_mover_advantage`, `order_sensitivity` | implemented |

## Docs

- [docs/config.md](docs/config.md): experiment YAML schema
- [docs/dataset.md](docs/dataset.md): dataset JSON format
- [docs/extending.md](docs/extending.md): adding models, policies, utilities, metrics
