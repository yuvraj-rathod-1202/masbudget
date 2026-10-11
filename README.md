# masbudget

Experiment framework for **multi-agent model/API selection under a shared budget**
(see [project_proposal/project_proposal.tex](project_proposal/project_proposal.tex)).
It compares centralized allocation with the decentralized sequential resource-allocation game
on the same agents, models, costs and budget.

## How an experiment is set up

- You give a dataset of tasks and `num_agents` (N). Each agent gets
  floor(number of tasks / N) tasks, drawn without replacement (seeded).
- Every task is one model call that gets its own model, paid from the shared budget B.
- **Centralized:** a controller sees all tasks at once and assigns every model.
- **Decentralized:** tasks are served one at a time. With no coordinator
  (`arrival_policy: null`) the agents ask round-robin: agent 1 task 1, agent 2 task 1, ...,
  agent N task 1, agent 1 task 2, ... Each agent sees only the budget left when its turn comes.
- Quality is read from the dataset (no live API calls); costs come from the config.

## Setup

```bash
pip install -r requirements.txt
pip install ipykernel
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

## Plot results

`masbudget.analysis.PolicyComparison`. It reads the newest result file of each experiment and
draws one bar chart per metric, one bar per policy:

```python
from masbudget.analysis import PolicyComparison

cmp = PolicyComparison.from_config("configs/all_individual_policy.yaml")
cmp = cmp.with_gap("central_optimal_mckp")   # optional: optimality_gap vs. a reference experiment
cmp.table(["average_quality", "total_cost"])  # policies x metrics
cmp.plot(["average_quality", "budget_utilization", "jains_fairness"], sort=True)
cmp.plot(["starvation_rate"], paradigms=["decentralized"])
```

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

tasks = load_dataset("data/example_tasks.json")
agents = AgentRegistry.from_task_lists([tasks[0:5], tasks[5:10]])   # 2 agents x 5 tasks
models = ModelRegistry()
models.add("gpt-4o-mini", 0.01)
models.add("o1-reasoning", 0.25)

result = run_experiment(
    "optimal_mckp",        # centralized policy  (None = skip centralized)
    "quasi_linear",        # decentralized utility (None = skip decentralized)
    None,                  # arrival policy (None = uncoordinated FCFS, round-robin over agents)
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
notebooks/                result plots
logs/  results/           outputs
masbudget/
  core.py                 Task, Agent, Job, Assignment, RunResult, BudgetLedger, ...
  models.py               BaseModel + model factory (fallback: quality 0, cost 0)
  registry.py             AgentRegistry, ModelRegistry (shared by both paradigms)
  dataset.py              JSON loading / validation / dealing tasks to agents
  experiment.py           Experiment, run_experiment()
  config.py  runner.py    YAML loading, batch runs, results/logs
  analysis.py             load results/*.json, bar-plot policies per metric (PolicyComparison)
  centralized/            BaseCentralizedPolicy + one file per policy
  decentralized/
    sequential.py         the sequential best-response game (implemented)
    arrival/              BaseArrivalPolicy + one file per ordering policy
    utilities/            BaseUtilityFunction + one file per utility
  metrics/                BaseMetrics + metric implementations
tests/
  helpers.py              toy instance (4 tasks, 3 models; 4x1 or 2x2 agents), builders, brute-force optimum
  test_centralized.py     test_arrival.py  test_utilities.py   policy tests
  test_metrics.py         test_framework.py                    framework tests
```

## Docs

- [docs/config.md](docs/config.md): experiment YAML schema, how agents get tasks, metric definitions
- [docs/dataset.md](docs/dataset.md): dataset JSON format
- [docs/extending.md](docs/extending.md): adding models, policies, utilities, metrics
