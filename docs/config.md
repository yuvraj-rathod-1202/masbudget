# Experiment config (YAML)

A config file has optional `defaults` and a required list of `experiments`. Each experiment
is `defaults` merged with its own keys (top-level keys only; the experiment's value wins, so
a `centralized:` block in an experiment replaces the whole default block).
Unknown keys are rejected to catch typos. See [configs/example.yaml](../configs/example.yaml).

```yaml
defaults:            # any experiment key below can go here
  dataset_path: data/example_tasks.json
  models: 
    gpt-4o-mini: 0.01
    o1-reasoning: 0.25
  metrics: 
    "average_quality" 
    "total_cost"

experiments:
  - name: my_run
    mode: both
    budget: 0.5
    centralized: {policy: optimal_mckp}
    decentralized: {arrival_policy: null, utility: quasi_linear, utility_params: {lambda_cost: 0.5}}
```

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
| `num_agents` | no | all tasks | N. Agents get a seeded random sample of N tasks; that sample order is also the FCFS arrival order. |
| `seed` | no | `0` | Seeds task sampling; run r uses `seed + r` for policy randomness. |
| `num_runs` | no | `1` | Repetitions on the same agents (e.g. 50 to 100 arrival permutations). Metrics report mean/std over runs. |
| `centralized.policy` | if mode uses it | | Centralized factory name. |
| `centralized.params` | no | `{}` | Passed to the policy constructor (`self.params`). |
| `decentralized.utility` | if mode uses it | | Utility factory name. |
| `decentralized.utility_params` | no | `{}` | Utility parameters. |
| `decentralized.arrival_policy` | no | `null` | Arrival factory name. `null` = no coordinator: uncoordinated FCFS (`fcfs`). |
| `decentralized.arrival_params` | no | `{}` | Arrival policy parameters. |
| `decentralized.policy` | no | `sequential_best_response` | Decentralized engine factory name. |
| `decentralized.params` | no | `{}` | Engine parameters. |
| `metrics` | no | none | Metric names, or `{name: {param: value}}`, e.g. `- success_rate: {tau: 0.8}`. |

## Output

`results/<name>_<timestamp>.json` holds the resolved config and, per metric, one entry per
paradigm, usually `{"mean", "std", "n"}` over runs (`null` when undefined; `"inf"` / `"nan"`
are written as strings). `optimality_gap` only appears in `mode: both`.
