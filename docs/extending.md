# Extending the framework

Every component type has an abstract base class and a registry. To add one:

1. Create a new file in the matching folder.
2. Subclass the base class and implement its one abstract method.
3. Decorate the class with `@<REGISTRY>.register("factory_name")`.

Files in these folders are imported automatically, so `factory_name` can be used in YAML
right away. Config params arrive as `self.params` (a dict). Check with `python -m masbudget --list`.

To implement one of the existing stubs, replace its `raise NotImplementedError` body.

## Centralized policy: `masbudget/centralized/`

```python
from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy
from masbudget.core import FALLBACK

@CENTRALIZED_POLICIES.register("my_policy")
class MyPolicy(BaseCentralizedPolicy):
    def allocate(self, agents, models, budget, rng):
        # agents: AgentRegistry, models: ModelRegistry, rng: random.Random (seeded)
        # q_{i,m} = model.quality(agent.task), c_m = model.cost
        return {agent.agent_id: models.cheapest().name for agent in agents}  # or FALLBACK
```

Return `{agent_id: model_name}`. Agents left out (or mapped to `FALLBACK`) get quality 0,
cost 0. The experiment charges every assignment to a `BudgetLedger` and fails if the total
exceeds the budget.

## Decentralized utility: `masbudget/decentralized/utilities/`

```python
from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction

@UTILITY_FUNCTIONS.register("my_utility")
class MyUtility(BaseUtilityFunction):
    def compute(self, agent, model, state):
        lam = self.params.get("lambda_cost", 1.0)
        return model.quality(agent.task) - lam * model.cost
```

`state` is an `AllocationState`: `step` (k), `remaining_budget` (B_k), `total_budget` (B),
`budget_fraction` (B_k/B), `history` (earlier assignments), `agents`, `models`, `rng`.
The engine only calls `compute` for affordable models and picks the argmax. Ties go to the
model listed first in the config.

## Arrival policy: `masbudget/decentralized/arrival/`

```python
from masbudget.decentralized.arrival.base import ARRIVAL_POLICIES, BaseArrivalPolicy

@ARRIVAL_POLICIES.register("my_order")
class MyOrder(BaseArrivalPolicy):
    def select_next(self, remaining, state):
        return max(remaining, key=lambda a: a.task.difficulty_rank)
```

Return one agent from `remaining`. Use `state.rng` for randomness so runs are reproducible.

## Decentralized engine: `masbudget/decentralized/`

The sequential best-response game is `sequential_best_response`. A different game subclasses
`BaseDecentralizedPolicy` (registry `DECENTRALIZED_POLICIES`). It receives
`self.arrival_policy` and `self.utility`, and `allocate(agents, models, budget, rng)` returns a
list of `Assignment` (one per agent, in acting order). Select it with `decentralized.policy`.

## Model: `masbudget/models.py` (or a new module imported from it)

Models default to type `dataset`: q_{i,m} is read from the task's `quality` table. For other
behaviour (e.g. a live API call or a learned estimator), register a model type:

```python
from masbudget.models import MODEL_TYPES, BaseModel

@MODEL_TYPES.register("my_type")
class MyModel(BaseModel):
    def quality(self, task):
        ...
```

Then use it in YAML: `models: {gpt-4o: {cost: 0.02, type: my_type, params: {...}}}`.
Adding a new model with dataset-backed quality needs no code. Add it to `models:` in the
config and to every task's `quality` table.

## Metric: `masbudget/metrics/`

```python
from masbudget.metrics.base import METRICS, BaseMetrics, per_run

@METRICS.register("max_cost_share")
class MaxCostShare(BaseMetrics):
    def compute(self, result):
        return per_run(result, lambda run: max(a.cost for a in run.assignments) / run.budget)
```

`compute(result)` gets the `ExperimentResult` (`result.runs[paradigm]` is a list of
`RunResult`s, each with per-agent `assignments`) and returns `{paradigm: value}`. `per_run`
applies a function to each run and reports mean/std per paradigm.

## Testing your policy

Tests use a toy instance with 4 agents and 3 models, defined in
[tests/helpers.py](../tests/helpers.py). It is small enough to check expected answers by hand.
Each policy test file has two parts:

- **Contract tests** run for *every* registered name, including new ones. They check that
  output is valid, stays within budget, and is deterministic for a fixed seed. No work is
  needed to get them for a new policy.
- **Policy-specific tests** (`class TestOptimalMCKP`, ...) check the behaviour expected from
  the proposal, e.g. MCKP matches a brute-force optimum, or `quasi_linear` equals
  q - lambda * c.

All of them skip while the policy raises `NotImplementedError`. Once you implement it:

1. If your policy needs parameters, add them to `POLICY_PARAMS` at the top of the test file.
2. Run `python -m pytest tests/test_centralized.py -k my_policy -rs`.
3. If a policy-specific test encodes an assumption your design deliberately differs from (they
   say so in a comment), update that test. Replace `@pytest.mark.skip(reason="TODO ...")`
   placeholders with real tests.
4. For a brand-new policy, add a `class TestMyPolicy` with at least one test of its intended
   behaviour on the toy instance, using `call_or_skip`, `make_state`, `brute_force_optimum`
   etc. from `tests/helpers.py`.
