# Extending the framework

Every component type has an abstract base class and a registry. To add one:

1. Create a new file in the matching folder.
2. Subclass the base class and implement its one abstract method.
3. Decorate the class with `@<REGISTRY>.register("factory_name")`.

Files in these folders are imported automatically, so `factory_name` can be used in YAML
right away. Config params arrive as `self.params` (a dict). Check with `python -m masbudget --list`.

A new component can start as a stub that raises `NotImplementedError`; its tests skip until
it is implemented (see [Testing your policy](#testing-your-policy)).

Every agent owns a list of tasks (`agent.tasks`), and each task is one model call, a `Job`
(`job.job_id`, `job.agent_id`, `job.task`). Policies assign a model to every job, so an
agent with 10 tasks gets 10 models. `agents.jobs` lists all jobs round-robin over agents
(also the null-arrival FCFS order); `agents.jobs_of(agent_id)` gives one agent's jobs.
How tasks are dealt to agents: [config.md](config.md#agents-and-tasks).

## Centralized policy: `masbudget/centralized/`

```python
from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy
from masbudget.core import FALLBACK

@CENTRALIZED_POLICIES.register("my_policy")
class MyPolicy(BaseCentralizedPolicy):
    def allocate(self, agents, models, budget, rng):
        # agents: AgentRegistry, models: ModelRegistry, rng: random.Random (seeded)
        # q_{j,m} = model.quality(job.task), c_m = model.cost
        return {job.job_id: models.cheapest().name for job in agents.jobs}  # or FALLBACK
```

Return `{job_id: model_name}`. Jobs left out (or mapped to `FALLBACK`) get quality 0,
cost 0. A value can also be a list of models tried in order, e.g. `["cheap", "mid"]` for a
cascade that escalated once: every attempt is paid for and the job keeps the last model's
answer (`Assignment.attempts` records the list). The experiment charges every call to a
`BudgetLedger` and fails if the total exceeds the budget.

## Decentralized utility: `masbudget/decentralized/utilities/`

```python
from masbudget.decentralized.utilities.base import UTILITY_FUNCTIONS, BaseUtilityFunction

@UTILITY_FUNCTIONS.register("my_utility")
class MyUtility(BaseUtilityFunction):
    def compute(self, job, model, state):
        lam = self.params.get("lambda_cost", 1.0)
        return model.quality(job.task) - lam * model.cost
```

`job` is the task being served; its owner is `state.agents.get(job.agent_id)`.

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
        return max(remaining, key=lambda job: job.task.difficulty_rank)
```

Return one job from `remaining` (the jobs not served yet, in round-robin order). Any waiting
job can be picked, whichever agent owns it. Use `state.rng` for randomness so runs are
reproducible.

## Decentralized engine: `masbudget/decentralized/`

The sequential best-response game is `sequential_best_response`. A different game subclasses
`BaseDecentralizedPolicy` (registry `DECENTRALIZED_POLICIES`). It receives
`self.arrival_policy` and `self.utility`, and `allocate(agents, models, budget, rng)` returns a
list of `Assignment` (one per job, in serving order). Select it with `decentralized.policy`.

## Model: `masbudget/models.py` (or a new module imported from it)

Models default to type `dataset`: q_{j,m} is read from the task's `quality` table. For other
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
`RunResult`s, each with one `assignment` per job; `run.by_agent()` groups them by agent and
`run.agent_payoffs` gives each agent's mean quality) and returns `{paradigm: value}`. `per_run`
applies a function to each run and reports mean/std per paradigm. Add the new metric to the
table in [config.md](config.md#metrics).

## Testing your policy

Tests use a toy instance with 4 tasks and 3 models (as 4 one-task agents, or 2 agents with
2 tasks each), defined in [tests/helpers.py](../tests/helpers.py). It is small enough to
check expected answers by hand.
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
