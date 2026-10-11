"""Tests for the framework itself: ledger, registries, sequential game, experiment, dataset, config."""

import json
import random

import pytest

from masbudget.centralized import CENTRALIZED_POLICIES, BaseCentralizedPolicy
from masbudget.config import ConfigError, build_experiment, load_config
from masbudget.core import CENTRALIZED, DECENTRALIZED, FALLBACK, BudgetExceededError, BudgetLedger, Task
from masbudget.dataset import DatasetError, load_dataset
from masbudget.decentralized import UTILITY_FUNCTIONS, BaseUtilityFunction, create_arrival_policy
from masbudget.decentralized.sequential import SequentialBestResponse
from masbudget.experiment import Experiment
from masbudget.registry import AgentRegistry
from tests.helpers import make_agents, make_models


class QualityUtility(BaseUtilityFunction):
    def compute(self, job, model, state):
        return model.quality(job.task)


class CheapestForAll(BaseCentralizedPolicy):
    def allocate(self, agents, models, budget, rng):
        return {j.job_id: models.cheapest().name for j in agents.jobs}


class Overspend(BaseCentralizedPolicy):
    def allocate(self, agents, models, budget, rng):
        return {j.job_id: "big" for j in agents.jobs}


class TwoAttempts(BaseCentralizedPolicy):
    def allocate(self, agents, models, budget, rng):
        return {j.job_id: ["cheap", "mid"] for j in agents.jobs}


# ---------------------------------------------------------------- ledger & registries


def test_ledger_deducts_and_rejects_overspending():
    ledger = BudgetLedger(5.0)
    ledger.charge(4.0)
    assert ledger.remaining == pytest.approx(1.0)
    with pytest.raises(BudgetExceededError):
        ledger.charge(2.0)


def test_model_registry_affordable_excludes_fallback(models):
    assert [m.name for m in models.affordable(2.0)] == ["cheap", "mid"]
    assert models.affordable(0.5) == []
    assert models.get(FALLBACK).cost == 0.0 and models.get(FALLBACK).quality(None) == 0.0
    assert FALLBACK not in models.names


def test_model_registry_rejects_duplicates_and_reserved_name(models):
    with pytest.raises(ValueError):
        models.add("cheap", 1.0)
    with pytest.raises(ValueError):
        models.add(FALLBACK, 0.0)


def test_agent_registry_interleaves_jobs_round_robin():
    a, b, c = (Task(name, "easy", {}) for name in "abc")
    agents = AgentRegistry.from_task_lists([[a, b, c], [a]])  # uneven task lists are fine
    assert [(j.job_id, j.agent_id, j.task.task_id) for j in agents.jobs] == [
        (0, 0, "a"), (1, 1, "a"), (2, 0, "b"), (3, 0, "c")
    ]
    assert [j.job_id for j in agents.jobs_of(0)] == [0, 2, 3]
    with pytest.raises(ValueError, match="no tasks"):
        AgentRegistry.from_task_lists([[]])


def test_unknown_factory_name_lists_available():
    with pytest.raises(ValueError, match="Available"):
        CENTRALIZED_POLICIES.create("does_not_exist")


# ---------------------------------------------------------------- sequential game


def test_sequential_game_deducts_budget_and_starves_late_agents(agents, models):
    game = SequentialBestResponse(create_arrival_policy(None), QualityUtility())
    out = game.allocate(agents, models, 6.0, random.Random(0))
    # easy-0 takes big (4) -> 2 left; medium-1 takes mid (2) -> 0 left; the rest starve.
    assert [a.model for a in out] == ["big", "mid", FALLBACK, FALLBACK]
    assert [a.remaining_budget for a in out] == pytest.approx([6.0, 2.0, 0.0, 0.0])
    assert [a.step for a in out] == [0, 1, 2, 3]


def test_sequential_game_serves_each_task_of_each_agent(multi_agents, models):
    game = SequentialBestResponse(create_arrival_policy(None), QualityUtility())
    out = game.allocate(multi_agents, models, 6.0, random.Random(0))
    # round-robin: agent 0's easy-0 takes big (4 -> 2 left), agent 1's hard-2 takes mid (-> 0 left)
    assert [(a.job_id, a.agent_id, a.model) for a in out] == [
        (0, 0, "big"), (1, 1, "mid"), (2, 0, FALLBACK), (3, 1, FALLBACK)
    ]


# ---------------------------------------------------------------- experiment


def test_experiment_runs_both_paradigms_on_shared_agents(agents, models, register):
    register(CENTRALIZED_POLICIES, "_cheapest_for_all", CheapestForAll)
    register(UTILITY_FUNCTIONS, "_quality", QualityUtility)
    exp = Experiment(agents, models, 6.0, centralized_policy="_cheapest_for_all", utility="_quality", num_runs=3)
    result = exp.run(metrics=["average_quality", "optimality_gap"])
    assert {p: len(r) for p, r in result.runs.items()} == {CENTRALIZED: 3, DECENTRALIZED: 3}
    assert set(result.metrics) == {"average_quality", "optimality_gap"}
    assert exp.evaluate(["total_cost"])["total_cost"][CENTRALIZED]["mean"] == pytest.approx(4.0)


def test_experiment_rejects_overspending_centralized_policy(agents, models, register):
    register(CENTRALIZED_POLICIES, "_overspend", Overspend)
    with pytest.raises(BudgetExceededError):
        Experiment(agents, models, 8.0, centralized_policy="_overspend").run()


def test_experiment_charges_every_attempt_and_keeps_the_last_answer(agents, models, register):
    register(CENTRALIZED_POLICIES, "_two_attempts", TwoAttempts)
    run = Experiment(agents, models, 12.0, centralized_policy="_two_attempts").run().runs[CENTRALIZED][0]
    first = run.assignments[0]  # easy-0: cheap (1) then mid (2)
    assert (first.model, first.quality, first.cost, first.attempts) == ("mid", 0.85, 3.0, ("cheap", "mid"))
    assert run.total_cost == pytest.approx(12.0)
    with pytest.raises(BudgetExceededError):  # 4 jobs x 3 > 11
        Experiment(agents, models, 11.0, centralized_policy="_two_attempts").run()


def test_experiment_needs_something_to_run(agents, models):
    with pytest.raises(ValueError):
        Experiment(agents, models, 8.0)


def test_experiment_rejects_tasks_without_quality_for_a_model(agents):
    models = make_models({"cheap": 1.0, "unknown_model": 2.0})
    with pytest.raises(ValueError, match="unknown_model"):
        Experiment(agents, models, 8.0, centralized_policy="optimal_mckp")


# ---------------------------------------------------------------- dataset & config


def write_dataset(path, tasks=None):
    tasks = tasks or [
        {"task_id": f"t{i}", "difficulty": "easy", "quality": {"cheap": 0.5, "mid": 0.7, "big": 0.9}}
        for i in range(5)
    ]
    path.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    return path


def test_dataset_loads_and_validates(tmp_path):
    assert len(load_dataset(write_dataset(tmp_path / "ok.json"))) == 5
    bad = write_dataset(tmp_path / "bad.json", [{"task_id": "x", "difficulty": "trivial", "quality": {"a": 0.5}}])
    with pytest.raises(DatasetError, match="difficulty"):
        load_dataset(bad)


def write_config(tmp_path, body):
    dataset = write_dataset(tmp_path / "data.json").as_posix()
    path = tmp_path / "config.yaml"
    path.write_text(
        f"defaults:\n  dataset_path: {dataset}\n  models: {{cheap: 1.0, mid: 2.0, big: 4.0}}\n  budget: 5.0\n"
        f"  num_agents: 2\n{body}",
        encoding="utf-8",
    )
    return path


def test_config_merges_defaults_and_infers_mode(tmp_path):
    path = write_config(
        tmp_path,
        "experiments:\n"
        "  - name: a\n    centralized: {policy: optimal_mckp}\n"
        "  - name: b\n    budget: 9\n    num_agents: 3\n    num_models: 2\n"
        "    decentralized: {arrival_policy: null, utility: quasi_linear}\n",
    )
    a, b = load_config(path)
    assert (a["mode"], a["budget"]) == ("centralized", 5.0)
    assert (b["mode"], b["budget"]) == ("decentralized", 9)
    exp = build_experiment(b)
    assert len(exp.agents) == 3 and exp.models.names == ["cheap", "mid"]
    assert len(exp.agents.jobs) == 3  # tasks_per_agent defaults to 5 tasks // 3 agents = 1
    assert exp.centralized is None and exp.decentralized.arrival_policy.registry_name == "fcfs"
    assert len(build_experiment(a).agents.jobs) == 4  # 2 agents (defaults) x 5 // 2 = 2 tasks


def test_config_deals_distinct_tasks_to_agents(tmp_path):
    path = write_config(tmp_path, "experiments:\n  - name: a\n    tasks_per_agent: 2\n    centralized: {policy: optimal_mckp}\n")
    agents = build_experiment(load_config(path)[0]).agents
    tasks = [t.task_id for agent in agents for t in agent.tasks]
    assert [len(agent.tasks) for agent in agents] == [2, 2] and len(set(tasks)) == 4


def test_config_rejects_more_tasks_than_the_dataset_has(tmp_path):
    path = write_config(tmp_path, "experiments:\n  - name: a\n    tasks_per_agent: 3\n    centralized: {policy: optimal_mckp}\n")
    with pytest.raises(DatasetError, match="needs 6 tasks"):
        build_experiment(load_config(path)[0])


@pytest.mark.parametrize(
    "body, message",
    [
        ("experiments:\n  - name: a\n    budjet: 3\n    centralized: {policy: optimal_mckp}\n", "unknown keys"),
        ("experiments:\n  - name: a\n    mode: both\n    centralized: {policy: optimal_mckp}\n", "decentralized.utility"),
        ("experiments:\n  - name: a\n    centralized: {policy: x}\n  - name: a\n    centralized: {policy: x}\n", "Duplicate"),
        ("experiments:\n  - name: a\n    num_agents: null\n    centralized: {policy: x}\n", "'num_agents' is required"),
    ],
)
def test_config_errors(tmp_path, body, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write_config(tmp_path, body))
