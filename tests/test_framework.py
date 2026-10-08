"""Tests for the framework itself: ledger, registries, sequential game, experiment, dataset, config."""

import json
import random

import pytest

from masbudget.centralized import CENTRALIZED_POLICIES, BaseCentralizedPolicy
from masbudget.config import ConfigError, build_experiment, load_config
from masbudget.core import CENTRALIZED, DECENTRALIZED, FALLBACK, BudgetExceededError, BudgetLedger
from masbudget.dataset import DatasetError, load_dataset
from masbudget.decentralized import UTILITY_FUNCTIONS, BaseUtilityFunction, create_arrival_policy
from masbudget.decentralized.sequential import SequentialBestResponse
from masbudget.experiment import Experiment
from tests.helpers import make_agents, make_models


class QualityUtility(BaseUtilityFunction):
    def compute(self, agent, model, state):
        return model.quality(agent.task)


class CheapestForAll(BaseCentralizedPolicy):
    def allocate(self, agents, models, budget, rng):
        return {a.agent_id: models.cheapest().name for a in agents}


class Overspend(BaseCentralizedPolicy):
    def allocate(self, agents, models, budget, rng):
        return {a.agent_id: "big" for a in agents}


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
        f"defaults:\n  dataset_path: {dataset}\n  models: {{cheap: 1.0, mid: 2.0, big: 4.0}}\n  budget: 5.0\n{body}",
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
    assert exp.centralized is None and exp.decentralized.arrival_policy.registry_name == "fcfs"


@pytest.mark.parametrize(
    "body, message",
    [
        ("experiments:\n  - name: a\n    budjet: 3\n    centralized: {policy: optimal_mckp}\n", "unknown keys"),
        ("experiments:\n  - name: a\n    mode: both\n    centralized: {policy: optimal_mckp}\n", "decentralized.utility"),
        ("experiments:\n  - name: a\n    centralized: {policy: x}\n  - name: a\n    centralized: {policy: x}\n", "Duplicate"),
    ],
)
def test_config_errors(tmp_path, body, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write_config(tmp_path, body))
