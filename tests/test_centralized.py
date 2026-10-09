"""Centralized policy tests.

Contract tests run for every registered centralized policy. Policy-specific tests encode the
behaviour described in the proposal. Tests skip while a policy raises NotImplementedError.
"""

import random

import pytest

from masbudget.centralized import CENTRALIZED_POLICIES, create_centralized_policy
from masbudget.core import FALLBACK
from tests.helpers import (
    BUDGETS,
    allocation_quality,
    assert_valid_centralized_allocation,
    brute_force_optimum,
    call_or_skip,
    make_agents,
    make_models,
)

# Params each policy gets in these tests. Add an entry if your policy needs parameters.
POLICY_PARAMS = {
    "confidence_cascade": {"threshold": 0.8},
}


def allocate(name, budget, seed=0):
    agents, models = make_agents(), make_models()
    policy = create_centralized_policy(name, **POLICY_PARAMS.get(name, {}))
    mapping = call_or_skip(policy.allocate, agents, models, budget, random.Random(seed))
    return mapping, agents, models


def model_of(mapping, agent_id):
    return mapping.get(agent_id, FALLBACK)


# ---------------------------------------------------------------- contract (all policies)


@pytest.mark.parametrize("budget", BUDGETS.values(), ids=BUDGETS.keys())
@pytest.mark.parametrize("name", CENTRALIZED_POLICIES.names())
def test_allocation_is_valid_and_within_budget(name, budget):
    mapping, agents, models = allocate(name, budget)
    assert_valid_centralized_allocation(mapping, agents, models, budget)


@pytest.mark.parametrize("name", CENTRALIZED_POLICIES.names())
def test_zero_budget_gives_everyone_fallback(name):
    mapping, agents, _ = allocate(name, 0.0)
    assert all(model_of(mapping, a.agent_id) == FALLBACK for a in agents)


@pytest.mark.parametrize("name", CENTRALIZED_POLICIES.names())
def test_same_seed_gives_same_allocation(name):
    first, _, _ = allocate(name, BUDGETS["moderate"], seed=7)
    second, _, _ = allocate(name, BUDGETS["moderate"], seed=7)
    assert first == second


# ---------------------------------------------------------------- policy-specific


class TestOptimalMCKP:
    @pytest.mark.parametrize("budget", BUDGETS.values(), ids=BUDGETS.keys())
    def test_matches_brute_force_optimum(self, budget):
        mapping, agents, models = allocate("optimal_mckp", budget)
        assert allocation_quality(mapping, agents, models) == pytest.approx(
            brute_force_optimum(agents, models, budget)
        )

    def test_generous_budget_gives_everyone_the_best_model(self):
        mapping, agents, _ = allocate("optimal_mckp", BUDGETS["generous"])
        assert all(model_of(mapping, a.agent_id) == "big" for a in agents)


class TestCheapestModel:
    def test_generous_budget_gives_everyone_the_cheapest_model(self):
        mapping, agents, _ = allocate("cheapest_model", BUDGETS["generous"])
        assert all(model_of(mapping, a.agent_id) == "cheap" for a in agents)


class TestStaticEqualQuota:
    @pytest.mark.parametrize("budget", [4.0, 8.0, 16.0])
    def test_each_agent_gets_best_model_within_its_slice(self, budget):
        mapping, agents, models = allocate("static_equal_quota", budget)
        slice_ = budget / len(agents)
        for agent in agents:
            affordable = models.affordable(slice_)
            expected = max(affordable, key=lambda m: m.quality(agent.task)).name if affordable else FALLBACK
            assert model_of(mapping, agent.agent_id) == expected


class TestConfidenceCascade:
    # Assumes "verification passes" means q >= threshold (0.8 here). Adjust if your rule differs.

    def test_keeps_cheap_model_when_it_passes(self):
        mapping, _, _ = allocate("confidence_cascade", BUDGETS["generous"])
        assert model_of(mapping, 0) == "cheap"  # easy-0: cheap quality 0.80 >= 0.8

    def test_escalates_when_cheap_model_fails(self):
        mapping, _, _ = allocate("confidence_cascade", BUDGETS["generous"])
        assert model_of(mapping, 2) == "big"  # hard-2: cheap 0.10, mid 0.40 fail; big 0.90 passes


class TestPredictiveRouter:
    def test_routes_hard_tasks_to_stronger_models(self):
        mapping, agents, models = allocate("predictive_router", BUDGETS["moderate"])
        # Hard tasks (hard-2) get routed to stronger models ('big', 'mid')
        # while easy task (easy-0) stays on 'cheap'
        assert model_of(mapping, 2) in ("mid", "big")
        assert model_of(mapping, 0) == "cheap"

