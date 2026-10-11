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
    allocation_cost,
    call_or_skip,
    final_model,
    make_agents,
    make_models,
    make_multi_agents,
)

# Params each policy gets in these tests. Add an entry if your policy needs parameters.
POLICY_PARAMS = {
    "confidence_cascade": {"threshold": 0.8},
}


def allocate(name, budget, seed=0, agents=None):
    agents, models = agents or make_agents(), make_models()
    policy = create_centralized_policy(name, **POLICY_PARAMS.get(name, {}))
    mapping = call_or_skip(policy.allocate, agents, models, budget, random.Random(seed))
    return mapping, agents, models


def model_of(mapping, job_id):
    """The model whose answer the job keeps (the last attempt for a cascade)."""
    return final_model(mapping, job_id)


# ---------------------------------------------------------------- contract (all policies)


@pytest.mark.parametrize("budget", BUDGETS.values(), ids=BUDGETS.keys())
@pytest.mark.parametrize("name", CENTRALIZED_POLICIES.names())
def test_allocation_is_valid_and_within_budget(name, budget):
    mapping, agents, models = allocate(name, budget)
    assert_valid_centralized_allocation(mapping, agents, models, budget)


@pytest.mark.parametrize("budget", BUDGETS.values(), ids=BUDGETS.keys())
@pytest.mark.parametrize("name", CENTRALIZED_POLICIES.names())
def test_multi_task_allocation_is_valid_and_within_budget(name, budget):
    mapping, agents, models = allocate(name, budget, agents=make_multi_agents())
    assert_valid_centralized_allocation(mapping, agents, models, budget)


@pytest.mark.parametrize("name", CENTRALIZED_POLICIES.names())
def test_zero_budget_gives_everyone_fallback(name):
    mapping, agents, _ = allocate(name, 0.0)
    assert all(model_of(mapping, j.job_id) == FALLBACK for j in agents.jobs)


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

    @pytest.mark.parametrize("budget", BUDGETS.values(), ids=BUDGETS.keys())
    def test_multi_task_matches_brute_force_optimum(self, budget):
        mapping, agents, models = allocate("optimal_mckp", budget, agents=make_multi_agents())
        assert allocation_quality(mapping, agents, models) == pytest.approx(
            brute_force_optimum(agents, models, budget)
        )

    def test_generous_budget_gives_everyone_the_best_model(self):
        mapping, agents, _ = allocate("optimal_mckp", BUDGETS["generous"])
        assert all(model_of(mapping, j.job_id) == "big" for j in agents.jobs)


class TestCheapestModel:
    def test_generous_budget_gives_everyone_the_cheapest_model(self):
        mapping, agents, _ = allocate("cheapest_model", BUDGETS["generous"])
        assert all(model_of(mapping, j.job_id) == "cheap" for j in agents.jobs)


class TestStaticEqualQuota:
    @pytest.mark.parametrize("budget", [4.0, 8.0, 16.0])
    def test_each_agent_gets_best_model_within_its_slice(self, budget):
        mapping, agents, models = allocate("static_equal_quota", budget)
        slice_ = budget / len(agents)
        for job in agents.jobs:
            affordable = models.affordable(slice_)
            expected = max(affordable, key=lambda m: m.quality(job.task)).name if affordable else FALLBACK
            assert model_of(mapping, job.job_id) == expected

    def test_each_agent_spends_only_its_own_slice(self):
        # B = 8 -> 4 per agent. Agent 0 (easy-0, medium-1): mid + mid = 1.60 is its best within 4.
        # Agent 1 (hard-2, hard-3): big on hard-2 (0.90) beats mid + mid (0.85), so hard-3 gets
        # nothing; agent 0's slice is not lent to it.
        mapping, _, _ = allocate("static_equal_quota", 8.0, agents=make_multi_agents())
        assert [model_of(mapping, j) for j in range(4)] == ["mid", "big", "mid", FALLBACK]


class TestConfidenceCascade:
    # Assumes "verification passes" means q >= threshold (0.8 here). Adjust if your rule differs.

    def test_keeps_cheap_model_when_it_passes(self):
        mapping, _, _ = allocate("confidence_cascade", BUDGETS["generous"])
        assert model_of(mapping, 0) == "cheap"  # easy-0: cheap quality 0.80 >= 0.8

    def test_escalates_when_cheap_model_fails(self):
        mapping, _, _ = allocate("confidence_cascade", BUDGETS["generous"])
        assert model_of(mapping, 2) == "big"  # hard-2: cheap 0.10, mid 0.40 fail; big 0.90 passes

    def test_pays_for_every_failed_attempt(self):
        # Full cascades: easy-0 cheap (1); medium-1, hard-2, hard-3 cheap -> mid -> big (7 each).
        mapping, agents, models = allocate("confidence_cascade", 22.0)
        full = ["cheap", "mid", "big"]
        assert mapping == {0: ["cheap"], 1: full, 2: full, 3: full}
        assert allocation_cost(mapping, agents, models) == pytest.approx(22.0)

    def test_tight_budget_escalates_best_quality_gain_per_cost(self):
        # B = 16 < 22: everyone gets cheap (4 spent), then failing jobs escalate by dq / c_next:
        # hard-2 -> mid (0.15), medium-1 -> mid (0.125, first of a tie), hard-2 -> big (0.125),
        # hard-3 -> mid (0.125); 14 spent and no affordable escalation is left.
        mapping, agents, models = allocate("confidence_cascade", BUDGETS["generous"])
        assert mapping == {0: ["cheap"], 1: ["cheap", "mid"], 2: ["cheap", "mid", "big"], 3: ["cheap", "mid"]}
        assert allocation_cost(mapping, agents, models) == pytest.approx(14.0)


class TestPredictiveRouter:
    def test_routes_hard_tasks_to_stronger_models(self):
        mapping, agents, models = allocate("predictive_router", BUDGETS["moderate"])
        # Hard tasks (hard-2) get routed to stronger models ('big', 'mid')
        # while easy task (easy-0) stays on 'cheap'
        assert model_of(mapping, 2) in ("mid", "big")
        assert model_of(mapping, 0) == "cheap"

