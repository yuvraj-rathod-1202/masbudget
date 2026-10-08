"""Decentralized utility function tests.

Contract tests run for every registered utility. Utility-specific tests encode the formulas
in the proposal. Tests skip while a utility raises NotImplementedError.
"""

import math
import random

import pytest

from masbudget.decentralized import UTILITY_FUNCTIONS, create_arrival_policy, create_utility_function
from masbudget.decentralized.sequential import SequentialBestResponse
from tests.helpers import BUDGETS, call_or_skip, make_agents, make_models, make_state

# Params each utility gets in these tests. Add an entry if your utility needs parameters.
POLICY_PARAMS = {
    "quasi_linear": {"lambda_cost": 0.1},
    "dynamic_shadow_price": {"p0": 0.1, "alpha": 1.5},
    "shapley_budget_share": {"lambda_cost": 0.1, "gamma": 1.0},
}


def make_utility(name, **override):
    return create_utility_function(name, **{**POLICY_PARAMS.get(name, {}), **override})


def utility(name, agent_id, model_name, remaining, total=BUDGETS["generous"], step=0, **override):
    agents, models = make_agents(), make_models()
    state = make_state(agents, models, remaining, total, step=step)
    return call_or_skip(make_utility(name, **override).compute, agents.get(agent_id), models.get(model_name), state)


def big_vs_cheap_preference(name, remaining):
    """How much more agent hard-2 prefers 'big' over 'cheap' when B_k = remaining."""
    return utility(name, 2, "big", remaining) - utility(name, 2, "cheap", remaining)


# ---------------------------------------------------------------- contract (all utilities)


@pytest.mark.parametrize("remaining", [BUDGETS["generous"], 4.0, 1.0], ids=["full", "quarter", "min"])
@pytest.mark.parametrize("name", UTILITY_FUNCTIONS.names())
def test_returns_a_number_for_every_affordable_model(name, remaining):
    agents, models = make_agents(), make_models()
    u = make_utility(name)
    for step, agent in enumerate(agents):
        state = make_state(agents, models, remaining, BUDGETS["generous"], step=step)
        for model in models.affordable(remaining):
            value = call_or_skip(u.compute, agent, model, state)
            assert isinstance(value, (int, float)) and not math.isnan(value)


@pytest.mark.parametrize("name", UTILITY_FUNCTIONS.names())
def test_is_deterministic(name):
    assert utility(name, 1, "mid", 8.0) == utility(name, 1, "mid", 8.0)


@pytest.mark.parametrize("budget", BUDGETS.values(), ids=BUDGETS.keys())
@pytest.mark.parametrize("name", UTILITY_FUNCTIONS.names())
def test_full_game_stays_within_budget(name, budget):
    game = SequentialBestResponse(create_arrival_policy(None), make_utility(name))
    assignments = call_or_skip(game.allocate, make_agents(), make_models(), budget, random.Random(0))
    assert sum(a.cost for a in assignments) <= budget + 1e-9


# ---------------------------------------------------------------- utility-specific


class TestQuasiLinear:
    def test_formula(self):
        # q - lambda * c = 0.90 - 0.1 * 4
        assert utility("quasi_linear", 2, "big", 8.0) == pytest.approx(0.90 - 0.1 * 4.0)


class TestDynamicShadowPrice:
    @pytest.mark.parametrize("remaining", [16.0, 8.0, 4.0])
    def test_formula(self, remaining):
        price = 0.1 * (16.0 / remaining) ** 1.5
        assert utility("dynamic_shadow_price", 2, "big", remaining) == pytest.approx(0.90 - price * 4.0)

    def test_expensive_models_lose_appeal_as_budget_depletes(self):
        assert big_vs_cheap_preference("dynamic_shadow_price", 16.0) > big_vs_cheap_preference(
            "dynamic_shadow_price", 4.0
        )


class TestShapleyBudgetShare:
    def test_gamma_zero_reduces_to_quasi_linear(self):
        assert utility("shapley_budget_share", 2, "big", 8.0, gamma=0.0) == pytest.approx(0.90 - 0.1 * 4.0)

    def test_critical_agents_get_lower_cost_penalty(self):
        # Critical agent (hard-2) has a larger delta_q (0.80) than easy agent (easy-0, delta_q 0.10)
        # So effective cost penalty for hard-2 is smaller than for easy-0.
        penalty_hard = 0.90 - utility("shapley_budget_share", 2, "big", 8.0)
        penalty_easy = 0.90 - utility("shapley_budget_share", 0, "big", 8.0)
        assert penalty_hard < penalty_easy



class TestMarginalROIRatio:
    @pytest.mark.parametrize("model, expected", [("mid", (0.40 - 0.10) / 1.0), ("big", (0.90 - 0.10) / 3.0)])
    def test_formula_against_cheapest_baseline(self, model, expected):
        assert utility("marginal_roi_ratio", 2, model, 8.0) == pytest.approx(expected)


class TestCostSensitiveAdaptive:
    def test_expensive_models_lose_appeal_as_budget_depletes(self):
        assert big_vs_cheap_preference("cost_sensitive_adaptive", 16.0) > big_vs_cheap_preference(
            "cost_sensitive_adaptive", 4.0
        )


class TestBestModel:
    def test_ranks_models_by_quality(self):
        ranking = sorted(["cheap", "mid", "big"], key=lambda m: utility("best_model", 1, m, 16.0))
        assert ranking == ["cheap", "mid", "big"]
