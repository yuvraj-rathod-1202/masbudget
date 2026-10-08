"""Arrival (ordering) policy tests.

Contract tests run for every registered arrival policy. Policy-specific tests encode the
behaviour described in the proposal. Tests skip while a policy raises NotImplementedError.
"""

import random

import pytest

from masbudget.decentralized import ARRIVAL_POLICIES, BaseUtilityFunction, create_arrival_policy
from masbudget.decentralized.sequential import SequentialBestResponse
from tests.helpers import BUDGETS, call_or_skip, make_agents, make_models, make_state

# Params each policy gets in these tests. Add an entry if your policy needs parameters.
POLICY_PARAMS: dict[str, dict] = {}


class QualityUtility(BaseUtilityFunction):
    """Test double: U = q, so only the ordering matters."""

    def compute(self, agent, model, state):
        return model.quality(agent.task)


def make_policy(name):
    return create_arrival_policy(name, **POLICY_PARAMS.get(name, {}))


def first_pick(name, seed=0):
    agents, models = make_agents(), make_models()
    state = make_state(agents, models, BUDGETS["moderate"], seed=seed)
    return call_or_skip(make_policy(name).select_next, list(agents), state)


def arrival_order(name, budget=BUDGETS["moderate"], seed=0):
    game = SequentialBestResponse(make_policy(name), QualityUtility())
    assignments = call_or_skip(game.allocate, make_agents(), make_models(), budget, random.Random(seed))
    return [a.agent_id for a in assignments]


# ---------------------------------------------------------------- contract (all policies)


@pytest.mark.parametrize("name", ARRIVAL_POLICIES.names())
def test_returns_a_waiting_agent(name):
    agents, models = make_agents(), make_models()
    waiting = list(agents)[1:]
    state = make_state(agents, models, BUDGETS["moderate"], step=1)
    picked = call_or_skip(make_policy(name).select_next, list(waiting), state)
    assert picked.agent_id in {a.agent_id for a in waiting}


@pytest.mark.parametrize("name", ARRIVAL_POLICIES.names())
def test_single_waiting_agent_is_picked(name):
    agents, models = make_agents(), make_models()
    last = list(agents)[-1]
    state = make_state(agents, models, BUDGETS["moderate"], step=len(agents) - 1)
    assert call_or_skip(make_policy(name).select_next, [last], state).agent_id == last.agent_id


@pytest.mark.parametrize("name", ARRIVAL_POLICIES.names())
def test_full_game_serves_every_agent_once(name):
    order = arrival_order(name)
    assert sorted(order) == make_agents().ids


@pytest.mark.parametrize("name", ARRIVAL_POLICIES.names())
def test_same_seed_gives_same_order(name):
    assert arrival_order(name, seed=3) == arrival_order(name, seed=3)


# ---------------------------------------------------------------- policy-specific


class TestFCFS:
    def test_serves_in_registration_order(self):
        assert arrival_order("fcfs") == [0, 1, 2, 3]


class TestRandomFCFS:
    def test_order_varies_across_seeds(self):
        orders = {tuple(arrival_order("random_fcfs", seed=s)) for s in range(30)}
        assert len(orders) > 1

    def test_every_agent_can_go_first(self):
        firsts = {first_pick("random_fcfs", seed=s).agent_id for s in range(200)}
        assert firsts == set(make_agents().ids)


class TestHardestTaskFirst:
    def test_hard_tasks_first_easy_last(self):
        order = arrival_order("hardest_task_first")
        assert set(order[:2]) == {2, 3}  # hard-2, hard-3
        assert order[-1] == 0  # easy-0


class TestEasiestTaskFirst:
    def test_easy_first_hard_last(self):
        order = arrival_order("easiest_task_first")
        assert order[:2] == [0, 1]  # easy-0, medium-1
        assert set(order[2:]) == {2, 3}


class TestMarginalROIPriority:
    def test_highest_quality_gain_per_cost_goes_first(self):
        # hard-2 has the best dq/dc over the cheap baseline (mid: 0.30/1, big: 0.80/3).
        assert first_pick("marginal_roi_priority").agent_id == 2
