"""Arrival (ordering) policy tests.

Contract tests run for every registered arrival policy. Policy-specific tests encode the
behaviour described in the proposal. Tests skip while a policy raises NotImplementedError.
"""

import random

import pytest

from masbudget.decentralized import ARRIVAL_POLICIES, BaseUtilityFunction, create_arrival_policy
from masbudget.decentralized.sequential import SequentialBestResponse
from tests.helpers import BUDGETS, call_or_skip, make_agents, make_models, make_multi_agents, make_state

# Params each policy gets in these tests. Add an entry if your policy needs parameters.
POLICY_PARAMS: dict[str, dict] = {}


class QualityUtility(BaseUtilityFunction):
    """Test double: U = q, so only the ordering matters."""

    def compute(self, job, model, state):
        return model.quality(job.task)


def make_policy(name):
    return create_arrival_policy(name, **POLICY_PARAMS.get(name, {}))


def first_pick(name, seed=0):
    agents, models = make_agents(), make_models()
    state = make_state(agents, models, BUDGETS["moderate"], seed=seed)
    return call_or_skip(make_policy(name).select_next, agents.jobs, state)


def arrival_assignments(name, budget=BUDGETS["moderate"], seed=0, agents=None):
    game = SequentialBestResponse(make_policy(name), QualityUtility())
    return call_or_skip(game.allocate, agents or make_agents(), make_models(), budget, random.Random(seed))


def arrival_order(name, budget=BUDGETS["moderate"], seed=0):
    """Job ids in serving order (equal to agent ids on the one-task-per-agent toy)."""
    return [a.job_id for a in arrival_assignments(name, budget, seed)]


# ---------------------------------------------------------------- contract (all policies)


@pytest.mark.parametrize("name", ARRIVAL_POLICIES.names())
def test_returns_a_waiting_job(name):
    agents, models = make_agents(), make_models()
    waiting = agents.jobs[1:]
    state = make_state(agents, models, BUDGETS["moderate"], step=1)
    picked = call_or_skip(make_policy(name).select_next, list(waiting), state)
    assert picked.job_id in {j.job_id for j in waiting}


@pytest.mark.parametrize("name", ARRIVAL_POLICIES.names())
def test_single_waiting_job_is_picked(name):
    agents, models = make_agents(), make_models()
    last = agents.jobs[-1]
    state = make_state(agents, models, BUDGETS["moderate"], step=len(agents.jobs) - 1)
    assert call_or_skip(make_policy(name).select_next, [last], state).job_id == last.job_id


@pytest.mark.parametrize("name", ARRIVAL_POLICIES.names())
def test_full_game_serves_every_job_once(name):
    order = arrival_order(name)
    assert sorted(order) == make_agents().job_ids


@pytest.mark.parametrize("name", ARRIVAL_POLICIES.names())
def test_multi_task_agents_act_once_per_task(name):
    out = arrival_assignments(name, agents=make_multi_agents())
    assert sorted(a.job_id for a in out) == [0, 1, 2, 3]
    assert sorted(a.agent_id for a in out) == [0, 0, 1, 1]


@pytest.mark.parametrize("name", ARRIVAL_POLICIES.names())
def test_same_seed_gives_same_order(name):
    assert arrival_order(name, seed=3) == arrival_order(name, seed=3)


# ---------------------------------------------------------------- policy-specific


class TestFCFS:
    def test_serves_in_registration_order(self):
        assert arrival_order("fcfs") == [0, 1, 2, 3]

    def test_interleaves_agents_round_robin(self):
        out = arrival_assignments("fcfs", agents=make_multi_agents())
        assert [a.agent_id for a in out] == [0, 1, 0, 1]
        assert [a.job_id for a in out] == [0, 1, 2, 3]


class TestRandomFCFS:
    def test_order_varies_across_seeds(self):
        orders = {tuple(arrival_order("random_fcfs", seed=s)) for s in range(30)}
        assert len(orders) > 1

    def test_every_agent_can_go_first(self):
        firsts = {first_pick("random_fcfs", seed=s).job_id for s in range(200)}
        assert firsts == set(make_agents().job_ids)


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
        assert first_pick("marginal_roi_priority").job_id == 2
