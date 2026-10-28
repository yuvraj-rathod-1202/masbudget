"""Tests for the decentralized game without the sequencing layer (``simultaneous_best_response``)
and the ``lottery_price`` utility used with it (experiment E9)."""

import random

import pytest

from masbudget.decentralized import create_arrival_policy, create_utility_function
from masbudget.decentralized.simultaneous import SimultaneousBestResponse
from masbudget.decentralized.utilities.lottery_price import calibrate_lottery
from tests.helpers import make_agents, make_models, make_state


def run_simultaneous(utility, budget, utility_params=None, seed=0, **params):
    engine = SimultaneousBestResponse(
        create_arrival_policy(None), create_utility_function(utility, **(utility_params or {})), **params
    )
    return engine.allocate(make_agents(), make_models(), budget, random.Random(seed))


@pytest.mark.parametrize("settlement", ["ration", "downgrade"])
@pytest.mark.parametrize("budget", [4.0, 6.0, 10.0, 16.0])
def test_simultaneous_never_overspends_and_serves_every_job_once(settlement, budget):
    history = run_simultaneous("best_model", budget, settlement=settlement)
    assert sorted(a.job_id for a in history) == [0, 1, 2, 3]
    assert sum(a.cost for a in history) <= budget + 1e-9


def test_agents_do_not_see_the_remaining_budget():
    history = run_simultaneous("best_model", 6.0)
    assert all(a.remaining_budget == 6.0 for a in history)


def test_ration_starves_requests_that_no_longer_fit():
    # Everyone requests 'big' (4.0); only one fits in 4.0, the other three starve.
    history = run_simultaneous("best_model", 4.0, settlement="ration")
    assert [a.model for a in history if not a.starved] == ["big"]


def test_downgrade_keeps_cheapest_model_for_every_job():
    history = run_simultaneous("best_model", 4.0, settlement="downgrade")
    assert all(a.model == "cheap" for a in history)


def test_job_allowance_caps_each_job_at_budget_per_job():
    history = run_simultaneous("best_model", 8.0, allowance="job")
    assert all(a.model == "mid" for a in history)


def test_unknown_settlement_is_rejected():
    with pytest.raises(ValueError):
        run_simultaneous("best_model", 8.0, settlement="auction")


def test_lottery_price_draws_once_per_job_and_run():
    utility = create_utility_function("lottery_price", p_low=0.0, p_high=100.0, theta=0.5)
    agents, models = make_agents(), make_models()
    state = make_state(agents, models, remaining_budget=10.0)
    job = agents.jobs[0]
    picks = {max(models, key=lambda m: utility.compute(job, m, state)).name for _ in range(20)}
    assert len(picks) == 1


def test_calibrated_lottery_spends_the_budget_in_expectation():
    agents, models = make_agents(), make_models()
    tasks = [j.task for j in agents.jobs]
    params = calibrate_lottery(tasks, models, budget=7.0)
    assert params["p_low"] < params["p_high"] and 0.0 <= params["theta"] <= 1.0

    def spend(lam):
        return sum(max(models, key=lambda m: m.quality(t) - lam * m.cost).cost for t in tasks)

    expected = params["theta"] * spend(params["p_low"]) + (1 - params["theta"]) * spend(params["p_high"])
    assert expected == pytest.approx(7.0, abs=1e-6)
