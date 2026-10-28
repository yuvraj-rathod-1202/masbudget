"""Tests for the extensions used by the experiment study: the engine's budget reserve,
the ``budget_pacing`` utility and its price calibration, the cascade's ``verify_key`` /
``skip_dominated`` options and the Shapley ``upgrade_gain`` characteristic."""

import math
import random

import pytest

from masbudget.centralized import create_centralized_policy
from masbudget.core import Task
from masbudget.decentralized import create_arrival_policy, create_utility_function
from masbudget.decentralized.sequential import SequentialBestResponse
from masbudget.decentralized.utilities.budget_pacing import calibrate_price
from masbudget.registry import AgentRegistry
from tests.helpers import brute_force_optimum, make_agents, make_models, make_multi_agents, make_state


def run_sequential(utility, budget, reserve, utility_params=None, agents=None):
    engine = SequentialBestResponse(
        create_arrival_policy(None), create_utility_function(utility, **(utility_params or {})), reserve=reserve
    )
    agents = agents or make_agents()
    return engine.allocate(agents, make_models(), budget, random.Random(0))


# ---------------------------------------------------------------- reserve


def test_without_reserve_greedy_agents_starve_late_jobs():
    # Severe budget = 4 * c_min: job 0 takes 'big' (4.0) and everyone else is starved.
    history = run_sequential("best_model", 4.0, reserve=False)
    assert history[0].model == "big"
    assert all(a.starved for a in history[1:])


@pytest.mark.parametrize("budget", [4.0, 5.0, 8.0, 16.0])
def test_reserve_prevents_starvation_when_budget_covers_cheapest(budget):
    history = run_sequential("best_model", budget, reserve=True)
    assert not any(a.starved for a in history)
    assert sum(a.cost for a in history) <= budget + 1e-9


def test_reserve_leaves_floor_for_waiting_jobs():
    # B = 7: job 0 may spend at most 7 - 3 * 1 = 4 -> 'big'; job 1 then has 3 - 2 = 1 -> 'cheap'.
    history = run_sequential("best_model", 7.0, reserve=True)
    assert [a.model for a in history] == ["big", "cheap", "cheap", "cheap"]


def test_reserve_serves_as_many_jobs_as_possible_when_floor_is_broken():
    # B = 3 < 4 * c_min: the floor cannot hold, so jobs take the cheapest model while it lasts.
    history = run_sequential("best_model", 3.0, reserve=True)
    assert [a.model for a in history] == ["cheap", "cheap", "cheap", "fallback"]


# ---------------------------------------------------------------- budget_pacing


def test_pacing_with_beta_zero_is_a_posted_price():
    agents, models = make_agents(), make_models()
    u = create_utility_function("budget_pacing", p0=0.2, beta=0.0)
    for remaining, step in [(16.0, 0), (5.0, 1), (1.0, 3)]:
        state = make_state(agents, models, remaining, 16.0, step=step)
        assert u.compute(agents.jobs[2], models.get("big"), state) == pytest.approx(0.90 - 0.2 * 4.0)


def test_pacing_price_rises_when_spending_ahead_of_schedule():
    agents, models = make_agents(), make_models()
    u = create_utility_function("budget_pacing", p0=0.1, beta=4.0)
    on_pace = u.price(make_state(agents, models, 8.0, 16.0, step=2))  # spent 1/2 after 2/4 jobs
    ahead = u.price(make_state(agents, models, 4.0, 16.0, step=2))  # spent 3/4 after 2/4 jobs
    behind = u.price(make_state(agents, models, 12.0, 16.0, step=2))  # spent 1/4 after 2/4 jobs
    assert on_pace == pytest.approx(0.1)
    assert ahead == pytest.approx(0.1 * math.exp(1.0))
    assert behind == pytest.approx(0.1 * math.exp(-1.0))


def test_pacing_price_floor_lets_feedback_act_from_zero():
    agents, models = make_agents(), make_models()
    zero = create_utility_function("budget_pacing", p0=0.0, beta=4.0)
    floored = create_utility_function("budget_pacing", p0=0.0, beta=4.0, p_min=0.1)
    ahead = make_state(agents, models, 4.0, 16.0, step=2)
    assert zero.price(ahead) == 0.0
    assert floored.price(ahead) == pytest.approx(0.1 * math.exp(1.0))
    assert create_utility_function("budget_pacing", p0=0.3, p_min=0.1, beta=0.0).price(ahead) == pytest.approx(0.3)


def test_pacing_horizon_param_overrides_job_count():
    agents, models = make_agents(), make_models()
    u = create_utility_function("budget_pacing", p0=1.0, beta=1.0, horizon=8)
    assert u.price(make_state(agents, models, 16.0, 16.0, step=4)) == pytest.approx(math.exp(-0.5))


def test_calibrated_price_makes_independent_choices_feasible():
    tasks = [job.task for job in make_agents().jobs]
    models = make_models()
    for budget in [4.0, 6.0, 8.0, 12.0]:
        lam = calibrate_price(tasks, models, budget)
        spend = sum(max(models, key=lambda m: m.quality(t) - lam * m.cost).cost for t in tasks)
        assert spend <= budget + 1e-9
    assert calibrate_price(tasks, models, 16.0) == 0.0  # everyone can afford 'big'


@pytest.mark.parametrize("budget", [5.0, 8.0, 10.0, 12.0])
def test_calibrated_posted_price_is_near_optimal_on_toy_instance(budget):
    # A posted price solves the LP relaxation; rounding loses at most one job's upgrade.
    agents, models = make_agents(), make_models()
    lam = calibrate_price([job.task for job in agents.jobs], models, budget)
    history = run_sequential("budget_pacing", budget, reserve=True, utility_params={"p0": lam, "beta": 0.0})
    optimum = brute_force_optimum(agents, models, budget)
    assert sum(a.cost for a in history) <= budget + 1e-9
    assert sum(a.quality for a in history) >= optimum - 0.8


# ---------------------------------------------------------------- cascade options


def make_verified_agents():
    """hard-2's realized outcome says 'mid' already passes although q_mid = 0.40."""
    tasks = [
        Task("easy-0", "easy", {"cheap": 0.80, "mid": 0.85, "big": 0.90}, metadata={"pass": {"cheap": 1, "mid": 1, "big": 1}}),
        Task("hard-2", "hard", {"cheap": 0.10, "mid": 0.40, "big": 0.90}, metadata={"pass": {"cheap": 0, "mid": 1, "big": 1}}),
    ]
    return AgentRegistry.from_task_lists([[t] for t in tasks])


def test_cascade_verify_key_escalates_on_realized_outcome():
    agents, models = make_verified_agents(), make_models()
    policy = create_centralized_policy("confidence_cascade", threshold=1.0, verify_key="pass")
    mapping = policy.allocate(agents, models, 100.0, random.Random(0))
    assert mapping == {0: ["cheap"], 1: ["cheap", "mid"]}


def test_cascade_jump_skips_models_with_lower_expected_quality():
    # Expected q of 'mid' on the hard task is below 'cheap', so the ladder would stall; jump goes to 'big'.
    task = Task("hard", "hard", {"cheap": 0.20, "mid": 0.10, "big": 0.60}, metadata={"pass": {"cheap": 0, "mid": 0, "big": 1}})
    agents, models = AgentRegistry.from_task_lists([[task]]), make_models()
    policy = create_centralized_policy("confidence_cascade", threshold=1.0, verify_key="pass", escalation="jump")
    assert policy.allocate(agents, models, 5.0, random.Random(0)) == {0: ["cheap", "big"]}
    assert policy.allocate(agents, models, 4.0, random.Random(0)) == {0: ["cheap"]}  # big no longer affordable


def test_cascade_skip_dominated_drops_dominated_models():
    agents = make_agents()
    models = make_models({"cheap": 1.0, "mid": 2.0, "big": 4.0})
    models.add("worse", 3.0)  # not in the task tables; give it a quality table via a fresh agent set
    tasks = [Task(t.task_id, t.difficulty, {**t.quality, "worse": 0.0}) for t in (j.task for j in agents.jobs)]
    agents = AgentRegistry.from_task_lists([[t] for t in tasks])
    policy = create_centralized_policy("confidence_cascade", threshold=0.95, skip_dominated=True)
    mapping = policy.allocate(agents, models, 100.0, random.Random(0))
    assert all("worse" not in attempts for attempts in mapping.values())


# ---------------------------------------------------------------- shapley upgrade_gain


def test_shapley_upgrade_gain_favours_agents_with_hard_tasks():
    agents, models = make_multi_agents(), make_models()  # agent 0: easy + medium, agent 1: two hard
    u = create_utility_function("shapley_budget_share", lambda_cost=0.1, gamma=1.0, characteristic="upgrade_gain")
    state = make_state(agents, models, 16.0)
    phi = {a.agent_id: u.compute_phi(a, state) for a in agents}
    assert phi[1] > phi[0]
    # Additive game: phi_i is the agent's own mean upgrade gain.
    assert phi[0] == pytest.approx(((0.90 - 0.80) + (0.85 - 0.50)) / 2)


# ---------------------------------------------------------------- serve-all MCKP and abstention


def test_serve_all_mckp_gives_every_job_a_model():
    agents, models = make_agents(), make_models({"cheap": 1.0, "mid": 2.0, "big": 2.5})
    free = create_centralized_policy("optimal_mckp").allocate(agents, models, 5.0, random.Random(0))
    served = create_centralized_policy("optimal_mckp", serve_all=True).allocate(agents, models, 5.0, random.Random(0))
    assert "fallback" in free.values()  # skipping a job to fund 'big' elsewhere pays off
    assert "fallback" not in served.values()
    assert sum(models.get(m).cost for m in served.values()) <= 5.0 + 1e-9


def test_abstaining_agent_declines_when_price_exceeds_value():
    engine = SequentialBestResponse(
        create_arrival_policy(None), create_utility_function("budget_pacing", p0=0.5, beta=0.0), abstain=True
    )
    history = engine.allocate(make_agents(), make_models(), 16.0, random.Random(0))
    # Jobs: easy-0, medium-1, hard-2, hard-3. U(cheap) = q - 0.5 < 0 for the hard tasks -> abstain;
    # medium-1 ties at 0 and the tie goes to the first candidate, cheap.
    assert [a.model for a in history] == ["cheap", "cheap", "fallback", "fallback"]


# ---------------------------------------------------------------- ucb_pacing (learning agents)


def test_ucb_learns_from_revealed_outcomes_not_from_q():
    # q says 'big' is great, but the revealed outcomes say only 'cheap' ever passes.
    tasks = [Task(f"t{i}", "hard", {"cheap": 0.1, "mid": 0.2, "big": 0.9}, metadata={"realized": {"cheap": 1, "mid": 0, "big": 0}})
             for i in range(12)]
    agents = AgentRegistry.from_task_lists([[t] for t in tasks])
    engine = SequentialBestResponse(
        create_arrival_policy(None), create_utility_function("ucb_pacing", c=0.0, beta=0.0, p0=0.0), reserve=True
    )
    history = engine.allocate(agents, make_models(), 100.0, random.Random(0))
    assert [a.model for a in history[-4:]] == ["cheap"] * 4  # after trying the others, it settles on 'cheap'


def test_ucb_resets_between_runs():
    agents, models = make_agents(), make_models()
    u = create_utility_function("ucb_pacing", c=0.0)
    engine = SequentialBestResponse(create_arrival_policy(None), u, reserve=True)
    first = engine.allocate(agents, models, 16.0, random.Random(0))
    second = engine.allocate(agents, models, 16.0, random.Random(0))
    assert [a.model for a in first] == [a.model for a in second]


def test_ucb_prior_sets_initial_estimate():
    u = create_utility_function("ucb_pacing", c=0.0, prior={"big": 0.8, "cheap": 0.3, "mid": 0.5}, prior_n=10)
    assert u.estimate("hard", "big") == pytest.approx(0.8)
