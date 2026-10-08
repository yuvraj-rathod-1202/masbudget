"""Metric tests on a hand-built ExperimentResult (values checked by hand)."""

import math

import pytest

from masbudget.core import CENTRALIZED, DECENTRALIZED, FALLBACK, Assignment, ExperimentResult, RunResult
from masbudget.metrics import METRICS, compute_metrics, get_metric

BUDGET = 8.0


def run(paradigm, rows, seed=0):
    """rows: (agent_id, model, quality, cost); list order = arrival order for decentralized."""
    assignments = [
        Assignment(agent_id, model, q, c, step=i if paradigm == DECENTRALIZED else None)
        for i, (agent_id, model, q, c) in enumerate(rows)
    ]
    return RunResult(paradigm, "test", seed, BUDGET, assignments)


# total quality 2.65, cost 8, no starvation
CENTRAL = [(0, "cheap", 0.80, 1.0), (1, "mid", 0.75, 2.0), (2, "big", 0.90, 4.0), (3, "cheap", 0.20, 1.0)]
# total quality 2.45, cost 8, no starvation
DEC_A = [(0, "mid", 0.85, 2.0), (1, "mid", 0.75, 2.0), (2, "mid", 0.40, 2.0), (3, "mid", 0.45, 2.0)]
# total quality 1.75, cost 8, two agents starved
DEC_B = [(0, "big", 0.90, 4.0), (1, "big", 0.85, 4.0), (2, FALLBACK, 0.0, 0.0), (3, FALLBACK, 0.0, 0.0)]


@pytest.fixture
def result():
    r = ExperimentResult("test", BUDGET, 4)
    r.runs[CENTRALIZED] = [run(CENTRALIZED, CENTRAL, 0), run(CENTRALIZED, CENTRAL, 1)]
    r.runs[DECENTRALIZED] = [run(DECENTRALIZED, DEC_A, 0), run(DECENTRALIZED, DEC_B, 1)]
    return r


def mean(metric, result, paradigm, **params):
    return get_metric(metric, **params).compute(result)[paradigm]["mean"]


# ---------------------------------------------------------------- contract (all metrics)


@pytest.mark.parametrize("name", METRICS.names())
def test_returns_values_keyed_by_paradigm(name, result):
    out = get_metric(name).compute(result)
    assert isinstance(out, dict)
    assert set(out) <= {CENTRALIZED, DECENTRALIZED}


# ---------------------------------------------------------------- metric-specific


def test_average_quality(result):
    assert mean("average_quality", result, CENTRALIZED) == pytest.approx(2.65 / 4)
    assert mean("average_quality", result, DECENTRALIZED) == pytest.approx((2.45 + 1.75) / 8)


def test_std_is_over_runs(result):
    out = get_metric("average_quality").compute(result)[DECENTRALIZED]
    assert out["std"] == pytest.approx((2.45 - 1.75) / 8)
    assert out["n"] == 2


def test_success_rate(result):
    assert mean("success_rate", result, CENTRALIZED, tau=0.8) == pytest.approx(0.5)
    assert mean("success_rate", result, DECENTRALIZED, tau=0.8) == pytest.approx((0.25 + 0.5) / 2)


def test_budget_metrics(result):
    assert mean("total_cost", result, CENTRALIZED) == pytest.approx(8.0)
    assert mean("budget_utilization", result, DECENTRALIZED) == pytest.approx(1.0)
    assert mean("cost_effectiveness", result, CENTRALIZED) == pytest.approx(2.65 / 8)


def test_social_welfare(result):
    assert mean("social_welfare", result, DECENTRALIZED) == pytest.approx((2.45 + 1.75) / 2)


def test_optimality_gap_pairs_runs(result):
    expected = ((2.65 - 2.45) / 2.65 + (2.65 - 1.75) / 2.65) / 2
    assert mean("optimality_gap", result, DECENTRALIZED) == pytest.approx(expected)


def test_optimality_gap_needs_both_paradigms(result):
    del result.runs[CENTRALIZED]
    assert get_metric("optimality_gap").compute(result) == {}


def test_jains_fairness(result):
    squares = 0.80**2 + 0.75**2 + 0.90**2 + 0.20**2
    assert mean("jains_fairness", result, CENTRALIZED) == pytest.approx(2.65**2 / (4 * squares))


def test_starvation_rate(result):
    assert mean("starvation_rate", result, CENTRALIZED) == 0.0
    assert mean("starvation_rate", result, DECENTRALIZED) == pytest.approx((0.0 + 0.5) / 2)


def test_first_mover_advantage(result):
    result.runs[DECENTRALIZED] = [run(DECENTRALIZED, DEC_A)]
    out = get_metric("first_mover_advantage").compute(result)
    assert set(out) == {DECENTRALIZED}  # undefined without an arrival order
    assert out[DECENTRALIZED]["mean"] == pytest.approx(0.85 / 0.45)


def test_first_mover_advantage_is_infinite_when_last_quartile_starves(result):
    result.runs[DECENTRALIZED] = [run(DECENTRALIZED, DEC_B)]
    assert math.isinf(mean("first_mover_advantage", result, DECENTRALIZED))


def test_order_sensitivity(result):
    out = get_metric("order_sensitivity").compute(result)
    assert out[CENTRALIZED] == pytest.approx(0.0)
    assert out[DECENTRALIZED] == pytest.approx(((2.45 - 1.75) / 2) ** 2)


def test_compute_metrics_accepts_names_and_params(result):
    out = compute_metrics(result, ["total_cost", {"success_rate": {"tau": 0.8}}])
    assert set(out) == {"total_cost", "success_rate"}
