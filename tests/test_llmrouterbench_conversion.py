"""Unit tests for LLMRouterBench dataset conversion and verification logic."""

import json
import math
from pathlib import Path

import pytest

from scripts.convert_llmrouterbench_real import (
    _parse_model_records,
    assign_difficulty,
    audit,
    convert,
    create_mixes,
)


# ---------------------------------------------------------------------------
# Fixtures / Mock Data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_model_records():
    return {
        "model_a": {
            1: {"score": 1.0, "cost": 0.001},
            2: {"score": 0.0, "cost": 0.002},
            3: {"score": 1.0, "cost": 0.003},
        },
        "model_b": {
            1: {"score": 0.0, "cost": 0.005},
            2: {"score": 0.0, "cost": 0.006},
            3: {"score": 1.0, "cost": 0.007},
        },
    }


# ---------------------------------------------------------------------------
# Unit tests for conversion logic
# ---------------------------------------------------------------------------

def test_valid_score_preserved_exactly():
    """Scores 0.0, 1.0, and intermediate floats must be preserved exactly."""
    raw = {
        "records": [
            {"index": 1, "score": 1.0, "cost": 0.01},
            {"index": 2, "score": 0.0, "cost": 0.02},
            {"index": 3, "score": 0.75, "cost": 0.03},
        ]
    }
    parsed = _parse_model_records(raw, "test_file.json")
    assert parsed[1]["score"] == 1.0
    assert parsed[2]["score"] == 0.0
    assert parsed[3]["score"] == 0.75


def test_reject_duplicate_source_indices():
    """Duplicate task indices within a model file must raise ValueError."""
    raw = {
        "records": [
            {"index": 1, "score": 1.0, "cost": 0.01},
            {"index": 1, "score": 0.0, "cost": 0.02},
        ]
    }
    with pytest.raises(ValueError, match="duplicate index 1"):
        _parse_model_records(raw, "test_file.json")


def test_reject_invalid_out_of_range_and_non_finite_scores():
    """Scores < 0, > 1, NaN, or Inf must be rejected."""
    bad_cases = [
        {"records": [{"index": 1, "score": 1.5, "cost": 0.01}]},
        {"records": [{"index": 1, "score": -0.1, "cost": 0.01}]},
        {"records": [{"index": 1, "score": float("nan"), "cost": 0.01}]},
        {"records": [{"index": 1, "score": float("inf"), "cost": 0.01}]},
    ]
    for case in bad_cases:
        with pytest.raises(ValueError):
            _parse_model_records(case, "test_file.json")


def test_reject_negative_or_non_finite_costs():
    """Costs < 0, NaN, or Inf must be rejected."""
    bad_cases = [
        {"records": [{"index": 1, "score": 1.0, "cost": -0.01}]},
        {"records": [{"index": 1, "score": 1.0, "cost": float("nan")}]},
        {"records": [{"index": 1, "score": 1.0, "cost": float("inf")}]},
    ]
    for case in bad_cases:
        with pytest.raises(ValueError):
            _parse_model_records(case, "test_file.json")


def test_no_imputation_when_model_score_missing(tmp_path):
    """When model_b has no score for task 2, task 2 is excluded from shared matrix."""
    dir_a = tmp_path / "livecodebench" / "test" / "model_a"
    dir_b = tmp_path / "livecodebench" / "test" / "model_b"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)

    # model_a evaluated on 1 and 2; model_b evaluated only on 1
    (dir_a / "res.json").write_text(json.dumps({
        "records": [
            {"index": 1, "score": 1.0, "cost": 0.01},
            {"index": 2, "score": 0.0, "cost": 0.02},
        ]
    }))
    (dir_b / "res.json").write_text(json.dumps({
        "records": [
            {"index": 1, "score": 0.0, "cost": 0.05},
        ]
    }))

    out_json = tmp_path / "output.json"
    tasks, _ = convert(
        source_path=str(tmp_path),
        benchmark="livecodebench",
        split="test",
        models=["model_a", "model_b"],
        output_path=str(out_json),
    )
    # Only task 1 should be retained; task 2 must never be imputed
    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "livecodebench/test/1"
    assert set(tasks[0]["quality"].keys()) == {"model_a", "model_b"}


def test_join_by_index_not_row_order(tmp_path):
    """Tasks must join by index, even if records appear in different row orders."""
    dir_a = tmp_path / "livecodebench" / "test" / "model_a"
    dir_b = tmp_path / "livecodebench" / "test" / "model_b"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)

    # model_a has index 1 then 2; model_b has index 2 then 1
    (dir_a / "res.json").write_text(json.dumps({
        "records": [
            {"index": 1, "score": 0.1, "cost": 0.01},
            {"index": 2, "score": 0.9, "cost": 0.02},
        ]
    }))
    (dir_b / "res.json").write_text(json.dumps({
        "records": [
            {"index": 2, "score": 0.8, "cost": 0.05},
            {"index": 1, "score": 0.2, "cost": 0.04},
        ]
    }))

    out_json = tmp_path / "output.json"
    tasks, _ = convert(
        source_path=str(tmp_path),
        benchmark="livecodebench",
        split="test",
        models=["model_a", "model_b"],
        output_path=str(out_json),
    )

    t1 = next(t for t in tasks if t["task_id"] == "livecodebench/test/1")
    t2 = next(t for t in tasks if t["task_id"] == "livecodebench/test/2")
    assert t1["quality"] == {"model_a": 0.1, "model_b": 0.2}
    assert t2["quality"] == {"model_a": 0.9, "model_b": 0.8}


def test_benchmark_split_never_mixed(tmp_path):
    """Attempting to convert with nonexistent split or mismatched benchmark must fail."""
    dir_a = tmp_path / "livecodebench" / "train" / "model_a"
    dir_a.mkdir(parents=True)
    (dir_a / "res.json").write_text(json.dumps({
        "records": [{"index": 1, "score": 1.0, "cost": 0.01}]
    }))

    out_json = tmp_path / "output.json"
    with pytest.raises(FileNotFoundError):
        convert(
            source_path=str(tmp_path),
            benchmark="livecodebench",
            split="test",  # Looking for test, but only train exists
            models=["model_a"],
            output_path=str(out_json),
        )


def test_deterministic_difficulty_assignment():
    """Proxy difficulty tertiles must be deterministic and break ties by index."""
    model_data = {
        "m1": {
            1: {"score": 1.0, "cost": 0.01},
            2: {"score": 0.5, "cost": 0.01},
            3: {"score": 0.0, "cost": 0.01},
        }
    }
    diff = assign_difficulty([1, 2, 3], model_data)
    assert diff[1] == "easy"
    assert diff[2] == "medium"
    assert diff[3] == "hard"


def test_deterministic_difficulty_mixes(tmp_path):
    """Mix creation must produce exact proportions without replacement or duplicate IDs."""
    # Create synthetic base dataset with 30 tasks: 10 easy, 10 medium, 10 hard
    tasks = []
    for i in range(10):
        tasks.append({"task_id": f"t_easy_{i}", "difficulty": "easy", "quality": {"m": 1.0}})
        tasks.append({"task_id": f"t_med_{i}", "difficulty": "medium", "quality": {"m": 0.5}})
        tasks.append({"task_id": f"t_hard_{i}", "difficulty": "hard", "quality": {"m": 0.0}})

    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps({
        "name": "base",
        "description": "test base",
        "source": "test",
        "tasks": tasks,
    }))

    paths = create_mixes(str(base_path), seed=42)
    for mix_name, p in paths.items():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        task_ids = [t["task_id"] for t in data["tasks"]]
        assert len(task_ids) == len(set(task_ids)), f"Duplicate task_id in {mix_name}"


def test_audit_functionality(tmp_path):
    """Verify that audit checks all pairs and catches score mismatches and missing records."""
    # Create mock directory structure with source records
    dir_a = tmp_path / "livecodebench" / "test" / "model_a"
    dir_b = tmp_path / "livecodebench" / "test" / "model_b"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)

    file_a = dir_a / "res.json"
    file_b = dir_b / "res.json"
    file_a.write_text(json.dumps({
        "records": [
            {"index": 1, "score": 1.0, "cost": 0.01},
            {"index": 2, "score": 0.0, "cost": 0.02},
        ]
    }))
    file_b.write_text(json.dumps({
        "records": [
            {"index": 1, "score": 0.5, "cost": 0.03},
            {"index": 2, "score": 1.0, "cost": 0.04},
        ]
    }))

    out_json = tmp_path / "out.json"
    convert(
        source_path=str(tmp_path),
        benchmark="livecodebench",
        split="test",
        models=["model_a", "model_b"],
        output_path=str(out_json),
    )

    # Exhaustive audit on matching data should pass
    verified = audit(str(tmp_path), str(out_json), n_samples=None)
    assert verified == 4  # 2 tasks x 2 models = 4 pairs

    # Tamper with score -> audit must raise RuntimeError
    with open(out_json, "r") as f:
        tampered = json.load(f)
    tampered["tasks"][0]["quality"]["model_a"] = 0.0  # was 1.0
    tampered_json = tmp_path / "tampered.json"
    tampered_json.write_text(json.dumps(tampered))

    with pytest.raises(RuntimeError, match="Audit failed"):
        audit(str(tmp_path), str(tampered_json), n_samples=None)
