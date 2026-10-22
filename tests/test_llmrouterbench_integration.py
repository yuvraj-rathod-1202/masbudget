"""Integration tests for generated real LiveCodeBench datasets and experiment config."""

from pathlib import Path

import pytest

from masbudget.config import build_experiment, load_config
from masbudget.dataset import load_dataset

BASE_DATASET = Path("data/llmrouterbench_livecodebench.json")
UNIFORM_DATASET = Path("data/llmrouterbench_livecodebench_uniform.json")
EASY_HEAVY_DATASET = Path("data/llmrouterbench_livecodebench_easy_heavy_80_20.json")
HARD_HEAVY_DATASET = Path("data/llmrouterbench_livecodebench_hard_heavy_20_80.json")
COSTS_FILE = Path("data/llmrouterbench_livecodebench_model_costs.json")
CONFIG_FILE = Path("configs/llmrouterbench_livecodebench.yaml")


@pytest.mark.skipif(not BASE_DATASET.exists(), reason="Real dataset not generated yet")
def test_generated_base_dataset_structure():
    """Verify base dataset conforms to masbudget expectations."""
    tasks = load_dataset(BASE_DATASET)
    assert len(tasks) == 1055

    expected_models = {"deepseek-v3.1-terminus", "deepseek-v3-0324", "gemini-2.5-flash", "deepseek-r1-0528"}
    task_ids = set()
    diff_counts = {"easy": 0, "medium": 0, "hard": 0}

    for t in tasks:
        assert t.task_id not in task_ids, f"Duplicate task_id: {t.task_id}"
        task_ids.add(t.task_id)
        assert set(t.quality.keys()) == expected_models
        for score in t.quality.values():
            assert 0.0 <= score <= 1.0
        assert t.difficulty in diff_counts
        diff_counts[t.difficulty] += 1

    assert diff_counts == {"easy": 351, "medium": 352, "hard": 352}


@pytest.mark.skipif(not UNIFORM_DATASET.exists(), reason="Mix dataset not generated yet")
def test_generated_difficulty_mixes():
    """Verify the 3 generated mix datasets load and have correct proportions."""
    uni_tasks = load_dataset(UNIFORM_DATASET)
    assert len(uni_tasks) == 1053
    uni_diffs = {d: sum(1 for t in uni_tasks if t.difficulty == d) for d in ("easy", "medium", "hard")}
    assert uni_diffs == {"easy": 351, "medium": 351, "hard": 351}

    eh_tasks = load_dataset(EASY_HEAVY_DATASET)
    assert len(eh_tasks) == 400
    eh_diffs = {d: sum(1 for t in eh_tasks if t.difficulty == d) for d in ("easy", "medium", "hard")}
    assert eh_diffs["easy"] == 320   # 80%
    assert eh_diffs["medium"] == 40  # 10%
    assert eh_diffs["hard"] == 40    # 10%

    hh_tasks = load_dataset(HARD_HEAVY_DATASET)
    assert len(hh_tasks) == 400
    hh_diffs = {d: sum(1 for t in hh_tasks if t.difficulty == d) for d in ("easy", "medium", "hard")}
    assert hh_diffs["easy"] == 40    # 10%
    assert hh_diffs["medium"] == 40  # 10%
    assert hh_diffs["hard"] == 320   # 80%


@pytest.mark.skipif(not CONFIG_FILE.exists(), reason="Config file not written yet")
def test_config_model_keys_match_dataset_model_keys():
    """Verify all model keys in config match the dataset quality keys exactly."""
    specs = load_config(CONFIG_FILE)
    base_tasks = load_dataset(BASE_DATASET)
    dataset_models = set(base_tasks[0].quality.keys())

    for spec in specs:
        config_models = set(spec["models"].keys())
        assert config_models == dataset_models, (
            f"Config {spec['name']} models {config_models} do not match dataset models {dataset_models}"
        )
        # Verify experiment builds cleanly
        exp = build_experiment(spec)
        assert len(exp.models.names) == len(dataset_models)
