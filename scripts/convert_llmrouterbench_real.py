#!/usr/bin/env python3
"""Convert LLMRouterBench archive to masbudget benchmark JSON.

Supports extracting from the official LLMRouterBench tar.gz archive or from an
extracted results directory, joining tasks across multiple models, calculating
per-task and aggregate model costs, generating proxy difficulty labels,
subsampling difficulty mixes, and auditing the output against original source records.

Usage examples:
  # List available models in livecodebench/test:
  python scripts/convert_llmrouterbench_real.py --archive BLOB --list-models

  # Convert livecodebench/test with 4 flagship/diverse models:
  python scripts/convert_llmrouterbench_real.py \
      --archive BLOB \
      --benchmark livecodebench --split test \
      --models Llama-3.1-8B-Instruct deepseek-v3-0324 gemini-2.5-flash deepseek-r1-0528 \
      --output data/llmrouterbench_livecodebench.json

  # Generate 3 difficulty mix datasets:
  python scripts/convert_llmrouterbench_real.py \
      --create-mixes data/llmrouterbench_livecodebench.json

  # Audit 20+ randomly sampled task-model pairs against source archive:
  python scripts/convert_llmrouterbench_real.py \
      --archive BLOB \
      --audit data/llmrouterbench_livecodebench.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import tarfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_BASE_DIR = "bench-release"
CATEGORY = "code_generation"
SOURCE_LABEL = "LLMRouterBench"
DESCRIPTION = (
    "LiveCodeBench pass@1 benchmark tasks from LLMRouterBench "
    "(NPULH/LLMRouterBench on HuggingFace), livecodebench/test split. "
    "Scores are binary pass@1 (0.0 = fail, 1.0 = pass) from LiveCodeBench evaluation. "
    "Difficulty labels are an explicit proxy derived from the mean score across the "
    "selected models (sorted descending by mean score; ties broken deterministically by "
    "task source index). Top tertile is 'easy', middle tertile is 'medium', bottom is 'hard'."
)


# ---------------------------------------------------------------------------
# Source Loading Helpers (Archive & Directory)
# ---------------------------------------------------------------------------

def _load_model_records_from_archive(
    archive_path: str | Path,
    benchmark: str,
    split: str,
    target_models: list[str],
) -> tuple[dict[str, dict[int, dict[str, float]]], dict[str, str]]:
    """Single-pass streaming read over the tar.gz archive to load only target models.

    Returns:
        (model_data, member_paths) where model_data is {model: {index: {score, cost}}}
        and member_paths is {model: archive_member_path}
    """
    prefix = f"{BENCHMARK_BASE_DIR}/{benchmark}/{split}/"
    needed = set(target_models)
    model_data: dict[str, dict[int, dict[str, float]]] = {}
    member_paths: dict[str, str] = {}

    print(f"[convert] Scanning archive in single pass: {archive_path}", file=sys.stderr)
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            if not name.startswith(prefix) or not name.endswith(".json"):
                continue

            rel = name[len(prefix):]
            parts = rel.split("/")
            if len(parts) != 2:
                continue

            model = parts[0]
            if model in needed:
                if model in model_data:
                    raise ValueError(
                        f"Duplicate result file found for model '{model}': "
                        f"already loaded {member_paths[model]}, also found {name}"
                    )
                f = tf.extractfile(member)
                if f is None:
                    raise RuntimeError(f"Could not extract {name}")
                raw = json.loads(f.read())
                records = _parse_model_records(raw, name)
                model_data[model] = records
                member_paths[model] = name
                print(f"[convert] Loaded {model}: {len(records)} records from {name}", file=sys.stderr)
                if len(model_data) == len(needed):
                    break

    missing = needed - set(model_data.keys())
    if missing:
        raise KeyError(f"Could not find result files for models: {sorted(missing)} in {archive_path}")

    return model_data, member_paths


def _load_model_records_from_dir(
    source_dir: str | Path,
    benchmark: str,
    split: str,
    target_models: list[str],
) -> tuple[dict[str, dict[int, dict[str, float]]], dict[str, str]]:
    """Load target models from an extracted results directory."""
    source_path = Path(source_dir)
    # Check if directory has bench-release prefix or starts directly with benchmark
    candidates = [
        source_path / BENCHMARK_BASE_DIR / benchmark / split,
        source_path / benchmark / split,
    ]
    base = None
    for cand in candidates:
        if cand.is_dir():
            base = cand
            break
    if base is None:
        raise FileNotFoundError(f"Could not find directory for {benchmark}/{split} under {source_dir}")

    model_data: dict[str, dict[int, dict[str, float]]] = {}
    member_paths: dict[str, str] = {}

    for model in target_models:
        model_dir = base / model
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        json_files = sorted(model_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON result files found in {model_dir}")
        if len(json_files) > 1:
            raise ValueError(f"Ambiguous: multiple JSON files found for model {model} in {model_dir}")

        chosen = json_files[0]
        with chosen.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        records = _parse_model_records(raw, str(chosen))
        model_data[model] = records
        member_paths[model] = str(chosen)
        print(f"[convert] Loaded {model}: {len(records)} records from {chosen}", file=sys.stderr)

    return model_data, member_paths


def _parse_model_records(data: Any, member_name: str) -> dict[int, dict[str, float]]:
    """Parse records from one model file into {index: {score, cost}}.

    Validates:
    - No duplicate indices
    - Scores finite and strictly in [0.0, 1.0]
    - Costs finite and non-negative
    """
    records: list[Any] = data if isinstance(data, list) else data.get("records", [])
    if not records:
        raise ValueError(f"{member_name}: No records found")

    result: dict[int, dict[str, float]] = {}
    for i, rec in enumerate(records):
        idx = rec.get("index")
        if idx is None:
            raise ValueError(f"{member_name}: record {i} missing 'index' field")
        idx = int(idx)
        if idx in result:
            raise ValueError(f"{member_name}: duplicate index {idx} at record {i}")

        score = rec.get("score")
        if score is None:
            raise ValueError(f"{member_name}: record {i} (index={idx}) missing 'score' field")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(f"{member_name}: record {i} (index={idx}) non-finite score {score}")
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"{member_name}: record {i} (index={idx}) score {score} out of [0, 1]")

        cost = rec.get("cost")
        cost = float(cost) if cost is not None else 0.0
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(f"{member_name}: record {i} (index={idx}) invalid cost {cost}")

        result[idx] = {"score": score, "cost": cost}

    return result


# ---------------------------------------------------------------------------
# Listing Models
# ---------------------------------------------------------------------------

def list_available_models(source_path: str, benchmark: str = "livecodebench", split: str = "test") -> None:
    """List all available models in the archive or directory with record counts."""
    path = Path(source_path)
    prefix = f"{BENCHMARK_BASE_DIR}/{benchmark}/{split}/"

    if path.is_file():
        print(f"Scanning archive {source_path} for {benchmark}/{split}...", file=sys.stderr)
        models: dict[str, str] = {}
        with tarfile.open(source_path, "r:gz") as tf:
            for member in tf:
                if member.isfile() and member.name.startswith(prefix) and member.name.endswith(".json"):
                    rel = member.name[len(prefix):]
                    parts = rel.split("/")
                    if len(parts) == 2:
                        models[parts[0]] = member.name
        print(f"\nAvailable models in {benchmark}/{split} ({len(models)} found):")
        for m, p in sorted(models.items()):
            print(f"  {m:35s} -> {p}")
    elif path.is_dir():
        cand = path / BENCHMARK_BASE_DIR / benchmark / split
        if not cand.is_dir():
            cand = path / benchmark / split
        if not cand.is_dir():
            print(f"Directory {cand} does not exist", file=sys.stderr)
            return
        subdirs = sorted([d.name for d in cand.iterdir() if d.is_dir()])
        print(f"\nAvailable models in {cand} ({len(subdirs)} found):")
        for m in subdirs:
            print(f"  {m}")
    else:
        print(f"Source not found: {source_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Difficulty Assignment (Deterministic Proxy)
# ---------------------------------------------------------------------------

def assign_difficulty(
    indices: list[int],
    model_data: dict[str, dict[int, dict[str, float]]],
) -> dict[int, str]:
    """Assign difficulty labels using tertiles of mean_score (descending).

    Proxy formulation:
    - High mean score (most models pass) -> 'easy'
    - Medium mean score -> 'medium'
    - Low mean score (most models fail) -> 'hard'

    Ties broken deterministically by source index (ascending).
    Tertiles divide n into:
      ranks 0 .. t1-1        -> easy (len = t1)
      ranks t1 .. t2-1       -> medium (len = t2 - t1)
      ranks t2 .. n-1        -> hard (len = n - t2)
    where t1 = n // 3, t2 = 2 * n // 3.
    """
    scores_by_idx: dict[int, float] = {}
    for idx in indices:
        m_scores = [model_data[m][idx]["score"] for m in model_data]
        scores_by_idx[idx] = sum(m_scores) / len(m_scores)

    # Sort descending by mean score, then ascending by index for deterministic tie breaking
    sorted_indices = sorted(indices, key=lambda i: (-scores_by_idx[i], i))

    n = len(sorted_indices)
    t1 = n // 3
    t2 = 2 * n // 3

    difficulty: dict[int, str] = {}
    for rank, idx in enumerate(sorted_indices):
        if rank < t1:
            difficulty[idx] = "easy"
        elif rank < t2:
            difficulty[idx] = "medium"
        else:
            difficulty[idx] = "hard"

    return difficulty


# ---------------------------------------------------------------------------
# Conversion Core
# ---------------------------------------------------------------------------

def convert(
    source_path: str,
    benchmark: str,
    split: str,
    models: list[str],
    output_path: str,
    seed: int = 42,
    max_tasks: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert benchmark results into masbudget base JSON dataset and model costs JSON.

    Returns:
        (tasks, costs_dict)
    """
    path = Path(source_path)
    if path.is_file():
        model_data, member_paths = _load_model_records_from_archive(path, benchmark, split, models)
    elif path.is_dir():
        model_data, member_paths = _load_model_records_from_dir(path, benchmark, split, models)
    else:
        raise FileNotFoundError(f"Source path not found: {source_path}")

    # Intersect indices
    sets = [set(model_data[m].keys()) for m in models]
    shared_indices = sorted(set.intersection(*sets))
    print(f"[convert] Total shared tasks across all {len(models)} models: {len(shared_indices)}", file=sys.stderr)

    if not shared_indices:
        raise ValueError("No common tasks found across selected models!")

    # Optional subsampling
    if max_tasks is not None and max_tasks < len(shared_indices):
        rng = random.Random(seed)
        shared_indices = sorted(rng.sample(shared_indices, max_tasks))
        print(f"[convert] Deterministically subsampled to {len(shared_indices)} tasks (seed={seed})", file=sys.stderr)

    # Proxy difficulty assignment
    difficulty_map = assign_difficulty(shared_indices, model_data)
    diff_counts = {"easy": 0, "medium": 0, "hard": 0}
    for d in difficulty_map.values():
        diff_counts[d] += 1
    print(f"[convert] Difficulty distribution: {diff_counts}", file=sys.stderr)

    # Build tasks list
    tasks: list[dict[str, Any]] = []
    for idx in shared_indices:
        quality = {m: model_data[m][idx]["score"] for m in models}
        task = {
            "task_id": f"{benchmark}/{split}/{idx}",
            "difficulty": difficulty_map[idx],
            "category": CATEGORY,
            "quality": quality,
            "metadata": {
                "source": SOURCE_LABEL,
                "benchmark": benchmark,
                "split": split,
                "source_index": idx,
                "source_files": {m: member_paths[m] for m in models},
            },
        }
        tasks.append(task)

    # Cost statistics per model
    costs_dict: dict[str, Any] = {}
    for model in models:
        costs = [model_data[model][idx]["cost"] for idx in shared_indices]
        costs_dict[model] = {
            "mean_cost_usd": statistics.mean(costs),
            "min_cost_usd": min(costs),
            "max_cost_usd": max(costs),
            "median_cost_usd": statistics.median(costs),
            "n_observations": len(costs),
            "source": f"per-record cost from LLMRouterBench {benchmark}/{split}",
            "archive_member": member_paths[model],
        }

    # Output structure
    output_data = {
        "name": f"llmrouterbench_{benchmark}",
        "description": DESCRIPTION,
        "source": (
            f"LLMRouterBench (NPULH/LLMRouterBench on HuggingFace), "
            f"{benchmark}/{split} split. "
            "Scores are binary pass@1 from LiveCodeBench evaluation."
        ),
        "tasks": tasks,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"[convert] Wrote {len(tasks)} tasks to {out_file}", file=sys.stderr)

    # Model costs companion JSON
    costs_file = out_file.parent / (out_file.stem + "_model_costs.json")
    with costs_file.open("w", encoding="utf-8") as f:
        json.dump(costs_dict, f, indent=2)
    print(f"[convert] Wrote model costs to {costs_file}", file=sys.stderr)

    return tasks, costs_dict


# ---------------------------------------------------------------------------
# Difficulty Mix Subsampling
# ---------------------------------------------------------------------------

def create_mixes(base_json_path: str, seed: int = 42, target_mix_size: int | None = 400) -> dict[str, Path]:
    """Generate 3 difficulty-mix dataset files from the base JSON:

    1. Uniform: balanced across all 3 buckets (min bucket count from each).
    2. Easy-heavy (80/20): 80% easy, 10% medium, 10% hard.
       Default target size = 400 (320 easy, 40 medium, 40 hard).
    3. Hard-heavy (20/80): 10% easy, 10% medium, 80% hard.
       Default target size = 400 (40 easy, 40 medium, 320 hard).

    Sampling is without replacement, deterministic with fixed seed.
    """
    path = Path(base_json_path)
    with path.open("r", encoding="utf-8") as f:
        base_data = json.load(f)

    tasks_by_diff: dict[str, list[dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for task in base_data["tasks"]:
        d = task["difficulty"]
        if d not in tasks_by_diff:
            raise ValueError(f"Unexpected difficulty {d!r}")
        tasks_by_diff[d].append(task)

    counts = {d: len(tasks_by_diff[d]) for d in ("easy", "medium", "hard")}
    print(f"[mixes] Base difficulty counts: {counts}", file=sys.stderr)

    generated_paths: dict[str, Path] = {}

    # 1. Uniform: take min(counts) from each bucket
    per_bucket = min(counts.values())
    rng_uniform = random.Random(seed)
    uniform_tasks: list[dict[str, Any]] = []
    for d in ("easy", "medium", "hard"):
        uniform_tasks.extend(rng_uniform.sample(tasks_by_diff[d], per_bucket))
    # Shuffle so order is not pure easy then medium then hard
    rng_uniform.shuffle(uniform_tasks)
    p_uni = _write_mix_file(path, "uniform", uniform_tasks, base_data)
    generated_paths["uniform"] = p_uni

    # 2. Easy-heavy (80% easy, 10% medium, 10% hard)
    # Unit size is 10: 8 easy, 1 medium, 1 hard
    max_units_eh = min(counts["easy"] // 8, counts["medium"], counts["hard"])
    if target_mix_size is not None:
        target_units = max(1, target_mix_size // 10)
        units_eh = min(max_units_eh, target_units)
    else:
        units_eh = max_units_eh

    if units_eh < 1:
        raise ValueError(f"Not enough tasks in base dataset to form an 80/10/10 mix: counts={counts}")

    n_easy_eh = units_eh * 8
    n_med_eh = units_eh * 1
    n_hard_eh = units_eh * 1

    rng_eh = random.Random(seed)
    easy_heavy_tasks = (
        rng_eh.sample(tasks_by_diff["easy"], n_easy_eh)
        + rng_eh.sample(tasks_by_diff["medium"], n_med_eh)
        + rng_eh.sample(tasks_by_diff["hard"], n_hard_eh)
    )
    rng_eh.shuffle(easy_heavy_tasks)
    p_eh = _write_mix_file(path, "easy_heavy_80_20", easy_heavy_tasks, base_data)
    generated_paths["easy_heavy_80_20"] = p_eh

    # 3. Hard-heavy (10% easy, 10% medium, 80% hard)
    # Unit size is 10: 1 easy, 1 medium, 8 hard
    max_units_hh = min(counts["easy"], counts["medium"], counts["hard"] // 8)
    if target_mix_size is not None:
        target_units = max(1, target_mix_size // 10)
        units_hh = min(max_units_hh, target_units)
    else:
        units_hh = max_units_hh

    if units_hh < 1:
        raise ValueError(f"Not enough tasks in base dataset to form a 10/10/80 mix: counts={counts}")

    n_easy_hh = units_hh * 1
    n_med_hh = units_hh * 1
    n_hard_hh = units_hh * 8

    rng_hh = random.Random(seed)
    hard_heavy_tasks = (
        rng_hh.sample(tasks_by_diff["easy"], n_easy_hh)
        + rng_hh.sample(tasks_by_diff["medium"], n_med_hh)
        + rng_hh.sample(tasks_by_diff["hard"], n_hard_hh)
    )
    rng_hh.shuffle(hard_heavy_tasks)
    p_hh = _write_mix_file(path, "hard_heavy_20_80", hard_heavy_tasks, base_data)
    generated_paths["hard_heavy_20_80"] = p_hh

    return generated_paths


def _write_mix_file(
    base_path: Path,
    mix_suffix: str,
    tasks: list[dict[str, Any]],
    base_data: dict[str, Any],
) -> Path:
    """Write mix dataset, ensuring unique task_ids and schema conformity."""
    ids = [t["task_id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Mix '{mix_suffix}' contains duplicate task_ids")

    diff_counts = {"easy": 0, "medium": 0, "hard": 0}
    for t in tasks:
        diff_counts[t["difficulty"]] += 1

    out_path = base_path.parent / f"{base_path.stem}_{mix_suffix}.json"
    mix_data = {
        "name": f"{base_data['name']}_{mix_suffix}",
        "description": f"{base_data['description']} Subsampled difficulty mix: {mix_suffix} ({diff_counts}).",
        "source": base_data["source"],
        "tasks": tasks,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(mix_data, f, indent=2)
    print(f"[mixes] Wrote {len(tasks)} tasks ({diff_counts}) to {out_path}", file=sys.stderr)
    return out_path


# ---------------------------------------------------------------------------
# Source-Trace Audit
# ---------------------------------------------------------------------------

def audit(
    source_path: str,
    dataset_json_path: str,
    seed: int = 42,
    n_samples: int | None = None,
) -> int:
    """Audit dataset task-model pairs and cost statistics against exact source records.

    If n_samples is None or <= 0, audits ALL task-model pairs across all tasks
    (exhaustive audit: len(tasks) * len(models) pairs).
    """
    print(f"[audit] Auditing {dataset_json_path} against source {source_path}", file=sys.stderr)
    dataset_file = Path(dataset_json_path)
    with dataset_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = data["tasks"]
    if not tasks:
        raise ValueError("No tasks to audit")

    models = sorted(list(tasks[0]["quality"].keys()))
    total_possible_pairs = len(tasks) * len(models)

    # Determine whether to do exhaustive or sampled audit
    is_exhaustive = (n_samples is None or n_samples <= 0 or n_samples >= total_possible_pairs)
    if is_exhaustive:
        print(f"[audit] Mode: EXHAUSTIVE full audit ({len(tasks)} tasks x {len(models)} models = {total_possible_pairs} pairs)", file=sys.stderr)
        pairs_to_check: list[tuple[dict[str, Any], str]] = [
            (t, m) for t in tasks for m in models
        ]
    else:
        print(f"[audit] Mode: SAMPLED audit ({n_samples} pairs out of {total_possible_pairs})", file=sys.stderr)
        rng = random.Random(seed)
        pairs_to_check = [
            (rng.choice(tasks), rng.choice(models)) for _ in range(n_samples)
        ]

    # Group by source member file
    by_file: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for t, m in pairs_to_check:
        s_file = t["metadata"]["source_files"][m]
        by_file.setdefault(s_file, []).append((t, m))

    path = Path(source_path)
    verified_pairs = 0
    errors: list[str] = []
    model_raw_records: dict[str, list[dict[str, Any]]] = {}

    if path.is_file():
        # Single-pass over the tarfile
        with tarfile.open(path, "r:gz") as tf:
            for member in tf:
                if member.name in by_file:
                    f = tf.extractfile(member)
                    raw = json.loads(f.read())
                    records = raw.get("records", [])
                    # Identify which model this member belongs to
                    model_name = next(m for t, m in by_file[member.name])
                    model_raw_records[model_name] = records
                    idx_map = {int(r["index"]): r for r in records}

                    for t, m in by_file[member.name]:
                        src_idx = t["metadata"]["source_index"]
                        expected_task_id = f"{t['metadata']['benchmark']}/{t['metadata']['split']}/{src_idx}"
                        if t["task_id"] != expected_task_id:
                            errors.append(f"Task ID mismatch: {t['task_id']} != {expected_task_id}")

                        if src_idx not in idx_map:
                            errors.append(f"Source index {src_idx} missing from {member.name}")
                            continue

                        src_rec = idx_map[src_idx]
                        src_score = float(src_rec["score"])
                        expected_score = float(t["quality"][m])
                        if src_score != expected_score:
                            errors.append(
                                f"SCORE MISMATCH: task {t['task_id']}, model {m}: "
                                f"expected {expected_score}, got source {src_score}"
                            )
                        else:
                            verified_pairs += 1

                    print(f"[audit] Checked {len(by_file[member.name])} pairs for member {member.name} (model={model_name})", file=sys.stderr)

    elif path.is_dir():
        for s_file, pair_list in by_file.items():
            f_path = Path(s_file)
            if not f_path.is_file():
                f_path = path / s_file
            with f_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            records = raw.get("records", [])
            model_name = next(m for t, m in pair_list)
            model_raw_records[model_name] = records
            idx_map = {int(r["index"]): r for r in records}

            for t, m in pair_list:
                src_idx = t["metadata"]["source_index"]
                expected_task_id = f"{t['metadata']['benchmark']}/{t['metadata']['split']}/{src_idx}"
                if t["task_id"] != expected_task_id:
                    errors.append(f"Task ID mismatch: {t['task_id']} != {expected_task_id}")

                if src_idx not in idx_map:
                    errors.append(f"Source index {src_idx} missing from {s_file}")
                    continue

                src_score = float(idx_map[src_idx]["score"])
                expected_score = float(t["quality"][m])
                if src_score != expected_score:
                    errors.append(
                        f"SCORE MISMATCH: task {t['task_id']}, model {m}: "
                        f"expected {expected_score}, got source {src_score}"
                    )
                else:
                    verified_pairs += 1

    # Check companion costs JSON if present
    costs_json_path = dataset_file.parent / (dataset_file.stem + "_model_costs.json")
    if costs_json_path.is_file():
        print(f"[audit] Validating companion costs file: {costs_json_path}", file=sys.stderr)
        with costs_json_path.open("r", encoding="utf-8") as f:
            costs_data = json.load(f)

        shared_indices = {t["metadata"]["source_index"] for t in tasks}
        for m, raw_recs in model_raw_records.items():
            if m not in costs_data:
                errors.append(f"Model {m} missing from costs JSON")
                continue
            costs_entry = costs_data[m]
            # Filter costs by shared indices
            m_costs = [float(r["cost"]) for r in raw_recs if int(r["index"]) in shared_indices]
            exp_mean = statistics.mean(m_costs)
            exp_min = min(m_costs)
            exp_max = max(m_costs)
            exp_median = statistics.median(m_costs)

            if abs(costs_entry["mean_cost_usd"] - exp_mean) > 1e-12:
                errors.append(f"Model {m} mean cost mismatch: {costs_entry['mean_cost_usd']} vs {exp_mean}")
            if abs(costs_entry["min_cost_usd"] - exp_min) > 1e-12:
                errors.append(f"Model {m} min cost mismatch: {costs_entry['min_cost_usd']} vs {exp_min}")
            if abs(costs_entry["max_cost_usd"] - exp_max) > 1e-12:
                errors.append(f"Model {m} max cost mismatch: {costs_entry['max_cost_usd']} vs {exp_max}")
            if abs(costs_entry["median_cost_usd"] - exp_median) > 1e-12:
                errors.append(f"Model {m} median cost mismatch: {costs_entry['median_cost_usd']} vs {exp_median}")
            if costs_entry["n_observations"] != len(m_costs):
                errors.append(f"Model {m} observations mismatch: {costs_entry['n_observations']} vs {len(m_costs)}")

    if errors:
        for err in errors[:50]:
            print(f"[audit ERROR] {err}", file=sys.stderr)
        raise RuntimeError(f"Audit failed with {len(errors)} mismatches!")

    mode_label = "EXHAUSTIVE" if is_exhaustive else "SAMPLED"
    print(
        f"\n[audit SUCCESS] {mode_label} audit passed: Verified {verified_pairs}/{len(pairs_to_check)} "
        f"task-model pairs and cost statistics identically match official source records.",
        file=sys.stderr,
    )
    return verified_pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LLMRouterBench benchmark data for MASBudget.")
    parser.add_argument("--archive", "--source", dest="source", help="Path to LLMRouterBench tar.gz or extracted directory")
    parser.add_argument("--benchmark", default="livecodebench", help="Benchmark name (default: livecodebench)")
    parser.add_argument("--split", default="test", help="Split name (default: test)")
    parser.add_argument("--models", nargs="+", help="Model names to extract")
    parser.add_argument("--output", help="Output path for base JSON dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional max tasks count")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--create-mixes", metavar="BASE_JSON", help="Generate 3 difficulty mix files from base JSON")
    parser.add_argument("--audit", metavar="BASE_JSON", help="Audit dataset against source archive/dir")
    parser.add_argument("--audit-samples", type=int, default=None, help="Number of pairs to audit (default: None for full exhaustive audit)")
    parser.add_argument("--audit-full", action="store_true", help="Perform full exhaustive audit of all task-model pairs")

    args = parser.parse_args()

    if args.list_models:
        if not args.source:
            parser.error("--archive / --source is required for --list-models")
        list_available_models(args.source, args.benchmark, args.split)
        return

    if args.create_mixes:
        create_mixes(args.create_mixes, seed=args.seed)
        return

    if args.audit:
        if not args.source:
            parser.error("--archive / --source is required for --audit")
        n_samples = None if args.audit_full else args.audit_samples
        audit(args.source, args.audit, seed=args.seed, n_samples=n_samples)
        return

    # Regular conversion
    if not args.source:
        parser.error("--archive / --source is required")
    if not args.models:
        parser.error("--models is required")
    if not args.output:
        parser.error("--output is required")

    convert(
        source_path=args.source,
        benchmark=args.benchmark,
        split=args.split,
        models=args.models,
        output_path=args.output,
        seed=args.seed,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main()
