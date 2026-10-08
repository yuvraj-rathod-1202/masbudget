"""Batch execution of YAML-defined experiments; writes results/*.json and logs/*.log."""

from __future__ import annotations

import json
import logging
import math
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

from masbudget.config import build_experiment, load_config

logger = logging.getLogger("masbudget")

RESULTS_DIR = Path("results")
LOGS_DIR = Path("logs")


def run_batch(
    config_path: str | Path,
    only: Sequence[str] | None = None,
    results_dir: str | Path = RESULTS_DIR,
    logs_dir: str | Path = LOGS_DIR,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Run every experiment in the config (or just ``only``). A failing experiment is logged
    and skipped. Returns ``({name: summary}, failed_names)``."""
    specs = load_config(config_path)
    if only:
        missing = set(only) - {s["name"] for s in specs}
        if missing:
            raise ValueError(f"No experiments named {sorted(missing)} in {config_path}")
        specs = [s for s in specs if s["name"] in only]

    results_dir, logs_dir = Path(results_dir), Path(logs_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    summaries: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for spec in specs:
        name = spec["name"]
        with _log_to_file(logs_dir / f"{name}_{timestamp}.log"):
            try:
                result = build_experiment(spec).run(metrics=spec.get("metrics") or [])
            except Exception:
                logger.exception("Experiment '%s' failed", name)
                failed.append(name)
                continue
            summary = {
                "experiment": name,
                "timestamp": timestamp,
                "config": {k: v for k, v in spec.items() if k != "metrics"},
                "num_agents": result.num_agents,
                "runs": {paradigm: len(runs) for paradigm, runs in result.runs.items()},
                "metrics": result.metrics,
            }
            out_path = results_dir / f"{name}_{timestamp}.json"
            out_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
            logger.info("Saved %s", out_path)
            summaries[name] = summary
    return summaries, failed


@contextmanager
def _log_to_file(path: Path) -> Iterator[None]:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()


def _json_safe(value: Any) -> Any:
    """Replace inf / nan (not valid JSON) with strings."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value
