"""CLI: python -m masbudget configs/example.yaml [--only NAME ...]."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from masbudget.centralized import CENTRALIZED_POLICIES
from masbudget.decentralized import ARRIVAL_POLICIES, DECENTRALIZED_POLICIES, UTILITY_FUNCTIONS
from masbudget.metrics import METRICS
from masbudget.models import MODEL_TYPES
from masbudget.runner import LOGS_DIR, RESULTS_DIR, run_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m masbudget", description="Run experiments from a YAML config.")
    parser.add_argument("config", nargs="?", help="path to the experiment YAML file")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="run only these experiments")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--logs-dir", default=LOGS_DIR)
    parser.add_argument("--list", action="store_true", help="list registered factory names and exit")
    args = parser.parse_args(argv)

    if args.list:
        for title, registry in [
            ("centralized policies", CENTRALIZED_POLICIES),
            ("decentralized policies", DECENTRALIZED_POLICIES),
            ("arrival policies", ARRIVAL_POLICIES),
            ("utility functions", UTILITY_FUNCTIONS),
            ("metrics", METRICS),
            ("model types", MODEL_TYPES),
        ]:
            print(f"== {title}: {', '.join(registry.names())}")
            print()
        return 0
    if not args.config:
        parser.error("config is required (or use --list)")

    root = logging.getLogger("masbudget")
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console)

    summaries, failed = run_batch(args.config, args.only, args.results_dir, args.logs_dir)
    for name, summary in summaries.items():
        print(f"\n== {name} ==\n{json.dumps(summary['metrics'], indent=2, default=str)}")
    if failed:
        print(f"\nFailed experiments ({len(failed)}): {', '.join(failed)} -- see {args.logs_dir}/", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
