"""Small tables derived from the raw runs, so the notebooks work without the (git-ignored) raw CSVs.

    python experiments/derive.py
"""

from __future__ import annotations

from lib import load, save

POS = [f"pos{k}" for k in range(10)]

if __name__ == "__main__":
    e03 = load("e03_raw")
    e03 = e03[e03.paradigm == "decentralized"]
    save(e03.groupby(["tpa", "tau", "policy"])[POS].mean().reset_index(), "e03_position")

    e08 = load("e08_raw")
    e08 = e08[e08.paradigm == "decentralized"]
    save(e08.groupby(["J", "tau", "policy"])[POS].mean().reset_index(), "e08_position")

    # Per-instance welfare of random vs. coordinated arrivals (order sensitivity, E3).
    sw = e03[e03.policy.str.startswith("Random")].assign(sw=lambda d: d.q_exp * d.J)
    save(sw.groupby(["tpa", "tau", "policy", "instance"]).sw.agg(["mean", "std", "min", "max"]).reset_index(), "e03_sw_spread")
    print("ok")
