"""Load saved results (results/*.json) and compare policies side by side with bar plots.

    from masbudget.analysis import PolicyComparison

    cmp = PolicyComparison.from_config("configs/all_individual_policy.yaml")
    cmp.table(["average_quality", "total_cost"])
    cmp.plot(["average_quality", "total_cost", "budget_utilization"])

Needs pandas and matplotlib (not imported by ``masbudget`` itself).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from masbudget.config import load_config
from masbudget.core import CENTRALIZED, DECENTRALIZED
from masbudget.runner import RESULTS_DIR

PARADIGMS = (CENTRALIZED, DECENTRALIZED)
COLUMNS = ["experiment", "timestamp", "paradigm", "policy", "metric", "mean", "std", "n"]

# Categorical slots 1-2 of the dataviz reference palette (light surface).
PARADIGM_COLORS = {CENTRALIZED: "#2a78d6", DECENTRALIZED: "#eb6834"}
TEXT_PRIMARY, TEXT_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_results(
    results_dir: str | Path = RESULTS_DIR, experiments: Sequence[str] | None = None
) -> pd.DataFrame:
    """Long table with one row per (experiment, paradigm, metric), read from the newest
    results file of each experiment. ``experiments`` filters (and orders) the experiments."""
    latest: dict[str, dict[str, Any]] = {}
    for path in Path(results_dir).glob("*.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        name = summary.get("experiment")
        if name is None or (experiments is not None and name not in experiments):
            continue
        if name not in latest or summary["timestamp"] > latest[name]["timestamp"]:
            latest[name] = summary

    order = list(experiments) if experiments is not None else sorted(latest)
    missing = [name for name in order if name not in latest]
    if missing:
        raise FileNotFoundError(f"No results in {results_dir}/ for experiments {missing}; run them first")

    rows = []
    for name in order:
        summary = latest[name]
        for metric, per_paradigm in summary["metrics"].items():
            for paradigm, value in (per_paradigm or {}).items():
                mean, std, n = _unpack(value)
                rows.append({
                    "experiment": name,
                    "timestamp": summary["timestamp"],
                    "paradigm": paradigm,
                    "policy": policy_label(summary["config"], paradigm),
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "n": n,
                })
    return pd.DataFrame(rows, columns=COLUMNS)


def policy_label(config: dict[str, Any], paradigm: str) -> str:
    """Short name of the policy a paradigm ran, e.g. ``optimal_mckp`` or ``quasi_linear``
    (a non-null arrival policy is appended: ``quasi_linear (random_fcfs)``)."""
    if paradigm == CENTRALIZED:
        return config["centralized"]["policy"]
    decentralized = config["decentralized"]
    arrival = decentralized.get("arrival_policy")
    return decentralized["utility"] + (f" ({arrival})" if arrival else "")


class PolicyComparison:
    """Results of several experiments, one bar per (experiment, paradigm)."""

    def __init__(self, data: pd.DataFrame):
        self.data = data.reset_index(drop=True)
        # Bar label: the policy name, or the experiment name if two experiments ran the same policy.
        runs = self.data[["experiment", "paradigm", "policy"]].drop_duplicates()
        clashes = runs[runs.duplicated(["paradigm", "policy"], keep=False)]
        clash_keys = set(zip(clashes["experiment"], clashes["paradigm"]))
        self.data["label"] = [
            exp if (exp, par) in clash_keys else pol
            for exp, par, pol in zip(self.data["experiment"], self.data["paradigm"], self.data["policy"])
        ]

    @classmethod
    def from_config(cls, config_path: str | Path, results_dir: str | Path = RESULTS_DIR) -> PolicyComparison:
        """The newest results of every experiment defined in ``config_path``."""
        return cls(load_results(results_dir, [spec["name"] for spec in load_config(config_path)]))

    @classmethod
    def from_results(
        cls, results_dir: str | Path = RESULTS_DIR, experiments: Sequence[str] | None = None
    ) -> PolicyComparison:
        return cls(load_results(results_dir, experiments))

    @property
    def metrics(self) -> list[str]:
        """Metric names available, in the order they appear in the results."""
        return list(dict.fromkeys(self.data["metric"]))

    def with_gap(
        self, reference: str, metric: str = "social_welfare", name: str = "optimality_gap"
    ) -> PolicyComparison:
        """Add ``name`` = (ref - x) / ref for ``metric``, relative to the ``reference``
        experiment (e.g. the optimal centralized policy). Computed on the run means."""
        rows = self.data[self.data["metric"] == metric]
        ref = rows[rows["experiment"] == reference]
        if len(ref) != 1:
            raise ValueError(f"Need exactly one '{metric}' result for reference '{reference}', got {len(ref)}")
        ref_value = float(ref["mean"].iloc[0])
        if ref_value == 0:
            raise ValueError(f"Reference '{reference}' has {metric} = 0; the gap is undefined")
        gap = rows.assign(metric=name, mean=(ref_value - rows["mean"]) / ref_value, std=rows["std"] / abs(ref_value))
        kept = self.data[self.data["metric"] != name]
        return PolicyComparison(pd.concat([kept, gap], ignore_index=True)[COLUMNS])

    def table(self, metrics: Sequence[str] | None = None, stat: str = "mean") -> pd.DataFrame:
        """Policies x metrics table of ``stat`` (``mean``, ``std`` or ``n``)."""
        metrics = list(metrics or self.metrics)
        self._check_metrics(metrics)
        data = self.data[self.data["metric"].isin(metrics)]
        table = data.pivot_table(index=["paradigm", "label"], columns="metric", values=stat, sort=False, dropna=False)
        return table.reindex(index=self._bars(), columns=metrics)

    def plot(
        self,
        metrics: Sequence[str],
        *,
        paradigms: Sequence[str] | None = None,
        ncols: int = 3,
        sort: bool = False,
        error_bars: bool = True,
        annotate: bool = True,
        title: str | None = None,
        panel_size: tuple[float, float] = (4.6, 3.4),
    ):
        """One horizontal bar chart per metric, one bar per policy, colored by paradigm.

        ``paradigms`` keeps only ``centralized`` and/or ``decentralized`` policies;
        ``sort`` orders each panel by value (otherwise every panel keeps the same order).
        Policies a metric does not apply to are marked ``n/a``. Returns the figure.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        metrics = [metrics] if isinstance(metrics, str) else list(metrics)
        self._check_metrics(metrics)
        bars = [b for b in self._bars() if paradigms is None or b[0] in paradigms]
        if not bars:
            raise ValueError(f"No policies for paradigms {paradigms}")

        ncols = max(1, min(ncols, len(metrics)))
        nrows = math.ceil(len(metrics) / ncols)
        height = max(panel_size[1], 0.32 * len(bars) + 1.2)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(panel_size[0] * ncols, height * nrows), squeeze=False, layout="constrained"
        )
        for ax, metric in zip(axes.flat, metrics):
            self._plot_panel(ax, metric, bars, sort=sort, error_bars=error_bars, annotate=annotate)
        for ax in axes.flat[len(metrics):]:
            ax.set_visible(False)

        shown = [p for p in PARADIGMS if any(b[0] == p for b in bars)]
        if len(shown) > 1:
            fig.legend(
                handles=[Patch(color=PARADIGM_COLORS[p], label=p) for p in shown],
                loc="outside lower center", ncols=len(shown), frameon=False, labelcolor=TEXT_PRIMARY,
            )
        if title:
            fig.suptitle(title, color=TEXT_PRIMARY, fontweight="bold")
        if "inline" in plt.get_backend():
            plt.close(fig)
        return fig

    def _plot_panel(self, ax, metric, bars, *, sort, error_bars, annotate):
        rows = self.data[self.data["metric"] == metric].set_index(["paradigm", "label"])
        values = [_finite_or_nan(rows["mean"].get(b, math.nan)) for b in bars]
        raw = [rows["mean"].get(b, math.nan) for b in bars]
        stds = [_finite_or_nan(rows["std"].get(b, math.nan)) for b in bars]
        order = list(range(len(bars)))
        if sort:  # largest first, n/a last
            order.sort(key=lambda i: (math.isnan(values[i]), 0.0 if math.isnan(values[i]) else -values[i]))

        y = list(range(len(order)))
        vals = [values[i] for i in order]
        errs = [stds[i] if not math.isnan(stds[i]) else 0.0 for i in order]
        ax.barh(
            y, [0.0 if math.isnan(v) else v for v in vals], height=0.62,
            color=[PARADIGM_COLORS.get(bars[i][0], TEXT_MUTED) for i in order],
            xerr=errs if error_bars and any(e > 0 for e in errs) else None,
            error_kw={"ecolor": TEXT_MUTED, "elinewidth": 1, "capsize": 2},
            zorder=2,
        )
        ax.set_yticks(y, [bars[i][1] for i in order])
        ax.invert_yaxis()

        if not sort:  # separator between the centralized and decentralized groups
            for k in range(1, len(order)):
                if bars[order[k]][0] != bars[order[k - 1]][0]:
                    ax.axhline(k - 0.5, color=GRID, linewidth=1, linestyle="--", zorder=1)

        finite = [v for v in vals if not math.isnan(v)]
        lo, hi = min([0.0] + finite), max([0.0] + finite)
        span = max(hi - lo, 1e-12)
        ax.set_xlim(lo - 0.15 * span if lo < 0 else 0.0, hi + 0.18 * span if hi > 0 else 0.05 * span)
        if annotate:
            for yi, i in zip(y, order):
                v = values[i]
                if math.isnan(v):
                    text = "inf" if _is_inf(raw[i]) else "n/a"
                    ax.text(0, yi, f" {text}", va="center", ha="left", fontsize=8, color=TEXT_MUTED)
                else:
                    ax.text(v + 0.01 * span * (1 if v >= 0 else -1), yi, _fmt(v), va="center",
                            ha="left" if v >= 0 else "right", fontsize=8, color=TEXT_PRIMARY)

        ax.set_title(metric.replace("_", " ").capitalize(), color=TEXT_PRIMARY, fontsize=11, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.axvline(0, color=TEXT_MUTED, linewidth=0.8, zorder=3)
        ax.tick_params(colors=TEXT_MUTED, labelsize=9, length=0)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)

    def _bars(self) -> list[tuple[str, str]]:
        """(paradigm, label) of every bar: centralized first, then config order."""
        keys = list(dict.fromkeys(zip(self.data["paradigm"], self.data["label"])))
        rank = {p: i for i, p in enumerate(PARADIGMS)}
        return sorted(keys, key=lambda k: rank.get(k[0], len(rank)))

    def _check_metrics(self, metrics: Sequence[str]) -> None:
        unknown = [m for m in metrics if m not in self.metrics]
        if unknown:
            raise ValueError(f"No results for metrics {unknown}. Available: {', '.join(self.metrics)}")


def plot_metrics(
    metrics: Sequence[str], config: str | Path, results_dir: str | Path = RESULTS_DIR, **plot_kwargs: Any
):
    """One-liner: bar plots of ``metrics`` for every experiment in ``config``."""
    return PolicyComparison.from_config(config, results_dir).plot(metrics, **plot_kwargs)


def _unpack(value: Any) -> tuple[float, float, int | None]:
    """A metric entry is ``{"mean", "std", "n"}``, a bare number (order_sensitivity) or None."""
    if isinstance(value, dict):
        return _to_float(value.get("mean")), _to_float(value.get("std")), value.get("n")
    return _to_float(value), math.nan, None


def _to_float(value: Any) -> float:
    return math.nan if value is None else float(value)  # also parses "inf" / "nan"


def _finite_or_nan(value: float) -> float:
    return value if isinstance(value, (int, float)) and math.isfinite(value) else math.nan


def _is_inf(value: Any) -> bool:
    return isinstance(value, float) and math.isinf(value)


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"
