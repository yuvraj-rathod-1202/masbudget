"""Matplotlib styling for the study figures (reference palette of the dataviz method, light mode)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": TEXT, "axes.titlecolor": TEXT, "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2.0, "lines.markersize": 5, "font.size": 9.5,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def colors(labels) -> dict[str, str]:
    """Fixed colour per label, in the order given (never cycled past 8)."""
    labels = list(dict.fromkeys(labels))
    if len(labels) > len(SERIES):
        raise ValueError(f"{len(labels)} series > {len(SERIES)} categorical slots: facet or fold")
    return dict(zip(labels, SERIES))


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
