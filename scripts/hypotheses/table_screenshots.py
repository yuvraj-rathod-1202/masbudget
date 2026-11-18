"""Render the paper's generated LaTeX tables to PNG images for the README.

    python scripts/hypotheses/table_screenshots.py

Needs pdflatex and pdftoppm (both ship with MiKTeX / TeX Live).
Reads report/generated/table_*.tex, writes figures/paper/table_*.png.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "report" / "generated"
OUT = ROOT / "figures" / "paper"

TEMPLATE = r"""\documentclass[border=6pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{xcolor}
\begin{document}
\small
\input{TABLE}
\end{document}
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for tex in sorted(GEN.glob("table_*.tex")):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shutil.copy(tex, tmp / tex.name)
            (tmp / "wrap.tex").write_text(TEMPLATE.replace("TABLE", tex.name), encoding="utf-8")
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "wrap.tex"], cwd=tmp, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["pdftoppm", "-r", "200", "-png", "-singlefile", "wrap.pdf", str(OUT / tex.stem)],
                           cwd=tmp, check=True)
        print("wrote", OUT / f"{tex.stem}.png")


if __name__ == "__main__":
    main()
