"""Generate readme_plot_{light,dark}.png — the hero image for the README.

Shows a 2x3 grid: Input (top row) vs Gauging-delta output (bottom row)
for jain, flame, and 3-spiral datasets.

Uses Catppuccin Mocha/Latte themes with transparent backgrounds for
GitHub dark/light mode support via <picture> tags.

Usage:
    uv run python scripts/plot_hero.py [--theme light|dark|both]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")

# Add project root so benchmarks._plotting and gauging_delta are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks._plotting import apply_theme

from gauging_delta import GaugingDelta


ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
DATA_DIR = ROOT / "data"

DATASETS = ["jain", "flame", "3-spiral"]

# Tufte-minimal base style (structural, not color)
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.bottom": False,
        "axes.spines.left": False,
        "axes.grid": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.transparent": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)


def load_dataset(name: str) -> np.ndarray:
    """Load a dataset from data/ (columns: x, y, label)."""
    return np.loadtxt(DATA_DIR / f"{name}.txt", delimiter=",")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate README hero image")
    parser.add_argument(
        "--theme",
        choices=["light", "dark", "both"],
        default="both",
        help="Which Catppuccin theme(s) to generate (default: both)",
    )
    args = parser.parse_args()
    themes = ["light", "dark"] if args.theme == "both" else [args.theme]

    # Load datasets and run clustering once (theme-independent)
    datasets = []
    for name in DATASETS:
        data = load_dataset(name)
        X = data[:, :2]
        labels = GaugingDelta().fit_predict(X)
        n_clusters = int(labels.max()) + 1
        datasets.append((name, X, labels, n_clusters))

    for theme in themes:
        tc = apply_theme(theme)

        fig, axes = plt.subplots(2, 3, figsize=(10, 6))

        for col, (name, X, labels, n_clusters) in enumerate(datasets):
            # -- Top row: raw input --
            ax_in = axes[0, col]
            ax_in.scatter(
                X[:, 0], X[:, 1], s=8, c=tc.input_points, alpha=0.7, edgecolors="none"
            )
            ax_in.set_title(name, fontstyle="italic")
            ax_in.set_xticks([])
            ax_in.set_yticks([])
            ax_in.set_aspect("equal")

            # -- Bottom row: clustered output --
            ax_out = axes[1, col]
            for k in range(n_clusters):
                mask = labels == k
                color = tc.cluster_colors[k % len(tc.cluster_colors)]
                ax_out.scatter(
                    X[mask, 0], X[mask, 1], s=8, c=color, alpha=0.8, edgecolors="none"
                )
            # k annotation at bottom-right
            ax_out.text(
                0.95,
                0.05,
                f"k = {n_clusters}",
                transform=ax_out.transAxes,
                fontsize=10,
                color=tc.text,
                ha="right",
                va="bottom",
                fontstyle="italic",
            )
            ax_out.set_xticks([])
            ax_out.set_yticks([])
            ax_out.set_aspect("equal")

        # Row labels
        axes[0, 0].set_ylabel("Input", fontweight="bold", fontsize=12)
        axes[1, 0].set_ylabel("Gauging-$\\delta$", fontweight="bold", fontsize=12)

        fig.tight_layout()
        ASSETS_DIR.mkdir(exist_ok=True)
        out = ASSETS_DIR / f"readme_plot_{theme}.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved plot \u2192 {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
