"""Generate a dendrogram plot for the flame dataset using SciencePlots styling."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

# Add project root so benchmarks._plotting is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks._plotting import use_science_style
from gauging_delta import GaugingDelta


def load_flame() -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(Path(__file__).resolve().parent.parent / "data" / "flame.txt", delimiter=",")
    return data[:, :2], data[:, 2].astype(int)


def main() -> None:
    use_science_style()

    X, y_true = load_flame()
    model = GaugingDelta().fit(X)
    Z = model.linkage_matrix_

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), width_ratios=[1.4, 1])

    # -- Left: dendrogram --
    ax = axes[0]
    dendrogram(
        Z,
        ax=ax,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=7,
        color_threshold=Z[-1, 2] * 0.7,
    )
    ax.set_xlabel("Cluster index")
    ax.set_ylabel("Distance")
    ax.set_title("Merge dendrogram (flame, truncated)")
    ax.tick_params(axis="x", labelsize=7)

    # -- Right: scatter coloured by cluster --
    ax = axes[1]
    for label in range(model.n_clusters_):
        mask = model.labels_ == label
        ax.scatter(X[mask, 0], X[mask, 1], s=12, alpha=0.8, label=f"Cluster {label}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Gauging-$\\delta$ clusters (flame)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal")

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "assets" / "dendrogram.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
