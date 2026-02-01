"""
Cluster visualization for Gauging-δ algorithm.

Provides plotting utilities for visualizing clustering results
and debugging the merge process.
"""

from pathlib import Path
from typing import Optional

import numpy as np


try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Circle

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from gauging_delta.core.cluster import Cluster


# Modern color palette (colorblind-friendly)
CLUSTER_COLORS = [
    "#4C72B0",  # Steel blue
    "#DD8452",  # Burnt orange
    "#55A868",  # Sage green
    "#C44E52",  # Brick red
    "#8172B3",  # Muted purple
    "#937860",  # Brown
    "#DA8BC3",  # Pink
    "#8C8C8C",  # Gray
    "#CCB974",  # Olive
    "#64B5CD",  # Light blue
]


def _get_color(idx: int, alpha: float = 1.0) -> tuple[float, ...]:
    """Get color for cluster index with optional alpha."""
    color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
    return to_rgba(color, alpha)


class ClusterPlotter:
    """
    Visualization utilities for Gauging-δ clustering.

    Provides methods for:
        - Plotting final cluster assignments
        - Visualizing merge process
        - Debugging proximity and continuity decisions

    Requires matplotlib: pip install gauging-delta[viz]

    Example:
        >>> from gauging_delta import GaugingDelta
        >>> from gauging_delta.visualization import ClusterPlotter
        >>>
        >>> model = GaugingDelta()
        >>> labels = model.fit(X)
        >>>
        >>> plotter = ClusterPlotter(X)
        >>> plotter.plot_results(labels, title="Clustering Results")
    """

    def __init__(self, X: np.ndarray, save_path: Optional[str] = None):
        """
        Initialize plotter with data.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            save_path: Directory to save figures (None = show interactively)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install gauging-delta[viz]"
            )
        self.X = np.asarray(X)
        self.save_path = Path(save_path) if save_path else None
        self.dimension = X.shape[1]

        # Style configuration
        plt.style.use("seaborn-v0_8-whitegrid")
        self._figsize_2d = (10, 8)
        self._figsize_3d = (12, 10)
        self._marker_size = 40
        self._alpha = 0.7

    def plot_results(
        self,
        labels: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        show_legend: bool = True,
        figsize: Optional[tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        Plot clustering results.

        Args:
            labels: Cluster labels for each point
            y_true: Optional ground truth labels for comparison
            title: Plot title
            show_legend: Whether to show cluster legend
            figsize: Optional figure size override

        Returns:
            matplotlib Figure object
        """
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        len(unique_labels)

        if self.dimension == 2:
            return self._plot_2d(labels, y_true, title, show_legend, figsize)
        elif self.dimension == 3:
            return self._plot_3d(labels, y_true, title, show_legend, figsize)
        else:
            return self._plot_2d_projection(labels, y_true, title, show_legend, figsize)

    def _plot_2d(
        self,
        labels: np.ndarray,
        y_true: Optional[np.ndarray],
        title: Optional[str],
        show_legend: bool,
        figsize: Optional[tuple[int, int]],
    ) -> plt.Figure:
        """2D scatter plot."""
        fig, axes = plt.subplots(
            1,
            2 if y_true is not None else 1,
            figsize=figsize or self._figsize_2d,
        )

        if y_true is None:
            axes = [axes]

        # Plot predicted clusters
        ax = axes[0]
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                self.X[mask, 0],
                self.X[mask, 1],
                c=[_get_color(i, self._alpha)],
                s=self._marker_size,
                label=f"Cluster {label}",
                edgecolors="white",
                linewidth=0.5,
            )

        ax.set_xlabel("Feature 1", fontsize=12)
        ax.set_ylabel("Feature 2", fontsize=12)
        ax.set_title(
            title or f"Gauging-δ: {len(unique_labels)} Clusters", fontsize=14, fontweight="bold"
        )
        if show_legend and len(unique_labels) <= 10:
            ax.legend(loc="best", framealpha=0.9)

        # Plot ground truth if provided
        if y_true is not None:
            ax = axes[1]
            unique_true = np.unique(y_true)
            for i, label in enumerate(unique_true):
                mask = y_true == label
                ax.scatter(
                    self.X[mask, 0],
                    self.X[mask, 1],
                    c=[_get_color(i, self._alpha)],
                    s=self._marker_size,
                    label=f"True {int(label)}",
                    edgecolors="white",
                    linewidth=0.5,
                )
            ax.set_xlabel("Feature 1", fontsize=12)
            ax.set_ylabel("Feature 2", fontsize=12)
            ax.set_title("Ground Truth", fontsize=14, fontweight="bold")
            if show_legend and len(unique_true) <= 10:
                ax.legend(loc="best", framealpha=0.9)

        plt.tight_layout()
        self._save_or_show(fig, "clusters_2d")
        return fig

    def _plot_3d(
        self,
        labels: np.ndarray,
        y_true: Optional[np.ndarray],
        title: Optional[str],
        show_legend: bool,
        figsize: Optional[tuple[int, int]],
    ) -> plt.Figure:
        """3D scatter plot."""
        fig = plt.figure(figsize=figsize or self._figsize_3d)

        if y_true is not None:
            ax1 = fig.add_subplot(121, projection="3d")
            ax2 = fig.add_subplot(122, projection="3d")
            axes = [ax1, ax2]
        else:
            ax1 = fig.add_subplot(111, projection="3d")
            axes = [ax1]

        # Plot predicted clusters
        ax = axes[0]
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                self.X[mask, 0],
                self.X[mask, 1],
                self.X[mask, 2],
                c=[_get_color(i, self._alpha)],
                s=self._marker_size,
                label=f"Cluster {label}",
                edgecolors="white",
                linewidth=0.3,
            )

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        ax.set_title(
            title or f"Gauging-δ: {len(unique_labels)} Clusters", fontsize=14, fontweight="bold"
        )

        # Plot ground truth if provided
        if y_true is not None:
            ax = axes[1]
            unique_true = np.unique(y_true)
            for i, label in enumerate(unique_true):
                mask = y_true == label
                ax.scatter(
                    self.X[mask, 0],
                    self.X[mask, 1],
                    self.X[mask, 2],
                    c=[_get_color(i, self._alpha)],
                    s=self._marker_size,
                    label=f"True {int(label)}",
                    edgecolors="white",
                    linewidth=0.3,
                )
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Feature 3")
            ax.set_title("Ground Truth", fontsize=14, fontweight="bold")

        plt.tight_layout()
        self._save_or_show(fig, "clusters_3d")
        return fig

    def _plot_2d_projection(
        self,
        labels: np.ndarray,
        y_true: Optional[np.ndarray],
        title: Optional[str],
        show_legend: bool,
        figsize: Optional[tuple[int, int]],
    ) -> plt.Figure:
        """Plot first 2 principal components for high-dimensional data."""
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(self.X)

        # Temporarily swap data for plotting
        original_X = self.X
        self.X = X_2d

        fig = self._plot_2d(
            labels,
            y_true,
            title=title or f"Gauging-δ (PCA Projection): {len(np.unique(labels))} Clusters",
            show_legend=show_legend,
            figsize=figsize,
        )

        self.X = original_X
        return fig

    def plot_clusters(
        self,
        clusters: dict[int, Cluster],
        show_centers: bool = True,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot cluster assignments from Cluster objects.

        Args:
            clusters: Dictionary of cluster_id -> Cluster
            show_centers: Whether to mark cluster centers
            title: Optional plot title

        Returns:
            matplotlib Figure object
        """
        # Build labels array
        n_points = len(self.X)
        labels = np.zeros(n_points, dtype=int)

        label_map = {cid: i for i, cid in enumerate(sorted(clusters.keys()))}

        for cid, cluster in clusters.items():
            for idx in cluster.point_indices:
                labels[idx] = label_map[cid]

        fig = self.plot_results(labels, title=title)

        # Add cluster centers
        if show_centers and self.dimension <= 3:
            ax = fig.axes[0]
            for _i, (_cid, cluster) in enumerate(sorted(clusters.items())):
                center = cluster.center
                if self.dimension == 2:
                    ax.scatter(
                        center[0],
                        center[1],
                        c="black",
                        marker="x",
                        s=100,
                        linewidth=2,
                        zorder=10,
                    )
                elif self.dimension == 3:
                    ax.scatter(
                        center[0],
                        center[1],
                        center[2],
                        c="black",
                        marker="x",
                        s=100,
                        linewidth=2,
                        zorder=10,
                    )

        return fig

    def plot_merge_decision(
        self,
        C_i: Cluster,
        C_j: Cluster,
        p_i: int,
        p_j: int,
        radius: float,
        decision: bool,
        rho: Optional[float] = None,
        threshold: Optional[float] = None,
        continuity: Optional[float] = None,
    ) -> plt.Figure:
        """
        Visualize a merge decision for debugging.

        Shows local neighborhoods and decision factors.

        Args:
            C_i, C_j: Clusters being considered
            p_i, p_j: Reference point indices
            radius: Search radius used
            decision: Whether merge was approved
            rho: Proximity statistic (optional)
            threshold: Adaptive threshold (optional)
            continuity: Continuity score (optional)

        Returns:
            matplotlib Figure object
        """
        if self.dimension > 2:
            raise ValueError("Merge visualization only supported for 2D data")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot cluster i points
        points_i = self.X[list(C_i.point_indices)]
        ax.scatter(
            points_i[:, 0],
            points_i[:, 1],
            c=[_get_color(0, 0.6)],
            s=60,
            label=f"Cluster {C_i.label}",
            edgecolors="white",
            linewidth=0.5,
        )

        # Plot cluster j points
        points_j = self.X[list(C_j.point_indices)]
        ax.scatter(
            points_j[:, 0],
            points_j[:, 1],
            c=[_get_color(1, 0.6)],
            s=60,
            label=f"Cluster {C_j.label}",
            edgecolors="white",
            linewidth=0.5,
        )

        # Highlight reference points
        ref_i = self.X[p_i]
        ref_j = self.X[p_j]
        ax.scatter(
            [ref_i[0]], [ref_i[1]], c="black", s=150, marker="*", zorder=10, label="Ref point i"
        )
        ax.scatter(
            [ref_j[0]], [ref_j[1]], c="red", s=150, marker="*", zorder=10, label="Ref point j"
        )

        # Draw line between reference points
        ax.plot([ref_i[0], ref_j[0]], [ref_i[1], ref_j[1]], "k--", alpha=0.5, linewidth=2)

        # Draw search radius circles
        circle_i = Circle(
            ref_i, radius, fill=False, color=_get_color(0), linestyle="--", linewidth=2
        )
        circle_j = Circle(
            ref_j, radius, fill=False, color=_get_color(1), linestyle="--", linewidth=2
        )
        ax.add_patch(circle_i)
        ax.add_patch(circle_j)

        # Build info text
        decision_str = "✓ MERGE" if decision else "✗ NO MERGE"
        info_lines = [decision_str]
        if rho is not None:
            info_lines.append(f"ρ = {rho:.3f}")
        if threshold is not None:
            info_lines.append(f"T = {threshold:.3f}")
        if continuity is not None:
            info_lines.append(f"Continuity = {continuity:.3f}")

        info_text = "\n".join(info_lines)
        color = "#55A868" if decision else "#C44E52"
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": color, "alpha": 0.3},
        )

        ax.set_xlabel("Feature 1", fontsize=12)
        ax.set_ylabel("Feature 2", fontsize=12)
        ax.set_title("Merge Decision Analysis", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_aspect("equal", adjustable="datalim")

        plt.tight_layout()
        self._save_or_show(fig, "merge_decision")
        return fig

    def _save_or_show(self, fig: plt.Figure, name: str) -> None:
        """Save figure to file or show interactively."""
        if self.save_path:
            filepath = self.save_path / f"{name}.png"
            fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"Saved: {filepath}")
        else:
            plt.show()


def plot_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Convenience function to plot clustering results.

    Args:
        X: Data matrix
        labels: Predicted cluster labels
        y_true: Optional ground truth labels
        title: Plot title
        save_path: Path to save figure (None = show)

    Returns:
        matplotlib Figure object

    Example:
        >>> from gauging_delta.visualization import plot_clusters
        >>> plot_clusters(X, labels, title="My Clusters")
    """
    plotter = ClusterPlotter(X, save_path)
    return plotter.plot_results(labels, y_true, title)
