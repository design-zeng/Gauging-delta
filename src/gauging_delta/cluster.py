"""Cluster data structure for the Gauging-delta algorithm.

Each cluster starts as a single data point and grows by absorbing other clusters.
It tracks its member points, centroid, internal statistics, and full merge history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cluster:
    """A single cluster tracking its member points, centroid, and merge history."""

    label: int
    point_indices: list[int] = field(default_factory=list)
    center: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mu_dist: float = 0.0  # mean distance from points to centroid
    sigma_dist: float = 0.0  # std dev of distances from points to centroid

    # Merge history
    merge_history: list[float] = field(default_factory=list)  # distances at which merges occurred
    sigma_history: list[float] = field(default_factory=list)  # sigma_dist after each merge
    merging_dists: list[float] = field(default_factory=list)  # merge distances (lead's own merges only)
    density_history: list[float] = field(default_factory=list)  # continuity scores from past merges
    merge_edges: list[tuple[int, int]] = field(default_factory=list)  # (point_a, point_b) bridging edges

    # Set by the algorithm before continuity check: index of the point in this
    # cluster that is closest to the partner cluster.
    ref_point: int = -1

    # Cache for T_stat computation: (merge_history_len, result)
    _t_stat_cache: tuple[int, float] | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.point_indices)
