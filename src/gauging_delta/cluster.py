"""Cluster data structure for the Gauging-delta algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cluster:
    """A single cluster tracking its points and merge history."""

    label: int
    point_indices: list[int] = field(default_factory=list)
    center: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mu_dist: float = 0.0
    sigma_dist: float = 0.0

    # Merge history — names match perception.py dict keys
    merge_history: list[float] = field(default_factory=list)  # past_dists
    sigma_history: list[float] = field(default_factory=list)  # past_std
    merging_dists: list[float] = field(default_factory=list)  # merging_dists
    density_history: list[float] = field(default_factory=list)  # past_densities
    merge_edges: list[tuple[int, int]] = field(default_factory=list)  # traces

    # Set by algorithm before continuity check — nearest point to partner cluster
    ref_point: int = -1

    def __len__(self) -> int:
        return len(self.point_indices)
