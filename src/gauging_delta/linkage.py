"""Cluster distance computation (swappable linkage).

Computes the distance between two clusters. The default is single-linkage
(minimum point-to-point distance), but any object implementing the
:class:`LinkageMetric` protocol can replace it.
"""

from __future__ import annotations

import numpy as np

from gauging_delta.cluster import Cluster


class DefaultLinkage:
    """Single-linkage: distance = minimum pairwise point-to-point distance."""

    def compute(self, C_i: Cluster, C_j: Cluster, X: np.ndarray) -> float:
        """Return the minimum distance between any two points in C_i, C_j."""
        pts_i = X[C_i.point_indices]
        pts_j = X[C_j.point_indices]
        diff = pts_i[:, np.newaxis, :] - pts_j[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        return float(dists.min())
