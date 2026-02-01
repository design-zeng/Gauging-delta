"""
Spatial indexing utilities for Gauging-δ algorithm.

Provides KDTree-based spatial queries for efficient neighbor lookups.
Time complexity: O(log n) for queries instead of O(n) linear scans.
"""

import numpy as np
from scipy.spatial import KDTree


class SpatialIndex:
    """
    KDTree-based spatial index for efficient neighbor queries.

    Provides O(log n) spatial queries instead of O(n) linear scans.

    Attributes:
        kdtree: scipy KDTree instance
        X: Original data points
    """

    def __init__(self, X: np.ndarray):
        """
        Build spatial index from data points.

        Args:
            X: Data matrix of shape (n_samples, n_features)
        """
        self.X = X
        self.kdtree = KDTree(X)

    def query_radius(
        self,
        point: np.ndarray,
        radius: float,
    ) -> list[int]:
        """
        Find all points within radius of query point.

        Args:
            point: Query point coordinates
            radius: Search radius

        Returns:
            List of point indices within radius
        """
        indices = self.kdtree.query_ball_point(point, radius)
        return indices

    def query_k_nearest(
        self,
        point: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors to query point.

        Args:
            point: Query point coordinates
            k: Number of neighbors

        Returns:
            Tuple of (distances, indices) arrays
        """
        # Ensure k doesn't exceed number of points
        k = min(k, len(self.X))
        distances, indices = self.kdtree.query(point, k=k)

        # Handle single neighbor case (scipy returns scalar)
        if k == 1:
            distances = np.array([distances])
            indices = np.array([indices])

        return distances, indices

    def query_radius_by_index(
        self,
        idx: int,
        radius: float,
        exclude_self: bool = True,
    ) -> list[int]:
        """
        Find all points within radius of point at given index.

        Args:
            idx: Index of query point in X
            radius: Search radius
            exclude_self: Whether to exclude the query point itself

        Returns:
            List of point indices within radius
        """
        point = self.X[idx]
        indices = self.query_radius(point, radius)

        if exclude_self and idx in indices:
            indices = [i for i in indices if i != idx]

        return indices

    def query_k_nearest_by_index(
        self,
        idx: int,
        k: int,
        exclude_self: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors to point at given index.

        Args:
            idx: Index of query point in X
            k: Number of neighbors
            exclude_self: Whether to exclude the query point itself

        Returns:
            Tuple of (distances, indices) arrays
        """
        point = self.X[idx]

        # If excluding self, query k+1 and remove self
        query_k = k + 1 if exclude_self else k
        distances, indices = self.query_k_nearest(point, query_k)

        if exclude_self:
            mask = indices != idx
            distances = distances[mask][:k]
            indices = indices[mask][:k]

        return distances, indices
