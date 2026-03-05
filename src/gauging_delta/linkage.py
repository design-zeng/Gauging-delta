"""Cluster distance computation (swappable linkage).

Port of ``Perception.closest_points_btn_2_clusters`` (perception.py L1497-1506)
and ``compute_2_clusters_dist`` (perception.py L222-244).

This is **swappable** — any object implementing the :class:`LinkageMetric`
protocol can replace the default single-linkage computation.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from gauging_delta.cluster import Cluster


class LinkageResult(NamedTuple):
    """Full distance info between two clusters."""

    near_dist: float
    near_ref_c1: int  # nearest point in C1
    near_ref_c2: int  # nearest point in C2
    center_dist: float
    center_ref_c1: int  # closest-to-center point in C1
    center_ref_c2: int  # closest-to-center point in C2
    mix_dist: float
    mix_ref_c1: int
    mix_ref_c2: int


class DefaultLinkage:
    """Single-linkage: closest pair with legacy iteration order.

    Exactly replicates ``Perception.compute_2_clusters_dist`` (L222-244)
    and ``closest_points_btn_2_clusters`` (L1497-1506).
    """

    def compute(self, C_i: Cluster, C_j: Cluster, X: np.ndarray) -> float:
        """Return the minimum distance between any two points in C_i, C_j."""
        return compute_linkage(C_i, C_j, X).near_dist

    def compute_full(self, C_i: Cluster, C_j: Cluster, X: np.ndarray) -> LinkageResult:
        """Return full distance info (near, center, mix) for DIST_MATRIX."""
        return compute_linkage(C_i, C_j, X)


def compute_linkage(C_i: Cluster, C_j: Cluster, X: np.ndarray) -> LinkageResult:
    """Port of ``compute_2_clusters_dist`` (perception.py L222-244).

    Computes three distances:
      - near_dist: min point-to-point distance (single linkage)
      - center_dist: distance between cluster centers
      - mix_dist: average of near_dist and center_dist
    """
    pts_i = X[C_i.point_indices]  # (n_i, d)
    pts_j = X[C_j.point_indices]  # (n_j, d)

    # --- Nearest points (single linkage) ---
    # Vectorized: compute all pairwise distances
    # diff[a, b, :] = pts_i[a] - pts_j[b]
    diff = pts_i[:, np.newaxis, :] - pts_j[np.newaxis, :, :]  # (n_i, n_j, d)
    dists = np.linalg.norm(diff, axis=2)  # (n_i, n_j)

    # Find minimum — use argmin to get the flat index, then unravel
    flat_idx = np.argmin(dists)
    idx_i, idx_j = np.unravel_index(flat_idx, dists.shape)
    near_dist = float(dists[idx_i, idx_j])
    near_ref_c1 = C_i.point_indices[idx_i]
    near_ref_c2 = C_j.point_indices[idx_j]

    # --- Center distance ---
    center_dist_val = float(np.linalg.norm(C_i.center - C_j.center))

    # center_ref_c1/c2 are write-only dead values — never read downstream.
    center_ref_c1 = -1
    center_ref_c2 = -1

    # --- Mix distance (perception.py L238-243) ---
    mix_dist_val = (near_dist + center_dist_val) / 2
    mix_ref_c1 = near_ref_c1
    mix_ref_c2 = near_ref_c2

    return LinkageResult(
        near_dist=near_dist,
        near_ref_c1=near_ref_c1,
        near_ref_c2=near_ref_c2,
        center_dist=center_dist_val,
        center_ref_c1=center_ref_c1,
        center_ref_c2=center_ref_c2,
        mix_dist=mix_dist_val,
        mix_ref_c1=mix_ref_c1,
        mix_ref_c2=mix_ref_c2,
    )
