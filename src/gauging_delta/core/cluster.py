"""
Cluster data structures for Gauging-δ algorithm.

Paper notation mapping:
    C_i, C_j  -> Cluster instances
    d_ij      -> ClusterPairInfo.d_near
    ρ         -> MergeabilityResult.rho
    T         -> MergeabilityResult.T_i, T_j
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Cluster:
    """
    Represents a cluster C_i in the Gauging-δ algorithm.

    Attributes:
        label: Unique identifier for this cluster
        point_indices: List of data point indices belonging to this cluster
        center: Centroid of the cluster (mean of all points)
        mu_dist: Mean distance of points from center
        sigma_dist: Standard deviation of distances from center
        merge_history: Historical merge distances (past_dists in original)
        sigma_history: Historical standard deviations (past_std in original)
        density_history: Historical density values (past_densities in original)
        merge_edges: Edge list tracking merge connections (traces in original)
    """

    label: int
    point_indices: list[int] = field(default_factory=list)
    center: Optional[np.ndarray] = None

    mu_dist: float = 0.0
    sigma_dist: float = 0.0

    merge_history: list[float] = field(default_factory=list)
    sigma_history: list[float] = field(default_factory=list)
    density_history: list[float] = field(default_factory=list)
    merging_dists: list[float] = field(default_factory=list)

    merge_edges: list[tuple[int, int]] = field(default_factory=list)

    def __len__(self) -> int:
        """Return number of points in cluster."""
        return len(self.point_indices)


@dataclass
class ClusterPairInfo:
    """
    Distance information between two clusters C_i and C_j.

    Paper notation:
        d_ij -> d_near (nearest point distance)

    Attributes:
        d_near: Distance between nearest points (d_ij in paper)
        p_i: Index of reference point in C_i
        p_j: Index of reference point in C_j
        d_center: Distance between cluster centers (optional)
    """

    d_near: float
    p_i: int
    p_j: int
    d_center: Optional[float] = None


@dataclass
class MergeabilityResult:
    """
    Result of the mergeability function evaluation.

    Paper notation:
        ρ   -> rho (proximity statistic, Eq. 3)
        T   -> T_i, T_j (adaptive thresholds, Eq. 4)

    Attributes:
        is_mergeable: Whether clusters should merge
        rho: Proximity statistic ρ = d_ij / μ_historical
        T_i: Adaptive threshold for cluster C_i
        T_j: Adaptive threshold for cluster C_j
        beta_ij: Environmental scaling factor β_ij
        xi_s: Shape similarity factor ξ_s
        continuity: Continuity score (0 to 1)
        lead_cluster: ID of larger cluster (absorbs the other)
        child_cluster: ID of smaller cluster (absorbed)
        d_ij: Distance between clusters
    """

    is_mergeable: bool
    rho: float
    T_i: float
    T_j: float
    beta_ij: float
    xi_s: float
    continuity: float
    lead_cluster: int
    child_cluster: int
    d_ij: float = 0.0
