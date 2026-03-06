"""Protocol definitions for swappable algorithm components."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol


if TYPE_CHECKING:
    import numpy as np

    from gauging_delta.cluster import Cluster


class ProximityResult(NamedTuple):
    """Result of proximity computation."""

    rho: float
    distance: float
    lead_id: int
    child_id: int


class ProximityMetric(Protocol):
    """Contract for proximity computation (Paper Eq. 3)."""

    def compute(
        self, C_i: Cluster, C_j: Cluster, d_ij: float, fallback: float
    ) -> ProximityResult: ...


class ContinuityMetric(Protocol):
    """Contract for continuity analysis (Paper Section II.C)."""

    def compute(
        self,
        lead: Cluster,
        child: Cluster,
        threshold: float,
        d_ij_norm: float,
        adp_prox: float,
        X: np.ndarray,
        point_dists: tuple[np.ndarray, np.ndarray],
    ) -> float: ...


class LinkageMetric(Protocol):
    """Contract for cluster-to-cluster distance computation."""

    def compute(self, C_i: Cluster, C_j: Cluster, X: np.ndarray) -> float: ...
