"""Proximity metric rho (Paper Eq. 3).

Measures how far apart two clusters are *relative to their recent merge history*:

    rho = d_ij / mu_historical

where d_ij is the distance between the two clusters, and mu_historical is the
mean of their recent merge distances. A rho near 1.0 means the gap is typical;
a large rho means the clusters are unusually far apart compared to past merges.

This component is **swappable** via the :class:`ProximityMetric` protocol.
"""

from __future__ import annotations

import numpy as np

from gauging_delta._types import ProximityResult
from gauging_delta.cluster import Cluster


class DefaultProximity:
    """Default proximity: rho = d_ij / mean(recent merge distances).

    Also designates the larger cluster as "lead" and the smaller as "child".
    """

    def compute(self, C_i: Cluster, C_j: Cluster, d_ij: float, fallback: float) -> ProximityResult:
        n1 = len(C_i.merge_history)
        n2 = len(C_j.merge_history)

        # Adaptive history window: use more history from the cluster with more merges,
        # but at least as much as the smaller cluster has.
        if n1 > n2:
            n = max(n1 // 2, n2)
        else:
            n = max(n2 // 2, n1)

        # mu_historical: mean of recent merge distances from both clusters.
        # Falls back to the global minimum inter-cluster distance when history is short.
        if n1 > 2 or n2 > 2:
            combined = [*C_i.merge_history[-n:], *C_j.merge_history[-n:]]
            mean_past = float(np.mean(combined))
        else:
            mean_past = fallback

        # Uses numpy division to produce nan/inf (not Python ZeroDivisionError)
        # when mean_past=0, which downstream logic relies on.
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = float(np.float64(d_ij) / np.float64(mean_past))

        # Lead = larger cluster (tie goes to C_i)
        if len(C_i) >= len(C_j):
            return ProximityResult(rho=rho, distance=d_ij, lead_id=C_i.label, child_id=C_j.label)
        return ProximityResult(rho=rho, distance=d_ij, lead_id=C_j.label, child_id=C_i.label)
