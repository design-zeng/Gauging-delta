"""Proximity metric rho (Paper Eq. 3).

    rho = d_ij / mu_historical

Port of ``Perception.compute_proximity`` (perception.py L575-587).
"""

from __future__ import annotations

import numpy as np

from gauging_delta._types import ProximityResult
from gauging_delta.cluster import Cluster


class DefaultProximity:
    """Default proximity: rho = d_ij / mean(recent merge distances)."""

    def compute(self, C_i: Cluster, C_j: Cluster, d_ij: float, fallback: float) -> ProximityResult:
        n1 = len(C_i.merge_history)
        n2 = len(C_j.merge_history)

        # History window — perception.py L578
        if n1 > n2:
            n = max(n1 // 2, n2)
        else:
            n = max(n2 // 2, n1)

        # Mean of recent merge distances — perception.py L580-582
        if n1 > 2 or n2 > 2:
            combined = [*C_i.merge_history[-n:], *C_j.merge_history[-n:]]
            mean_past = float(np.mean(combined))
        else:
            mean_past = fallback

        # Legacy uses numpy division which produces nan/inf instead of
        # ZeroDivisionError when mean_past_dist=0 (perception.py L585).
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = float(np.float64(d_ij) / np.float64(mean_past))

        # Lead = larger cluster — perception.py L584-587
        if len(C_i) >= len(C_j):
            return ProximityResult(rho=rho, distance=d_ij, lead_id=C_i.label, child_id=C_j.label)
        return ProximityResult(rho=rho, distance=d_ij, lead_id=C_j.label, child_id=C_i.label)
