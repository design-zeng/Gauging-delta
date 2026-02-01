"""
Mergeability functions for Gauging-δ algorithm.

Implements the adaptive mergeability function from Section II of the paper,
including proximity (ρ), adaptive threshold (T), and continuity statistics.
"""

from gauging_delta.mergeability.continuity import compute_continuity
from gauging_delta.mergeability.proximity import compute_proximity_rho
from gauging_delta.mergeability.threshold import (
    compute_adaptive_threshold_T,
    compute_beta_ij,
    compute_xi_s,
)


__all__ = [
    "compute_adaptive_threshold_T",
    "compute_beta_ij",
    "compute_continuity",
    "compute_proximity_rho",
    "compute_xi_s",
]
