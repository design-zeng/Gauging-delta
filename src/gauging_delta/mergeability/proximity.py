"""
Proximity statistic ρ for Gauging-δ algorithm.

Paper reference: Equation 3
    ρ = d_ij / μ_historical

Where:
    d_ij = distance between cluster reference points
    μ_historical = mean of historical merge distances
"""

import numpy as np

from gauging_delta.core.cluster import Cluster


def compute_proximity_rho(
    d_ij: float,
    C_i: Cluster,
    C_j: Cluster,
    min_dist_fallback: float = 0.0001,
) -> tuple[float, float, int, int]:
    """
    Compute proximity statistic ρ (Eq. 3 from paper).

    ρ = d_ij / μ_historical

    The proximity statistic measures whether the current distance between
    clusters is small relative to their historical merge patterns.

    Args:
        d_ij: Distance between cluster reference points
        C_i: First cluster
        C_j: Second cluster
        min_dist_fallback: Fallback value when no history exists

    Returns:
        Tuple of (rho, d_ij, lead_cluster_id, child_cluster_id)
        - lead_cluster is the larger cluster (will absorb child)
    """
    n_i = len(C_i.merge_history)
    n_j = len(C_j.merge_history)

    # Compute effective history length
    n = _compute_history_length(n_i, n_j)

    # Compute historical mean
    mu_historical = _compute_historical_mean(
        C_i.merge_history,
        C_j.merge_history,
        n,
        min_dist_fallback,
    )

    # Compute proximity statistic (handle edge case of zero/nan)
    if mu_historical <= 0 or np.isnan(mu_historical):
        rho = float("inf") if d_ij > 0 else 1.0
    else:
        rho = d_ij / mu_historical

    # Determine lead (larger) and child (smaller) clusters
    if len(C_i) >= len(C_j):
        lead_cluster = C_i.label
        child_cluster = C_j.label
    else:
        lead_cluster = C_j.label
        child_cluster = C_i.label

    return rho, d_ij, lead_cluster, child_cluster


def _compute_history_length(n_i: int, n_j: int) -> int:
    """
    Compute effective history length for proximity calculation.

    Uses asymmetric weighting: max(n_larger/2, n_smaller)

    Args:
        n_i: Length of C_i merge history
        n_j: Length of C_j merge history

    Returns:
        Effective history length n
    """
    if n_i > n_j:
        return max(n_i // 2, n_j)
    else:
        return max(n_j // 2, n_i)


def _compute_historical_mean(
    history_i: list[float],
    history_j: list[float],
    n: int,
    fallback: float,
) -> float:
    """
    Compute mean of recent merge distances from both clusters.

    Args:
        history_i: Merge history of C_i
        history_j: Merge history of C_j
        n: Number of recent entries to consider
        fallback: Value to return if insufficient history

    Returns:
        μ_historical
    """
    n_i = len(history_i)
    n_j = len(history_j)

    # Require at least some history
    if n_i <= 2 and n_j <= 2:
        return fallback

    if n <= 0:
        return fallback

    # Compute mean without list concatenation
    recent_i = history_i[-n:]
    recent_j = history_j[-n:]
    count = len(recent_i) + len(recent_j)
    
    if count == 0:
        return fallback

    total = sum(recent_i) + sum(recent_j)
    return total / count
