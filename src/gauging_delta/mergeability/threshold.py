"""
Adaptive threshold T for Gauging-δ algorithm.

Paper reference: Equation 4
    T = β_ij × T_stat × ξ_s

Where:
    β_ij = environmental scaling factor (force-based)
    T_stat = statistical threshold from merge history
    ξ_s = shape similarity adjustment
"""

import math
from collections.abc import Callable

import numpy as np

from gauging_delta.core.cluster import Cluster


def compute_adaptive_threshold_T(
    C: Cluster,
    beta_ij: float,
    xi_s: float,
) -> float:
    """
    Compute adaptive threshold T (Eq. 4 from paper).

    T = β_ij × T_stat × ξ_s

    Where T_stat = 3 / (1 + e^(0.3 × μ/σ)) + 1.4

    Note: Paper uses σ/μ but code uses μ/σ. We preserve code behavior
    for parity, with this comment noting the divergence.

    Args:
        C: Cluster to compute threshold for
        beta_ij: Environmental scaling factor
        xi_s: Shape similarity factor

    Returns:
        Adaptive threshold T
    """
    T_stat = compute_T_stat(C.merge_history)
    return beta_ij * T_stat * xi_s


def compute_T_stat(merge_history: list[float]) -> float:
    """
    Compute statistical threshold from merge history.

    T_stat = 3 / (1 + e^(0.3 × μ/σ)) + 1.4

    Returns 2.7 when:
        - History has ≤3 samples
        - σ = 0 or μ = 0

    Args:
        merge_history: List of historical merge distances

    Returns:
        T_stat in range [1.4, 4.4]
    """
    DEFAULT_T_STAT = 2.7

    # Need more than 3 samples for meaningful statistics
    if len(merge_history) <= 3:
        return DEFAULT_T_STAT

    mean = np.mean(merge_history)
    std = np.std(merge_history)

    # Avoid division by zero
    if std == 0 or mean == 0:
        return DEFAULT_T_STAT

    # T_stat = 3 / (1 + e^(0.3 × μ/σ)) + 1.4
    ratio = mean / std

    # Clamp exponent to avoid overflow (exp(700) overflows)
    exponent = 0.3 * ratio
    if exponent > 700:
        # When exp -> inf, T_stat -> 1.4
        return 1.4

    T_stat = 3.0 / (1.0 + math.exp(exponent)) + 1.4

    return T_stat


def compute_beta_ij(
    C_i: Cluster,
    C_j: Cluster,
    d_ij: float,
    neighbor_clusters: list[Cluster],
    neighbor_distances: list[float],
    *,
    neighbor_clusters_j: list[Cluster] | None = None,
    neighbor_distances_j: list[float] | None = None,
    get_near_distance: Callable[[Cluster, Cluster], float] | None = None,
    get_mix_distance: Callable[[Cluster, Cluster], float] | None = None,
) -> float:
    """
    Compute environmental scaling factor β_ij (vision_scale in legacy).

    Matches legacy's vision_scale calculation exactly:
    1. Start with forces = [(1, C_j)] - always include the other cluster
    2. For clusters in BOTH neighbors lists, add geometric mean of forces / base_force
    3. Take top N_rc=2 clusters by force
    4. Compute weighted sum: β = Σ (S_k × w_k / Total_affect)
       where S_k = 2 / (1 + e^(5 × d_ij/avg_dist_k)) + 1

    Args:
        C_i: First cluster
        C_j: Second cluster
        d_ij: Distance between C_i and C_j (near distance)
        neighbor_clusters: List of nearby contextual clusters (from C_i's perspective)
        neighbor_distances: Distances from C_i to each neighbor (near distances)
        neighbor_clusters_j: Optional contextual clusters for C_j (legacy requires overlap)
        neighbor_distances_j: Distances from C_j to each neighbor (near distances)
        get_near_distance: Optional callback to compute near distance between clusters
        get_mix_distance: Optional callback to compute mix distance between clusters

    Returns:
        Environmental scaling factor β_ij
    """
    neighbor_clusters_j = neighbor_clusters_j or []
    neighbor_distances_j = neighbor_distances_j or []

    neighbor_map_i = {
        cluster.label: dist
        for cluster, dist in zip(neighbor_clusters, neighbor_distances)
    }
    neighbor_map_j = {
        cluster.label: dist
        for cluster, dist in zip(neighbor_clusters_j, neighbor_distances_j)
    }

    cluster_lookup = {cluster.label: cluster for cluster in neighbor_clusters}
    for cluster in neighbor_clusters_j:
        cluster_lookup.setdefault(cluster.label, cluster)

    if neighbor_clusters_j:
        overlap_ids = set(neighbor_map_i.keys()) & set(neighbor_map_j.keys())
    else:
        overlap_ids = set(neighbor_map_i.keys())

    def _near_distance(C_a: Cluster, C_b: Cluster) -> float:
        if get_near_distance is not None:
            return get_near_distance(C_a, C_b)
        if {C_a.label, C_b.label} == {C_i.label, C_j.label}:
            return d_ij
        if C_a.label == C_i.label:
            return neighbor_map_i.get(C_b.label, d_ij)
        if C_b.label == C_i.label:
            return neighbor_map_i.get(C_a.label, d_ij)
        if C_a.label == C_j.label:
            return neighbor_map_j.get(C_b.label, d_ij)
        if C_b.label == C_j.label:
            return neighbor_map_j.get(C_a.label, d_ij)
        return d_ij

    def _mix_distance(C_a: Cluster, C_b: Cluster) -> float:
        if get_mix_distance is not None:
            return get_mix_distance(C_a, C_b)
        near_dist = _near_distance(C_a, C_b)
        if C_a.center is not None and C_b.center is not None:
            center_dist = float(np.linalg.norm(C_a.center - C_b.center))
        else:
            center_dist = near_dist
        return (near_dist + center_dist) / 2.0

    base_force = compute_F_ij(C_i, C_j, _mix_distance(C_i, C_j))
    forces: list[tuple[float, int]] = [(1.0, C_j.label)]

    if overlap_ids and d_ij > 0:
        if base_force > 0:
            for cid in overlap_ids:
                if cid == C_j.label:
                    continue
                C_k = cluster_lookup.get(cid)
                if C_k is None:
                    continue
                force_i = compute_F_ij(C_k, C_i, _mix_distance(C_k, C_i))
                force_j = compute_F_ij(C_k, C_j, _mix_distance(C_k, C_j))
                rel_force = math.sqrt(force_i * force_j) / base_force
                forces.append((rel_force, cid))

    forces_sorted = sorted(forces, key=lambda x: x[0], reverse=True)
    cont_clusters: list[tuple[float, float]] = []
    total_affect = 0.0

    for rel_force, cid in forces_sorted:
        if len(cont_clusters) == 2:
            break
        C_k = C_j if cid == C_j.label else cluster_lookup.get(cid)
        if C_k is None:
            continue
        dist_c_cluster1 = _near_distance(C_k, C_i)
        if cid == C_j.label:
            dist_c_cluster2 = dist_c_cluster1
        else:
            dist_c_cluster2 = _near_distance(C_k, C_j)
        avg_dist = (dist_c_cluster2 + dist_c_cluster1) / 2.0
        cont_clusters.append((rel_force, avg_dist))
        total_affect += rel_force

    if total_affect == 0:
        return 1.0

    vision_scale = 0.0
    for rel_force, avg_dist in cont_clusters:
        dist_ratio = d_ij / avg_dist if avg_dist != 0 else 1.0
        vision_scale += (2.0 / (1.0 + math.exp(5.0 * dist_ratio)) + 1.0) * rel_force / total_affect

    return vision_scale if vision_scale > 0 else 1.0


def compute_F_ij(C_i: Cluster, C_j: Cluster, d_ij: float) -> float:
    """
    Compute interaction force between two clusters.

    F_ij = |C_i| × |C_j| / d_ij²

    Gravitational-like force based on cluster sizes and distance.

    Args:
        C_i: First cluster
        C_j: Second cluster
        d_ij: Distance between clusters

    Returns:
        Interaction force F_ij
    """
    if d_ij == 0:
        return float("inf")

    return len(C_i) * len(C_j) / (d_ij**2)


def compute_xi_s(
    C_i: Cluster,
    C_j: Cluster,
) -> float:
    """
    Compute shape similarity factor ξ_s (legacy shape_diff).

    Legacy logic compares recent and early sigma history using a sliding
    window and returns:
        ξ_s = diff / (1 + diff) + 0.5
    when N > 4, otherwise 1.0.

    Args:
        C_i: First cluster
        C_j: Second cluster

    Returns:
        Shape similarity factor ξ_s
    """
    sigma_i = C_i.sigma_history
    sigma_j = C_j.sigma_history

    N1, N2 = len(sigma_i), len(sigma_j)
    N = min(N1, N2)

    if N <= 4 or len(sigma_j) <= 1:
        return 1.0

    m5 = np.mean(sigma_i[-N:])
    m6 = np.mean(sigma_j[-N:])

    Nt = N
    if N1 < N2:
        for i in range(N - 1, N2):
            if sigma_j[i] >= sigma_i[-1]:
                Nt = i + 1
                break
        m1 = np.mean(sigma_i[:N])
        m2 = np.mean(sigma_j[:Nt])
        std1 = np.std(sigma_i[:N]) * m1
        std2 = np.std(sigma_j[:Nt]) * m2
    else:
        for i in range(N - 1, N1):
            if sigma_i[i] >= sigma_j[-1]:
                Nt = i + 1
                break
        m1 = np.mean(sigma_i[:Nt])
        m2 = np.mean(sigma_j[:N])
        std1 = np.std(sigma_i[:Nt]) * m1
        std2 = np.std(sigma_j[:N]) * m2

    most_same_std = round(min(std1, std2) / max(std1, std2), 3) if std1 != 0 and std2 != 0 else 1

    std5 = np.std(sigma_i[-N:]) * m5
    std6 = np.std(sigma_j[-N:]) * m6

    l_same_std = round(min(std5, std6) / max(std5, std6), 3) if std5 != 0 and std6 != 0 else 1

    diff = max(l_same_std, most_same_std)

    return diff / (1 + diff) + 0.5
