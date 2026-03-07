"""Adaptive threshold T = beta * T_stat * xi_s  (Paper Eq. 4).

Port of ``Perception.compute_adaptive_threshold`` (perception.py L350-573).
This is the fixed composition-glue layer — NOT swappable.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from gauging_delta.cluster import Cluster
from gauging_delta.config import GaugingDeltaConfig


def _point_dist(a: np.ndarray, b: np.ndarray, metric: str | callable = "euclidean") -> float:
    """Distance between two points using the given metric."""
    if metric == "euclidean":
        diff = a - b
        return float(math.sqrt(float(diff @ diff)))
    from scipy.spatial.distance import cdist

    return float(cdist(a.reshape(1, -1), b.reshape(1, -1), metric=metric)[0, 0])


def _row_dists(X: np.ndarray, y: np.ndarray, metric: str | callable = "euclidean") -> np.ndarray:
    """Distances from each row of X to point y."""
    if metric == "euclidean":
        return np.linalg.norm(X - y, axis=1)
    from scipy.spatial.distance import cdist

    return cdist(X, y.reshape(1, -1), metric=metric).ravel()


class ThresholdResult(NamedTuple):
    T_i: float
    T_j: float
    xi_s: float
    adp_prox: float


def compute_adaptive_threshold(
    lead: Cluster,
    child: Cluster,
    d_ij: float,
    rho: float,
    all_clusters: dict[int, Cluster],
    dist_matrix: np.ndarray,
    cfg: GaugingDeltaConfig,
) -> ThresholdResult:
    """Compute adaptive thresholds T_i, T_j, shape similarity xi_s.

    Direct port of perception.py L350-573 (minus plotting).
    """
    size1 = len(lead)
    size2 = len(child)
    c1_id = lead.label
    c2_id = child.label

    k = min(len(all_clusters) - 1, cfg.num_nearest_clusters)

    # --- Step 1: nearest cluster distances (perception.py L354-360) ---
    row1 = dist_matrix[c1_id, :]
    row2 = dist_matrix[c2_id, :]
    idx1 = _k_smallest_indices(row1, k)
    idx2 = _k_smallest_indices(row2, k)

    nd1 = row1[idx1]
    nd2 = row2[idx2]
    nd1 = nd1[~np.isinf(nd1)]
    nd2 = nd2[~np.isinf(nd2)]

    # --- Step 2: vision scale / beta (perception.py L362-400) ---
    vision_scale = _compute_vision_scale(
        c1_id, c2_id, d_ij, idx1, idx2, all_clusters, dist_matrix, cfg
    )

    # --- Step 3: T_stat per cluster (perception.py L402-417) ---
    t1 = _compute_t_stat(lead, cfg) * vision_scale
    t2 = _compute_t_stat(child, cfg) * vision_scale

    # --- Step 4: shape similarity xi_s (perception.py L419-468) ---
    xi_s = _compute_xi_s(lead, child, cfg)
    t1 *= xi_s
    t2 *= xi_s

    adp_prox = t1 * size1 / (size1 + size2) + t2 * size2 / (size1 + size2)
    return ThresholdResult(T_i=t1, T_j=t2, xi_s=xi_s, adp_prox=adp_prox)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _k_smallest_indices(row: np.ndarray, k: int) -> np.ndarray:
    """Return sorted indices of k smallest values in *row* (like perception.py get_indices_of_k_smallest)."""
    k = min(k, len(row))
    idx = np.argpartition(row, k)[:k]
    return idx[np.argsort(row[idx])]


def _compute_vision_scale(
    c1_id: int,
    c2_id: int,
    d_ij: float,
    idx1: np.ndarray,
    idx2: np.ndarray,
    all_clusters: dict[int, Cluster],
    dist_matrix: np.ndarray,
    cfg: GaugingDeltaConfig,
) -> float:
    """Force-weighted environmental scaling (perception.py L362-400)."""
    # base force between the two clusters
    base_force = _compute_force(c1_id, c2_id, all_clusters, dist_matrix)
    if base_force == 0:
        return 1.0

    # Find shared contextual clusters (perception.py L365-370)
    set2 = set(idx2.tolist())
    forces: list[tuple[float, int]] = [(1.0, c2_id)]
    for c in idx1:
        c_int = int(c)
        if c_int in set2:
            f1 = _compute_force(c_int, c1_id, all_clusters, dist_matrix)
            f2 = _compute_force(c_int, c2_id, all_clusters, dist_matrix)
            forces.append((math.sqrt(f1 * f2) / base_force, c_int))

    if not forces:
        return 1.0

    forces.sort(key=lambda x: x[0], reverse=True)

    # Take top N_rc contextual clusters (perception.py L374-390)
    cont_clusters: list[tuple[float, float]] = []
    total_affect = 0.0
    for weight, cid in forces:
        d_c1 = float(dist_matrix[cid, c1_id])
        d_c2 = float(dist_matrix[cid, c2_id]) if cid != c2_id else d_c1
        cont_clusters.append((weight, (d_c2 + d_c1) / 2))
        total_affect += weight
        if len(cont_clusters) == cfg.n_contextual_clusters:
            break

    # Weighted vision scale (perception.py L392-396)
    vs = 0.0
    for weight, avg_d in cont_clusters:
        dist_ratio = d_ij / avg_d if avg_d != 0 else cfg.vision_scale_dist_ratio_fallback
        _vs = (
            cfg.vision_scale_coeff / (1 + math.e ** (cfg.vision_scale_exp * dist_ratio))
            + cfg.vision_scale_offset
        )
        vs += _vs * weight / total_affect
    return vs


def _compute_force(
    c1_id: int,
    c2_id: int,
    all_clusters: dict[int, Cluster],
    dist_matrix: np.ndarray,
) -> float:
    """Gravitational force = m1*m2 / d^2 (perception.py L344-348)."""
    near_d = dist_matrix[c1_id, c2_id]
    if np.isinf(near_d):
        return 0.0
    center_d = float(np.linalg.norm(all_clusters[c1_id].center - all_clusters[c2_id].center))
    d = (float(near_d) + center_d) / 2
    m1 = len(all_clusters[c1_id])
    m2 = len(all_clusters[c2_id])
    return float(np.float64(m1 * m2) / np.float64(d) ** 2)


def _compute_t_stat(cluster: Cluster, cfg: GaugingDeltaConfig) -> float:
    """Statistical threshold from merge history (perception.py L402-417).

    T_stat = numerator / (1 + e^(coeff * mu/sigma)) + offset
    Cached on cluster; invalidated when merge_history length changes.
    """
    mh_len = len(cluster.merge_history)
    cache = getattr(cluster, "_t_stat_cache", None)
    if cache is not None and cache[0] == mh_len:
        return cache[1]
    if mh_len > cfg.t_stat_min_history:
        mu = float(np.mean(cluster.merge_history))
        sigma = float(np.std(cluster.merge_history))
        if sigma != 0 and mu != 0:
            result = float(
                cfg.t_stat_numerator / (1 + math.e ** (cfg.t_stat_exp_coeff * mu / sigma))
                + cfg.t_stat_offset
            )
            cluster._t_stat_cache = (mh_len, result)  # type: ignore[attr-defined]
            return result
        cluster._t_stat_cache = (mh_len, cfg.t_stat_fallback)  # type: ignore[attr-defined]
        return cfg.t_stat_fallback
    cluster._t_stat_cache = (mh_len, cfg.t_stat_fallback)  # type: ignore[attr-defined]
    return cfg.t_stat_fallback


def compute_adaptive_threshold_lite(
    lead: Cluster,
    child: Cluster,
    d_ij: float,
    rho: float,
    all_clusters: dict[int, Cluster],
    centers: np.ndarray,  # (K, D) array of active cluster centers
    center_ids: list[int],  # maps position → cluster ID
    center_id_to_pos: dict[int, int],  # maps cluster ID → position
    cfg: GaugingDeltaConfig,
    metric: str | callable = "euclidean",
) -> ThresholdResult:
    """Lite-mode adaptive threshold using brute-force centroid scans.

    Replaces KDTree queries with O(K·D) numpy norm computations.
    In lite mode, all inter-cluster distances are centroid distances.
    """
    size1 = len(lead)
    size2 = len(child)
    c1_id = lead.label
    c2_id = child.label

    k = min(len(all_clusters) - 1, cfg.num_nearest_clusters)
    K = len(center_ids)

    # --- Step 1: K nearest clusters via brute-force (O(K·D) each) ---
    if K < 2:
        nd1 = np.array([d_ij])
        nd2 = np.array([d_ij])
        idx1_ids = np.array([c2_id])
        idx2_ids = np.array([c1_id])
    else:
        k_actual = min(k, K - 1)

        # Brute-force KNN for lead
        dists1 = _row_dists(centers, lead.center, metric)
        pos1 = center_id_to_pos[c1_id]
        dists1[pos1] = np.inf  # exclude self
        if k_actual > 0:
            idx1 = np.argpartition(dists1, k_actual)[:k_actual]
            order1 = np.argsort(dists1[idx1])
            idx1 = idx1[order1]
        else:
            idx1 = np.array([], dtype=int)

        # Brute-force KNN for child
        dists2 = _row_dists(centers, child.center, metric)
        pos2 = center_id_to_pos[c2_id]
        dists2[pos2] = np.inf  # exclude self
        if k_actual > 0:
            idx2 = np.argpartition(dists2, k_actual)[:k_actual]
            order2 = np.argsort(dists2[idx2])
            idx2 = idx2[order2]
        else:
            idx2 = np.array([], dtype=int)

        nd1 = dists1[idx1]
        nd2 = dists2[idx2]
        idx1_ids = np.array([center_ids[j] for j in idx1])
        idx2_ids = np.array([center_ids[j] for j in idx2])

    nd1 = nd1[~np.isinf(nd1)]
    nd2 = nd2[~np.isinf(nd2)]

    # --- Step 2: vision scale / beta ---
    vision_scale = _compute_vision_scale_lite(
        c1_id, c2_id, d_ij, idx1_ids, idx2_ids, all_clusters, cfg, metric
    )

    # --- Step 3: T_stat per cluster ---
    t1 = _compute_t_stat(lead, cfg) * vision_scale
    t2 = _compute_t_stat(child, cfg) * vision_scale

    # --- Step 4: shape similarity xi_s ---
    xi_s = _compute_xi_s(lead, child, cfg)
    t1 *= xi_s
    t2 *= xi_s

    adp_prox = t1 * size1 / (size1 + size2) + t2 * size2 / (size1 + size2)
    return ThresholdResult(T_i=t1, T_j=t2, xi_s=xi_s, adp_prox=adp_prox)


def _compute_force_lite(
    c1_id: int,
    c2_id: int,
    all_clusters: dict[int, Cluster],
    metric: str | callable = "euclidean",
) -> float:
    """Gravitational force using centroid distance only (lite mode)."""
    if c1_id not in all_clusters or c2_id not in all_clusters:
        return 0.0
    d = _point_dist(all_clusters[c1_id].center, all_clusters[c2_id].center, metric)
    if d == 0:
        # Legacy behavior: numpy division produces inf
        return float(
            np.float64(len(all_clusters[c1_id]) * len(all_clusters[c2_id])) / np.float64(0) ** 2
        )
    m1 = len(all_clusters[c1_id])
    m2 = len(all_clusters[c2_id])
    return float(np.float64(m1 * m2) / np.float64(d) ** 2)


def _compute_vision_scale_lite(
    c1_id: int,
    c2_id: int,
    d_ij: float,
    idx1_ids: np.ndarray,
    idx2_ids: np.ndarray,
    all_clusters: dict[int, Cluster],
    cfg: GaugingDeltaConfig,
    metric: str | callable = "euclidean",
) -> float:
    """Force-weighted environmental scaling using direct centroid distances (lite mode)."""
    base_force = _compute_force_lite(c1_id, c2_id, all_clusters, metric)
    if base_force == 0:
        return 1.0

    set2 = set(idx2_ids.tolist()) if len(idx2_ids) > 0 else set()
    forces: list[tuple[float, int]] = [(1.0, c2_id)]
    for c_int in idx1_ids:
        c_int = int(c_int)
        if c_int in set2:
            f1 = _compute_force_lite(c_int, c1_id, all_clusters, metric)
            f2 = _compute_force_lite(c_int, c2_id, all_clusters, metric)
            forces.append((math.sqrt(f1 * f2) / base_force, c_int))

    if not forces:
        return 1.0

    forces.sort(key=lambda x: x[0], reverse=True)

    cont_clusters: list[tuple[float, float]] = []
    total_affect = 0.0
    for weight, cid in forces:
        d_c1 = _point_dist(all_clusters[cid].center, all_clusters[c1_id].center, metric)
        if cid != c2_id:
            d_c2 = _point_dist(all_clusters[cid].center, all_clusters[c2_id].center, metric)
        else:
            d_c2 = d_c1
        cont_clusters.append((weight, (d_c2 + d_c1) / 2))
        total_affect += weight
        if len(cont_clusters) == cfg.n_contextual_clusters:
            break

    vs = 0.0
    for weight, avg_d in cont_clusters:
        dist_ratio = d_ij / avg_d if avg_d != 0 else cfg.vision_scale_dist_ratio_fallback
        _vs = (
            cfg.vision_scale_coeff / (1 + math.e ** (cfg.vision_scale_exp * dist_ratio))
            + cfg.vision_scale_offset
        )
        vs += _vs * weight / total_affect
    return vs


def _compute_xi_s(lead: Cluster, child: Cluster, cfg: GaugingDeltaConfig) -> float:
    """Shape similarity xi_s (perception.py L419-466).

    xi_s = max(l_same_std, most_same_std) / (1 + max(...)) + 0.5
    Returns 1.0 when history is too short (N <= xi_s_min_history).
    """
    n1 = len(lead.sigma_history)
    n2 = len(child.sigma_history)
    n = min(n1, n2)

    if n2 <= 1 or n <= cfg.xi_s_min_history:
        return 1.0

    # perception.py L423-456
    m5 = float(np.mean(lead.sigma_history[-n:])) if n > 0 else 0.0
    m6 = float(np.mean(child.sigma_history[-n:])) if n > 0 else 0.0

    nt = n
    if n1 < n2:
        for i in range(n - 1, n2):
            if child.sigma_history[i] >= lead.sigma_history[-1]:
                nt = i + 1
                break
        m1_val = float(np.mean(lead.sigma_history[:n]))
        m2_val = float(np.mean(child.sigma_history[:nt]))
        s1 = float(np.std(lead.sigma_history[:n])) * m1_val
        s2 = float(np.std(child.sigma_history[:nt])) * m2_val
    else:
        for i in range(n - 1, n1):
            if lead.sigma_history[i] >= child.sigma_history[-1]:
                nt = i + 1
                break
        m1_val = float(np.mean(lead.sigma_history[:nt]))
        m2_val = float(np.mean(child.sigma_history[:n]))
        s1 = float(np.std(lead.sigma_history[:nt])) * m1_val
        s2 = float(np.std(child.sigma_history[:n])) * m2_val

    most_same = (
        round(min(s1, s2) / max(s1, s2), cfg.std_ratio_rounding) if s1 != 0 and s2 != 0 else 1.0
    )

    s5 = float(np.std(lead.sigma_history[-n:])) * m5
    s6 = float(np.std(child.sigma_history[-n:])) * m6
    l_same = (
        round(min(s5, s6) / max(s5, s6), cfg.std_ratio_rounding) if s5 != 0 and s6 != 0 else 1.0
    )

    diff = max(l_same, most_same)
    if n > cfg.xi_s_min_history:
        return diff / (1 + diff) + cfg.xi_s_offset
    return 1.0
