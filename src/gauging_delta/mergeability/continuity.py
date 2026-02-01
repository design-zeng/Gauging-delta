"""
Continuity analysis for Gauging-δ algorithm.

Paper reference: Section II.C
Implements the "Good Continuation" Gestalt principle by evaluating:
    - Density transition (are densities similar across boundary?)
    - Angle transition (is direction consistent?)
    - Shape consistency
    - Orientation consistency
"""

import math

import numpy as np

from gauging_delta.core.cluster import Cluster
from gauging_delta.geometry.angles import compute_angle


def compute_continuity(
    C_i: Cluster,
    C_j: Cluster,
    p_i: int,
    p_j: int,
    X: np.ndarray,
    T_continuity: float,
    d_ij: float,
    T_adaptive: float,
    _debug: bool = False,
) -> float:
    """
    Compute continuity score matching original's compute_local_transition.
    
    This is a faithful port of the original algorithm's continuity logic,
    including surrounded detection, boundary cases, and multi-scale analysis.
    
    Returns:
        Smoothness score in [0, 1], higher = more continuous
    """
    if _debug:
        print(f"    [continuity] C_i={C_i.label} (size={len(C_i)}), C_j={C_j.label} (size={len(C_j)})")
        print(f"    [continuity] p_i={p_i}, p_j={p_j}")
        print(f"    [continuity] merge_hist_i={C_i.merge_history[-5:]}, merge_hist_j={C_j.merge_history[-5:]}")
    
    points_dist = np.linalg.norm(X[p_i] - X[p_j])
    middle_point = (X[p_i] + X[p_j]) / 2.0
    
    # Compute past distances for base length calculation
    past_dists_i = C_i.merge_history[-5:] if C_i.merge_history else []
    past_dists_j = C_j.merge_history[-5:] if C_j.merge_history else []
    past_dists = past_dists_i + past_dists_j
    
    # Compute compactness (matches original)
    size_i, size_j = len(C_i), len(C_j)
    
    if size_i > 2 and C_i.sigma_history and C_i.merge_history:
        compact_i = (C_i.sigma_history[-1] / np.mean(C_i.merge_history) / np.log(size_i)) * size_i / (size_i + size_j)
    else:
        compact_i = 0.5
        
    if size_j > 2 and C_j.sigma_history and C_j.merge_history:
        compact_j = (C_j.sigma_history[-1] / np.mean(C_j.merge_history) / np.log(size_j)) * size_j / (size_i + size_j)
    else:
        compact_j = 0.5
    
    compact = compact_i + compact_j
    
    # Compute base length (matches original)
    c_past_dists = [max(past_dists), points_dist] if past_dists else [points_dist]
    
    # Handle case where centers might be None
    if C_i.center is not None and C_j.center is not None:
        center_dist = np.linalg.norm(C_i.center - C_j.center) / compact
    else:
        center_dist = points_dist
    
    rate = max(center_dist / points_dist, 3) - 3
    enlarge_rate = 2 / (1 + math.exp(-rate / 20))
    base_length = np.mean(c_past_dists) * enlarge_rate
    
    if _debug:
        print(f"    [continuity] points_dist={points_dist:.4f}, compact={compact:.4f}")
        print(f"    [continuity] c_past_dists={c_past_dists}, enlarge_rate={enlarge_rate:.4f}")
        print(f"    [continuity] base_length={base_length:.4f}, radius[0]={2*base_length:.4f}")
    
    # Multi-scale exploration (matches original: [2, 2.5, 3])
    explore_range = [2.0, 2.5, 3.0]
    
    min_radius_idx = None
    min_radius_smoothness = 1.0  # Store smoothness at min_radius index (like legacy)
    surrounded = False
    prev_smoothness = 1.0
    prev_mass_smoothness = 1.0
    prev_is_boundary = False
    last_smoothness = 1.0  # Track last iteration's smoothness
    
    # Legacy searches ALL dataset points, not just cluster points
    all_point_indices = list(range(len(X)))
    
    for i, radius_mult in enumerate(explore_range):
        radius = radius_mult * base_length
        
        # Find local points from ALL dataset points (matches legacy behavior)
        local_points_i = _find_local_points_original(middle_point, p_i, p_j, radius, X, C_i.point_indices, all_point_indices)
        local_points_j = _find_local_points_original(middle_point, p_j, p_i, radius, X, C_j.point_indices, all_point_indices)
        
        N_i = len(local_points_i) + 1
        N_j = len(local_points_j) + 1
        
        if _debug:
            # Show detailed info for singleton merges
            if len(C_i) == 1 or len(C_j) == 1:
                print(f"    [continuity] i={i} r={radius:.2f}, local_i={len(local_points_i)}, local_j={len(local_points_j)}, N_j={N_j}")
        
        # Remove outliers (matches original)
        local_points_i = _remove_outliers(local_points_i, middle_point, X)
        local_points_j = _remove_outliers(local_points_j, middle_point, X)
        
        angle_smoothness = 1.0
        mass_smoothness = 1.0
        transition_smoothness = 1.0
        is_boundary = False
        
        if len(local_points_i) > 1 and len(local_points_j) > 1:
            # Surrounded detection (matches original)
            if not surrounded:
                count_in_i = sum(1 for p, _ in local_points_j if int(p) in C_i.point_indices)
                surrounded = count_in_i >= len(C_j) and count_in_i > N_j / 2
                if _debug and surrounded:
                    print(f"    [continuity] SURROUNDED at i={i}: count_in_i={count_in_i}, len(C_j)={len(C_j)}, N_j/2={N_j/2}")
            
            # Compute max angles (matches original compute_max_angle)
            max_angle_i = _compute_max_angle(local_points_i, p_i, X)
            max_angle_j = _compute_max_angle(local_points_j, p_j, X)
            
            area_i = max_angle_i[2] if max_angle_i else 0.0
            area_j = max_angle_j[2] if max_angle_j else 0.0
            
            # Compute mass (matches original)
            if min(area_i, area_j) == 0 and max(area_i, area_j) <= 0.2:
                mass_i = mass_j = 1.0
            else:
                mass_i = N_i * area_i
                mass_j = N_j * area_j
            
            if max(mass_i, mass_j) > 0:
                mass_smoothness = min(mass_i, mass_j) / max(mass_i, mass_j)
            
            # Compute angle transition (matches original compute_angle_transition_2)
            if max_angle_i and max_angle_j:
                angle_smoothness = _compute_angle_transition_original(
                    p_i, p_j, max_angle_i, max_angle_j, local_points_i, local_points_j, X
                )
            
            # Compute transition smoothness (matches original)
            transition_smoothness = _compute_transition_smoothness_original(
                p_i, p_j, middle_point, N_i, N_j, radius, radius_mult - 1, X, C_i, C_j, _debug
            )
        
        # Orientation smoothness (matches original)
        orientation_smoothness = math.sqrt(mass_smoothness * angle_smoothness)
        
        # Final smoothness (matches original)
        smoothness = min(transition_smoothness * orientation_smoothness, 1.0)
        
        if _debug:
            print(f"    [continuity] i={i} trans={transition_smoothness:.4f}, orient={orientation_smoothness:.4f}, smooth={smoothness:.4f}, surrounded={surrounded}")
        
        # Apply surrounded override (matches original)
        if surrounded:
            smoothness = 1.0
        
        # Boundary detection (matches original)
        if transition_smoothness > 2:
            is_boundary = True
        
        # Track min_radius where smoothness <= threshold (store smoothness like legacy)
        if smoothness <= T_continuity:
            min_radius_idx = i
            min_radius_smoothness = smoothness
        
        # Early break conditions (matches original)
        if i > 0:
            # Boundary case
            if is_boundary and prev_is_boundary:
                return 1.0  # Legacy sets smoothness=1 and returns it
            
            # Both smoothness == 1
            if prev_smoothness == 1.0 and smoothness == 1.0:
                return smoothness
            
            # Mass smoothness ratio jump
            if prev_mass_smoothness > 0 and mass_smoothness / prev_mass_smoothness > math.e:
                return smoothness
        
        prev_smoothness = smoothness
        prev_mass_smoothness = mass_smoothness
        prev_is_boundary = is_boundary
        last_smoothness = smoothness
    
    # Legacy returns locality_info[min_radius]['smoothness']
    # If min_radius was set, return that smoothness; otherwise return last
    if min_radius_idx is not None:
        return min_radius_smoothness
    else:
        return last_smoothness


def _find_local_points_simple(
    midpoint: np.ndarray,
    ref_point: int,
    exclude_point: int,
    radius: float,
    X: np.ndarray,
    cluster_indices: list[int],
) -> list[tuple[int, float]]:
    """Simple implementation of finding local points with angles."""
    local_points = []

    for idx in cluster_indices:
        if idx == exclude_point:
            continue

        dist_to_mid = np.linalg.norm(X[idx] - midpoint)
        if dist_to_mid <= radius:
            # Compute angle from midpoint
            angle = compute_angle(midpoint, X[ref_point], X[idx])

            # Only include points roughly on same side (angle < π/2)
            if angle <= math.pi / 2 or angle >= 3 * math.pi / 2:
                local_points.append((idx, angle))

    # Sort by angle
    local_points.sort(key=lambda x: x[1])
    return local_points


def compute_density_transition(
    local_i: list[tuple[int, float]],
    local_j: list[tuple[int, float]],
    p_i: int,
    p_j: int,
    midpoint: np.ndarray,
    radius: float,
    X: np.ndarray,
) -> float:
    """
    Compute density transition smoothness.

    Checks for sudden drops in point density at merge interface.

    Args:
        local_i: Local points near p_i with angles
        local_j: Local points near p_j with angles
        p_i, p_j: Reference point indices
        midpoint: Midpoint between p_i and p_j
        radius: Search radius
        X: Data matrix

    Returns:
        Density transition score
    """
    n_i = len(local_i)
    n_j = len(local_j)

    if n_i == 0 and n_j == 0:
        return 1.0

    if n_i == 0 or n_j == 0:
        return 0.5

    # Density ratio: min/max for balance
    density_ratio = min(n_i, n_j) / max(n_i, n_j)

    return density_ratio


def compute_angle_transition(
    local_i: list[tuple[int, float]],
    local_j: list[tuple[int, float]],
    p_i: int,
    p_j: int,
    X: np.ndarray,
) -> float:
    """
    Compute angle transition smoothness.

    Checks for abrupt changes in cluster growth direction.

    Args:
        local_i: Local points near p_i with angles
        local_j: Local points near p_j with angles
        p_i, p_j: Reference point indices
        X: Data matrix

    Returns:
        Angle transition score in [0, 1]
    """
    if len(local_i) < 2 or len(local_j) < 2:
        return 1.0

    midpoint = (X[p_i] + X[p_j]) / 2.0

    # Get extreme points from each local set
    local_i[-1][0] if local_i else p_i
    local_j[-1][0] if local_j else p_j

    # Compute transition angles
    transition_angles = []

    for idx_i, _ in local_i[-2:]:  # Use last 2 from local_i
        for idx_j, _ in local_j[-2:]:  # Use last 2 from local_j
            angle = compute_angle(midpoint, X[idx_i], X[idx_j])
            transition_angles.append(angle)

    if not transition_angles:
        return 1.0

    # Score based on how close angles are to π/2 (perpendicular is bad)
    # 2 * |min(θ, |θ - π|) - π/2| / π
    scores = []
    for angle in transition_angles:
        deviation = min(angle, abs(angle - math.pi))
        score = 2 * abs(deviation - math.pi / 2) / math.pi
        scores.append(score)

    return float(np.mean(scores))


def compute_mass_smoothness(
    local_i: list[tuple[int, float]],
    local_j: list[tuple[int, float]],
    max_angle_i: float,
    max_angle_j: float,
) -> float:
    """
    Compute mass smoothness between local neighborhoods.

    mass_i = N_i × max_angle_i
    mass_j = N_j × max_angle_j
    smoothness = min(mass_i, mass_j) / max(mass_i, mass_j)

    Args:
        local_i: Local points near p_i
        local_j: Local points near p_j
        max_angle_i: Maximum angle spread in local_i
        max_angle_j: Maximum angle spread in local_j

    Returns:
        Mass smoothness in [0, 1]
    """
    n_i = len(local_i)
    n_j = len(local_j)

    # Mass = count × angle spread
    mass_i = n_i * max_angle_i if max_angle_i > 0 else n_i
    mass_j = n_j * max_angle_j if max_angle_j > 0 else n_j

    if mass_i == 0 and mass_j == 0:
        return 1.0

    if mass_i == 0 or mass_j == 0:
        return 0.0

    return min(mass_i, mass_j) / max(mass_i, mass_j)


def compute_compactness(C: Cluster, X: np.ndarray) -> float:
    """
    Compute compactness factor for a cluster.

    compact = (σ_history[-1] / μ_merge_history / log(size)) × weight

    Args:
        C: Cluster to evaluate
        X: Data matrix

    Returns:
        Compactness factor
    """
    size = len(C)

    if size <= 2:
        return 0.5

    if len(C.sigma_history) == 0 or len(C.merge_history) == 0:
        return 0.5

    sigma = C.sigma_history[-1]
    mean_dist = np.mean(C.merge_history)

    if mean_dist == 0:
        return 0.5

    # Compactness formula from legacy code
    compact = (sigma / mean_dist / math.log(size)) if size > 1 else sigma / mean_dist

    return max(0.0, compact)


def find_local_points(
    midpoint: np.ndarray,
    reference_point: int,
    exclude_point: int,
    radius: float,
    X: np.ndarray,
    point_distances: dict,
) -> list[tuple[int, float]]:
    """
    Find points within radius that are on the same side as reference.

    Points must satisfy angle constraint: within π/2 of reference direction.

    Args:
        midpoint: Center of search
        reference_point: Index of reference point
        exclude_point: Index of point to exclude (other cluster's reference)
        radius: Search radius
        X: Data matrix
        point_distances: Pre-computed point-to-point distances

    Returns:
        List of (point_index, angle) tuples, sorted by angle
    """
    local_points = []

    for idx in range(len(X)):
        if idx == exclude_point:
            continue

        # Check distance to midpoint
        dist_to_mid = np.linalg.norm(X[idx] - midpoint)
        if dist_to_mid > radius:
            continue

        # Compute angle from midpoint
        angle = compute_angle(midpoint, X[reference_point], X[idx])

        # Only include points on same side (angle constraint)
        if angle <= math.pi / 2:
            local_points.append((idx, angle))

    # Sort by angle
    local_points.sort(key=lambda x: x[1])
    return local_points


# =============================================================================
# Helper functions matching original algorithm exactly
# =============================================================================

def _find_local_points_original(
    middle_point: np.ndarray,
    point: int,
    exclude_point: int,
    radius: float,
    X: np.ndarray,
    cluster_indices: list[int],
    all_indices: list[int] | None = None,
) -> list[tuple[int, float]]:
    """
    Find local points matching original's find_local_points.
    
    IMPORTANT: Original searches ALL points within radius, not just cluster points.
    Points within radius from 'point' that are on the same side as middle_point.
    
    Args:
        all_indices: If provided, search these indices (should be all data points).
                    If None, falls back to cluster_indices for backward compatibility.
    """
    local_points = []
    
    # Original searches ALL points, not just cluster points
    search_indices = all_indices if all_indices is not None else cluster_indices
    
    for p in search_indices:
        if p == exclude_point or p == point:
            continue
            
        # Check distance from reference point (matches original)
        dist = np.linalg.norm(X[point] - X[p])
        if dist > radius:
            continue
        
        # Compute angle (matches original)
        angle = compute_angle(middle_point, X[point], X[p])
        
        # Angle constraint (matches original)
        if angle <= math.pi / 2 or angle >= 3 * math.pi / 2:
            _angle = angle if angle <= math.pi / 2 else angle - 2 * math.pi
            local_points.append((p, _angle))
    
    # Sort by angle (matches original)
    local_points.sort(key=lambda x: x[1])
    return local_points


def _remove_outliers(
    local_points: list[tuple[int, float]],
    middle_point: np.ndarray,
    X: np.ndarray,
) -> list[tuple[int, float]]:
    """
    Remove outliers matching original's remove_outliers.
    """
    n = len(local_points)
    if n <= 3:
        return local_points
    
    # Check if last point is an outlier (matches original contain_outliers)
    last_point, last_angle = local_points[-1]
    is_outlier = True
    
    for i in range(n - 1):
        p, _ = local_points[i]
        theta = compute_angle(middle_point, X[last_point], X[p])
        if theta < abs(last_angle) / 2:
            is_outlier = False
            break
    
    if is_outlier:
        return local_points[:-1]
    return local_points


def _compute_max_angle(
    local_points: list[tuple[int, float]],
    ref_point: int,
    X: np.ndarray,
) -> tuple[int, int, float] | None:
    """
    Compute max angle info matching original's compute_max_angle.
    
    Returns (max_p1, max_p2, max_angle) or None if insufficient points.
    """
    if len(local_points) < 2:
        return None
    
    max_p1, max_angle1 = local_points[-1]
    
    for i in reversed(range(len(local_points) - 1)):
        p, _ = local_points[i]
        angle = compute_angle(X[ref_point], X[max_p1], X[p])
        if angle > max_angle1:
            return (max_p1, p, angle)
    
    # Fallback
    p0, _ = local_points[0]
    angle = compute_angle(X[ref_point], X[max_p1], X[p0])
    return (max_p1, p0, angle if angle != 0 else max_angle1)


def _compute_angle_transition_original(
    p_i: int,
    p_j: int,
    max_angle_i: tuple[int, int, float],
    max_angle_j: tuple[int, int, float],
    local_points_i: list[tuple[int, float]],
    local_points_j: list[tuple[int, float]],
    X: np.ndarray,
) -> float:
    """
    Compute angle transition matching original's compute_angle_transition_2.
    """
    if not local_points_i or not local_points_j:
        return 1.0
    
    middle_point = (X[p_i] + X[p_j]) / 2.0
    
    _p1, _p2, _ = max_angle_i
    _p3, _p4, _ = max_angle_j
    
    def find_smallest_angle(start_p, p, rp, points):
        angle = math.pi
        for v, _ in points:
            theta = compute_angle(start_p, X[p], X[v])
            r_theta = compute_angle(start_p, X[rp], X[v])
            if theta < angle and r_theta >= theta:
                angle = theta
        return angle
    
    transition_angles = [
        find_smallest_angle(middle_point, _p1, p_i, local_points_j),
        find_smallest_angle(middle_point, _p2, p_i, local_points_j),
        find_smallest_angle(middle_point, _p3, p_j, local_points_i),
        find_smallest_angle(middle_point, _p4, p_j, local_points_i),
    ]
    
    angle_transition = [
        2 * abs(min(angle, abs(angle - math.pi)) - math.pi / 2) / math.pi
        for angle in transition_angles
    ]
    
    return float(np.mean(angle_transition))


def _compute_transition_smoothness_original(
    p_i: int,
    p_j: int,
    middle_point: np.ndarray,
    N_i: int,
    N_j: int,
    radius: float,
    r_rate: float,
    X: np.ndarray,
    C_i: Cluster,
    C_j: Cluster,
    _debug: bool = False,
) -> float:
    """
    Compute transition smoothness matching original's compute_transition_smoothness.
    """
    # Compute transition state (r_e, r_i, g_i, g_e)
    # Legacy searches ALL dataset points via points_dist ordering with early break.
    n_points = len(X)

    def _points_within_radius(point_idx: int) -> list[int]:
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        order = np.argsort(distances)
        selected: list[int] = []
        for idx in order:
            if idx == point_idx:
                continue
            dist = distances[idx]
            # Legacy uses `if dist > radius: break` on sorted distances
            if dist > radius:
                break
            selected.append(int(idx))
        return selected

    all_local_i = []
    all_local_j = []

    for p in _points_within_radius(p_i):
        angle = compute_angle(middle_point, X[p_i], X[p])
        angle = angle if angle <= math.pi else angle - 2 * math.pi
        all_local_i.append((p, angle))

    for p in _points_within_radius(p_j):
        angle = compute_angle(middle_point, X[p_j], X[p])
        angle = angle if angle <= math.pi else angle - 2 * math.pi
        all_local_j.append((p, angle))
    
    # g_i = points from cluster_i in local_j area - N_j's own points
    g_i = len(all_local_i) + 1 - N_i
    r_i = len(all_local_j) + 1 - N_j
    
    r_e = N_i - r_i  # external red
    g_e = N_j - g_i  # external green
    
    # Ensure non-negative
    r_e = max(0, r_e)
    r_i = max(0, r_i)
    g_i = max(0, g_i)
    g_e = max(0, g_e)
    
    # Debug transition state
    if _debug:  # Set True to trace
        print(f"      [trans] all_local_i={len(all_local_i)}, all_local_j={len(all_local_j)}, N_i={N_i}, N_j={N_j}")
        print(f"      [trans] r_e={r_e}, r_i={r_i}, g_i={g_i}, g_e={g_e}")
    
    # Compute transition smoothness (matches original logic)
    if r_e == 0 or g_e == 0:
        if r_rate > 0:
            transition_smoothness = max(2, (max(N_i, N_j) / min(N_i, N_j)) / r_rate)
        else:
            transition_smoothness = 2
    elif ((r_i < g_i or r_i < g_e) and r_i < r_e) or (g_i < g_e and (g_i < r_i or g_i < r_e)):
        denom1 = max(g_i, g_e) if max(g_i, g_e) > 0 else 1
        denom2 = max(r_i, r_e) if max(r_i, r_e) > 0 else 1
        transition_smoothness = min(min(r_i, g_i) / denom1, min(r_i, g_i) / denom2)
    else:
        if (max([g_i, g_e, r_i]) > 0 and r_e / max([g_i, g_e, r_i]) <= 0.1) or \
           (max([g_i, r_e, r_i]) > 0 and g_e / max([g_i, r_e, r_i]) <= 0.1):
            transition_smoothness = 1.5
        else:
            transition_smoothness = 1.0
    
    return transition_smoothness
