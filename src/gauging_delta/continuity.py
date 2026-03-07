"""Continuity analysis (Paper Section II.C).

Determines whether two clusters are separated by a genuine gap or connected by
a smooth bridge of points. It examines the region between the two closest boundary
points (one from each cluster) at multiple radii, checking:

1. **Density transition** — whether point counts change smoothly across the boundary.
2. **Angular distribution** — whether points in each cluster's half fan out symmetrically
   or concentrate in a narrow wedge.
3. **Mass balance** — whether both sides of the boundary have comparable point densities.

A high smoothness score means the clusters share a continuous region and should merge.
A low score means there is a gap or density drop, and the merge should be rejected.

This component is **swappable** — any object implementing the :class:`ContinuityMetric`
protocol can replace it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from gauging_delta.angles import compute_angle, compute_angle_batch
from gauging_delta.cluster import Cluster
from gauging_delta.config import GaugingDeltaConfig


class DefaultContinuity:
    """Default continuity gate: multi-scale angle-based local transition analysis."""

    def __init__(self, cfg: GaugingDeltaConfig | None = None) -> None:
        self.cfg = cfg or GaugingDeltaConfig()

    def compute(
        self,
        lead: Cluster,
        child: Cluster,
        threshold: float,
        d_ij_norm: float,
        adp_prox: float,
        X: np.ndarray,
        point_dists: tuple[np.ndarray, np.ndarray],
    ) -> float:
        """Compute continuity (smoothness) score between *lead* and *child*.

        Returns a float in [0, 1] where 1.0 = perfectly smooth transition (merge),
        and values below *threshold* trigger rejection.
        ``point_dists`` is ``(pd_indices, pd_dists)`` — per-point sorted neighbor arrays.
        """
        return _compute_local_transition(
            lead, child, threshold, d_ij_norm, adp_prox, X, point_dists, self.cfg
        )


# ---------------------------------------------------------------------------
# Core transition logic
# ---------------------------------------------------------------------------


def _compute_local_transition(
    lead: Cluster,
    child: Cluster,
    threshold_cont: float,
    d_ij_norm: float,
    adp_prox: float,
    X: np.ndarray,
    point_dists: tuple[np.ndarray, np.ndarray],
    cfg: GaugingDeltaConfig,
) -> float:
    """Analyze the spatial transition between two clusters at multiple radii.

    Examines the region around the midpoint of the two closest boundary points
    (p1 from lead, p2 from child). At each radius, it measures density balance,
    angular coverage, and transition smoothness, combining them into an overall
    smoothness score.
    """
    p1 = lead.ref_point
    p2 = child.ref_point
    points_dist = float(np.linalg.norm(X[p1] - X[p2]))
    middle_point = (X[p1] + X[p2]) / 2

    # --- Compactness: how tightly packed each cluster is ---
    past_dists = [*lead.merge_history[-cfg.compact_history_window:], *child.merge_history[-cfg.compact_history_window:]]
    size1, size2 = len(lead), len(child)

    if size1 > cfg.compact_min_size:
        mean_pd1 = float(np.mean(lead.merge_history))
        compact1 = (lead.sigma_history[-1] / mean_pd1 / math.log(size1)) * size1 / (size1 + size2)
    else:
        compact1 = cfg.compact_fallback
    if size2 > cfg.compact_min_size:
        mean_pd2 = float(np.mean(child.merge_history))
        compact2 = (child.sigma_history[-1] / mean_pd2 / math.log(size2)) * size2 / (size1 + size2)
    else:
        compact2 = cfg.compact_fallback
    compact = compact1 + compact2

    # --- Base length: determines the scale for the multi-radius exploration ---
    c_past_dists = [max(past_dists), points_dist] if past_dists else [points_dist]
    center_dist = float(np.linalg.norm(lead.center - child.center)) / compact

    # When points_dist=0 (duplicate/coincident points), numpy 0/0 produces nan.
    # nan propagates to radius=nan, and since `dist > nan` is always False,
    # ALL points pass the radius filter. This behavior is intentional.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = float(np.float64(center_dist) / np.float64(points_dist))
    rate = max(ratio, cfg.enlarge_rate_floor) - cfg.enlarge_rate_floor
    enlarge_rate = 2 / (1 + math.e ** (-rate / cfg.enlarge_rate_divisor))
    base_length = float(np.mean(c_past_dists)) * enlarge_rate

    # --- Multi-radius exploration: analyze the boundary at increasing radii ---
    explore_range = cfg.explore_radii

    locality_info: list[dict[str, Any]] = []
    for r in explore_range:
        locality_info.append(
            {
                "radius": r * base_length,
                "red": {"max angle": 0.0, "N": 0, "mass": 1.0, "local_points": np.empty((0, 2))},
                "green": {"max angle": 0.0, "N": 0, "mass": 1.0, "local_points": np.empty((0, 2))},
                "angle_smoothness": 1.0,
                "mass_smoothness": 1.0,
                "transition_smoothness": 1.0,
                "orientation_smoothness": 0.0,
                "smoothness": 1.0,
                "is_boundary": False,
            }
        )

    min_radius = None
    surrounded = False

    c1_points_set = set(lead.point_indices)

    pd_indices, pd_dists = point_dists

    # Pre-compute angles at max radius (once per merge attempt)
    max_r = max(explore_range) * base_length
    max_cutoff_1 = int(np.searchsorted(pd_dists[p1], max_r, side="right"))
    max_cutoff_2 = int(np.searchsorted(pd_dists[p2], max_r, side="right"))

    if max_cutoff_1 > 0:
        cands_1 = pd_indices[p1, :max_cutoff_1]
        all_angles_1 = compute_angle_batch(middle_point, X[cands_1], X[p1], rounding=4)
    else:
        cands_1 = np.empty(0, dtype=int)
        all_angles_1 = np.empty(0)

    if max_cutoff_2 > 0:
        cands_2 = pd_indices[p2, :max_cutoff_2]
        all_angles_2 = compute_angle_batch(middle_point, X[cands_2], X[p2], rounding=4)
    else:
        cands_2 = np.empty(0, dtype=int)
        all_angles_2 = np.empty(0)

    for i, radius_mult in enumerate(explore_range):
        _r = radius_mult * base_length

        # Slice pre-computed angles to this radius's candidates
        cutoff_1 = int(np.searchsorted(pd_dists[p1], _r, side="right"))
        cutoff_2 = int(np.searchsorted(pd_dists[p2], _r, side="right"))

        _local_points1 = _filter_local_points(cands_1[:cutoff_1], all_angles_1[:cutoff_1], p2)
        _local_points2 = _filter_local_points(cands_2[:cutoff_2], all_angles_2[:cutoff_2], p1)

        locality_info[i]["red"]["local_points"] = _local_points1
        locality_info[i]["green"]["local_points"] = _local_points2

        N1 = len(_local_points1) + 1
        N2 = len(_local_points2) + 1
        locality_info[i]["radius"] = _r
        locality_info[i]["red"]["N"] = N1
        locality_info[i]["green"]["N"] = N2

        # Remove angular outliers (points with extreme angles relative to the group)
        _local_points1 = _remove_outliers(_local_points1, middle_point, X, cfg)
        _local_points2 = _remove_outliers(_local_points2, middle_point, X, cfg)

        if len(_local_points1) > 1 and len(_local_points2) > 1:
            # Check if child is completely surrounded by lead's points
            if not surrounded:
                count2 = sum(1 for p in _local_points2[:, 0] if int(p) in c1_points_set)
                surrounded = count2 >= len(child) and count2 > N2 / 2

            # Find the widest angular gap in each side's point distribution
            max_angle_info1 = _compute_max_angle(_local_points1, p1, X, cfg)
            max_angle_info2 = _compute_max_angle(_local_points2, p2, X, cfg)

            area1, area2 = max_angle_info1[2], max_angle_info2[2]
            if min(area1, area2) == 0 and max(area1, area2) <= cfg.max_angle_zero_threshold:
                mass1 = mass2 = 1.0
            else:
                mass1 = N1 * area1
                mass2 = N2 * area2

            locality_info[i]["red"]["mass"] = mass1
            locality_info[i]["green"]["mass"] = mass2
            locality_info[i]["red"]["max angle"] = area1
            locality_info[i]["green"]["max angle"] = area2
            locality_info[i]["mass_smoothness"] = min(mass1, mass2) / max(mass1, mass2)

            # Angle transition: how well the angular extremes from each side
            # align with points on the other side
            angle_smoothness = _compute_angle_transition(
                p1,
                p2,
                [max_angle_info1[0], max_angle_info1[1], max_angle_info2[0], max_angle_info2[1]],
                _local_points1,
                _local_points2,
                X,
                cfg,
            )
            locality_info[i]["angle_smoothness"] = angle_smoothness

            # Density transition: how evenly points are distributed across the boundary
            transition_smoothness = _compute_transition_smoothness(
                p1,
                p2,
                N1,
                N2,
                _r,
                radius_mult - 1,
                point_dists,
                cfg,
            )
            locality_info[i]["transition_smoothness"] = transition_smoothness

        # Orientation smoothness = geometric mean of mass balance and angle alignment
        orientation_smoothness = math.sqrt(
            locality_info[i]["mass_smoothness"] * locality_info[i]["angle_smoothness"]
        )
        locality_info[i]["orientation_smoothness"] = orientation_smoothness

        smoothness = min(locality_info[i]["transition_smoothness"] * orientation_smoothness, 1.0)
        locality_info[i]["smoothness"] = smoothness if not surrounded else 1.0

        if locality_info[i]["transition_smoothness"] > cfg.boundary_threshold:
            locality_info[i]["is_boundary"] = True

        if locality_info[i]["smoothness"] <= threshold_cont:
            min_radius = i

        # Early termination: stop exploring larger radii when the signal is clear
        if i > 0:
            if locality_info[i]["is_boundary"] and locality_info[i - 1]["is_boundary"]:
                min_radius = i
                locality_info[i]["smoothness"] = 1.0
                break

            if locality_info[i - 1]["smoothness"] == 1.0 and locality_info[i]["smoothness"] == 1.0:
                min_radius = i
                break

            # Unguarded division: when prev=0 and curr>0, numpy produces inf.
            # inf > mass_ratio_jump triggers the early break. This is intentional.
            with np.errstate(divide="ignore", invalid="ignore"):
                mass_ratio = float(
                    np.float64(locality_info[i]["mass_smoothness"])
                    / np.float64(locality_info[i - 1]["mass_smoothness"])
                )
            if mass_ratio > cfg.mass_ratio_jump:
                min_radius = i
                break

    if min_radius is None:
        min_radius = -1

    return float(locality_info[min_radius]["smoothness"])


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _filter_local_points(
    cand_indices: np.ndarray,
    angles: np.ndarray,
    exclude_point: int,
    *,
    find_all: bool = False,
) -> np.ndarray:
    """Filter candidate points to those on one side of the boundary midpoint.

    By default, keeps only points whose angle from the midpoint is within [-pi/2, pi/2]
    of the reference direction (i.e., on the same side as the reference cluster).
    Returns an (n, 2) array with columns [point_index, angle], sorted by angle.
    """
    if len(cand_indices) == 0:
        return np.empty((0, 2))

    half_pi = math.pi / 2
    if not find_all:
        three_half_pi = 3 * math.pi / 2
        mask = ((angles <= half_pi) | (angles >= three_half_pi)) & (cand_indices != exclude_point)
        if not np.any(mask):
            return np.empty((0, 2))
        filt_idx = cand_indices[mask]
        filt_ang = angles[mask].copy()
        wrap = filt_ang >= three_half_pi
        filt_ang[wrap] -= 2 * math.pi
    else:
        filt_idx = cand_indices
        filt_ang = angles.copy()
        wrap = filt_ang > math.pi
        filt_ang[wrap] -= 2 * math.pi

    order = np.argsort(filt_ang, kind="stable")
    return np.column_stack((filt_idx[order].astype(float), filt_ang[order]))


def _remove_outliers(
    local_points: np.ndarray,
    middle_point: np.ndarray,
    X: np.ndarray,
    cfg: GaugingDeltaConfig,
) -> np.ndarray:
    """Remove the last (widest-angle) point if it is an angular outlier."""
    n = len(local_points)
    if n > cfg.outlier_min_points and _contain_outliers(local_points, middle_point, X):
        return local_points[:-1]
    return local_points


def _contain_outliers(
    local_points: np.ndarray,
    middle_point: np.ndarray,
    X: np.ndarray,
) -> bool:
    """Check if the last point (widest angle) is an outlier.

    Returns True if no other point has an angle less than half the last point's angle
    when measured from the midpoint, indicating the last point is isolated.
    """
    last_angle = local_points[-1][1]
    last_point = int(local_points[-1][0])
    if len(local_points) <= 1:
        return True
    indices = local_points[:-1, 0].astype(int)
    thetas = compute_angle_batch(middle_point, X[indices], X[last_point])
    return not np.any(thetas < last_angle / 2)


def _compute_max_angle(
    local_points: np.ndarray,
    ref_point: int,
    X: np.ndarray,
    cfg: GaugingDeltaConfig,
) -> tuple[int, int, float]:
    """Find the widest angular gap in the local point distribution.

    Returns (point_a, point_b, angle) where point_a and point_b are the two
    points that define the largest angle as seen from ref_point.
    """
    max_p1 = int(local_points[-1][0])
    max_angle1 = local_points[-1][1]

    if len(local_points) > 1:
        indices = local_points[:-1, 0].astype(int)
        angles = compute_angle_batch(X[ref_point], X[indices], X[max_p1])
        # Reversed iteration: find largest index where angle > max_angle1
        exceeds = np.where(angles > max_angle1)[0]
        if len(exceeds) > 0:
            idx = int(exceeds[-1])
            return max_p1, int(local_points[idx][0]), float(angles[idx])

    # Fallback: use the angle between the last and first points
    angle = compute_angle(X[ref_point], X[max_p1], X[int(local_points[0][0])])
    return max_p1, int(local_points[0][0]), angle if angle != 0 else max_angle1


def _compute_angle_transition(
    p1: int,
    p2: int,
    max_points: list[int],
    local_points1: np.ndarray,
    local_points2: np.ndarray,
    X: np.ndarray,
    cfg: GaugingDeltaConfig,
) -> float:
    """Measure how well the angular extremes from one side align with the other side.

    For each side's widest-angle pair, find the smallest matching angle among points
    on the opposite side. Returns the mean alignment score (0 = perpendicular/no match,
    1 = perfectly aligned).
    """
    if len(local_points1) == 0 or len(local_points2) == 0:
        return 1.0

    middle_point = (X[p1] + X[p2]) / 2
    _p1, _p2, _p3, _p4 = max_points

    # Fused: calls 1-2 share local_points2 and start_p=middle_point
    # Calls 3-4 share local_points1 and start_p=middle_point
    a1, a2 = _find_smallest_angle_pair(middle_point, _p1, _p2, p1, local_points2, X)
    a3, a4 = _find_smallest_angle_pair(middle_point, _p3, _p4, p2, local_points1, X)

    transition_angles = [a1, a2, a3, a4]
    angle_transition = [
        2 * math.fabs(min(a, math.fabs(a - math.pi)) - math.pi / 2) / math.pi
        for a in transition_angles
    ]
    return float(np.mean(angle_transition))


def _find_smallest_angle_pair(
    start_p: np.ndarray,
    pa: int,
    pb: int,
    rp: int,
    points: np.ndarray,
    X: np.ndarray,
) -> tuple[float, float]:
    """Find the smallest angle from start_p to each of pa and pb among candidate points.

    Only considers candidates whose angle to rp (the reference point) is at least as
    large as their angle to pa/pb, ensuring directional consistency. Returns (min_angle_pa,
    min_angle_pb), defaulting to pi (no match) when no qualifying candidate exists.
    """
    if len(points) == 0:
        return math.pi, math.pi

    v_indices = points[:, 0].astype(int)
    v1 = X[v_indices] - start_p  # (n, d)
    norms1 = np.linalg.norm(v1, axis=1)  # (n,)

    # Batch all 3 reference vectors into one matmul
    v2_all = X[np.array([rp, pa, pb])] - start_p  # (3, d)
    norms_all = np.linalg.norm(v2_all, axis=1)  # (3,)
    dots_all = v1 @ v2_all.T  # (n, 3)
    norm_prods_all = norms1[:, None] * norms_all  # (n, 3)

    nonzero_all = np.round(norm_prods_all, 4) != 0
    safe_norm_prods = np.where(nonzero_all, norm_prods_all, 1.0)
    cos_all = np.clip(np.round(dots_all / safe_norm_prods, 4), -1.0, 1.0)
    thetas_all = np.where(nonzero_all, np.arccos(cos_all), 0.0)

    r_thetas = thetas_all[:, 0]

    # For pa and pb: find min angle where r_thetas >= thetas
    pa_filtered = np.where(r_thetas >= thetas_all[:, 1], thetas_all[:, 1], np.inf)
    pb_filtered = np.where(r_thetas >= thetas_all[:, 2], thetas_all[:, 2], np.inf)
    min_a = float(np.min(pa_filtered))
    min_b = float(np.min(pb_filtered))

    return (min_a if not np.isinf(min_a) else math.pi,
            min_b if not np.isinf(min_b) else math.pi)


def _compute_transition_smoothness(
    p1: int,
    p2: int,
    N1: int,
    N2: int,
    radius: float,
    r_rate: float,
    point_dists: tuple[np.ndarray, np.ndarray],
    cfg: GaugingDeltaConfig,
) -> float:
    """Measure how smoothly point density transitions across the boundary.

    Counts points from each cluster that fall inside vs. outside the partner's radius,
    producing four counts (r_e, r_i, g_i, g_e = external/internal for each side).
    A smooth transition has balanced internal-to-external ratios on both sides.
    """
    r_e, r_i, g_i, g_e = _compute_transition_state(p1, p2, N1, N2, radius, point_dists)
    if r_e == 0 or g_e == 0:
        return max(cfg.transition_external_zero_fallback, (max(N1, N2) / min(N1, N2)) / r_rate) if r_rate != 0 else cfg.transition_external_zero_fallback
    elif ((r_i < g_i or r_i < g_e) and r_i < r_e) or (g_i < g_e and (g_i < r_i or g_i < r_e)):
        return min(
            min(r_i, g_i) / max(g_i, g_e) if max(g_i, g_e) != 0 else 0.0,
            min(r_i, g_i) / max(r_i, r_e) if max(r_i, r_e) != 0 else 0.0,
        )
    else:
        if r_e / max(g_i, g_e, r_i) <= cfg.transition_lopsided_ratio or g_e / max(g_i, r_e, r_i) <= cfg.transition_lopsided_ratio:
            return cfg.transition_lopsided_override
        return 1.0


def _compute_transition_state(
    p1: int,
    p2: int,
    N1: int,
    N2: int,
    radius: float,
    point_dists: tuple[np.ndarray, np.ndarray],
) -> tuple[float, float, float, float]:
    """Count how points from each cluster distribute inside vs. outside the boundary.

    Returns (r_e, r_i, g_i, g_e) where:
      - r_i = "red internal"  — points near p2 that belong to p1's cluster
      - r_e = "red external"  — points near p1 that belong to p1's cluster
      - g_i = "green internal" — points near p1 that belong to p2's cluster
      - g_e = "green external" — points near p2 that belong to p2's cluster

    Uses searchsorted on pre-sorted distance arrays for O(log N) lookups.
    """
    _, pd_dists = point_dists
    count_1 = int(np.searchsorted(pd_dists[p1], radius, side="right"))
    count_2 = int(np.searchsorted(pd_dists[p2], radius, side="right"))

    g_i = count_1 + 1 - N1
    r_i = count_2 + 1 - N2
    r_e = N1 - r_i
    g_e = N2 - g_i
    return float(r_e), float(r_i), float(g_i), float(g_e)
