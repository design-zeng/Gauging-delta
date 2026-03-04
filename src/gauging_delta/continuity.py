"""Continuity analysis (Paper Section II.C).

Port of ``Perception.compute_local_transition`` (perception.py L713-860)
and its helper functions.  This is **swappable** — any object implementing
the :class:`ContinuityMetric` protocol can replace it.
"""

from __future__ import annotations

import math

import numpy as np

from gauging_delta.angles import compute_angle
from gauging_delta.cluster import Cluster
from gauging_delta.config import GaugingDeltaConfig


class DefaultContinuity:
    """Default continuity: angle-based local transition analysis."""

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
        point_dists: dict[int, list[list]],
    ) -> float:
        """Compute continuity score between *lead* and *child*.

        Direct port of ``Perception.compute_local_transition`` (L713-860).
        Returns the smoothness value used as the continuity gate.
        """
        return _compute_local_transition(
            lead, child, threshold, d_ij_norm, adp_prox, X, point_dists, self.cfg
        )


# ---------------------------------------------------------------------------
# Core transition logic (perception.py L713-860)
# ---------------------------------------------------------------------------


def _compute_local_transition(
    lead: Cluster,
    child: Cluster,
    threshold_cont: float,
    d_ij_norm: float,
    adp_prox: float,
    X: np.ndarray,
    point_dists: dict[int, list[list]],
    cfg: GaugingDeltaConfig,
) -> float:
    """Full port of ``compute_local_transition`` (perception.py L713-860)."""
    p1 = lead.ref_point
    p2 = child.ref_point
    points_dist = float(np.linalg.norm(X[p1] - X[p2]))
    middle_point = (X[p1] + X[p2]) / 2

    # --- Compactness (L728-745) ---
    past_dists = [*lead.merge_history[-5:], *child.merge_history[-5:]]
    size1, size2 = len(lead), len(child)

    if size1 > 2:
        mean_pd1 = float(np.mean(lead.merge_history))
        compact1 = (lead.sigma_history[-1] / mean_pd1 / math.log(size1)) * size1 / (size1 + size2)
    else:
        compact1 = 0.5
    if size2 > 2:
        mean_pd2 = float(np.mean(child.merge_history))
        compact2 = (child.sigma_history[-1] / mean_pd2 / math.log(size2)) * size2 / (size1 + size2)
    else:
        compact2 = 0.5
    compact = compact1 + compact2

    # --- Base length / enlarge rate (L747-753) ---
    c_past_dists = [max(past_dists), points_dist] if past_dists else [points_dist]
    center_dist = float(np.linalg.norm(lead.center - child.center)) / compact

    # When points_dist=0 (duplicate points), legacy gets nan for rate
    # via numpy 0/0.  nan propagates to radius=nan, and since `dist > nan`
    # is always False, ALL points pass the radius filter.  We replicate
    # this exactly by using numpy division (nan instead of ZeroDivisionError).
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = float(np.float64(center_dist) / np.float64(points_dist))
    rate = max(ratio, cfg.enlarge_rate_floor) - cfg.enlarge_rate_floor
    enlarge_rate = 2 / (1 + math.e ** (-rate / cfg.enlarge_rate_divisor))
    base_length = float(np.mean(c_past_dists)) * enlarge_rate

    # --- Explore range loop (L755-860) ---
    explore_range = cfg.explore_radii

    locality_info: list[dict] = []
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

    for i, radius_mult in enumerate(explore_range):
        _r = radius_mult * base_length
        _local_points1 = _find_local_points(middle_point, p1, _r, p2, X, point_dists)
        _local_points2 = _find_local_points(middle_point, p2, _r, p1, X, point_dists)

        locality_info[i]["red"]["local_points"] = _local_points1
        locality_info[i]["green"]["local_points"] = _local_points2

        N1 = len(_local_points1) + 1
        N2 = len(_local_points2) + 1
        locality_info[i]["radius"] = _r
        locality_info[i]["red"]["N"] = N1
        locality_info[i]["green"]["N"] = N2

        # Remove outliers (perception.py L799-800)
        _local_points1 = _remove_outliers(_local_points1, middle_point, X, cfg)
        _local_points2 = _remove_outliers(_local_points2, middle_point, X, cfg)

        if len(_local_points1) > 1 and len(_local_points2) > 1:
            # Surrounded check (perception.py L803-806)
            if not surrounded:
                count2 = sum(1 for p in _local_points2[:, 0] if int(p) in c1_points_set)
                surrounded = count2 >= len(child) and count2 > N2 / 2

            # Max angle (perception.py L808-809)
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

            # Angle transition (perception.py L822-827)
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

            # Transition smoothness (perception.py L829)
            transition_smoothness = _compute_transition_smoothness(
                p1,
                p2,
                middle_point,
                N1,
                N2,
                _r,
                radius_mult - 1,
                X,
                point_dists,
                cfg,
            )
            locality_info[i]["transition_smoothness"] = transition_smoothness

        # Orientation = sqrt(mass * angle) (perception.py L832-834)
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

        # Early termination checks (perception.py L847-859)
        if i > 0:
            if locality_info[i]["is_boundary"] and locality_info[i - 1]["is_boundary"]:
                min_radius = i
                locality_info[i]["smoothness"] = 1.0
                break

            if locality_info[i - 1]["smoothness"] == 1.0 and locality_info[i]["smoothness"] == 1.0:
                min_radius = i
                break

            # Legacy has NO guard on this division (perception.py L857).
            # When prev=0 and curr>0, numpy gives +inf, and inf > e → True,
            # triggering the early break.  We must replicate this exactly.
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


def _find_local_points(
    middle_point: np.ndarray,
    point: int,
    radius: float,
    exclude_point: int,
    X: np.ndarray,
    point_dists: dict[int, list[list]],
    *,
    find_all: bool = False,
) -> np.ndarray:
    """Port of ``find_local_points`` (perception.py L1410-1427).

    Returns (n, 2) array: columns are [point_index, angle].
    """
    local_points: list[list[float]] = []
    ref = X[point]

    for p_dist in point_dists[point]:
        p_idx = int(p_dist[0])
        dist = p_dist[1]

        if dist > radius:
            break

        angle = compute_angle(middle_point, ref, X[p_idx], rounding=4)

        if not find_all:
            if (angle <= math.pi / 2 or angle >= 3 * math.pi / 2) and p_idx != exclude_point:
                _angle = angle if angle <= math.pi / 2 else angle - 2 * math.pi
                local_points.append([p_idx, _angle])
        else:
            angle = angle if angle <= math.pi else angle - 2 * math.pi
            local_points.append([p_idx, angle])

    if not local_points:
        return np.empty((0, 2))
    result = np.array(sorted(local_points, key=lambda x: x[1]))
    return result


def _remove_outliers(
    local_points: np.ndarray,
    middle_point: np.ndarray,
    X: np.ndarray,
    cfg: GaugingDeltaConfig,
) -> np.ndarray:
    """Port of ``remove_outliers`` (perception.py L1037-1043)."""
    n = len(local_points)
    if n > cfg.outlier_min_points and _contain_outliers(local_points, middle_point, X):
        return local_points[:-1]
    return local_points


def _contain_outliers(
    local_points: np.ndarray,
    middle_point: np.ndarray,
    X: np.ndarray,
) -> bool:
    """Port of ``contain_outliers`` (perception.py L1017-1023)."""
    last_angle = local_points[-1][1]
    last_point = int(local_points[-1][0])
    for i in range(len(local_points) - 1):
        theta = compute_angle(middle_point, X[last_point], X[int(local_points[i][0])])
        if theta < last_angle / 2:
            return False
    return True


def _compute_max_angle(
    local_points: np.ndarray,
    ref_point: int,
    X: np.ndarray,
    cfg: GaugingDeltaConfig,
) -> tuple[int, int, float]:
    """Port of ``compute_max_angle`` (perception.py L701-711).

    Returns (max_point_1, max_point_2, angle).
    """
    max_p1 = int(local_points[-1][0])
    max_angle1 = local_points[-1][1]

    for i in reversed(range(len(local_points) - 1)):
        angle = compute_angle(X[ref_point], X[max_p1], X[int(local_points[i][0])])
        if angle > max_angle1:
            return max_p1, int(local_points[i][0]), angle

    # Fallback: compare last with first (perception.py L709-710)
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
    """Port of ``compute_angle_transition_2`` (perception.py L1045-1069)."""
    if len(local_points1) == 0 or len(local_points2) == 0:
        return 1.0

    middle_point = (X[p1] + X[p2]) / 2
    _p1, _p2, _p3, _p4 = max_points

    transition_angles = [
        _find_smallest_angle(middle_point, _p1, p1, local_points2, X),
        _find_smallest_angle(middle_point, _p2, p1, local_points2, X),
        _find_smallest_angle(middle_point, _p3, p2, local_points1, X),
        _find_smallest_angle(middle_point, _p4, p2, local_points1, X),
    ]
    angle_transition = [
        2 * math.fabs(min(a, math.fabs(a - math.pi)) - math.pi / 2) / math.pi
        for a in transition_angles
    ]
    return float(np.mean(angle_transition))


def _find_smallest_angle(
    start_p: np.ndarray,
    p: int,
    rp: int,
    points: np.ndarray,
    X: np.ndarray,
) -> float:
    """Port of inner ``find_smallest_angle`` (perception.py L1046-1053)."""
    angle = math.pi
    for v in points:
        v_idx = int(v[0])
        theta = compute_angle(start_p, X[p], X[v_idx])
        r_theta = compute_angle(start_p, X[rp], X[v_idx])
        if theta < angle and r_theta >= theta:
            angle = theta
    return angle


def _compute_transition_smoothness(
    p1: int,
    p2: int,
    middle_point: np.ndarray,
    N1: int,
    N2: int,
    radius: float,
    r_rate: float,
    X: np.ndarray,
    point_dists: dict[int, list[list]],
    cfg: GaugingDeltaConfig,
) -> float:
    """Port of ``compute_transition_smoothness`` (perception.py L1176-1188)."""
    r_e, r_i, g_i, g_e = _compute_transition_state(
        p1, p2, middle_point, N1, N2, radius, X, point_dists
    )
    if r_e == 0 or g_e == 0:
        return max(2.0, (max(N1, N2) / min(N1, N2)) / r_rate) if r_rate != 0 else 2.0
    elif ((r_i < g_i or r_i < g_e) and r_i < r_e) or (g_i < g_e and (g_i < r_i or g_i < r_e)):
        return min(
            min(r_i, g_i) / max(g_i, g_e) if max(g_i, g_e) != 0 else 0.0,
            min(r_i, g_i) / max(r_i, r_e) if max(r_i, r_e) != 0 else 0.0,
        )
    else:
        if r_e / max(g_i, g_e, r_i) <= 0.1 or g_e / max(g_i, r_e, r_i) <= 0.1:
            return 1.5
        return 1.0


def _compute_transition_state(
    p1: int,
    p2: int,
    middle_point: np.ndarray,
    N1: int,
    N2: int,
    radius: float,
    X: np.ndarray,
    point_dists: dict[int, list[list]],
) -> tuple[float, float, float, float]:
    """Port of ``compute_transition_state`` (perception.py L1363-1370).

    Returns (r_e, r_i, g_i, g_e) — external/internal counts per side.
    """
    all_local_points1 = _find_local_points(
        middle_point, p1, radius, p2, X, point_dists, find_all=True
    )
    all_local_points2 = _find_local_points(
        middle_point, p2, radius, p1, X, point_dists, find_all=True
    )

    g_i = len(all_local_points1) + 1 - N1
    r_i = len(all_local_points2) + 1 - N2

    r_e = N1 - r_i
    g_e = N2 - g_i
    return float(r_e), float(r_i), float(g_i), float(g_e)
