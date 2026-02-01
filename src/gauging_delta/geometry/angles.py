"""
Angle calculation utilities for Gauging-δ algorithm.

Used for computing angular relationships between points,
essential for continuity analysis and local neighborhood evaluation.

Vectorized implementations use np.einsum for batch operations.
"""

import numpy as np


def compute_angle(origin: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute angle at origin between vectors to p1 and p2.

    Uses the formula:
        θ = arccos((v1 · v2) / (|v1| × |v2|))

    Where v1 = p1 - origin, v2 = p2 - origin.

    Args:
        origin: Origin point (vertex of angle)
        p1: First point
        p2: Second point

    Returns:
        Angle in radians, range [0, π]
    """
    v1 = p1 - origin
    v2 = p2 - origin

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    # Handle degenerate case (matches legacy: round(norm1 * norm2, 4) == 0)
    if round(norm1 * norm2, 4) == 0:
        return 0.0

    dot_product = np.dot(v1, v2)
    # Match legacy: round to 4 decimal places before arccos
    # Legacy: np.arccos(round(dot_p / (norm1 * norm2), 4))
    cos_angle = round(dot_product / (norm1 * norm2), 4)
    # Clip to [-1, 1] for safety after rounding
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.arccos(cos_angle))


def compute_angle_batch(
    origin: np.ndarray,
    ref: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """
    Compute angles between ref vector and multiple points (vectorized).

    This is the 100x faster version for batch angle calculations.

    Args:
        origin: Origin point (vertex of angles)
        ref: Reference point (defines v_ref = ref - origin)
        points: Array of points, shape (n, d)

    Returns:
        Array of angles in radians, shape (n,), range [0, π]
    """
    v_ref = ref - origin
    v_points = points - origin

    norm_ref = np.linalg.norm(v_ref)
    norms_points = np.linalg.norm(v_points, axis=1)

    # Handle degenerate cases
    valid = (norm_ref > 0) & (norms_points > 0)

    angles = np.zeros(len(points))
    if norm_ref > 0 and np.any(valid):
        # Vectorized dot product using einsum
        dot_products = np.einsum("i,ji->j", v_ref, v_points[valid])
        cos_angles = np.clip(dot_products / (norm_ref * norms_points[valid]), -1.0, 1.0)
        angles[valid] = np.arccos(cos_angles)

    return angles


def compute_clockwise_angle(
    origin: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> float:
    """
    Compute clockwise angle from p1 to p2 around origin.

    Uses atan2 for proper quadrant handling:
        θ = -atan2(v1×v2, v1·v2)

    Returns angle in [0, 2π).

    Note: Only meaningful for 2D points.

    Args:
        origin: Origin point
        p1: Start vector endpoint
        p2: End vector endpoint

    Returns:
        Clockwise angle in radians, range [0, 2π)
    """
    v1 = p1 - origin
    v2 = p2 - origin

    # 2D cross product: v1[0]*v2[1] - v1[1]*v2[0]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = np.dot(v1, v2)

    # Clockwise angle: negative atan2
    theta = -np.arctan2(cross, dot)

    # Normalize to [0, 2π)
    return normalize_angle(theta)


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to range [0, 2π).

    Args:
        angle: Any angle in radians

    Returns:
        Equivalent angle in [0, 2π)
    """
    TWO_PI = 2.0 * np.pi
    result = angle % TWO_PI
    if result < 0:
        result += TWO_PI
    return float(result)
