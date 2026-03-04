"""Parity-safe angle computation.

The rounding in ``compute_angle`` exactly matches ``Perception.to_find_angle``
from perception.py (lines 1438-1448):

1. ``round(norm1 * norm2, 4)`` — if zero, return 0
2. ``round(dot / (norm1 * norm2), 4)`` — then arccos

Changing the rounding precision or order **will break parity**.
"""

from __future__ import annotations

import numpy as np


def compute_angle(
    start: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    rounding: int = 4,
) -> float:
    """Angle at *start* between rays to *left* and *right*.

    Reproduces ``Perception.to_find_angle`` exactly.
    """
    v1 = left - start
    v2 = right - start

    dot_p = float(np.dot(v1, v2))
    norm_prod_raw = float(np.linalg.norm(v1)) * float(np.linalg.norm(v2))
    # Legacy rounds product only for the zero check (perception.py L1444),
    # but divides by the UNROUNDED product (perception.py L1447).
    if round(norm_prod_raw, rounding) == 0:
        return 0.0
    cos_val = round(dot_p / norm_prod_raw, rounding)
    # Clamp to [-1, 1] to guard against FP drift after rounding
    cos_val = max(-1.0, min(1.0, cos_val))
    return float(np.arccos(cos_val))


def compute_angle_batch(
    start: np.ndarray,
    points: np.ndarray,
    right: np.ndarray,
    rounding: int = 4,
) -> np.ndarray:
    """Vectorised version of :func:`compute_angle`.

    Parameters
    ----------
    start : (d,)
    points : (n, d) — each row is a "left" point
    right : (d,)
    rounding : int

    Returns
    -------
    angles : (n,)  in radians
    """
    v1 = points - start  # (n, d)
    v2 = right - start  # (d,)

    dots = np.einsum("ij,j->i", v1, v2)  # (n,)
    norms1 = np.linalg.norm(v1, axis=1)  # (n,)
    norm2 = float(np.linalg.norm(v2))

    norm_prods_raw = norms1 * norm2  # (n,) — unrounded
    norm_prods_rounded = np.round(norm_prods_raw, rounding)  # for zero check only

    angles = np.zeros(len(points), dtype=float)
    nonzero = norm_prods_rounded != 0
    # Divide by unrounded product (matching legacy perception.py L1447)
    cos_vals = np.round(dots[nonzero] / norm_prods_raw[nonzero], rounding)
    cos_vals = np.clip(cos_vals, -1.0, 1.0)
    angles[nonzero] = np.arccos(cos_vals)
    return angles
