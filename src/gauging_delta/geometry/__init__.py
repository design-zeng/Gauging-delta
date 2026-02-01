"""
Geometric utilities for Gauging-δ algorithm.

Provides angle calculations and spatial query operations.
"""

from gauging_delta.geometry.angles import compute_angle, compute_clockwise_angle
from gauging_delta.geometry.spatial import SpatialIndex


__all__ = [
    "SpatialIndex",
    "compute_angle",
    "compute_clockwise_angle",
]
