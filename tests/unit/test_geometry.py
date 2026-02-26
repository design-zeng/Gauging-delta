"""
Tests for geometry module (angles and spatial queries).

These tests are written FIRST per TDD approach.
Implementation in src/gauging_delta/geometry/ must pass these tests.
"""

import math

import numpy as np
from hypothesis import assume, given, settings

from tests.strategies import three_points_2d


class TestComputeAngle:
    """Tests for compute_angle function."""

    def test_zero_angle_same_direction(self):
        """Vectors in same direction should have angle 0."""
        from gauging_delta.geometry.angles import compute_angle

        origin = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([2.0, 0.0])  # Same direction, just farther

        angle = compute_angle(origin, p1, p2)
        assert np.isclose(angle, 0.0, atol=1e-10)

    def test_right_angle(self):
        """Perpendicular vectors should have angle π/2."""
        from gauging_delta.geometry.angles import compute_angle

        origin = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([0.0, 1.0])

        angle = compute_angle(origin, p1, p2)
        assert np.isclose(angle, math.pi / 2, atol=1e-10)

    def test_opposite_direction_pi(self):
        """Opposite vectors should have angle π."""
        from gauging_delta.geometry.angles import compute_angle

        origin = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([-1.0, 0.0])

        angle = compute_angle(origin, p1, p2)
        assert np.isclose(angle, math.pi, atol=1e-10)

    def test_angle_range(self):
        """Angle should always be in [0, π]."""
        from gauging_delta.geometry.angles import compute_angle

        origin = np.array([0.0, 0.0])
        for theta in np.linspace(0, 2 * math.pi, 20):
            p1 = np.array([1.0, 0.0])
            p2 = np.array([math.cos(theta), math.sin(theta)])

            angle = compute_angle(origin, p1, p2)
            assert 0 <= angle <= math.pi + 1e-10

    @given(three_points_2d())
    @settings(max_examples=100)
    def test_angle_always_in_valid_range(self, points):
        """Property: angle must always be in [0, π]."""
        from gauging_delta.geometry.angles import compute_angle

        origin, p1, p2 = points

        # Skip degenerate cases
        assume(np.linalg.norm(p1 - origin) > 1e-10)
        assume(np.linalg.norm(p2 - origin) > 1e-10)

        angle = compute_angle(origin, p1, p2)
        assert 0 <= angle <= math.pi + 1e-10

    @given(three_points_2d())
    @settings(max_examples=100)
    def test_angle_symmetric(self, points):
        """Property: angle(origin, p1, p2) == angle(origin, p2, p1)."""
        from gauging_delta.geometry.angles import compute_angle

        origin, p1, p2 = points

        assume(np.linalg.norm(p1 - origin) > 1e-10)
        assume(np.linalg.norm(p2 - origin) > 1e-10)

        angle1 = compute_angle(origin, p1, p2)
        angle2 = compute_angle(origin, p2, p1)

        assert np.isclose(angle1, angle2, atol=1e-10)

    def test_3d_right_angle(self):
        """compute_angle should work in 3D."""
        from gauging_delta.geometry.angles import compute_angle

        origin = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])

        angle = compute_angle(origin, p1, p2)
        assert np.isclose(angle, math.pi / 2, atol=1e-10)


class TestComputeClockwiseAngle:
    """Tests for compute_clockwise_angle function."""

    def test_quarter_turn_clockwise(self):
        """90° clockwise from +x to +y should be 3π/2 (or 270°)."""
        from gauging_delta.geometry.angles import compute_clockwise_angle

        origin = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])  # +x
        p2 = np.array([0.0, 1.0])  # +y

        angle = compute_clockwise_angle(origin, p1, p2)
        # Clockwise from +x to +y is 270° = 3π/2
        # Or counter-clockwise is 90° = π/2
        # Depends on convention - verify against original code
        assert 0 <= angle < 2 * math.pi

    def test_full_circle(self):
        """Full rotation should give angle close to 0 (or 2π)."""
        from gauging_delta.geometry.angles import compute_clockwise_angle

        origin = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([1.0, 0.0])  # Same point

        angle = compute_clockwise_angle(origin, p1, p2)
        assert np.isclose(angle, 0.0, atol=1e-10) or np.isclose(angle, 2 * math.pi, atol=1e-10)

    def test_range_always_0_to_2pi(self):
        """Clockwise angle should always be in [0, 2π)."""
        from gauging_delta.geometry.angles import compute_clockwise_angle

        origin = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])

        for theta in np.linspace(0, 2 * math.pi, 20):
            p2 = np.array([math.cos(theta), math.sin(theta)])
            angle = compute_clockwise_angle(origin, p1, p2)
            assert 0 <= angle < 2 * math.pi + 1e-10


class TestSpatialIndex:
    """Tests for SpatialIndex class."""

    def test_query_radius_finds_nearby(self):
        """Points within radius should be found."""
        from gauging_delta.geometry.spatial import SpatialIndex

        X = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [5.0, 5.0],
            ]
        )

        idx = SpatialIndex(X)
        nearby = idx.query_radius(np.array([0.0, 0.0]), radius=1.5)

        assert 0 in nearby
        assert 1 in nearby
        assert 2 in nearby
        assert 3 not in nearby

    def test_query_radius_empty(self):
        """No points within tiny radius should return empty."""
        from gauging_delta.geometry.spatial import SpatialIndex

        X = np.array(
            [
                [0.0, 0.0],
                [10.0, 10.0],
            ]
        )

        idx = SpatialIndex(X)
        nearby = idx.query_radius(np.array([5.0, 5.0]), radius=0.1)

        assert len(nearby) == 0

    def test_query_k_nearest(self):
        """k-nearest should return correct number of neighbors."""
        from gauging_delta.geometry.spatial import SpatialIndex

        X = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ]
        )

        idx = SpatialIndex(X)
        _distances, indices = idx.query_k_nearest(np.array([0.0, 0.0]), k=2)

        assert len(indices) == 2
        assert 0 in indices  # Self
        assert 1 in indices  # Nearest neighbor
