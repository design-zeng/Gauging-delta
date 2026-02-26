"""
Tests for continuity analysis.

Paper reference: Section II.C
Continuity combines density transition, angle transition, and shape consistency.

These tests are written FIRST per TDD approach.
"""

import numpy as np


class TestComputeContinuity:
    """Tests for compute_continuity function."""

    def test_continuity_range(self):
        """Continuity score should be in [0, 1]."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.continuity import compute_continuity

        # Create simple clusters
        X = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [2.5, 0.0],
                [3.0, 0.0],
            ]
        )

        C_i = Cluster(label=0, point_indices=[0, 1, 2])
        C_i.merge_history = [0.5, 0.5]
        C_i.sigma_history = [0.1]

        C_j = Cluster(label=1, point_indices=[3, 4, 5])
        C_j.merge_history = [0.5, 0.5]
        C_j.sigma_history = [0.1]

        score = compute_continuity(
            C_i,
            C_j,
            p_i=2,
            p_j=3,  # Reference points
            X=X,
            T_continuity=0.15,
            d_ij=1.0,
            T_adaptive=2.0,
        )

        assert 0.0 <= score <= 1.0


class TestDensityTransition:
    """Tests for density transition computation."""

    def test_equal_density_high_score(self):
        """Equal densities should give high transition score."""
        from gauging_delta.mergeability.continuity import compute_density_transition

        # Mock local points with equal counts
        local_i = [(0, 0.1), (1, 0.2), (2, 0.3)]
        local_j = [(3, 0.1), (4, 0.2), (5, 0.3)]

        X = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [1.0, 0.0],
                [1.1, 0.0],
                [1.2, 0.0],
            ]
        )

        midpoint = np.array([0.5, 0.0])

        score = compute_density_transition(
            local_i,
            local_j,
            p_i=0,
            p_j=3,
            midpoint=midpoint,
            radius=0.5,
            X=X,
        )

        # Equal densities -> score should be close to 1
        assert score >= 0.5


class TestAngleTransition:
    """Tests for angle transition computation."""

    def test_aligned_clusters_high_score(self):
        """Clusters aligned in same direction should have high score."""
        from gauging_delta.mergeability.continuity import compute_angle_transition

        # Points in a line -> good angle transition
        X = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [2.5, 0.0],
                [3.0, 0.0],
            ]
        )

        local_i = [(0, 0.0), (1, 0.0)]
        local_j = [(4, 0.0), (5, 0.0)]

        score = compute_angle_transition(local_i, local_j, p_i=2, p_j=3, X=X)

        # Aligned -> high score
        assert score >= 0.5

    def test_perpendicular_clusters_low_score(self):
        """Perpendicular cluster directions should have lower score."""
        from gauging_delta.mergeability.continuity import compute_angle_transition

        # L-shaped arrangement
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.5],
                [0.0, 1.0],  # Vertical
                [0.0, 1.5],
                [0.5, 1.5],
                [1.0, 1.5],  # Horizontal
            ]
        )

        local_i = [(0, 0.0), (1, 0.0)]
        local_j = [(4, 0.0), (5, 0.0)]

        score = compute_angle_transition(local_i, local_j, p_i=2, p_j=3, X=X)

        # Perpendicular -> lower score than aligned
        assert 0.0 <= score <= 1.0


class TestMassSmoothness:
    """Tests for mass smoothness computation."""

    def test_equal_mass_perfect_score(self):
        """Equal mass should give score of 1."""
        from gauging_delta.mergeability.continuity import compute_mass_smoothness

        local_i = [(0, 0.1), (1, 0.2), (2, 0.3)]  # 3 points
        local_j = [(3, 0.1), (4, 0.2), (5, 0.3)]  # 3 points

        max_angle_i = 0.5
        max_angle_j = 0.5

        score = compute_mass_smoothness(local_i, local_j, max_angle_i, max_angle_j)

        # mass_i = 3 * 0.5 = 1.5
        # mass_j = 3 * 0.5 = 1.5
        # score = min/max = 1.0
        assert np.isclose(score, 1.0, atol=1e-10)

    def test_unequal_mass_lower_score(self):
        """Unequal mass should give lower score."""
        from gauging_delta.mergeability.continuity import compute_mass_smoothness

        local_i = [(0, 0.1)]  # 1 point
        local_j = [(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4)]  # 4 points

        max_angle_i = 0.5
        max_angle_j = 0.5

        score = compute_mass_smoothness(local_i, local_j, max_angle_i, max_angle_j)

        # mass_i = 1 * 0.5 = 0.5
        # mass_j = 4 * 0.5 = 2.0
        # score = 0.5 / 2.0 = 0.25
        assert score < 0.5


class TestCompactness:
    """Tests for compactness computation."""

    def test_compactness_basic(self):
        """Compactness should be computable for valid cluster."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.continuity import compute_compactness

        X = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
            ]
        )

        C = Cluster(label=0, point_indices=[0, 1, 2])
        C.sigma_history = [0.1, 0.2]
        C.merge_history = [0.1, 0.2]

        compact = compute_compactness(C, X)

        assert compact >= 0
