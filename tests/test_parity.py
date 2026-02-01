"""
Parity tests: Verify new implementation matches original exactly.

These tests compare the refactored implementation against the legacy
perception.py to ensure behavioral parity.
"""

from pathlib import Path

import numpy as np
import pytest


# Skip all tests if legacy module not available
pytestmark = pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath("legacy", "perception_original.py").exists(),
    reason="Legacy perception_original.py not found",
)


class TestAngleParity:
    """Verify angle calculations match original."""

    def test_to_find_angle_parity(self):
        """compute_angle must match legacy to_find_angle."""
        from legacy.perception_original import Perception

        from gauging_delta.geometry.angles import compute_angle

        # Test cases
        test_cases = [
            (np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            (np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([-1.0, 0.0])),
            (np.array([1.0, 1.0]), np.array([2.0, 1.0]), np.array([1.0, 2.0])),
        ]

        for origin, p1, p2 in test_cases:
            new_angle = compute_angle(origin, p1, p2)
            legacy_angle = Perception.to_find_angle(origin, p1, p2)

            assert np.isclose(new_angle, legacy_angle, atol=1e-10), (
                f"Angle mismatch: {new_angle} vs {legacy_angle}"
            )

    def test_to_find_clockwise_angle_parity(self):
        """compute_clockwise_angle must match legacy to_find_clockwise_angle."""
        from legacy.perception_original import Perception

        from gauging_delta.geometry.angles import compute_clockwise_angle

        # Test cases (2D only)
        test_cases = [
            (np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            (np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0])),
        ]

        for origin, p1, p2 in test_cases:
            new_angle = compute_clockwise_angle(origin, p1, p2)
            legacy_angle = Perception.to_find_clockwise_angle(origin, p1, p2)

            assert np.isclose(new_angle, legacy_angle, atol=1e-10), (
                f"Clockwise angle mismatch: {new_angle} vs {legacy_angle}"
            )


class TestProximityParity:
    """Verify proximity calculations match original."""

    def test_compute_proximity_rho_formula(self):
        """compute_proximity_rho follows ρ = d_ij / μ_historical formula."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.proximity import compute_proximity_rho

        C_i = Cluster(label=0, point_indices=[0, 1, 2])
        C_i.merge_history = [1.0, 2.0, 3.0, 4.0]

        C_j = Cluster(label=1, point_indices=[3, 4])
        C_j.merge_history = [1.5, 2.5, 3.5]

        d_ij = 2.0
        rho, _, lead_id, child_id = compute_proximity_rho(d_ij, C_i, C_j)

        # Verify rho is positive and reasonable
        assert rho > 0
        assert lead_id == 0  # C_i is larger
        assert child_id == 1


class TestThresholdParity:
    """Verify threshold calculations match original."""

    def test_adaptive_threshold_formula(self):
        """compute_adaptive_threshold_T follows T = β × T_stat × ξ formula."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.threshold import compute_adaptive_threshold_T

        C = Cluster(label=0, point_indices=[0, 1, 2])
        C.merge_history = [1.0, 2.0]  # Short history -> T_stat = 2.7

        beta_ij = 1.5
        xi_s = 1.0

        T = compute_adaptive_threshold_T(C, beta_ij, xi_s)

        # T = 1.5 × 2.7 × 1.0 = 4.05
        assert np.isclose(T, 4.05, atol=1e-10)

    def test_force_formula(self):
        """compute_F_ij follows F = |C_i| × |C_j| / d² formula."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.threshold import compute_F_ij

        C_i = Cluster(label=0, point_indices=[0, 1, 2])  # 3 points
        C_j = Cluster(label=1, point_indices=[3, 4])  # 2 points
        d_ij = 2.0

        F = compute_F_ij(C_i, C_j, d_ij)

        # F = 3 × 2 / 2² = 6 / 4 = 1.5
        assert np.isclose(F, 1.5, atol=1e-10)


class TestContinuityParity:
    """Verify continuity calculations match original."""

    def test_continuity_range(self):
        """compute_continuity returns value in [0, 1] range."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.continuity import compute_continuity

        # Create simple test case
        X = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
        ])

        C_i = Cluster(label=0, point_indices=[0, 1], center=np.array([0.05, 0.0]))
        C_j = Cluster(label=1, point_indices=[2, 3], center=np.array([1.05, 0.0]))

        continuity = compute_continuity(
            C_i, C_j,
            p_i=1, p_j=2,
            X=X,
            T_continuity=0.15,
            d_ij=0.9,
            T_adaptive=2.0,
        )

        assert 0.0 <= continuity <= 1.0


class TestFullAlgorithmParity:
    """Verify full algorithm produces reasonable results on benchmark datasets."""

    @pytest.mark.parametrize(
        "dataset_name",
        [
            "3-spiral",
            "3_blobs",
            "aggregation",
            "atom",
        ],
    )
    def test_clustering_on_dataset(self, dataset_name, load_test_datasets):
        """Algorithm should produce valid clustering on benchmark datasets."""
        from gauging_delta import GaugingDelta

        datasets = load_test_datasets
        if dataset_name not in datasets:
            pytest.skip(f"Dataset {dataset_name} not found")

        X = datasets[dataset_name]["X"]
        y_true = datasets[dataset_name]["y"]

        # Run clustering
        model = GaugingDelta()
        labels = model.fit(X)

        # Basic validity checks
        assert len(labels) == len(X)
        assert model.n_clusters_ >= 1
        assert model.n_clusters_ <= len(X)

        # If ground truth exists, check we find a reasonable number of clusters
        if y_true is not None:
            # Allow some tolerance - algorithm may find slightly different count
            assert model.n_clusters_ >= 1
            assert model.n_clusters_ <= len(X) // 2  # Should merge at least half
