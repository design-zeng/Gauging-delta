"""
Tests for adaptive threshold T.

Paper reference: Equation 4
    T = β_ij × T_stat × ξ_s
    T_stat = 3 / (1 + e^(0.3 × μ/σ)) + 1.4

These tests are written FIRST per TDD approach.
"""

import numpy as np
from hypothesis import assume, given, settings

from tests.strategies import merge_history


class TestComputeTStat:
    """Tests for compute_T_stat function."""

    def test_T_stat_default_insufficient_history(self):
        """T_stat = 2.7 when history has ≤3 samples."""
        from gauging_delta.mergeability.threshold import compute_T_stat

        assert compute_T_stat([]) == 2.7
        assert compute_T_stat([1.0]) == 2.7
        assert compute_T_stat([1.0, 2.0]) == 2.7
        assert compute_T_stat([1.0, 2.0, 3.0]) == 2.7

    def test_T_stat_default_zero_std(self):
        """T_stat = 2.7 when σ = 0."""
        from gauging_delta.mergeability.threshold import compute_T_stat

        # All same values -> std = 0
        assert compute_T_stat([1.0, 1.0, 1.0, 1.0, 1.0]) == 2.7

    def test_T_stat_bounds(self):
        """T_stat should be in [1.4, 4.4]."""
        from gauging_delta.mergeability.threshold import compute_T_stat

        # Formula: 3 / (1 + e^(0.3 × μ/σ)) + 1.4
        # When μ/σ → ∞: 3/(1+∞) + 1.4 = 0 + 1.4 = 1.4
        # When μ/σ → 0: 3/(1+1) + 1.4 = 1.5 + 1.4 = 2.9
        # When μ/σ → -∞: 3/(1+0) + 1.4 = 3 + 1.4 = 4.4

        # Low variance relative to mean
        T_low = compute_T_stat([10.0, 10.1, 10.0, 10.1, 10.0])
        assert 1.4 <= T_low <= 4.4

        # High variance relative to mean
        T_high = compute_T_stat([1.0, 10.0, 1.0, 10.0, 1.0])
        assert 1.4 <= T_high <= 4.4

    @given(merge_history(min_length=5, max_length=20))
    @settings(max_examples=100)
    def test_T_stat_always_in_bounds(self, history):
        """Property: T_stat ∈ [1.4, 4.4] for all valid inputs."""
        from gauging_delta.mergeability.threshold import compute_T_stat

        assume(len(history) >= 4)
        assume(all(h > 0 for h in history))

        T = compute_T_stat(history)

        # Either default (2.7) or computed value
        assert 1.4 <= T <= 4.4


class TestComputeFij:
    """Tests for compute_F_ij (interaction force)."""

    def test_force_basic(self):
        """F_ij = |C_i| × |C_j| / d_ij²."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.threshold import compute_F_ij

        C_i = Cluster(label=0, point_indices=[0, 1, 2])  # 3 points
        C_j = Cluster(label=1, point_indices=[3, 4])  # 2 points
        d_ij = 2.0

        F = compute_F_ij(C_i, C_j, d_ij)

        # F = 3 × 2 / 2² = 6 / 4 = 1.5
        assert np.isclose(F, 1.5, atol=1e-10)

    def test_force_increases_with_size(self):
        """Larger clusters have stronger force."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.threshold import compute_F_ij

        C_small = Cluster(label=0, point_indices=[0])
        C_large = Cluster(label=1, point_indices=list(range(1, 11)))
        C_other = Cluster(label=2, point_indices=[11, 12])

        d = 1.0
        F_small = compute_F_ij(C_small, C_other, d)
        F_large = compute_F_ij(C_large, C_other, d)

        assert F_large > F_small

    def test_force_decreases_with_distance(self):
        """Force decreases with distance squared."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.threshold import compute_F_ij

        C_i = Cluster(label=0, point_indices=[0, 1])
        C_j = Cluster(label=1, point_indices=[2, 3])

        F_near = compute_F_ij(C_i, C_j, d_ij=1.0)
        F_far = compute_F_ij(C_i, C_j, d_ij=2.0)

        # F ∝ 1/d² -> doubling distance quarters force
        assert np.isclose(F_far, F_near / 4, atol=1e-10)


class TestComputeAdaptiveThresholdT:
    """Tests for compute_adaptive_threshold_T."""

    def test_threshold_composition(self):
        """T = β_ij × T_stat × ξ_s."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.threshold import compute_adaptive_threshold_T

        C = Cluster(label=0, point_indices=[0, 1, 2])
        C.merge_history = [1.0, 2.0]  # ≤3 samples -> T_stat = 2.7

        beta_ij = 1.5
        xi_s = 1.0

        T = compute_adaptive_threshold_T(C, beta_ij, xi_s)

        # T = 1.5 × 2.7 × 1.0 = 4.05
        assert np.isclose(T, 4.05, atol=1e-10)


class TestComputeXiS:
    """Tests for compute_xi_s (shape similarity)."""

    def test_xi_s_default_insufficient_samples(self):
        """ξ_s = 1.0 when fewer than 5 samples."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.threshold import compute_xi_s

        C_i = Cluster(label=0, point_indices=[0, 1])
        C_i.sigma_history = [0.1, 0.2]

        C_j = Cluster(label=1, point_indices=[2, 3])
        C_j.sigma_history = [0.1, 0.2]

        xi = compute_xi_s(C_i, C_j)
        assert xi == 1.0

    def test_xi_s_range(self):
        """ξ_s should be in [0.5, ~1.5] range."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.threshold import compute_xi_s

        C_i = Cluster(label=0, point_indices=list(range(10)))
        C_i.sigma_history = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        C_j = Cluster(label=1, point_indices=list(range(10, 20)))
        C_j.sigma_history = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        xi = compute_xi_s(C_i, C_j)

        # ξ_s = r / (1 + r) + 0.5
        # When r → 0: 0 + 0.5 = 0.5
        # When r → 1: 0.5 + 0.5 = 1.0
        # When r → ∞: 1 + 0.5 = 1.5
        assert 0.5 <= xi <= 1.5
