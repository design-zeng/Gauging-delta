"""
Tests for proximity statistic ρ.

Paper reference: Equation 3
    ρ = d_ij / μ_historical

These tests are written FIRST per TDD approach.
"""

import numpy as np
from hypothesis import assume, given, settings

from tests.strategies import merge_history, positive_floats


class TestComputeProximityRho:
    """Tests for compute_proximity_rho function."""

    def test_rho_basic_calculation(self):
        """Basic ρ calculation: d / mean(history)."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.proximity import compute_proximity_rho

        C_i = Cluster(label=0, point_indices=[0, 1, 2])
        C_i.merge_history = [1.0, 2.0, 3.0]  # mean = 2.0

        C_j = Cluster(label=1, point_indices=[3, 4])
        C_j.merge_history = [2.0, 2.0, 2.0]  # mean = 2.0

        d_ij = 2.0
        rho, dist, _lead, _child = compute_proximity_rho(d_ij, C_i, C_j)

        # ρ = 2.0 / 2.0 = 1.0
        assert np.isclose(rho, 1.0, atol=1e-10)
        assert dist == d_ij

    def test_rho_fallback_no_history(self):
        """When no history, use fallback value."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.proximity import compute_proximity_rho

        C_i = Cluster(label=0, point_indices=[0])
        C_j = Cluster(label=1, point_indices=[1])

        d_ij = 1.0
        fallback = 0.5
        rho, _dist, _lead, _child = compute_proximity_rho(
            d_ij, C_i, C_j, min_dist_fallback=fallback
        )

        # ρ = 1.0 / 0.5 = 2.0
        assert np.isclose(rho, 2.0, atol=1e-10)

    def test_lead_cluster_is_larger(self):
        """Lead cluster should be the one with more points."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.proximity import compute_proximity_rho

        C_i = Cluster(label=0, point_indices=[0, 1, 2, 3, 4])  # 5 points
        C_j = Cluster(label=1, point_indices=[5, 6])  # 2 points

        _rho, _dist, lead, child = compute_proximity_rho(1.0, C_i, C_j)

        assert lead == 0  # C_i has more points
        assert child == 1

    def test_lead_cluster_when_equal(self):
        """When equal size, C_i should be lead."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.proximity import compute_proximity_rho

        C_i = Cluster(label=0, point_indices=[0, 1])
        C_j = Cluster(label=1, point_indices=[2, 3])

        _rho, _dist, lead, child = compute_proximity_rho(1.0, C_i, C_j)

        assert lead == 0  # C_i by convention
        assert child == 1

    @given(
        d_ij=positive_floats,
        history_i=merge_history(min_length=3, max_length=10),
        history_j=merge_history(min_length=3, max_length=10),
    )
    @settings(max_examples=100)
    def test_rho_always_positive(self, d_ij, history_i, history_j):
        """Property: ρ must always be positive."""
        from gauging_delta.core.cluster import Cluster
        from gauging_delta.mergeability.proximity import compute_proximity_rho

        assume(len(history_i) >= 3 and len(history_j) >= 3)
        assume(all(h > 0 for h in history_i))
        assume(all(h > 0 for h in history_j))

        C_i = Cluster(label=0, point_indices=list(range(5)))
        C_i.merge_history = history_i

        C_j = Cluster(label=1, point_indices=list(range(5, 10)))
        C_j.merge_history = history_j

        rho, _dist, _lead, _child = compute_proximity_rho(d_ij, C_i, C_j)

        assert rho > 0


class TestHistoryLength:
    """Tests for _compute_history_length helper."""

    def test_asymmetric_weighting(self):
        """History length uses max(n_larger/2, n_smaller)."""
        from gauging_delta.mergeability.proximity import _compute_history_length

        # n_i=10, n_j=3 -> max(10/2, 3) = max(5, 3) = 5
        assert _compute_history_length(10, 3) == 5

        # n_i=4, n_j=10 -> max(10/2, 4) = max(5, 4) = 5
        assert _compute_history_length(4, 10) == 5

        # n_i=6, n_j=6 -> max(6/2, 6) = max(3, 6) = 6
        assert _compute_history_length(6, 6) == 6
