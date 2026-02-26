"""
Tests for Cluster data structures.

These tests are written FIRST per TDD approach.
"""

from hypothesis import given, settings

from tests.strategies import cluster_data


class TestCluster:
    """Tests for Cluster dataclass."""

    def test_cluster_creation(self):
        """Basic cluster creation."""
        from gauging_delta.core.cluster import Cluster

        c = Cluster(label=0, point_indices=[0, 1, 2])

        assert c.label == 0
        assert c.point_indices == [0, 1, 2]
        assert len(c) == 3

    def test_cluster_defaults(self):
        """Default values should be sensible."""
        from gauging_delta.core.cluster import Cluster

        c = Cluster(label=0)

        assert c.point_indices == []
        assert c.center is None
        assert c.mu_dist == 0.0
        assert c.sigma_dist == 0.0
        assert c.merge_history == []
        assert c.sigma_history == []
        assert c.density_history == []
        assert c.merge_edges == []

    def test_cluster_len(self):
        """__len__ should return number of points."""
        from gauging_delta.core.cluster import Cluster

        c = Cluster(label=0, point_indices=[0, 1, 2, 3, 4])
        assert len(c) == 5

        c2 = Cluster(label=1, point_indices=[])
        assert len(c2) == 0

    @given(cluster_data())
    @settings(max_examples=50)
    def test_cluster_from_data(self, data):
        """Property: cluster should preserve all data."""
        from gauging_delta.core.cluster import Cluster

        c = Cluster(
            label=data["label"],
            point_indices=data["point_indices"],
            merge_history=data["merge_history"],
            sigma_history=data["sigma_history"],
        )

        assert c.label == data["label"]
        assert c.point_indices == data["point_indices"]
        assert c.merge_history == data["merge_history"]
        assert c.sigma_history == data["sigma_history"]


class TestClusterPairInfo:
    """Tests for ClusterPairInfo dataclass."""

    def test_pair_info_creation(self):
        """Basic pair info creation."""
        from gauging_delta.core.cluster import ClusterPairInfo

        info = ClusterPairInfo(d_near=1.5, p_i=0, p_j=10)

        assert info.d_near == 1.5
        assert info.p_i == 0
        assert info.p_j == 10
        assert info.d_center is None

    def test_pair_info_with_center(self):
        """Pair info with center distance."""
        from gauging_delta.core.cluster import ClusterPairInfo

        info = ClusterPairInfo(d_near=1.5, p_i=0, p_j=10, d_center=3.0)

        assert info.d_center == 3.0


class TestMergeabilityResult:
    """Tests for MergeabilityResult dataclass."""

    def test_result_creation(self):
        """Basic result creation."""
        from gauging_delta.core.cluster import MergeabilityResult

        result = MergeabilityResult(
            is_mergeable=True,
            rho=0.5,
            T_i=2.0,
            T_j=2.5,
            beta_ij=1.1,
            xi_s=0.9,
            continuity=0.8,
            lead_cluster=0,
            child_cluster=1,
        )

        assert result.is_mergeable is True
        assert result.rho == 0.5
        assert result.T_i == 2.0
        assert result.T_j == 2.5
        assert result.beta_ij == 1.1
        assert result.xi_s == 0.9
        assert result.continuity == 0.8
        assert result.lead_cluster == 0
        assert result.child_cluster == 1

    def test_result_not_mergeable(self):
        """Result when clusters should not merge."""
        from gauging_delta.core.cluster import MergeabilityResult

        result = MergeabilityResult(
            is_mergeable=False,
            rho=5.0,  # High proximity = far apart
            T_i=2.0,
            T_j=2.5,
            beta_ij=1.0,
            xi_s=1.0,
            continuity=0.0,
            lead_cluster=0,
            child_cluster=1,
        )

        assert result.is_mergeable is False
        assert result.rho > result.T_i  # ρ > T means not mergeable
