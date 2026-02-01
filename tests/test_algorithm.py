"""
Tests for the main GaugingDelta algorithm.

These tests verify Algorithm 1 from the paper.
"""

import numpy as np
from hypothesis import given, settings

from tests.strategies import points_nd


class TestGaugingDeltaInit:
    """Tests for GaugingDelta initialization."""

    def test_default_init(self):
        """Default initialization should work."""
        from gauging_delta import GaugingDelta

        model = GaugingDelta()

        assert model.k is None
        assert model.T_continuity == 0.15
        assert model.K_neighbors == 5
        assert model.labels_ is None
        assert model.n_clusters_ == 0

    def test_custom_init(self):
        """Custom parameters should be stored."""
        from gauging_delta import GaugingDelta

        model = GaugingDelta(k=5, threshold_continuity=0.2, n_neighbors=10)

        assert model.k == 5
        assert model.T_continuity == 0.2
        assert model.K_neighbors == 10


class TestGaugingDeltaFit:
    """Tests for GaugingDelta.fit() method."""

    def test_fit_returns_labels(self, sample_2d_data):
        """fit() should return array of labels."""
        from gauging_delta import GaugingDelta

        model = GaugingDelta()
        labels = model.fit(sample_2d_data)

        assert labels is not None
        assert len(labels) == len(sample_2d_data)

    def test_labels_shape_matches_input(self, sample_2d_data):
        """Labels array should have same length as input."""
        from gauging_delta import GaugingDelta

        model = GaugingDelta()
        labels = model.fit(sample_2d_data)

        assert len(labels) == len(sample_2d_data)

    def test_single_point_own_cluster(self):
        """Each point starts as its own cluster."""
        from gauging_delta import GaugingDelta

        X = np.array([[0.0, 0.0]])
        model = GaugingDelta()
        labels = model.fit(X)

        assert len(labels) == 1

    def test_identical_points_same_cluster(self):
        """Identical points should end up in same cluster."""
        from gauging_delta import GaugingDelta

        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        model = GaugingDelta()
        labels = model.fit(X)

        assert len(set(labels)) == 1  # All same label

    def test_well_separated_clusters(self):
        """Well-separated clusters should be identified."""
        from gauging_delta import GaugingDelta

        # Two clearly separated clusters
        cluster1 = np.random.randn(20, 2) + np.array([0, 0])
        cluster2 = np.random.randn(20, 2) + np.array([100, 100])
        X = np.vstack([cluster1, cluster2])

        model = GaugingDelta()
        labels = model.fit(X)

        # Should find 2 clusters
        assert len(set(labels)) == 2

        # First 20 should have same label
        assert len(set(labels[:20])) == 1
        # Last 20 should have same label
        assert len(set(labels[20:])) == 1
        # But different from each other
        assert labels[0] != labels[20]

    @given(points_nd(min_dim=2, max_dim=5, min_points=5, max_points=20))
    @settings(max_examples=10)
    def test_fit_handles_nd_data(self, X):
        """fit() should work with N-dimensional data."""
        from gauging_delta import GaugingDelta

        model = GaugingDelta()
        labels = model.fit(X)

        assert len(labels) == len(X)
        assert all(isinstance(l, (int, np.integer)) for l in labels)


class TestGaugingDeltaWithK:
    """Tests for GaugingDelta with target cluster count k."""

    def test_k_limits_clusters(self):
        """Setting k should limit number of clusters."""
        from gauging_delta import GaugingDelta

        X = np.random.randn(50, 2)

        model = GaugingDelta(k=3)
        labels = model.fit(X)

        # Should have at most k clusters
        assert len(set(labels)) <= 3


class TestNeighborGraph:
    """Tests for sparse neighbor graph operations."""

    def test_neighbor_graph_creation(self, sample_2d_data):
        """NeighborGraph should be created with KDTree."""
        from gauging_delta.core.neighbor_graph import NeighborGraph

        graph = NeighborGraph(X=sample_2d_data, n_neighbors=10)

        assert graph.kdtree is not None
        assert graph.X is sample_2d_data
        assert graph.n_neighbors == 10

    def test_neighbor_graph_initialize(self, sample_2d_data):
        """Initialize should build heap with k-nearest edges."""
        from gauging_delta.core.neighbor_graph import NeighborGraph

        graph = NeighborGraph(X=sample_2d_data, n_neighbors=5)
        graph.initialize()

        # Each point should have neighbors
        assert len(graph.neighbors) == len(sample_2d_data)
        # Heap should have edges
        assert len(graph.merge_heap) > 0

    def test_pop_closest_pair_returns_minimum(self, sample_2d_data):
        """pop_closest_pair should return smallest distance edge."""
        from gauging_delta.core.neighbor_graph import NeighborGraph

        graph = NeighborGraph(X=sample_2d_data, n_neighbors=5)
        graph.initialize()

        result = graph.pop_closest_pair()
        assert result is not None
        _c_i, _c_j, dist = result
        assert dist >= 0

    def test_lazy_deletion_skips_dead_clusters(self, sample_2d_data):
        """pop_closest_pair should skip edges with dead clusters."""
        from gauging_delta.core.neighbor_graph import NeighborGraph

        graph = NeighborGraph(X=sample_2d_data, n_neighbors=5)
        graph.initialize()

        # Mark some clusters as dead
        graph.dead_clusters.add(0)
        graph.dead_clusters.add(1)

        # Should skip edges involving 0 or 1
        result = graph.pop_closest_pair()
        if result is not None:
            c_i, c_j, _ = result
            assert c_i not in graph.dead_clusters
            assert c_j not in graph.dead_clusters

    def test_memory_is_sparse(self):
        """NeighborGraph should use O(N×k) memory, not O(N²)."""
        import tracemalloc

        from gauging_delta.core.neighbor_graph import NeighborGraph

        # 10k points with k=50 neighbors
        np.random.seed(42)
        X = np.random.randn(10000, 2)

        tracemalloc.start()
        graph = NeighborGraph(X=X, n_neighbors=50)
        graph.initialize()
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Should be well under 400MB (dense would be ~800MB for 10k×10k float64)
        # Sparse should be ~10-50MB
        assert peak < 100 * 1024 * 1024  # 100 MB max
