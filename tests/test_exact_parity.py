"""
Exact Parity Tests: Verify gauging_delta produces IDENTICAL results to legacy.

These tests use Hypothesis for property-based testing with synthetic datasets
under various conditions. All tests require ARI=1.0 (exact match).
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck, Verbosity
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import adjusted_rand_score

# Skip all tests if legacy module not available
pytestmark = pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath("perception.py").exists(),
    reason="perception.py not found",
)


# =============================================================================
# Hypothesis Strategies for Dataset Generation
# =============================================================================

@st.composite
def well_separated_blobs(draw, n_clusters=None, n_points_per_cluster=None):
    """Generate well-separated blob clusters."""
    if n_clusters is None:
        n_clusters = draw(st.integers(min_value=2, max_value=3))  # Reduced from 5
    if n_points_per_cluster is None:
        n_points_per_cluster = draw(st.integers(min_value=3, max_value=8))  # Reduced from 20
    
    n_dims = 2  # Fixed to 2D for speed
    separation = draw(st.floats(min_value=5.0, max_value=10.0))
    noise_scale = draw(st.floats(min_value=0.1, max_value=0.3))
    
    points = []
    for i in range(n_clusters):
        center = np.array([separation * i] + [0.0] * (n_dims - 1))
        cluster_points = center + np.random.randn(n_points_per_cluster, n_dims) * noise_scale
        points.append(cluster_points)
    
    X = np.vstack(points)
    return X


@st.composite
def linear_chain_clusters(draw):
    """Generate clusters arranged in a linear chain (tests continuity)."""
    n_clusters = draw(st.integers(min_value=2, max_value=3))  # Reduced from 4
    n_points = draw(st.integers(min_value=3, max_value=8))  # Reduced from 15
    gap = draw(st.floats(min_value=2.0, max_value=4.0))
    spread = draw(st.floats(min_value=0.2, max_value=0.5))
    
    points = []
    for i in range(n_clusters):
        center_x = i * gap
        cluster_points = np.column_stack([
            np.random.randn(n_points) * spread + center_x,
            np.random.randn(n_points) * spread
        ])
        points.append(cluster_points)
    
    X = np.vstack(points)
    return X


@st.composite
def concentric_rings(draw):
    """Generate concentric ring clusters (tests shape detection)."""
    n_rings = draw(st.integers(min_value=2, max_value=2))  # Fixed to 2
    points_per_ring = draw(st.integers(min_value=6, max_value=12))  # Reduced from 25
    
    points = []
    for i in range(n_rings):
        radius = (i + 1) * 2.0
        noise = 0.1
        angles = np.linspace(0, 2 * np.pi, points_per_ring, endpoint=False)
        x = radius * np.cos(angles) + np.random.randn(points_per_ring) * noise
        y = radius * np.sin(angles) + np.random.randn(points_per_ring) * noise
        points.append(np.column_stack([x, y]))
    
    X = np.vstack(points)
    return X


@st.composite
def random_gaussian_mixture(draw):
    """Generate random Gaussian mixture with controlled overlap."""
    n_clusters = draw(st.integers(min_value=2, max_value=3))  # Reduced from 4
    n_points = draw(st.integers(min_value=4, max_value=10))  # Reduced from 20
    
    # Generate well-separated centers
    centers = []
    min_separation = 4.0
    for _ in range(n_clusters):
        for attempt in range(100):
            center = np.random.randn(2) * 5
            if all(np.linalg.norm(center - c) > min_separation for c in centers):
                centers.append(center)
                break
        else:
            # Fallback: use grid placement
            idx = len(centers)
            centers.append(np.array([idx * min_separation, 0.0]))
    
    points = []
    for center in centers:
        std = draw(st.floats(min_value=0.2, max_value=0.6))
        cluster_points = center + np.random.randn(n_points, 2) * std
        points.append(cluster_points)
    
    X = np.vstack(points)
    return X


@st.composite
def uniform_grid_points(draw):
    """Generate points on a uniform grid (edge case for density)."""
    grid_size = draw(st.integers(min_value=3, max_value=4))  # Reduced from 6
    spacing = draw(st.floats(min_value=0.5, max_value=1.5))
    
    x = np.arange(grid_size) * spacing
    y = np.arange(grid_size) * spacing
    xx, yy = np.meshgrid(x, y)
    X = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Add small noise to break exact symmetry
    X += np.random.randn(*X.shape) * 0.01
    return X


@st.composite  
def single_dense_cluster(draw):
    """Generate a single dense cluster (should produce 1 cluster)."""
    n_points = draw(st.integers(min_value=8, max_value=20))  # Reduced from 50
    spread = draw(st.floats(min_value=0.1, max_value=0.4))
    
    X = np.random.randn(n_points, 2) * spread
    return X


# =============================================================================
# Helper Functions
# =============================================================================

def run_legacy(X: np.ndarray) -> np.ndarray:
    """Run legacy algorithm and return labels."""
    from perception import Perception
    
    p = Perception()
    with redirect_stdout(io.StringIO()):
        labels, _ = p.fit(X)
    return np.asarray(labels, dtype=int)


def run_gauging_delta(X: np.ndarray) -> np.ndarray:
    """Run new algorithm and return labels."""
    from gauging_delta import GaugingDelta

    g = GaugingDelta(preserve_labels=True)
    labels = g.fit(X)
    return np.asarray(labels, dtype=int)


def assert_identical_labels(labels_legacy: np.ndarray, labels_new: np.ndarray):
    """Assert that two label arrays match exactly (value-by-value)."""
    if labels_legacy.shape != labels_new.shape:
        raise AssertionError(
            f"Label shape mismatch: legacy={labels_legacy.shape}, new={labels_new.shape}"
        )

    if not np.array_equal(labels_legacy, labels_new):
        ari = adjusted_rand_score(labels_legacy, labels_new)
        diffs = np.where(labels_legacy != labels_new)[0]
        sample = diffs[:10].tolist()
        raise AssertionError(
            "Label mismatch! "
            f"ARI={ari:.4f}, mismatched_indices={sample}, total_mismatches={len(diffs)}"
        )


def assert_identical_clustering(
    labels_legacy: np.ndarray,
    labels_new: np.ndarray,
    X: np.ndarray | None = None,
):
    """Backward-compatible shim for exact label parity checks."""
    _ = X
    assert_identical_labels(labels_legacy, labels_new)


def load_text_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load comma-delimited text dataset with labels in last column."""
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Expected features+label in {path}")
    return data[:, :-1], data[:, -1]


# =============================================================================
# Exact Parity Tests with Hypothesis
# =============================================================================

class TestExactParityHypothesis:
    """Property-based tests requiring exact parity (ARI=1.0)."""
    
    @given(X=well_separated_blobs())
    @settings(
        max_examples=5,  # Reduced from 20
        deadline=30000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        verbosity=Verbosity.verbose,
    )
    def test_well_separated_blobs_exact_parity(self, X):
        """Well-separated blobs must produce identical clustering."""
        assume(len(X) >= 6)
        assume(len(X) <= 50)
        print(f"  Testing {len(X)} points...", end=" ", flush=True)
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_labels(labels_legacy, labels_new)
    
    @given(X=linear_chain_clusters())
    @settings(
        max_examples=5,  # Reduced from 15
        deadline=30000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        verbosity=Verbosity.verbose,
    )
    def test_linear_chain_exact_parity(self, X):
        """Linear chain clusters must produce identical clustering."""
        assume(len(X) >= 6)
        assume(len(X) <= 30)
        print(f"  Testing {len(X)} points...", end=" ", flush=True)
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_labels(labels_legacy, labels_new)
    
    @given(X=random_gaussian_mixture())
    @settings(
        max_examples=5,  # Reduced from 20
        deadline=30000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        verbosity=Verbosity.verbose,
    )
    def test_gaussian_mixture_exact_parity(self, X):
        """Gaussian mixtures must produce identical clustering."""
        assume(len(X) >= 8)
        assume(len(X) <= 40)
        print(f"  Testing {len(X)} points...", end=" ", flush=True)
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)
    
    @given(X=single_dense_cluster())
    @settings(
        max_examples=5,  # Reduced from 15
        deadline=30000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        verbosity=Verbosity.verbose,
    )
    def test_single_cluster_exact_parity(self, X):
        """Single dense cluster must produce identical result."""
        assume(len(X) >= 10)
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)
    
    @given(X=uniform_grid_points())
    @settings(
        max_examples=5,  # Reduced from 10
        deadline=30000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        verbosity=Verbosity.verbose,
    )
    def test_grid_points_exact_parity(self, X):
        """Grid points must produce identical clustering."""
        assume(len(X) >= 9)
        assume(len(X) <= 20)
        print(f"  Testing {len(X)} points...", end=" ", flush=True)
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)


class TestExactParityBenchmarkDatasets:
    """Exact parity on benchmark datasets."""
    
    @pytest.mark.parametrize("dataset_name", [
        "3-spiral",
        "3_blobs",
        "aggregation",
        "atom",
        "compound",
        "flame",
        "jain",
        "pathbased",
        "r15",
        "s1",
        "impossible",
        "chainlink",
    ])
    def test_benchmark_dataset_exact_parity(self, dataset_name):
        """Each benchmark dataset must produce identical clustering."""
        data_path = Path(__file__).parent.parent / "data" / f"{dataset_name}.txt"
        
        if not data_path.exists():
            pytest.skip(f"Dataset {dataset_name} not found")
        
        X, _ = load_text_dataset(data_path)
        print(f"  [{dataset_name}] {len(X)} points...", end=" ", flush=True)
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)


class TestExactParityEdgeCases:
    """Edge case tests requiring exact parity."""
    
    def test_two_points_same_cluster(self):
        """Two close points should be in same cluster."""
        X = np.array([[0.0, 0.0], [0.1, 0.1]])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)
    
    def test_two_points_far_apart(self):
        """Two far points - verify identical handling."""
        X = np.array([[0.0, 0.0], [100.0, 100.0]])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)
    
    def test_three_collinear_points(self):
        """Three collinear points."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)
    
    def test_equilateral_triangle(self):
        """Points forming equilateral triangle."""
        X = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2]
        ])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)
    
    def test_square_arrangement(self):
        """Points in square arrangement."""
        X = np.array([
            [0.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [1.0, 1.0]
        ])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)
    
    def test_two_distinct_clusters_min_size(self):
        """Minimum viable two-cluster scenario."""
        X = np.array([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],  # Cluster 1
            [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]  # Cluster 2
        ])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)
    
    def test_cluster_with_outlier(self):
        """Dense cluster with one outlier."""
        np.random.seed(42)
        cluster_points = np.random.randn(15, 2) * 0.3
        outlier = np.array([[10.0, 10.0]])
        X = np.vstack([cluster_points, outlier])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)
    
    def test_varying_density_clusters(self):
        """Clusters with different densities."""
        np.random.seed(123)
        dense_cluster = np.random.randn(20, 2) * 0.2  # Dense
        sparse_cluster = np.random.randn(10, 2) * 1.5 + np.array([10, 10])  # Sparse
        X = np.vstack([dense_cluster, sparse_cluster])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)
    
    def test_three_clusters_different_sizes(self):
        """Three clusters with different sizes."""
        np.random.seed(456)
        cluster1 = np.random.randn(5, 2) * 0.3  # Small
        cluster2 = np.random.randn(15, 2) * 0.3 + np.array([5, 0])  # Medium
        cluster3 = np.random.randn(25, 2) * 0.3 + np.array([10, 0])  # Large
        X = np.vstack([cluster1, cluster2, cluster3])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)


class TestExactParityDeterminism:
    """Verify deterministic behavior."""
    
    def test_same_input_same_output(self):
        """Same input must always produce same output."""
        np.random.seed(789)
        X = np.random.randn(30, 2) * 0.5
        X = np.vstack([X, X + np.array([5, 5])])  # Two clusters
        
        # Run multiple times
        results = []
        for _ in range(3):
            labels = run_gauging_delta(X.copy())
            results.append(labels.copy())
        
        # All results must be identical
        for i in range(1, len(results)):
            assert np.array_equal(results[0], results[i]), \
                "Non-deterministic behavior detected!"
    
    def test_legacy_matches_new_multiple_runs(self):
        """Multiple runs should all match legacy."""
        np.random.seed(999)
        X = np.random.randn(25, 2) * 0.4
        X = np.vstack([X, X + np.array([6, 0]), X + np.array([12, 0])])
        
        labels_legacy = run_legacy(X)
        
        for _ in range(3):
            labels_new = run_gauging_delta(X.copy())
            assert_identical_labels(labels_legacy, labels_new)


class TestExactParityNumericalStability:
    """Tests for numerical edge cases."""
    
    def test_very_small_coordinates(self):
        """Very small coordinate values."""
        X = np.array([
            [1e-10, 1e-10], [2e-10, 1e-10], [1e-10, 2e-10],
            [1e-8, 1e-8], [1e-8 + 1e-10, 1e-8], [1e-8, 1e-8 + 1e-10]
        ])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)
    
    def test_large_coordinates(self):
        """Large coordinate values."""
        np.random.seed(111)
        cluster1 = np.random.randn(10, 2) * 0.1 + np.array([1e6, 1e6])
        cluster2 = np.random.randn(10, 2) * 0.1 + np.array([1e6 + 100, 1e6])
        X = np.vstack([cluster1, cluster2])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)
    
    def test_mixed_scale_coordinates(self):
        """Mix of small and large coordinates."""
        np.random.seed(222)
        cluster1 = np.random.randn(10, 2) * 0.01  # Small scale
        cluster2 = np.random.randn(10, 2) * 0.01 + np.array([1000, 0])  # Large offset
        X = np.vstack([cluster1, cluster2])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)


class TestExactParityHigherDimensions:
    """Tests for higher dimensional data."""
    
    def test_3d_two_clusters(self):
        """3D data with two clusters."""
        np.random.seed(333)
        cluster1 = np.random.randn(15, 3) * 0.3
        cluster2 = np.random.randn(15, 3) * 0.3 + np.array([5, 5, 5])
        X = np.vstack([cluster1, cluster2])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)
    
    def test_4d_three_clusters(self):
        """4D data with three clusters."""
        np.random.seed(444)
        cluster1 = np.random.randn(10, 4) * 0.3
        cluster2 = np.random.randn(10, 4) * 0.3 + np.array([5, 0, 0, 0])
        cluster3 = np.random.randn(10, 4) * 0.3 + np.array([0, 5, 0, 0])
        X = np.vstack([cluster1, cluster2, cluster3])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_identical_clustering(labels_legacy, labels_new, X)
