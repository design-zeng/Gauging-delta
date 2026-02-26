"""
Parity Tests: Verify GaugingDelta produces IDENTICAL results to perception.py.

Uses Hypothesis for property-based testing with various synthetic datasets.
All tests require ARI=1.0 (exact label match) for parity verification.

Dataset sizes: <2000 points for testing efficiency.
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck, Phase
from hypothesis import strategies as st
from sklearn.metrics import adjusted_rand_score

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Skip all tests if perception.py not found
pytestmark = pytest.mark.skipif(
    not Path(__file__).parent.parent.parent.joinpath("perception.py").exists(),
    reason="perception.py not found",
)


# =============================================================================
# Hypothesis Strategies for Dataset Generation
# =============================================================================

@st.composite
def well_separated_blobs(draw, n_clusters=None, n_points_per_cluster=None):
    """Generate well-separated blob clusters."""
    if n_clusters is None:
        n_clusters = draw(st.integers(min_value=2, max_value=4))
    if n_points_per_cluster is None:
        n_points_per_cluster = draw(st.integers(min_value=5, max_value=50))
    
    n_dims = draw(st.integers(min_value=2, max_value=3))
    separation = draw(st.floats(min_value=8.0, max_value=15.0))
    noise_scale = draw(st.floats(min_value=0.1, max_value=0.5))
    
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
    n_clusters = draw(st.integers(min_value=2, max_value=4))
    n_points = draw(st.integers(min_value=5, max_value=30))
    gap = draw(st.floats(min_value=3.0, max_value=6.0))
    spread = draw(st.floats(min_value=0.2, max_value=0.6))
    
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
def random_gaussian_mixture(draw):
    """Generate random Gaussian mixture with controlled overlap."""
    n_clusters = draw(st.integers(min_value=2, max_value=4))
    n_points = draw(st.integers(min_value=5, max_value=40))
    
    # Generate well-separated centers
    centers = []
    min_separation = 5.0
    for _ in range(n_clusters):
        for attempt in range(100):
            center = np.random.randn(2) * 8
            if all(np.linalg.norm(center - c) > min_separation for c in centers):
                centers.append(center)
                break
        else:
            idx = len(centers)
            centers.append(np.array([idx * min_separation, 0.0]))
    
    points = []
    for center in centers:
        std = draw(st.floats(min_value=0.2, max_value=0.8))
        cluster_points = center + np.random.randn(n_points, 2) * std
        points.append(cluster_points)
    
    X = np.vstack(points)
    return X


@st.composite
def uniform_grid_points(draw):
    """Generate points on a uniform grid (edge case for density)."""
    grid_size = draw(st.integers(min_value=3, max_value=6))
    spacing = draw(st.floats(min_value=0.5, max_value=2.0))
    
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
    n_points = draw(st.integers(min_value=10, max_value=100))
    spread = draw(st.floats(min_value=0.1, max_value=0.5))
    
    X = np.random.randn(n_points, 2) * spread
    return X


@st.composite
def varying_density_clusters(draw):
    """Generate clusters with different densities."""
    n_dense = draw(st.integers(min_value=15, max_value=40))
    n_sparse = draw(st.integers(min_value=8, max_value=25))
    
    dense_cluster = np.random.randn(n_dense, 2) * 0.3
    sparse_cluster = np.random.randn(n_sparse, 2) * 1.5 + np.array([15, 15])
    
    X = np.vstack([dense_cluster, sparse_cluster])
    return X


@st.composite
def three_clusters_different_sizes(draw):
    """Generate three clusters with different sizes."""
    size_small = draw(st.integers(min_value=5, max_value=15))
    size_medium = draw(st.integers(min_value=15, max_value=30))
    size_large = draw(st.integers(min_value=25, max_value=50))
    
    cluster1 = np.random.randn(size_small, 2) * 0.4
    cluster2 = np.random.randn(size_medium, 2) * 0.4 + np.array([8, 0])
    cluster3 = np.random.randn(size_large, 2) * 0.4 + np.array([16, 0])
    
    X = np.vstack([cluster1, cluster2, cluster3])
    return X


@st.composite
def higher_dimensional_clusters(draw):
    """Generate 3D or 4D clusters."""
    n_dims = draw(st.integers(min_value=3, max_value=4))
    n_clusters = draw(st.integers(min_value=2, max_value=3))
    n_points = draw(st.integers(min_value=10, max_value=30))
    
    points = []
    for i in range(n_clusters):
        center = np.zeros(n_dims)
        center[i % n_dims] = 8.0 * (i + 1)
        cluster_points = center + np.random.randn(n_points, n_dims) * 0.4
        points.append(cluster_points)
    
    X = np.vstack(points)
    return X


# =============================================================================
# Helper Functions
# =============================================================================

def run_legacy(X: np.ndarray) -> np.ndarray:
    """Run legacy perception.py algorithm and return labels."""
    # Import here to avoid issues if file doesn't exist
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from perception import Perception
    
    p = Perception()
    with redirect_stdout(io.StringIO()):
        labels, _ = p.fit(X.copy())
    return np.asarray(labels, dtype=int)


def run_gauging_delta(X: np.ndarray) -> np.ndarray:
    """Run new GaugingDelta algorithm and return labels."""
    from gauging_delta import GaugingDelta
    
    g = GaugingDelta(preserve_labels=True)
    labels = g.fit(X.copy())
    return np.asarray(labels, dtype=int)


def assert_exact_parity(labels_legacy: np.ndarray, labels_new: np.ndarray, X: np.ndarray = None):
    """Assert that two label arrays match exactly (ARI=1.0)."""
    if labels_legacy.shape != labels_new.shape:
        raise AssertionError(
            f"Label shape mismatch: legacy={labels_legacy.shape}, new={labels_new.shape}"
        )
    
    ari = adjusted_rand_score(labels_legacy, labels_new)
    
    if ari < 1.0:
        n_legacy = len(set(labels_legacy))
        n_new = len(set(labels_new))
        diffs = np.where(labels_legacy != labels_new)[0]
        sample = diffs[:10].tolist() if len(diffs) > 0 else []
        
        raise AssertionError(
            f"Parity FAILED! ARI={ari:.6f}, "
            f"legacy_clusters={n_legacy}, new_clusters={n_new}, "
            f"mismatched_indices={sample}, total_mismatches={len(diffs)}"
        )


def load_text_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load comma-delimited text dataset with labels in last column."""
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Expected features+label in {path}")
    return data[:, :-1], data[:, -1]


def generate_random_cluster_data(
    n_points: int,
    n_dims: int,
    n_clusters: int = 4,
    separation: float = 10.0,
    spread: float = 0.8,
    seed: int = 123,
) -> np.ndarray:
    """Generate deterministic random clusters with well-separated centers."""
    rng = np.random.default_rng(seed + n_dims + n_clusters)
    points_per_cluster = n_points // n_clusters
    remainder = n_points % n_clusters

    points = []
    for i in range(n_clusters):
        n = points_per_cluster + (1 if i < remainder else 0)
        center = np.zeros(n_dims)
        center[i % n_dims] = separation * (i + 1)
        cluster_points = rng.normal(loc=center, scale=spread, size=(n, n_dims))
        points.append(cluster_points)

    X = np.vstack(points)
    rng.shuffle(X)
    return X


# =============================================================================
# Hypothesis-based Parity Tests
# =============================================================================

class TestParityHypothesis:
    """Property-based tests requiring exact parity (ARI=1.0)."""
    
    @given(X=well_separated_blobs())
    @settings(
        max_examples=5,
        deadline=120000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        phases=[Phase.generate, Phase.target, Phase.shrink],
    )
    def test_well_separated_blobs(self, X):
        """Well-separated blobs must produce identical clustering."""
        assume(len(X) >= 10)
        assume(len(X) <= 200)
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)
    
    @given(X=linear_chain_clusters())
    @settings(
        max_examples=5,
        deadline=120000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        phases=[Phase.generate, Phase.target, Phase.shrink],
    )
    def test_linear_chain(self, X):
        """Linear chain clusters must produce identical clustering."""
        assume(len(X) >= 10)
        assume(len(X) <= 100)
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)
    
    @given(X=random_gaussian_mixture())
    @settings(
        max_examples=5,
        deadline=120000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        phases=[Phase.generate, Phase.target, Phase.shrink],
    )
    def test_gaussian_mixture(self, X):
        """Gaussian mixtures must produce identical clustering."""
        assume(len(X) >= 10)
        assume(len(X) <= 100)
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)
    
    @given(X=single_dense_cluster())
    @settings(
        max_examples=5,
        deadline=120000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        phases=[Phase.generate, Phase.target, Phase.shrink],
    )
    def test_single_cluster(self, X):
        """Single dense cluster must produce identical result."""
        assume(len(X) >= 15)
        assume(len(X) <= 80)
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)
    
    @given(X=uniform_grid_points())
    @settings(
        max_examples=5,
        deadline=120000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        phases=[Phase.generate, Phase.target, Phase.shrink],
    )
    def test_grid_points(self, X):
        """Grid points must produce identical clustering."""
        assume(len(X) >= 9)
        assume(len(X) <= 36)
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)
    
    @given(X=varying_density_clusters())
    @settings(
        max_examples=5,
        deadline=120000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        phases=[Phase.generate, Phase.target, Phase.shrink],
    )
    def test_varying_density(self, X):
        """Varying density clusters must produce identical clustering."""
        assume(len(X) >= 20)
        assume(len(X) <= 60)
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)
    
    @given(X=three_clusters_different_sizes())
    @settings(
        max_examples=5,
        deadline=120000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        phases=[Phase.generate, Phase.target, Phase.shrink],
    )
    def test_different_sizes(self, X):
        """Different sized clusters must produce identical clustering."""
        assume(len(X) >= 30)
        assume(len(X) <= 80)
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)
    
    @given(X=higher_dimensional_clusters())
    @settings(
        max_examples=5,
        deadline=120000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        phases=[Phase.generate, Phase.target, Phase.shrink],
    )
    def test_higher_dimensions(self, X):
        """Higher dimensional data must produce identical clustering."""
        assume(len(X) >= 20)
        assume(len(X) <= 60)
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)


class TestParityRandomLarge:
    """Randomized larger datasets with exact parity."""

    @pytest.mark.parametrize("n_dims", [2, 5, 10])
    def test_random_clusters_800(self, n_dims):
        """~800 points with <=10 dimensions should match exactly."""
        X = generate_random_cluster_data(n_points=800, n_dims=n_dims, n_clusters=4)

        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)

        assert_exact_parity(labels_legacy, labels_new, X)


class TestParityBenchmarkDatasets:
    """Exact parity on benchmark datasets from data/ directory (small datasets only)."""
    
    @pytest.mark.parametrize("dataset_name", [
        "3-spiral",      # 312 pts
        "3_blobs",       # 300 pts
        "flame",         # 240 pts
        "jain",          # 373 pts
        "pathbased",     # 300 pts
        "compound",      # 399 pts
    ])
    def test_benchmark_dataset(self, dataset_name):
        """Each benchmark dataset must produce identical clustering."""
        data_path = Path(__file__).parent.parent.parent / "data" / f"{dataset_name}.txt"
        
        if not data_path.exists():
            pytest.skip(f"Dataset {dataset_name} not found")
        
        X, _ = load_text_dataset(data_path)
        
        if len(X) > 500:
            pytest.skip(f"Dataset {dataset_name} too large ({len(X)} points)")
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)


class TestParityEdgeCases:
    """Edge case tests requiring exact parity."""
    
    def test_two_points_close(self):
        """Two close points should produce identical clustering."""
        X = np.array([[0.0, 0.0], [0.1, 0.1]])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)
    
    def test_two_points_far(self):
        """Two far points should produce identical clustering."""
        X = np.array([[0.0, 0.0], [100.0, 100.0]])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)
    
    def test_three_collinear_points(self):
        """Three collinear points should produce identical clustering."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)
    
    def test_equilateral_triangle(self):
        """Equilateral triangle should produce identical clustering."""
        X = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2]
        ])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)
    
    def test_square_arrangement(self):
        """Square arrangement should produce identical clustering."""
        X = np.array([
            [0.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [1.0, 1.0]
        ])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)
    
    def test_two_distinct_clusters_minimal(self):
        """Minimum two-cluster scenario should produce identical clustering."""
        X = np.array([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
            [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]
        ])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)
    
    def test_cluster_with_outlier(self):
        """Cluster with outlier should produce identical clustering."""
        np.random.seed(42)
        cluster_points = np.random.randn(15, 2) * 0.3
        outlier = np.array([[10.0, 10.0]])
        X = np.vstack([cluster_points, outlier])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)


class TestParityDeterminism:
    """Verify deterministic behavior of both implementations."""
    
    def test_same_input_same_output(self):
        """Same input must always produce same output."""
        np.random.seed(789)
        X = np.random.randn(30, 2) * 0.5
        X = np.vstack([X, X + np.array([8, 8])])
        
        # Run multiple times
        results_legacy = []
        results_new = []
        for _ in range(3):
            results_legacy.append(run_legacy(X.copy()))
            results_new.append(run_gauging_delta(X.copy()))
        
        # All results must be identical within each implementation
        for i in range(1, len(results_legacy)):
            assert np.array_equal(results_legacy[0], results_legacy[i]), \
                "Legacy non-deterministic!"
            assert np.array_equal(results_new[0], results_new[i]), \
                "New non-deterministic!"
        
        # And both must match
        assert_exact_parity(results_legacy[0], results_new[0], X)


class TestParityNumericalStability:
    """Tests for numerical edge cases."""
    
    def test_very_small_coordinates(self):
        """Very small coordinate values."""
        X = np.array([
            [1e-10, 1e-10], [2e-10, 1e-10], [1e-10, 2e-10],
            [1e-8, 1e-8], [1e-8 + 1e-10, 1e-8], [1e-8, 1e-8 + 1e-10]
        ])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)
    
    def test_large_coordinates(self):
        """Large coordinate values."""
        np.random.seed(111)
        cluster1 = np.random.randn(10, 2) * 0.1 + np.array([1e6, 1e6])
        cluster2 = np.random.randn(10, 2) * 0.1 + np.array([1e6 + 100, 1e6])
        X = np.vstack([cluster1, cluster2])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)
    
    def test_mixed_scale_coordinates(self):
        """Mix of small and large coordinates."""
        np.random.seed(222)
        cluster1 = np.random.randn(10, 2) * 0.01
        cluster2 = np.random.randn(10, 2) * 0.01 + np.array([1000, 0])
        X = np.vstack([cluster1, cluster2])
        
        labels_legacy = run_legacy(X)
        labels_new = run_gauging_delta(X)
        
        assert_exact_parity(labels_legacy, labels_new, X)
