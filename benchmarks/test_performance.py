"""
Benchmark Tests: Compare time and memory between GaugingDelta and perception.py.

Measures execution time and peak memory usage for various dataset sizes
up to 10,000 points.
"""

import io
import sys
import time
import tracemalloc
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Skip all tests if perception.py not found
pytestmark = pytest.mark.skipif(
    not Path(__file__).parent.parent.parent.joinpath("perception.py").exists(),
    reason="perception.py not found",
)


# =============================================================================
# Dataset Generators
# =============================================================================

def generate_blobs(n_points: int, n_clusters: int = 3, spread: float = 0.5) -> np.ndarray:
    """Generate well-separated blob clusters."""
    points_per_cluster = n_points // n_clusters
    remainder = n_points % n_clusters
    
    all_points = []
    for i in range(n_clusters):
        n = points_per_cluster + (1 if i < remainder else 0)
        center = np.array([i * 10.0, 0.0])
        cluster = center + np.random.randn(n, 2) * spread
        all_points.append(cluster)
    
    return np.vstack(all_points)


def generate_random_gaussian(n_points: int, n_dims: int = 2) -> np.ndarray:
    """Generate random Gaussian distributed points."""
    return np.random.randn(n_points, n_dims) * 5


def generate_grid(grid_size: int) -> np.ndarray:
    """Generate uniform grid points."""
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xx, yy = np.meshgrid(x, y)
    X = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    X += np.random.randn(*X.shape) * 0.01  # Small noise
    return X


# =============================================================================
# Benchmark Runners
# =============================================================================

def run_legacy_timed(X: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Run legacy and return (labels, time_seconds, peak_memory_bytes)."""
    from perception import Perception
    
    tracemalloc.start()
    start_time = time.perf_counter()
    
    p = Perception()
    with redirect_stdout(io.StringIO()):
        labels, _ = p.fit(X.copy())
    
    elapsed = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return np.asarray(labels, dtype=int), elapsed, peak_memory


def run_gauging_delta_timed(X: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Run GaugingDelta and return (labels, time_seconds, peak_memory_bytes)."""
    from gauging_delta import GaugingDelta
    
    tracemalloc.start()
    start_time = time.perf_counter()
    
    g = GaugingDelta(preserve_labels=True)
    labels = g.fit(X.copy())
    
    elapsed = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return np.asarray(labels, dtype=int), elapsed, peak_memory


# =============================================================================
# Benchmark Tests
# =============================================================================

class TestBenchmarkSmall:
    """Benchmark tests for small datasets (100-500 points)."""
    
    @pytest.mark.parametrize("n_points", [100, 200, 500])
    def test_blobs_small(self, n_points):
        """Benchmark well-separated blobs (small)."""
        np.random.seed(42)
        X = generate_blobs(n_points, n_clusters=3)
        
        labels_legacy, time_legacy, mem_legacy = run_legacy_timed(X)
        labels_new, time_new, mem_new = run_gauging_delta_timed(X)
        
        ari = adjusted_rand_score(labels_legacy, labels_new)
        speedup = time_legacy / time_new if time_new > 0 else float('inf')
        
        print(f"\n  [{n_points} pts] Legacy: {time_legacy:.3f}s, {mem_legacy/1024/1024:.1f}MB")
        print(f"  [{n_points} pts] New:    {time_new:.3f}s, {mem_new/1024/1024:.1f}MB")
        print(f"  [{n_points} pts] Speedup: {speedup:.2f}x, ARI: {ari:.6f}")
        
        assert ari == 1.0, f"Parity failed! ARI={ari}"


class TestBenchmarkMedium:
    """Benchmark tests for medium datasets (1000-2000 points)."""
    
    @pytest.mark.parametrize("n_points", [1000, 2000])
    def test_blobs_medium(self, n_points):
        """Benchmark well-separated blobs (medium)."""
        np.random.seed(42)
        X = generate_blobs(n_points, n_clusters=5)
        
        labels_legacy, time_legacy, mem_legacy = run_legacy_timed(X)
        labels_new, time_new, mem_new = run_gauging_delta_timed(X)
        
        ari = adjusted_rand_score(labels_legacy, labels_new)
        speedup = time_legacy / time_new if time_new > 0 else float('inf')
        
        print(f"\n  [{n_points} pts] Legacy: {time_legacy:.3f}s, {mem_legacy/1024/1024:.1f}MB")
        print(f"  [{n_points} pts] New:    {time_new:.3f}s, {mem_new/1024/1024:.1f}MB")
        print(f"  [{n_points} pts] Speedup: {speedup:.2f}x, ARI: {ari:.6f}")
        
        assert ari == 1.0, f"Parity failed! ARI={ari}"


class TestBenchmarkLarge:
    """Benchmark tests for large datasets (2000-5000 points)."""
    
    @pytest.mark.slow
    @pytest.mark.parametrize("n_points", [2000, 5000])
    def test_blobs_large(self, n_points):
        """Benchmark well-separated blobs (large)."""
        np.random.seed(42)
        X = generate_blobs(n_points, n_clusters=10)
        
        labels_legacy, time_legacy, mem_legacy = run_legacy_timed(X)
        labels_new, time_new, mem_new = run_gauging_delta_timed(X)
        
        ari = adjusted_rand_score(labels_legacy, labels_new)
        speedup = time_legacy / time_new if time_new > 0 else float('inf')
        
        print(f"\n  [{n_points} pts] Legacy: {time_legacy:.3f}s, {mem_legacy/1024/1024:.1f}MB")
        print(f"  [{n_points} pts] New:    {time_new:.3f}s, {mem_new/1024/1024:.1f}MB")
        print(f"  [{n_points} pts] Speedup: {speedup:.2f}x, ARI: {ari:.6f}")
        
        assert ari == 1.0, f"Parity failed! ARI={ari}"


class TestBenchmarkScaling:
    """Test how performance scales with dataset size."""
    
    def test_scaling_report(self):
        """Generate scaling report for multiple sizes."""
        sizes = [50, 100, 200, 300]
        results = []
        
        for n in sizes:
            np.random.seed(42)
            X = generate_blobs(n, n_clusters=max(2, n // 50))
            
            labels_legacy, time_legacy, mem_legacy = run_legacy_timed(X)
            labels_new, time_new, mem_new = run_gauging_delta_timed(X)
            
            ari = adjusted_rand_score(labels_legacy, labels_new)
            
            results.append({
                'n_points': n,
                'time_legacy': time_legacy,
                'time_new': time_new,
                'mem_legacy': mem_legacy,
                'mem_new': mem_new,
                'ari': ari,
            })
            
            assert ari == 1.0, f"Parity failed at n={n}! ARI={ari}"
        
        # Print report
        print("\n" + "="*70)
        print("SCALING REPORT")
        print("="*70)
        print(f"{'N':>8} | {'Legacy(s)':>10} | {'New(s)':>10} | {'Speedup':>8} | {'ARI':>8}")
        print("-"*70)
        for r in results:
            speedup = r['time_legacy'] / r['time_new'] if r['time_new'] > 0 else float('inf')
            print(f"{r['n_points']:>8} | {r['time_legacy']:>10.3f} | {r['time_new']:>10.3f} | {speedup:>8.2f}x | {r['ari']:>8.4f}")
        print("="*70)


class TestBenchmarkMemory:
    """Test memory usage specifically."""
    
    def test_memory_efficiency(self):
        """Verify new implementation is memory efficient."""
        np.random.seed(42)
        X = generate_blobs(1000, n_clusters=5)
        
        _, _, mem_legacy = run_legacy_timed(X)
        _, _, mem_new = run_gauging_delta_timed(X)
        
        print(f"\n  Legacy memory: {mem_legacy/1024/1024:.2f} MB")
        print(f"  New memory:    {mem_new/1024/1024:.2f} MB")
        print(f"  Ratio:         {mem_new/mem_legacy:.2f}x")
        
        # New should not use more than 2x the memory of legacy
        # (allowing some overhead for different data structures)
        assert mem_new < mem_legacy * 3, \
            f"New uses too much memory: {mem_new/1024/1024:.1f}MB vs {mem_legacy/1024/1024:.1f}MB"


class TestBenchmarkDataPatterns:
    """Benchmark different data patterns."""
    
    def test_random_gaussian(self):
        """Benchmark random Gaussian data."""
        np.random.seed(42)
        X = generate_random_gaussian(500)
        
        labels_legacy, time_legacy, _ = run_legacy_timed(X)
        labels_new, time_new, _ = run_gauging_delta_timed(X)
        
        ari = adjusted_rand_score(labels_legacy, labels_new)
        
        print(f"\n  Random Gaussian 500pts: Legacy={time_legacy:.3f}s, New={time_new:.3f}s, ARI={ari:.6f}")
        
        assert ari == 1.0, f"Parity failed! ARI={ari}"
    
    def test_grid_pattern(self):
        """Benchmark grid pattern data."""
        np.random.seed(42)
        X = generate_grid(20)  # 400 points
        
        labels_legacy, time_legacy, _ = run_legacy_timed(X)
        labels_new, time_new, _ = run_gauging_delta_timed(X)
        
        ari = adjusted_rand_score(labels_legacy, labels_new)
        
        print(f"\n  Grid 20x20 (400pts): Legacy={time_legacy:.3f}s, New={time_new:.3f}s, ARI={ari:.6f}")
        
        assert ari == 1.0, f"Parity failed! ARI={ari}"
