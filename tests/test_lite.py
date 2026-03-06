"""Tests for mode='lite' (centroid linkage, no continuity)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from gauging_delta.algorithm import GaugingDelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _well_separated_blobs(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated blobs — any reasonable algorithm should ace this."""
    rng = np.random.RandomState(seed)
    centers = np.array([[0, 0], [10, 0], [5, 10]], dtype=float)
    parts_X, parts_y = [], []
    for i, c in enumerate(centers):
        parts_X.append(rng.randn(30, 2) * 0.5 + c)
        parts_y.append(np.full(30, i))
    return np.vstack(parts_X), np.concatenate(parts_y)


# ---------------------------------------------------------------------------
# API / smoke tests
# ---------------------------------------------------------------------------

class TestLiteAPI:

    def test_fit_returns_self(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
        gd = GaugingDelta(mode="lite")
        assert gd.fit(X) is gd

    def test_labels_and_n_clusters(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
        gd = GaugingDelta(mode="lite").fit(X)
        assert hasattr(gd, "labels_")
        assert hasattr(gd, "n_clusters_")
        assert isinstance(gd.n_clusters_, int)
        assert len(gd.labels_) == 4

    def test_fit_predict(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
        assert np.array_equal(
            GaugingDelta(mode="lite").fit_predict(X),
            GaugingDelta(mode="lite").fit(X).labels_,
        )

    def test_preserve_labels(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(15, 2), rng.randn(15, 2) + [8, 0]])
        gd = GaugingDelta(mode="lite", preserve_labels=True).fit(X)
        assert gd.n_clusters_ >= 1

    def test_n_clusters_target(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(20, 2), rng.randn(20, 2) + [10, 10]])
        gd = GaugingDelta(mode="lite", n_clusters=2).fit(X)
        assert gd.n_clusters_ == 2


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

class TestLiteValidation:

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            GaugingDelta(mode="bogus").fit(np.array([[0.0, 0.0], [1.0, 1.0]]))

    def test_mode_stored(self):
        assert GaugingDelta(mode="lite").mode == "lite"

    def test_full_is_default(self):
        assert GaugingDelta().mode == "full"


# ---------------------------------------------------------------------------
# Memory footprint
# ---------------------------------------------------------------------------

class TestLiteMemory:

    def test_no_near_ref(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
        gd = GaugingDelta(mode="lite").fit(X)
        assert not hasattr(gd, "_near_ref")

    def test_no_pd_arrays(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
        gd = GaugingDelta(mode="lite").fit(X)
        assert not hasattr(gd, "_pd_indices")
        assert not hasattr(gd, "_pd_dists")

    def test_has_dist_matrix(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
        gd = GaugingDelta(mode="lite").fit(X)
        assert hasattr(gd, "_dist_matrix")


# ---------------------------------------------------------------------------
# Clustering quality
# ---------------------------------------------------------------------------

class TestLiteQuality:

    def test_well_separated_blobs(self):
        """Lite mode should produce multiple clusters for well-separated data."""
        X, y = _well_separated_blobs()
        gd = GaugingDelta(mode="lite").fit(X)
        ari = adjusted_rand_score(y, gd.labels_)
        # Centroid linkage tends to under-merge (more clusters than ground truth)
        assert ari > 0.4, f"ARI={ari:.3f} too low for well-separated blobs"
        assert gd.n_clusters_ >= 3, "Should find at least 3 clusters"

    @pytest.mark.parametrize("name", ["flame", "3_blobs", "pathbased", "jain", "compound"])
    def test_benchmark_runs(self, name, datasets):
        """Lite mode should run without error on all benchmark datasets."""
        X, _y_true = datasets[name]
        gd = GaugingDelta(mode="lite").fit(X)
        assert gd.n_clusters_ >= 1
        assert len(gd.labels_) == len(X)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestLiteEdgeCases:

    def test_two_points(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0]])
        gd = GaugingDelta(mode="lite").fit(X)
        assert gd.n_clusters_ == 1
        assert gd.labels_[0] == gd.labels_[1]

    def test_three_collinear(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        gd = GaugingDelta(mode="lite").fit(X)
        assert gd.n_clusters_ == 1

    def test_duplicate_points(self):
        X = np.array([[1.0, 1.0], [1.0, 1.0], [5.0, 5.0], [5.0, 5.0]])
        gd = GaugingDelta(mode="lite").fit(X)
        assert gd.n_clusters_ >= 1
        assert len(gd.labels_) == 4

    def test_single_point(self):
        X = np.array([[0.0, 0.0]])
        gd = GaugingDelta(mode="lite").fit(X)
        assert gd.n_clusters_ == 1
        assert len(gd.labels_) == 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestLiteDeterminism:

    def test_deterministic(self):
        rng = np.random.RandomState(99)
        X = np.vstack([rng.randn(20, 2), rng.randn(20, 2) + [5, 0]])
        labels1 = GaugingDelta(mode="lite").fit_predict(X)
        labels2 = GaugingDelta(mode="lite").fit_predict(X)
        np.testing.assert_array_equal(labels1, labels2)
