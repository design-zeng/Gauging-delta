"""Unit tests for individual components.

Tests angle computation, proximity, threshold edge cases, config override,
and swappable component interfaces.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gauging_delta import GaugingDelta, GaugingDeltaConfig
from gauging_delta.angles import compute_angle, compute_angle_batch
from gauging_delta.cluster import Cluster
from gauging_delta.proximity import DefaultProximity


# ---------------------------------------------------------------------------
# Angle computation
# ---------------------------------------------------------------------------


class TestComputeAngle:
    """Tests for compute_angle matching legacy to_find_angle."""

    def test_right_angle(self) -> None:
        """90-degree angle at origin."""
        start = np.array([0.0, 0.0])
        left = np.array([1.0, 0.0])
        right = np.array([0.0, 1.0])
        angle = compute_angle(start, left, right)
        assert angle == pytest.approx(math.pi / 2, abs=1e-6)

    def test_zero_angle_same_point(self) -> None:
        """Zero angle when left == right."""
        start = np.array([0.0, 0.0])
        left = np.array([1.0, 0.0])
        angle = compute_angle(start, left, left)
        assert angle == pytest.approx(0.0, abs=1e-10)

    def test_straight_angle(self) -> None:
        """180-degree angle (opposite directions)."""
        start = np.array([0.0, 0.0])
        left = np.array([1.0, 0.0])
        right = np.array([-1.0, 0.0])
        angle = compute_angle(start, left, right)
        assert angle == pytest.approx(math.pi, abs=1e-6)

    def test_zero_vector_returns_zero(self) -> None:
        """When start == left, norm product is 0, should return 0."""
        start = np.array([1.0, 1.0])
        angle = compute_angle(start, start, np.array([2.0, 2.0]))
        assert angle == 0.0

    def test_rounding_parity(self) -> None:
        """Ensure rounding matches legacy: round(norm_prod, 4) for zero check,
        but divide by unrounded product."""
        start = np.array([0.0, 0.0])
        left = np.array([0.0001, 0.0])  # very small
        right = np.array([0.0, 0.0001])
        angle = compute_angle(start, left, right, rounding=4)
        # norm_prod_raw = 1e-8, round(1e-8, 4) = 0.0 → returns 0
        assert angle == 0.0

    def test_batch_matches_scalar(self) -> None:
        """compute_angle_batch must match compute_angle for each point."""
        rng = np.random.RandomState(42)
        start = rng.randn(2)
        points = rng.randn(10, 2)
        right = rng.randn(2)

        batch_angles = compute_angle_batch(start, points, right)
        scalar_angles = np.array([compute_angle(start, p, right) for p in points])
        np.testing.assert_allclose(batch_angles, scalar_angles, atol=1e-12)


# ---------------------------------------------------------------------------
# Proximity
# ---------------------------------------------------------------------------


class TestProximity:
    """Tests for DefaultProximity edge cases."""

    def _make_cluster(self, label: int, n_points: int, **kwargs) -> Cluster:
        return Cluster(
            label=label,
            point_indices=list(range(n_points)),
            center=np.zeros(2),
            **kwargs,
        )

    def test_lead_is_larger(self) -> None:
        """Larger cluster should be lead."""
        c1 = self._make_cluster(0, 5, merging_dists=[1.0, 2.0, 3.0])
        c2 = self._make_cluster(1, 3, merging_dists=[1.0, 2.0])
        prox = DefaultProximity()
        result = prox.compute(c1, c2, d_ij=1.0, fallback=0.5)
        assert result.lead_id == 0

    def test_equal_size_c1_is_lead(self) -> None:
        """When equal size, C_i (first arg) should be lead."""
        c1 = self._make_cluster(0, 5, merging_dists=[1.0, 2.0])
        c2 = self._make_cluster(1, 5, merging_dists=[1.0, 2.0])
        prox = DefaultProximity()
        result = prox.compute(c1, c2, d_ij=1.0, fallback=0.5)
        assert result.lead_id == 0

    def test_singleton_uses_fallback(self) -> None:
        """Singletons with no merge history should use fallback_dist."""
        c1 = self._make_cluster(0, 1)
        c2 = self._make_cluster(1, 1)
        prox = DefaultProximity()
        result = prox.compute(c1, c2, d_ij=1.0, fallback=0.5)
        # rho = d_ij / fallback = 1.0 / 0.5 = 2.0
        assert result.rho == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Config override
# ---------------------------------------------------------------------------


class TestConfig:
    """GaugingDeltaConfig should be overridable."""

    def test_default_values(self) -> None:
        cfg = GaugingDeltaConfig()
        assert cfg.threshold_continuity == 0.15
        assert cfg.explore_radii == (2.0, 2.5, 3.0)

    def test_override(self) -> None:
        cfg = GaugingDeltaConfig(threshold_continuity=0.3)
        assert cfg.threshold_continuity == 0.3

    def test_frozen(self) -> None:
        cfg = GaugingDeltaConfig()
        with pytest.raises(AttributeError):
            cfg.threshold_continuity = 0.5  # type: ignore[misc]

    def test_compactness_defaults(self) -> None:
        """Compactness fields match legacy hardcoded constants."""
        cfg = GaugingDeltaConfig()
        assert cfg.compact_fallback == 0.5
        assert cfg.compact_min_size == 2
        assert cfg.compact_history_window == 5

    def test_compactness_override(self) -> None:
        cfg = GaugingDeltaConfig(compact_fallback=0.8, compact_min_size=4, compact_history_window=10)
        assert cfg.compact_fallback == 0.8
        assert cfg.compact_min_size == 4
        assert cfg.compact_history_window == 10

    def test_transition_defaults(self) -> None:
        """Transition smoothness fields match legacy hardcoded constants."""
        cfg = GaugingDeltaConfig()
        assert cfg.transition_external_zero_fallback == 2.0
        assert cfg.transition_lopsided_override == 1.5
        assert cfg.transition_lopsided_ratio == 0.1

    def test_transition_override(self) -> None:
        cfg = GaugingDeltaConfig(
            transition_external_zero_fallback=3.0,
            transition_lopsided_override=2.0,
            transition_lopsided_ratio=0.2,
        )
        assert cfg.transition_external_zero_fallback == 3.0
        assert cfg.transition_lopsided_override == 2.0
        assert cfg.transition_lopsided_ratio == 0.2

    def test_vision_scale_and_boundary_defaults(self) -> None:
        """Vision scale fallback, boundary threshold, and mass ratio jump."""
        cfg = GaugingDeltaConfig()
        assert cfg.vision_scale_dist_ratio_fallback == 1.0
        assert cfg.boundary_threshold == 2.0
        assert cfg.mass_ratio_jump == pytest.approx(math.e)

    def test_vision_scale_and_boundary_override(self) -> None:
        cfg = GaugingDeltaConfig(
            vision_scale_dist_ratio_fallback=2.0,
            boundary_threshold=3.0,
            mass_ratio_jump=3.0,
        )
        assert cfg.vision_scale_dist_ratio_fallback == 2.0
        assert cfg.boundary_threshold == 3.0
        assert cfg.mass_ratio_jump == 3.0


# ---------------------------------------------------------------------------
# Swappable components
# ---------------------------------------------------------------------------


class TestSwappableComponents:
    """GaugingDelta should accept custom proximity/continuity/linkage."""

    def test_custom_proximity(self) -> None:
        """A custom proximity that always returns rho=0 should merge everything."""

        class AlwaysMerge:
            def compute(self, C_i, C_j, d_ij, fallback):
                from gauging_delta._types import ProximityResult

                lead = C_i if len(C_i) >= len(C_j) else C_j
                child = C_j if lead is C_i else C_i
                return ProximityResult(
                    rho=0.0, distance=d_ij, lead_id=lead.label, child_id=child.label
                )

        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(10, 2), rng.randn(10, 2) + [10, 0]])
        gd = GaugingDelta(proximity=AlwaysMerge())
        gd.fit(X)
        # With rho=0 (< any threshold), many merges should succeed
        # but continuity can still reject, so just check it runs
        assert hasattr(gd, "labels_")
        assert len(gd.labels_) == 20

    def test_config_passthrough(self) -> None:
        """Custom config should be used by the algorithm."""
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(20, 2), rng.randn(20, 2) + [10, 0]])
        cfg = GaugingDeltaConfig(threshold_continuity=0.15)
        gd = GaugingDelta(config=cfg)
        gd.fit(X)
        assert hasattr(gd, "labels_")


# ---------------------------------------------------------------------------
# sklearn API
# ---------------------------------------------------------------------------


class TestSklearnAPI:
    """Test sklearn-compatible interface."""

    def test_fit_returns_self(self) -> None:
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
        gd = GaugingDelta()
        result = gd.fit(X)
        assert result is gd

    def test_labels_dtype(self) -> None:
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
        gd = GaugingDelta().fit(X)
        assert gd.labels_.dtype == np.int64 or gd.labels_.dtype == int

    def test_n_clusters_attribute(self) -> None:
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
        gd = GaugingDelta().fit(X)
        assert isinstance(gd.n_clusters_, int)
        assert gd.n_clusters_ >= 1


# ---------------------------------------------------------------------------
# Metric validation
# ---------------------------------------------------------------------------


class TestMetricValidation:
    """Triangle inequality validation for custom metrics."""

    def test_euclidean_skips_validation(self) -> None:
        """Euclidean metric should not trigger validation (always valid)."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        GaugingDelta(metric="euclidean").fit(X)  # no error

    def test_valid_scipy_metric_passes(self) -> None:
        """Standard scipy metrics satisfy triangle inequality."""
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [5.0, 5.0]])
        GaugingDelta(metric="cityblock").fit(X)
        GaugingDelta(metric="cosine").fit(X)

    def test_invalid_metric_raises(self) -> None:
        """A callable violating triangle inequality should be rejected."""

        def bad_metric(u, v):
            # Squared euclidean violates triangle inequality
            return float(np.sum((u - v) ** 2))

        # Collinear spread-out points guarantee violation for squared euclidean
        X = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        with pytest.raises(ValueError, match="triangle inequality"):
            GaugingDelta(metric=bad_metric).fit(X)

    def test_valid_callable_passes(self) -> None:
        """A callable satisfying triangle inequality should work."""

        def manhattan(u, v):
            return float(np.sum(np.abs(u - v)))

        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
        gd = GaugingDelta(metric=manhattan).fit(X)
        assert gd.n_clusters_ >= 1


# ---------------------------------------------------------------------------
# Continuity toggle
# ---------------------------------------------------------------------------


class TestContinuityToggle:
    """Test continuity=False disables the continuity gate."""

    def test_continuity_false_full_mode(self) -> None:
        """Full mode with continuity=False should run and produce labels."""
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(20, 2), rng.randn(20, 2) + [10, 0]])
        gd = GaugingDelta(continuity=False).fit(X)
        assert hasattr(gd, "labels_")
        assert len(gd.labels_) == 40
        assert gd.n_clusters_ >= 1

    def test_continuity_false_merges_more(self) -> None:
        """Disabling continuity should merge at least as aggressively as with it."""
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(30, 2), rng.randn(30, 2) + [5, 0]])
        n_with = GaugingDelta().fit(X).n_clusters_
        n_without = GaugingDelta(continuity=False).fit(X).n_clusters_
        # Without continuity gate, fewer or equal clusters (more merges accepted)
        assert n_without <= n_with

    def test_continuity_none_uses_mode_default(self) -> None:
        """continuity=None should resolve to DefaultContinuity in full mode at fit time."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
        gd = GaugingDelta(continuity=None).fit(X)
        assert gd._cont is not None  # full mode default: on

    def test_continuity_none_lite_mode(self) -> None:
        """continuity=None in lite mode should resolve to None (off) at fit time."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
        gd = GaugingDelta(mode="lite", continuity=None).fit(X)
        assert gd._cont is None

    def test_continuity_false_lite_mode(self) -> None:
        """continuity=False in lite mode should work (same as default)."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
        gd = GaugingDelta(mode="lite", continuity=False).fit(X)
        assert gd.n_clusters_ >= 1


# ---------------------------------------------------------------------------
# Precomputed distance matrix
# ---------------------------------------------------------------------------


class TestPrecomputed:
    """Test metric='precomputed' with various distance matrix forms."""

    def _blobs_X(self) -> np.ndarray:
        rng = np.random.RandomState(42)
        return np.vstack([rng.randn(15, 2), rng.randn(15, 2) + [10, 0]])

    def test_square_matrix(self) -> None:
        """Square (N, N) distance matrix should work."""
        from scipy.spatial.distance import squareform, pdist

        X = self._blobs_X()
        D = squareform(pdist(X))
        gd = GaugingDelta(metric="precomputed").fit(D)
        assert gd.n_clusters_ >= 1
        assert len(gd.labels_) == len(X)

    def test_condensed_vector(self) -> None:
        """Condensed 1-D vector (from pdist) should work."""
        from scipy.spatial.distance import pdist

        X = self._blobs_X()
        D_cond = pdist(X)
        gd = GaugingDelta(metric="precomputed").fit(D_cond)
        assert gd.n_clusters_ >= 1
        assert len(gd.labels_) == len(X)

    def test_square_and_condensed_agree(self) -> None:
        """Square and condensed input should produce identical labels."""
        from scipy.spatial.distance import squareform, pdist

        X = self._blobs_X()
        D_sq = squareform(pdist(X))
        D_cond = pdist(X)
        labels_sq = GaugingDelta(metric="precomputed").fit_predict(D_sq)
        labels_cond = GaugingDelta(metric="precomputed").fit_predict(D_cond)
        np.testing.assert_array_equal(labels_sq, labels_cond)

    def test_consistent_with_euclidean(self) -> None:
        """Precomputed euclidean matrix should produce same result as metric='euclidean'
        with continuity disabled (precomputed forces continuity off)."""
        from scipy.spatial.distance import squareform, pdist

        X = self._blobs_X()
        D = squareform(pdist(X))
        labels_pre = GaugingDelta(metric="precomputed").fit_predict(D)
        labels_euc = GaugingDelta(continuity=False).fit_predict(X)
        np.testing.assert_array_equal(labels_pre, labels_euc)

    def test_n_clusters_target(self) -> None:
        """n_clusters should work with precomputed."""
        from scipy.spatial.distance import squareform, pdist

        X = self._blobs_X()
        D = squareform(pdist(X))
        gd = GaugingDelta(metric="precomputed", n_clusters=2).fit(D)
        assert gd.n_clusters_ == 2

    def test_dataframe_input(self) -> None:
        """Pandas DataFrame distance matrix should work."""
        pd = pytest.importorskip("pandas")
        from scipy.spatial.distance import squareform, pdist

        X = self._blobs_X()
        D = squareform(pdist(X))
        df = pd.DataFrame(D)
        gd = GaugingDelta(metric="precomputed").fit(df)
        assert gd.n_clusters_ >= 1

    # --- Validation ---

    def test_rejects_non_square(self) -> None:
        with pytest.raises(ValueError, match="square"):
            GaugingDelta(metric="precomputed").fit(np.ones((3, 4)))

    def test_rejects_negative_values(self) -> None:
        D = np.array([[0, 1, -1], [1, 0, 1], [-1, 1, 0]], dtype=float)
        with pytest.raises(ValueError, match="negative"):
            GaugingDelta(metric="precomputed").fit(D)

    def test_rejects_nonzero_diagonal(self) -> None:
        D = np.array([[1, 2], [2, 1]], dtype=float)
        with pytest.raises(ValueError, match="diagonal"):
            GaugingDelta(metric="precomputed").fit(D)

    def test_rejects_asymmetric(self) -> None:
        D = np.array([[0, 1, 2], [1, 0, 3], [2, 4, 0]], dtype=float)
        with pytest.raises(ValueError, match="symmetric"):
            GaugingDelta(metric="precomputed").fit(D)

    def test_rejects_lite_mode(self) -> None:
        D = np.array([[0, 1], [1, 0]], dtype=float)
        with pytest.raises(ValueError, match="mode='full'"):
            GaugingDelta(metric="precomputed", mode="lite").fit(D)

    def test_rejects_continuity_with_precomputed(self) -> None:
        from gauging_delta.continuity import DefaultContinuity

        D = np.array([[0, 1], [1, 0]], dtype=float)
        with pytest.raises(ValueError, match="continuity"):
            GaugingDelta(metric="precomputed", continuity=DefaultContinuity()).fit(D)

    def test_continuity_auto_disabled(self) -> None:
        D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        gd = GaugingDelta(metric="precomputed").fit(D)
        assert gd._cont is None


# ---------------------------------------------------------------------------
# sklearn integration
# ---------------------------------------------------------------------------


class TestSklearnIntegration:
    """Tests for BaseEstimator/ClusterMixin compliance."""

    X = np.array([
        [0.0, 0.0], [0.5, 0.3], [0.2, 0.1],
        [8.0, 8.0], [8.5, 8.3], [8.2, 8.1],
    ])

    def test_get_params_returns_raw(self) -> None:
        """get_params() must return the raw constructor values, not resolved defaults."""
        params = GaugingDelta().get_params()
        assert params["config"] is None
        assert params["proximity"] is None
        assert params["continuity"] is None
        assert params["linkage"] is None
        assert params["n_clusters"] is None
        assert params["mode"] == "full"
        assert params["metric"] == "euclidean"
        assert params["preserve_labels"] is False

    def test_set_params_roundtrip(self) -> None:
        gd = GaugingDelta()
        gd.set_params(n_clusters=5, mode="lite")
        assert gd.get_params()["n_clusters"] == 5
        assert gd.get_params()["mode"] == "lite"

    def test_clone(self) -> None:
        from sklearn.base import clone

        original = GaugingDelta(n_clusters=3, mode="lite")
        cloned = clone(original)
        assert cloned.get_params() == original.get_params()
        assert cloned is not original

    def test_repr(self) -> None:
        r = repr(GaugingDelta(n_clusters=3))
        assert "GaugingDelta" in r
        assert "n_clusters=3" in r

    def test_fit_predict_inherited(self) -> None:
        """fit_predict comes from ClusterMixin, not a custom method."""
        labels = GaugingDelta().fit_predict(self.X)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(self.X)

    def test_fit_accepts_y(self) -> None:
        """fit(X, y=...) must accept and ignore y."""
        gd = GaugingDelta().fit(self.X, y=np.zeros(len(self.X)))
        assert hasattr(gd, "labels_")

    def test_pipeline(self) -> None:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([("scaler", StandardScaler()), ("cluster", GaugingDelta())])
        labels = pipe.fit_predict(self.X)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(self.X)


# ---------------------------------------------------------------------------
# Input validation + sklearn tags
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for validate_data and __sklearn_tags__."""

    X = np.array([
        [0.0, 0.0], [0.5, 0.3], [0.2, 0.1],
        [8.0, 8.0], [8.5, 8.3], [8.2, 8.1],
    ])

    def test_nan_rejected(self) -> None:
        X_nan = self.X.copy()
        X_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="Input .* contains NaN"):
            GaugingDelta().fit(X_nan)

    def test_inf_rejected(self) -> None:
        X_inf = self.X.copy()
        X_inf[0, 0] = np.inf
        with pytest.raises(ValueError, match="Input .* contains (infinity|NaN)"):
            GaugingDelta().fit(X_inf)

    def test_n_features_in(self) -> None:
        gd = GaugingDelta().fit(self.X)
        assert gd.n_features_in_ == 2

    def test_n_features_in_precomputed(self) -> None:
        D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        gd = GaugingDelta(metric="precomputed").fit(D)
        assert gd.n_features_in_ == 3

    def test_sklearn_tags_default(self) -> None:
        tags = GaugingDelta().__sklearn_tags__()
        assert tags.input_tags.pairwise is False

    def test_sklearn_tags_precomputed(self) -> None:
        tags = GaugingDelta(metric="precomputed").__sklearn_tags__()
        assert tags.input_tags.pairwise is True


# ---------------------------------------------------------------------------
# Fitted attributes: cluster_centers_
# ---------------------------------------------------------------------------


class TestClusterCenters:
    """Tests for cluster_centers_ attribute."""

    X = np.array([
        [0.0, 0.0], [0.5, 0.3], [0.2, 0.1],
        [8.0, 8.0], [8.5, 8.3], [8.2, 8.1],
    ])

    def test_shape(self) -> None:
        gd = GaugingDelta().fit(self.X)
        assert gd.cluster_centers_.shape == (gd.n_clusters_, 2)

    def test_values_are_means(self) -> None:
        gd = GaugingDelta().fit(self.X)
        for label in range(gd.n_clusters_):
            mask = gd.labels_ == label
            expected = self.X[mask].mean(axis=0)
            np.testing.assert_allclose(gd.cluster_centers_[label], expected)

    def test_not_set_for_precomputed(self) -> None:
        D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        gd = GaugingDelta(metric="precomputed").fit(D)
        assert not hasattr(gd, "cluster_centers_")


# ---------------------------------------------------------------------------
# Hierarchical outputs
# ---------------------------------------------------------------------------


class TestHierarchicalOutputs:
    """Tests for children_, distances_, n_leaves_, linkage_matrix_."""

    X = np.array([
        [0.0, 0.0], [0.5, 0.3], [0.2, 0.1],
        [8.0, 8.0], [8.5, 8.3], [8.2, 8.1],
    ])

    def test_children_shape(self) -> None:
        gd = GaugingDelta().fit(self.X)
        n = len(self.X)
        assert gd.children_.shape == (n - 1, 2)

    def test_distances_shape(self) -> None:
        gd = GaugingDelta().fit(self.X)
        n = len(self.X)
        assert gd.distances_.shape == (n - 1,)

    def test_distances_nonnegative(self) -> None:
        gd = GaugingDelta().fit(self.X)
        assert np.all(gd.distances_ >= 0)

    def test_n_leaves(self) -> None:
        gd = GaugingDelta().fit(self.X)
        assert gd.n_leaves_ == len(self.X)

    def test_linkage_matrix_shape(self) -> None:
        gd = GaugingDelta().fit(self.X)
        Z = gd.linkage_matrix_
        assert Z.shape == (len(self.X) - 1, 4)

    def test_linkage_valid(self) -> None:
        from scipy.cluster.hierarchy import is_valid_linkage

        gd = GaugingDelta().fit(self.X)
        assert is_valid_linkage(gd.linkage_matrix_)

    def test_dendrogram_compatible(self) -> None:
        from scipy.cluster.hierarchy import dendrogram

        gd = GaugingDelta().fit(self.X)
        result = dendrogram(gd.linkage_matrix_, no_plot=True)
        assert "ivl" in result

    def test_children_indices_valid(self) -> None:
        gd = GaugingDelta().fit(self.X)
        n = gd.n_leaves_
        n_merges = len(gd.children_)
        assert np.all(gd.children_ >= 0)
        assert np.all(gd.children_ < n + n_merges)

    def test_linkage_matrix_unfitted(self) -> None:
        from sklearn.exceptions import NotFittedError

        gd = GaugingDelta()
        with pytest.raises(NotFittedError):
            _ = gd.linkage_matrix_

    def test_lite_mode(self) -> None:
        """Hierarchical outputs work in lite mode too."""
        from scipy.cluster.hierarchy import is_valid_linkage

        gd = GaugingDelta(mode="lite").fit(self.X)
        assert is_valid_linkage(gd.linkage_matrix_)
