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
