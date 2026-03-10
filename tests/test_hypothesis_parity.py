"""Hypothesis property-based parity tests: GaugingDelta vs legacy Perception.

Generates diverse random datasets (~800 points) and verifies ARI=1.000
between implementations. Targets numerical hazards identified in the
stress test suite.

Usage:
    uv run pytest tests/test_hypothesis_parity.py -v
    uv run pytest tests/test_hypothesis_parity.py -v --hypothesis-seed=12345
    uv run pytest tests/test_hypothesis_parity.py::test_parity_well_separated_blobs -v
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, note, settings
from hypothesis import strategies as st
from sklearn.metrics import adjusted_rand_score

from gauging_delta import GaugingDelta


# ---------------------------------------------------------------------------
# Legacy import setup (matches tests/conftest.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent / "legacy"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_legacy(X: np.ndarray) -> np.ndarray:
    """Run Perception.fit and extract raw cluster labels."""
    from perception import Perception

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p = Perception(k=None)
            p.fit(X)
    finally:
        sys.stdout = old_stdout

    labels = np.full(len(X), -1, dtype=int)
    for cid, cl in p.initial_clusters.items():
        for pt in cl["data"]:
            labels[pt] = cid
    return labels


def _run_new(X: np.ndarray) -> np.ndarray:
    """Run GaugingDelta full mode and return labels."""
    gd = GaugingDelta(preserve_labels=True)
    gd.fit(X)
    return gd.labels_


def _assert_parity(X: np.ndarray, label: str) -> None:
    """Run both implementations and assert ARI=1.000."""
    try:
        legacy_labels = _run_legacy(X.copy())
    except Exception as exc:
        note(f"Legacy crashed: {exc}")
        assume(False)
        return

    new_labels = _run_new(X.copy())

    n_legacy = len(set(legacy_labels))
    n_new = len(set(new_labels))
    ari = adjusted_rand_score(legacy_labels, new_labels)

    note(f"generator={label}")
    note(f"shape={X.shape}")
    note(f"n_clusters: legacy={n_legacy}, new={n_new}")
    note(f"ARI={ari:.6f}")
    note(f"data range: min={X.min():.6g}, max={X.max():.6g}, mean={X.mean():.6g}")

    assert ari == pytest.approx(1.0, abs=1e-9), (
        f"ARI={ari:.6f} (legacy={n_legacy} clusters, new={n_new} clusters)"
    )


# ---------------------------------------------------------------------------
# Composite strategies
# ---------------------------------------------------------------------------

COMMON_SETTINGS = dict(
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@st.composite
def well_separated_blobs(draw):
    """Control group: trivially separable blobs."""
    n_clusters = draw(st.integers(min_value=2, max_value=5))
    n_per = draw(st.integers(min_value=100, max_value=800 // n_clusters))
    n_dims = draw(st.sampled_from([2, 3]))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)

    parts = []
    for _ in range(n_clusters):
        center = rng.uniform(-20, 20, size=n_dims)
        pts = rng.normal(loc=center, scale=0.5, size=(n_per, n_dims))
        parts.append(pts)
    return np.vstack(parts)


@st.composite
def near_zero_separation(draw):
    """Clusters with near-zero separation — proximity denominator hazard."""
    n_clusters = draw(st.integers(min_value=2, max_value=4))
    n_per = draw(st.integers(min_value=100, max_value=800 // n_clusters))
    separation = draw(st.sampled_from([1e-4, 1e-5, 1e-6, 1e-8]))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)

    parts = []
    for i in range(n_clusters):
        center = np.zeros(2)
        center[0] = i * separation
        pts = rng.normal(loc=center, scale=separation * 0.1, size=(n_per, 2))
        parts.append(pts)
    return np.vstack(parts)


@st.composite
def duplicate_points(draw):
    """Dataset with exact duplicate points — zero distance hazard."""
    n_clusters = draw(st.integers(min_value=2, max_value=4))
    dup_ratio = draw(st.floats(min_value=0.1, max_value=0.5))
    n_total = draw(st.integers(min_value=200, max_value=800))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)

    n_unique = int(n_total * (1 - dup_ratio))
    n_dup = n_total - n_unique

    parts = []
    for _ in range(n_clusters):
        center = rng.uniform(-10, 10, size=2)
        n_per = n_unique // n_clusters
        pts = rng.normal(loc=center, scale=1.0, size=(n_per, 2))
        parts.append(pts)

    unique = np.vstack(parts)
    dup_idx = rng.choice(len(unique), size=n_dup, replace=True)
    return np.vstack([unique, unique[dup_idx]])


@st.composite
def extreme_scale(draw):
    """Data at extreme scales — float precision loss hazard."""
    n_clusters = draw(st.integers(min_value=2, max_value=4))
    n_per = draw(st.integers(min_value=100, max_value=800 // n_clusters))
    scale = draw(st.sampled_from([1e-8, 1e-6, 1e6, 1e8]))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)

    parts = []
    for i in range(n_clusters):
        center = np.full(2, i * 5.0) * scale
        pts = rng.normal(loc=center, scale=scale * 0.5, size=(n_per, 2))
        parts.append(pts)
    return np.vstack(parts)


@st.composite
def collinear_clusters(draw):
    """Clusters along a line — degenerate angle hazard."""
    n_clusters = draw(st.integers(min_value=2, max_value=5))
    n_per = draw(st.integers(min_value=100, max_value=800 // n_clusters))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)

    parts = []
    for i in range(n_clusters):
        center = np.zeros(2)
        center[0] = i * 5.0
        pts = rng.normal(loc=center, scale=0.3, size=(n_per, 2))
        parts.append(pts)
    return np.vstack(parts)


@st.composite
def singleton_merges(draw):
    """Mostly singletons with a few clusters — empty merge history hazard."""
    n_clusters = draw(st.integers(min_value=2, max_value=4))
    n_total = draw(st.integers(min_value=200, max_value=800))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)

    n_clustered = n_total // 2
    n_scattered = n_total - n_clustered

    parts = []
    for _ in range(n_clusters):
        center = rng.uniform(-5, 5, size=2)
        n_per = n_clustered // n_clusters
        pts = rng.normal(loc=center, scale=0.5, size=(n_per, 2))
        parts.append(pts)

    scattered = rng.uniform(-20, 20, size=(n_scattered, 2))
    parts.append(scattered)
    return np.vstack(parts)


@st.composite
def unequal_mass(draw):
    """Highly unequal cluster sizes — mass smoothness hazard."""
    large_ratio = draw(st.floats(min_value=0.90, max_value=0.99))
    n_total = draw(st.integers(min_value=200, max_value=800))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)

    n_large = int(n_total * large_ratio)
    n_small = max(n_total - n_large, 2)

    large = rng.normal(loc=[0, 0], scale=1.0, size=(n_large, 2))
    small = rng.normal(loc=[8, 8], scale=0.3, size=(n_small, 2))
    return np.vstack([large, small])


@st.composite
def high_dimensional(draw):
    """Higher-dimensional data — angle computation hazard."""
    n_clusters = draw(st.integers(min_value=2, max_value=4))
    n_per = draw(st.integers(min_value=50, max_value=200))
    n_dims = draw(st.sampled_from([3, 5]))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)

    parts = []
    for _ in range(n_clusters):
        center = rng.uniform(-5, 5, size=n_dims)
        pts = rng.normal(loc=center, scale=1.0, size=(n_per, n_dims))
        parts.append(pts)
    return np.vstack(parts)


@st.composite
def spirals(draw):
    """Non-convex spiral shapes — the shape that broke during rewrite."""
    n_spirals = draw(st.integers(min_value=2, max_value=3))
    n_per = draw(st.integers(min_value=100, max_value=800 // n_spirals))
    noise = draw(st.floats(min_value=0.05, max_value=0.25))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.RandomState(seed)

    parts = []
    for k in range(n_spirals):
        theta = np.linspace(0, 4 * np.pi, n_per) + k * 2 * np.pi / n_spirals
        r = np.linspace(0.5, 5, n_per)
        x = r * np.cos(theta) + rng.normal(0, noise, n_per)
        y = r * np.sin(theta) + rng.normal(0, noise, n_per)
        parts.append(np.column_stack([x, y]))
    return np.vstack(parts)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


@pytest.mark.slow
@settings(max_examples=15, **COMMON_SETTINGS)
@given(data=well_separated_blobs())
def test_parity_well_separated_blobs(data):
    """Control group: well-separated blobs should always match."""
    _assert_parity(data, "well_separated_blobs")


@pytest.mark.slow
@settings(max_examples=15, **COMMON_SETTINGS)
@given(data=near_zero_separation())
def test_parity_near_zero_separation(data):
    """Near-zero cluster separation — proximity denominator hazard."""
    _assert_parity(data, "near_zero_separation")


@pytest.mark.slow
@settings(max_examples=15, **COMMON_SETTINGS)
@given(data=duplicate_points())
def test_parity_duplicate_points(data):
    """Exact duplicate points — zero distance hazard."""
    _assert_parity(data, "duplicate_points")


@pytest.mark.slow
@settings(max_examples=15, **COMMON_SETTINGS)
@given(data=extreme_scale())
def test_parity_extreme_scale(data):
    """Extreme data scales — float precision loss hazard."""
    _assert_parity(data, "extreme_scale")


@pytest.mark.slow
@settings(max_examples=15, **COMMON_SETTINGS)
@given(data=collinear_clusters())
def test_parity_collinear_clusters(data):
    """Collinear clusters — degenerate angle hazard."""
    _assert_parity(data, "collinear_clusters")


@pytest.mark.slow
@settings(max_examples=15, **COMMON_SETTINGS)
@given(data=singleton_merges())
def test_parity_singleton_merges(data):
    """Scattered singletons — empty merge history hazard."""
    _assert_parity(data, "singleton_merges")


@pytest.mark.slow
@settings(max_examples=15, **COMMON_SETTINGS)
@given(data=unequal_mass())
def test_parity_unequal_mass(data):
    """Unequal cluster sizes — mass smoothness hazard."""
    _assert_parity(data, "unequal_mass")


@pytest.mark.slow
@settings(max_examples=10, **COMMON_SETTINGS)
@given(data=high_dimensional())
def test_parity_high_dimensional(data):
    """Higher dimensions (3D/5D) — angle computation hazard."""
    _assert_parity(data, "high_dimensional")


@pytest.mark.slow
@settings(max_examples=15, **COMMON_SETTINGS)
@given(data=spirals())
def test_parity_spirals(data):
    """Non-convex spirals — shape that broke during rewrite."""
    _assert_parity(data, "spirals")


@pytest.mark.slow
@settings(max_examples=20, **COMMON_SETTINGS)
@given(
    data=st.one_of(
        well_separated_blobs(),
        near_zero_separation(),
        duplicate_points(),
        extreme_scale(),
        collinear_clusters(),
        singleton_merges(),
        unequal_mass(),
        high_dimensional(),
        spirals(),
    )
)
def test_parity_any_hazard(data):
    """Random mix of all hazard generators."""
    _assert_parity(data, "any_hazard")
