"""
Intermediate parity tests for GaugingDelta vs legacy Perception.

These tests validate step-by-step mergeability metrics to ensure the
refactored implementation preserves intermediate data parity.
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, Phase, assume, given, settings
from hypothesis import strategies as st


# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Skip all tests if perception.py not found
pytestmark = pytest.mark.skipif(
    not Path(__file__).parent.parent.parent.joinpath("perception.py").exists(),
    reason="perception.py not found",
)

FLOAT_RTOL = 1e-9
FLOAT_ATOL = 1e-9


# =============================================================================
# Hypothesis Strategy
# =============================================================================


@st.composite
def small_random_clusters(draw):
    """Generate small random Gaussian clusters for intermediate parity tests."""
    n_clusters = draw(st.integers(min_value=2, max_value=3))
    n_points = draw(st.integers(min_value=10, max_value=20))
    separation = draw(
        st.floats(min_value=3.0, max_value=8.0, allow_nan=False, allow_infinity=False)
    )
    spread = draw(
        st.floats(min_value=0.1, max_value=0.6, allow_nan=False, allow_infinity=False)
    )
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))

    rng = np.random.default_rng(seed)

    counts = [n_points // n_clusters] * n_clusters
    remainder = n_points % n_clusters
    counts = [count + (1 if i < remainder else 0) for i, count in enumerate(counts)]

    points = []
    for i, count in enumerate(counts):
        center = np.array([separation * (i + 1), 0.0])
        cluster_points = rng.normal(loc=center, scale=spread, size=(count, 2))
        points.append(cluster_points)

    X = np.vstack(points)
    rng.shuffle(X)
    return X


@st.composite
def wide_range_random_clusters(draw):
    """Generate diverse random datasets for broader parity testing."""
    n_clusters = draw(st.integers(min_value=2, max_value=5))
    n_points = draw(st.integers(min_value=15, max_value=80))
    separation = draw(
        st.floats(min_value=2.0, max_value=15.0, allow_nan=False, allow_infinity=False)
    )
    spread = draw(
        st.floats(min_value=0.05, max_value=1.2, allow_nan=False, allow_infinity=False)
    )
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    # Optional 2D rotation for variety
    rotation_deg = draw(st.floats(min_value=0.0, max_value=360.0, allow_nan=False, allow_infinity=False))

    rng = np.random.default_rng(seed)

    counts = [n_points // n_clusters] * n_clusters
    remainder = n_points % n_clusters
    counts = [count + (1 if i < remainder else 0) for i, count in enumerate(counts)]

    points = []
    for i, count in enumerate(counts):
        # Place clusters in varied patterns (linear, L-shape, scattered)
        if n_clusters == 2:
            center = np.array([separation * (i + 1), 0.0])
        elif n_clusters == 3:
            center = np.array([separation * (i + 1), separation * ((i % 2) * 2 - 1)])
        else:
            angle = 2 * np.pi * i / n_clusters
            center = np.array([separation * np.cos(angle), separation * np.sin(angle)])
        cluster_points = rng.normal(loc=center, scale=spread, size=(count, 2))
        points.append(cluster_points)

    X = np.vstack(points)
    rng.shuffle(X)

    # Apply rotation if non-zero
    if rotation_deg != 0:
        theta = np.radians(rotation_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        X = X @ rot.T

    return X


# =============================================================================
# Logging Helpers
# =============================================================================


def _safe_mean_distance(distance: float, proximity: float) -> float:
    if proximity == 0 or np.isnan(proximity) or np.isinf(proximity):
        return distance
    return distance / proximity


def _make_log_entry(
    *,
    pair: tuple[int, int],
    decision: str,
    proximity: float,
    distance: float,
    T1: float,
    T2: float,
    shape_diff: float,
    adp_prox: float,
    continuity: float,
    lead_id: int,
    child_id: int,
    lead_size: int,
    child_size: int,
) -> dict:
    return {
        "pair": pair,
        "decision": decision,
        "prox": float(proximity),
        "d_ij": float(distance),
        "T1": float(T1),
        "T2": float(T2),
        "shape_diff": float(shape_diff),
        "adp_prox": float(adp_prox),
        "continuity": float(continuity),
        "mean_historical": float(_safe_mean_distance(distance, proximity)),
        "lead_id": int(lead_id),
        "child_id": int(child_id),
        "lead_size": int(lead_size),
        "child_size": int(child_size),
    }


def run_legacy_with_logging(X: np.ndarray) -> list[dict]:
    """Run legacy perception.py with intermediate logging."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from perception import Perception

    class LegacyWithLogging(Perception):
        def __init__(self):
            super().__init__()
            self.candidate_log: list[dict] = []

        def vision_generic(self, cluster1, cluster2):
            proximity, distance, lead_cluster, child_cluster = self.compute_proximity(
                cluster1, cluster2
            )
            T1, T2, shape_diff, adp_prox = self.compute_adaptive_threshold(
                self.NUM_NEAREST_CLUSTERS,
                lead_cluster,
                child_cluster,
                distance,
                proximity,
            )

            lead_size = len(self.initial_clusters[lead_cluster]["data"])
            child_size = len(self.initial_clusters[child_cluster]["data"])

            if proximity > T1 or proximity > T2:
                self.candidate_log.append(
                    _make_log_entry(
                        pair=(cluster1, cluster2),
                        decision="reject_proximity",
                        proximity=proximity,
                        distance=distance,
                        T1=T1,
                        T2=T2,
                        shape_diff=shape_diff,
                        adp_prox=adp_prox,
                        continuity=0.0,
                        lead_id=lead_cluster,
                        child_id=child_cluster,
                        lead_size=lead_size,
                        child_size=child_size,
                    )
                )
                self.continuation_data["merged"].append(False)
                return False, cluster1, cluster2

            continuation = self.compute_continuation(
                lead_cluster,
                child_cluster,
                self.THRESHOLD_CONTINUATION / shape_diff,
                distance / proximity,
                adp_prox,
            )

            if continuation > self.THRESHOLD_CONTINUATION / shape_diff:
                self.candidate_log.append(
                    _make_log_entry(
                        pair=(cluster1, cluster2),
                        decision="accept",
                        proximity=proximity,
                        distance=distance,
                        T1=T1,
                        T2=T2,
                        shape_diff=shape_diff,
                        adp_prox=adp_prox,
                        continuity=continuation,
                        lead_id=lead_cluster,
                        child_id=child_cluster,
                        lead_size=lead_size,
                        child_size=child_size,
                    )
                )
                self.initial_clusters[lead_cluster]["past_densities"].append(continuation)
                self.continuation_data["merged"].append(True)
                self.initial_clusters[lead_cluster]["past_dists"].append(distance)
                self.initial_clusters[lead_cluster]["merging_dists"].append(distance)
                return True, lead_cluster, child_cluster

            self.candidate_log.append(
                _make_log_entry(
                    pair=(cluster1, cluster2),
                    decision="reject_continuity",
                    proximity=proximity,
                    distance=distance,
                    T1=T1,
                    T2=T2,
                    shape_diff=shape_diff,
                    adp_prox=adp_prox,
                    continuity=continuation,
                    lead_id=lead_cluster,
                    child_id=child_cluster,
                    lead_size=lead_size,
                    child_size=child_size,
                )
            )
            self.continuation_data["merged"].append(False)
            return False, lead_cluster, child_cluster

    legacy = LegacyWithLogging()
    with redirect_stdout(io.StringIO()):
        legacy.fit(X.copy())
    return legacy.candidate_log


def run_refactored_with_logging(X: np.ndarray) -> list[dict]:
    """Run refactored GaugingDelta with intermediate logging."""
    from gauging_delta import GaugingDelta

    class GDWithLogging(GaugingDelta):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.candidate_log: list[dict] = []

        def _compute_mergeability(self, C_i, C_j, d_ij):
            result = super()._compute_mergeability(C_i, C_j, d_ij)

            if result.is_mergeable:
                decision = "accept"
            elif result.rho > result.T_i or result.rho > result.T_j:
                decision = "reject_proximity"
            else:
                decision = "reject_continuity"

            if result.lead_cluster == C_i.label:
                lead_size = len(C_i)
                child_size = len(C_j)
            else:
                lead_size = len(C_j)
                child_size = len(C_i)

            total_size = lead_size + child_size
            adp_prox = result.T_i * lead_size / total_size + result.T_j * child_size / total_size
            shape_diff = result.xi_s if result.xi_s > 0 else 1.0

            self.candidate_log.append(
                _make_log_entry(
                    pair=(C_i.label, C_j.label),
                    decision=decision,
                    proximity=result.rho,
                    distance=d_ij,
                    T1=result.T_i,
                    T2=result.T_j,
                    shape_diff=shape_diff,
                    adp_prox=adp_prox,
                    continuity=result.continuity,
                    lead_id=result.lead_cluster,
                    child_id=result.child_cluster,
                    lead_size=lead_size,
                    child_size=child_size,
                )
            )
            return result

    gd = GDWithLogging(preserve_labels=True)
    gd.fit(X.copy())
    return gd.candidate_log


# =============================================================================
# Assertions
# =============================================================================


def _assert_float_close(name: str, left: float, right: float, idx: int) -> None:
    if np.isnan(left) and np.isnan(right):
        return
    if np.isinf(left) or np.isinf(right):
        if left == right:
            return
        raise AssertionError(f"[{idx}] {name} mismatch: legacy={left} refactored={right}")
    if not np.isclose(left, right, rtol=FLOAT_RTOL, atol=FLOAT_ATOL):
        raise AssertionError(f"[{idx}] {name} mismatch: legacy={left} refactored={right}")


def assert_candidate_logs_match(legacy_log: list[dict], refactored_log: list[dict]) -> None:
    if len(legacy_log) != len(refactored_log):
        raise AssertionError(
            f"Candidate log length mismatch: legacy={len(legacy_log)} refactored={len(refactored_log)}"
        )

    float_keys = [
        "prox",
        "d_ij",
        "T1",
        "T2",
        "shape_diff",
        "adp_prox",
        "continuity",
        "mean_historical",
    ]

    for idx, (legacy, refactored) in enumerate(zip(legacy_log, refactored_log)):
        if legacy["pair"] != refactored["pair"]:
            raise AssertionError(
                f"[{idx}] pair mismatch: legacy={legacy['pair']} refactored={refactored['pair']}"
            )
        if legacy["decision"] != refactored["decision"]:
            raise AssertionError(
                f"[{idx}] decision mismatch: legacy={legacy['decision']} refactored={refactored['decision']}"
            )
        for key in ["lead_id", "child_id", "lead_size", "child_size"]:
            if legacy[key] != refactored[key]:
                raise AssertionError(
                    f"[{idx}] {key} mismatch: legacy={legacy[key]} refactored={refactored[key]}"
                )
        for key in float_keys:
            _assert_float_close(key, legacy[key], refactored[key], idx)


# =============================================================================
# Tests
# =============================================================================


class TestIntermediateParity:
    """Parity tests for intermediate mergeability metrics."""

    @given(X=small_random_clusters())
    @settings(
        max_examples=3,
        deadline=120000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        phases=[Phase.generate, Phase.target, Phase.shrink],
    )
    def test_intermediate_metrics_match(self, X):
        assume(len(X) >= 10)
        legacy_log = run_legacy_with_logging(X)
        refactored_log = run_refactored_with_logging(X)
        assert_candidate_logs_match(legacy_log, refactored_log)

    @given(X=wide_range_random_clusters())
    @settings(
        max_examples=5,
        deadline=180000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        phases=[Phase.generate, Phase.target, Phase.shrink],
    )
    def test_intermediate_metrics_match_wide_range(self, X):
        assume(len(X) >= 15)
        legacy_log = run_legacy_with_logging(X)
        refactored_log = run_refactored_with_logging(X)
        assert_candidate_logs_match(legacy_log, refactored_log)

    @given(X=small_random_clusters())
    @settings(
        max_examples=1000,
        deadline=600000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        phases=[Phase.generate, Phase.target, Phase.shrink],
    )
    def test_intermediate_metrics_match_1000_examples(self, X):
        assume(len(X) >= 10)
        legacy_log = run_legacy_with_logging(X)
        refactored_log = run_refactored_with_logging(X)
        assert_candidate_logs_match(legacy_log, refactored_log)
