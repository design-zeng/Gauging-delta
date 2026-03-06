"""Parity tests: verify GaugingDelta matches legacy Perception exactly.

Three categories:
1. Dataset parity — ARI=1.000 on all 6 benchmark datasets
2. Seed parity — ARI=1.000 on 10 random blob datasets
3. Intermediate parity — merge sequence matches step-by-step on flame + 3_blobs
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from gauging_delta import GaugingDelta


# ---------------------------------------------------------------------------
# 1. Dataset parity
# ---------------------------------------------------------------------------

DATASET_NAMES = ["flame", "3_blobs", "pathbased", "3-spiral", "jain", "compound"]


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_dataset_parity(
    name: str,
    datasets: dict,
    legacy_labels: dict,
) -> None:
    """GaugingDelta must achieve ARI=1.000 vs legacy on each benchmark dataset."""
    X, _ = datasets[name]
    gd = GaugingDelta(preserve_labels=True)
    gd.fit(X)
    ari = adjusted_rand_score(legacy_labels[name], gd.labels_)
    assert ari == pytest.approx(1.0, abs=1e-9), (
        f"{name}: ARI={ari:.6f}, clusters: legacy="
        f"{len(set(legacy_labels[name]))}, new={gd.n_clusters_}"
    )


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_cluster_count_matches(
    name: str,
    datasets: dict,
    legacy_labels: dict,
) -> None:
    """Cluster count must match legacy exactly."""
    X, _ = datasets[name]
    gd = GaugingDelta()
    gd.fit(X)
    n_legacy = len(set(legacy_labels[name]))
    assert gd.n_clusters_ == n_legacy, f"{name}: legacy={n_legacy}, new={gd.n_clusters_}"


# ---------------------------------------------------------------------------
# 2. Seed parity — random blobs
# ---------------------------------------------------------------------------

BLOB_SEEDS = [42, 123, 256, 512, 777, 1024, 2048, 4096, 7777, 9999]


@pytest.mark.parametrize("seed", BLOB_SEEDS)
@pytest.mark.slow
def test_seed_parity(seed: int) -> None:
    """ARI=1.000 on random 3-blob datasets with well-separated clusters."""
    rng = np.random.RandomState(seed)
    centers = np.array([[0, 0], [8, 0], [4, 7]], dtype=float)
    n_per_cluster = 25
    X_parts, y_parts = [], []
    for i, c in enumerate(centers):
        pts = rng.randn(n_per_cluster, 2) * 0.8 + c
        X_parts.append(pts)
        y_parts.append(np.full(n_per_cluster, i))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Shuffle deterministically
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    # Legacy
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from perception import Perception

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        p = Perception(k=None)
        p.fit(X)
    finally:
        sys.stdout = old_stdout

    legacy_labels = np.full(len(X), -1, dtype=int)
    for cid, cl in p.initial_clusters.items():
        for pt in cl["data"]:
            legacy_labels[pt] = cid

    # New
    gd = GaugingDelta(preserve_labels=True)
    gd.fit(X)
    ari = adjusted_rand_score(legacy_labels, gd.labels_)
    assert ari == pytest.approx(1.0, abs=1e-9), f"seed={seed}: ARI={ari:.6f}"


# ---------------------------------------------------------------------------
# 3. Intermediate parity — merge sequence matches step-by-step
# ---------------------------------------------------------------------------

INTERMEDIATE_DATASETS = ["flame", "3_blobs"]


@pytest.mark.parametrize("name", INTERMEDIATE_DATASETS)
def test_merge_sequence_parity(
    name: str,
    datasets: dict,
) -> None:
    """Merge sequence (lead, child, distance) must match legacy exactly."""
    X, _ = datasets[name]

    # --- Legacy merge log ---
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from perception import Perception

    legacy_merges: list[tuple[int, int, float]] = []
    _orig_vg = Perception.vision_generic

    def _log_vg(self, c1, c2):
        result = _orig_vg(self, c1, c2)
        is_merge, lead, child = result
        if is_merge:
            d = self.clusters_dist[frozenset((c1, c2))]["distance_info"]["near_dist"]["distance"]
            legacy_merges.append((int(lead), int(child), float(d)))
        return result

    Perception.vision_generic = _log_vg
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        p = Perception(k=None)
        p.fit(X)
    finally:
        sys.stdout = old_stdout
    # Restore
    Perception.vision_generic = _orig_vg

    # --- New merge log ---
    new_merges: list[tuple[int, int, float]] = []
    _orig_tm = GaugingDelta._try_merge

    def _log_tm(self, c1, c2):
        d = float(self._dist_matrix[c1, c2])  # capture before merge invalidates child row
        result = _orig_tm(self, c1, c2)
        if result is not None:
            lead_id = result
            child_id = c2 if lead_id == c1 else c1
            new_merges.append((lead_id, child_id, d))
        return result

    GaugingDelta._try_merge = _log_tm
    gd = GaugingDelta()
    gd.fit(X)
    GaugingDelta._try_merge = _orig_tm

    # --- Compare ---
    assert len(legacy_merges) == len(new_merges), (
        f"{name}: legacy={len(legacy_merges)} merges, new={len(new_merges)}"
    )
    for i, (lm, nm) in enumerate(zip(legacy_merges, new_merges, strict=True)):
        assert lm[0] == nm[0] and lm[1] == nm[1], (
            f"{name} merge #{i}: legacy=(lead={lm[0]},child={lm[1]}), "
            f"new=(lead={nm[0]},child={nm[1]})"
        )
        assert lm[2] == pytest.approx(nm[2], abs=1e-12), (
            f"{name} merge #{i}: legacy d={lm[2]}, new d={nm[2]}"
        )


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------


def test_two_points() -> None:
    """Two points should produce one cluster."""
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    gd = GaugingDelta()
    gd.fit(X)
    assert gd.n_clusters_ == 1
    assert len(gd.labels_) == 2
    assert gd.labels_[0] == gd.labels_[1]


def test_three_collinear() -> None:
    """Three collinear points should produce one cluster."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    gd = GaugingDelta()
    gd.fit(X)
    assert gd.n_clusters_ == 1


def test_duplicate_points() -> None:
    """Duplicate points should not crash (nan propagation)."""
    X = np.array([[1.0, 1.0], [1.0, 1.0], [5.0, 5.0], [5.0, 5.0]])
    gd = GaugingDelta()
    gd.fit(X)
    assert gd.n_clusters_ >= 1
    assert len(gd.labels_) == 4


def test_n_clusters_target() -> None:
    """n_clusters parameter should force the target cluster count."""
    rng = np.random.RandomState(42)
    X = np.vstack([rng.randn(20, 2) + [0, 0], rng.randn(20, 2) + [10, 10]])
    gd = GaugingDelta(n_clusters=2)
    gd.fit(X)
    assert gd.n_clusters_ == 2


def test_fit_predict_returns_labels() -> None:
    """fit_predict should return the same labels as fit().labels_."""
    rng = np.random.RandomState(42)
    X = np.vstack([rng.randn(15, 2), rng.randn(15, 2) + [8, 0]])
    labels_fp = GaugingDelta().fit_predict(X)
    labels_f = GaugingDelta().fit(X).labels_
    np.testing.assert_array_equal(labels_fp, labels_f)


def test_preserve_labels_flag() -> None:
    """preserve_labels=True should keep original cluster IDs."""
    rng = np.random.RandomState(42)
    X = np.vstack([rng.randn(15, 2), rng.randn(15, 2) + [8, 0]])
    gd_norm = GaugingDelta(preserve_labels=False).fit(X)
    gd_pres = GaugingDelta(preserve_labels=True).fit(X)
    # Both should have same cluster count
    assert gd_norm.n_clusters_ == gd_pres.n_clusters_
    # Normal labels should be 0..n-1
    assert set(gd_norm.labels_) == set(range(gd_norm.n_clusters_))
    # ARI between them should be 1.0
    ari = adjusted_rand_score(gd_norm.labels_, gd_pres.labels_)
    assert ari == pytest.approx(1.0)
