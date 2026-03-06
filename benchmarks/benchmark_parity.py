"""
Unified parity benchmark: per-stage timing/memory + binary-search parity verification.

Four levels of granularity (binary search for parity):

  Level 1 — holistic:  End-to-end ARI + total time + peak memory
  Level 2 — stage:     Per-stage timing/memory + cluster state comparison
  Level 3 — component: Per-merge-decision component parity (rho, T_i, T_j, xi_s, smoothness)
  Level 4 — decision:  Full dump of a specific merge decision (identified by level 3)

Usage:
    uv run python benchmarks/benchmark_parity.py
    uv run python benchmarks/benchmark_parity.py --level stage --sizes 100 500
    uv run python benchmarks/benchmark_parity.py --level component --sizes 500
    uv run python benchmarks/benchmark_parity.py --level decision --sizes 500 --decision 58
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import sys
import time
import tracemalloc
import warnings
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"
ASSETS_DIR = ROOT / "assets"
sys.path.insert(0, str(ROOT))

from benchmarks._plotting import use_science_style  # noqa: E402, I001

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
BENCHMARK_DATASETS = {
    "flame": DATA_DIR / "flame.txt",
    "3_blobs": DATA_DIR / "3_blobs.txt",
    "pathbased": DATA_DIR / "pathbased.txt",
    "3-spiral": DATA_DIR / "3-spiral.txt",
    "jain": DATA_DIR / "jain.txt",
    "compound": DATA_DIR / "compound.txt",
}


def load_dataset(path: Path) -> np.ndarray:
    data = np.loadtxt(str(path), delimiter=",")
    return data[:, :2]


def make_blobs(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-20, 20, size=(5, 2))
    parts = []
    per = n // 5
    for i, c in enumerate(centers):
        count = per + (1 if i < n - per * 5 else 0)
        parts.append(rng.normal(loc=c, scale=1.0, size=(count, 2)))
    return np.vstack(parts)


# ---------------------------------------------------------------------------
# Merge decision log (for component-level parity)
# ---------------------------------------------------------------------------


@dataclass
class MergeDecisionLog:
    """Captured intermediate values for a single merge attempt."""

    step: int
    c1: int
    c2: int
    d_ij: float
    rho: float
    lead_id: int
    child_id: int
    T_i: float
    T_j: float
    xi_s: float
    adp_prox: float
    smoothness: float | None  # None if rejected before continuity
    merged: bool


@dataclass
class StageResult:
    """Timing and memory for a single stage."""

    name: str
    time_s: float
    peak_bytes: float


@dataclass
class Divergence:
    """Documented non-parity between legacy and new."""

    dataset: str
    level: str
    merge_step: int | None = None
    divergent_field: str = ""
    legacy_value: Any = None
    new_value: Any = None
    delta: float | None = None
    context: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Legacy runner (with optional instrumentation)
# ---------------------------------------------------------------------------


def _run_legacy_holistic(X: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Run legacy Perception, return (labels, time_s, peak_bytes)."""
    from perception import Perception

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    p = Perception(k=None)
    with redirect_stdout(io.StringIO()):
        labels_raw, _ = p.fit(X.copy())

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return np.asarray(labels_raw, dtype=int), elapsed, float(peak)


def _run_new_holistic(X: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Run GaugingDelta, return (labels, time_s, peak_bytes)."""
    from gauging_delta import GaugingDelta

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    model = GaugingDelta(preserve_labels=True)
    model.fit(X.copy())

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return np.asarray(model.labels_, dtype=int), elapsed, float(peak)


# ---------------------------------------------------------------------------
# Stage-level instrumentation
# ---------------------------------------------------------------------------


def _run_legacy_staged(X: np.ndarray) -> tuple[np.ndarray, list[StageResult]]:
    """Run legacy with per-stage timing. Returns (labels, stages)."""
    from perception import Perception

    stages: list[StageResult] = []
    p = Perception(k=None)
    p.X = X.copy()
    p.DIMENSION = len(X[0])
    p.X_RANGE = min(X[:, 0]), max(X[:, 0])
    p.Y_RANGE = min(X[:, 1]), max(X[:, 1])

    # Stage 1: Init clusters
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    p.DIST_MATRIX = np.full((len(X), len(X)), np.inf)
    p.FAIL_MATRIX = np.empty((len(X), len(X), 2))
    p.initial_clusters = {
        id: {
            "label": id,
            "data": [id],
            "center": x,
            "mean_dist": 0,
            "std_dist": 0,
            "past_dists": [],
            "merging_dists": [],
            "past_densities": [],
            "past_std": [],
            "traces": [],
        }
        for id, x in enumerate(p.X)
    }
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stages.append(StageResult("init_clusters", t1 - t0, float(peak)))

    # Stage 2: Distance matrix
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        p.clusters_dist, p.points_dist = p.initiate_dists()
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stages.append(StageResult("distance_matrix", t1 - t0, float(peak)))

    # Stages 3-6: Merge loop (timed as a block)
    gc.collect()
    tracemalloc.start()
    t_loop_start = time.perf_counter()
    t_sorted = 0.0
    t_merge_check = 0.0
    t_merge_exec = 0.0

    early_stop = False
    with redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        while len(p.clusters_dist):
            pre_length = len(p.initial_clusters)

            t0 = time.perf_counter()
            k_nearest = p.get_indices_of_k_smallest(
                k=X.shape[0], matrix=p.DIST_MATRIX, sorted=True
            )
            t_sorted += time.perf_counter() - t0

            last_merged = None
            i = 0
            while i < len(k_nearest[0]):
                c1, c2 = k_nearest[:, i]

                if p.k is not None:
                    early_stop = p.k == len(p.initial_clusters)
                    if early_stop:
                        break

                if ~np.isinf(p.DIST_MATRIX[c1][c2]):
                    if last_merged is not None:
                        _c = p.get_indices_of_k_smallest(
                            1, p.DIST_MATRIX[last_merged, :]
                        )[:, 0][0]
                        if p.DIST_MATRIX[last_merged][_c] < p.DIST_MATRIX[c1][c2]:
                            c1, c2 = last_merged, _c
                        else:
                            i += 1
                        last_merged = None
                    else:
                        i += 1

                    t0 = time.perf_counter()
                    is_complete, merged = p.merge_clusters(c1, c2)
                    dt = time.perf_counter() - t0
                    if is_complete:
                        t_merge_exec += dt
                        last_merged = merged
                    else:
                        t_merge_check += dt
                        i += 1
                else:
                    i += 1

            if pre_length == len(p.initial_clusters) or early_stop:
                break

    t_loop_end = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stages.append(StageResult("sorted_pairs", t_sorted, 0.0))
    stages.append(
        StageResult(
            "merge_loop_total", t_loop_end - t_loop_start, float(peak)
        )
    )
    stages.append(StageResult("mergeability_check", t_merge_check, 0.0))
    stages.append(StageResult("merge_execution", t_merge_exec, 0.0))

    # Stage 7: Post-processing (no-op when k=None)
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    # skip post_processing since k=None
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stages.append(StageResult("post_processing", t1 - t0, float(peak)))

    # Stage 8: Label building
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    labels = np.full(len(X), -1, dtype=int)
    for cid, cl in p.initial_clusters.items():
        for pt in cl["data"]:
            labels[pt] = cid
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stages.append(StageResult("label_building", t1 - t0, float(peak)))

    return labels, stages


def _run_new_staged(X: np.ndarray) -> tuple[np.ndarray, list[StageResult]]:
    """Run GaugingDelta with per-stage timing. Returns (labels, stages)."""
    from gauging_delta import GaugingDelta

    stages: list[StageResult] = []
    model = GaugingDelta(preserve_labels=True)
    model._X = np.asarray(X.copy(), dtype=float)
    n = len(model._X)

    # Stage 1: Init clusters
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    from gauging_delta.cluster import Cluster

    model._clusters = {
        i: Cluster(label=i, point_indices=[i], center=model._X[i].copy())
        for i in range(n)
    }
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stages.append(StageResult("init_clusters", t1 - t0, float(peak)))

    # Stage 2: Distance matrix
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    model._dist_matrix = np.full((n, n), np.inf)
    model._near_ref = np.empty((n, n), dtype=int)
    model._point_dists = {}
    model._init_distances()
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stages.append(StageResult("distance_matrix", t1 - t0, float(peak)))

    # Stages 3-6: Merge loop
    gc.collect()
    tracemalloc.start()
    t_loop_start = time.perf_counter()
    t_sorted = 0.0
    t_merge_check = 0.0
    t_merge_exec = 0.0

    early_stop = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        while len(model._clusters) > 1:
            pre_length = len(model._clusters)

            t0 = time.perf_counter()
            k_nearest = model._get_sorted_pairs()
            t_sorted += time.perf_counter() - t0

            last_merged = None
            i = 0
            while i < k_nearest.shape[1]:
                c1, c2 = int(k_nearest[0, i]), int(k_nearest[1, i])

                if model.n_clusters is not None and len(model._clusters) == model.n_clusters:
                    early_stop = True
                    break

                if np.isinf(model._dist_matrix[c1, c2]):
                    i += 1
                    continue

                if last_merged is not None:
                    _c = model._get_nearest_cluster(last_merged)
                    if (
                        _c is not None
                        and model._dist_matrix[last_merged, _c]
                        < model._dist_matrix[c1, c2]
                    ):
                        c1, c2 = last_merged, _c
                    else:
                        i += 1
                    last_merged = None
                else:
                    i += 1

                t0 = time.perf_counter()
                merged_id = model._try_merge(c1, c2)
                dt = time.perf_counter() - t0
                if merged_id is not None:
                    t_merge_exec += dt
                    last_merged = merged_id
                else:
                    t_merge_check += dt

            if pre_length == len(model._clusters) or early_stop:
                break

    t_loop_end = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stages.append(StageResult("sorted_pairs", t_sorted, 0.0))
    stages.append(
        StageResult("merge_loop_total", t_loop_end - t_loop_start, float(peak))
    )
    stages.append(StageResult("mergeability_check", t_merge_check, 0.0))
    stages.append(StageResult("merge_execution", t_merge_exec, 0.0))

    # Stage 7: Post-processing
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    if model.n_clusters is not None and model.n_clusters < len(model._clusters):
        model._post_processing()
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stages.append(StageResult("post_processing", t1 - t0, float(peak)))

    # Stage 8: Label building
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    model._build_labels()
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stages.append(StageResult("label_building", t1 - t0, float(peak)))

    return np.asarray(model.labels_, dtype=int), stages


# ---------------------------------------------------------------------------
# Component-level instrumentation (monkey-patching)
# ---------------------------------------------------------------------------


def _run_legacy_component(X: np.ndarray) -> tuple[np.ndarray, list[MergeDecisionLog]]:
    """Run legacy with per-merge-decision component logging.

    Uses observe-only patches: monkey-patches leaf methods to capture their
    return values as they're called naturally during a single execution of
    vision_generic.  No double-computation.
    """
    from perception import Perception

    logs: list[MergeDecisionLog] = []
    step_counter = [0]

    # Closure lists for captured values
    captured_prox: list[tuple] = []
    captured_thresh: list[tuple] = []

    # --- Observe-only patches on leaf methods ---
    _orig_prox = Perception.compute_proximity
    _orig_thresh = Perception.compute_adaptive_threshold
    _orig_vg = Perception.vision_generic

    def _patched_prox(self, c1, c2):
        result = _orig_prox(self, c1, c2)
        captured_prox.clear()
        captured_prox.append(result)  # (rho, distance, lead_id, child_id)
        return result

    def _patched_thresh(self, k, lead_id, child_id, d_ij, rho):
        result = _orig_thresh(self, k, lead_id, child_id, d_ij, rho)
        captured_thresh.clear()
        captured_thresh.append(result)  # (T_i, T_j, xi_s, adp_prox)
        return result

    def _patched_vg(self, c1, c2):
        d_ij = self.clusters_dist[frozenset((c1, c2))]["distance_info"][
            "near_dist"
        ]["distance"]

        result = _orig_vg(self, c1, c2)
        is_merge = result[0]

        rho, _, lead_id, child_id = captured_prox[0] if captured_prox else (float("nan"), 0, c1, c2)
        T_i, T_j, xi_s, adp_prox = captured_thresh[0] if captured_thresh else (0, 0, 1, 0)

        logs.append(
            MergeDecisionLog(
                step=step_counter[0],
                c1=c1,
                c2=c2,
                d_ij=d_ij,
                rho=rho,
                lead_id=lead_id,
                child_id=child_id,
                T_i=T_i,
                T_j=T_j,
                xi_s=xi_s,
                adp_prox=adp_prox,
                smoothness=None,
                merged=is_merge,
            )
        )
        step_counter[0] += 1
        return result

    Perception.compute_proximity = _patched_prox
    Perception.compute_adaptive_threshold = _patched_thresh
    Perception.vision_generic = _patched_vg
    try:
        p = Perception(k=None)
        with redirect_stdout(io.StringIO()):
            labels_raw, _ = p.fit(X.copy())
    finally:
        Perception.compute_proximity = _orig_prox
        Perception.compute_adaptive_threshold = _orig_thresh
        Perception.vision_generic = _orig_vg

    return np.asarray(labels_raw, dtype=int), logs


def _run_new_component(X: np.ndarray) -> tuple[np.ndarray, list[MergeDecisionLog]]:
    """Run GaugingDelta with per-merge-decision component logging.

    Uses observe-only patches: monkey-patches leaf methods to capture their
    return values as they're called naturally during a single _try_merge
    execution.  No double-computation.

    Critical: algorithm.py does ``from gauging_delta.threshold import
    compute_adaptive_threshold`` which binds a local name.  We must patch
    ``gauging_delta.algorithm.compute_adaptive_threshold`` (the module-level
    binding), NOT ``gauging_delta.threshold.compute_adaptive_threshold``.
    """
    import gauging_delta.algorithm as _alg_mod
    from gauging_delta import GaugingDelta
    from gauging_delta.proximity import DefaultProximity

    logs: list[MergeDecisionLog] = []
    step_counter = [0]

    # Closure lists for captured values
    captured_prox: list = []
    captured_thresh: list = []

    # --- Observe-only patches on leaf methods ---
    _orig_prox_compute = DefaultProximity.compute
    _orig_thresh = _alg_mod.compute_adaptive_threshold
    _orig_tm = GaugingDelta._try_merge

    def _patched_prox(self, C_i, C_j, d_ij, fallback):
        result = _orig_prox_compute(self, C_i, C_j, d_ij, fallback)
        captured_prox.clear()
        captured_prox.append(result)
        return result

    def _patched_thresh(lead, child, d_ij, rho, all_clusters, dist_matrix, cfg):
        result = _orig_thresh(lead, child, d_ij, rho, all_clusters, dist_matrix, cfg)
        captured_thresh.clear()
        captured_thresh.append(result)
        return result

    def _patched_tm(self, c1, c2):
        d_ij = float(self._dist_matrix[c1, c2])

        result = _orig_tm(self, c1, c2)

        if captured_prox:
            prox = captured_prox[0]
            rho, lead_id, child_id = prox.rho, prox.lead_id, prox.child_id
        else:
            rho, lead_id, child_id = float("nan"), c1, c2

        if captured_thresh:
            thr = captured_thresh[0]
            T_i, T_j, xi_s, adp_prox = thr.T_i, thr.T_j, thr.xi_s, thr.adp_prox
        else:
            T_i, T_j, xi_s, adp_prox = 0.0, 0.0, 1.0, 0.0

        logs.append(
            MergeDecisionLog(
                step=step_counter[0],
                c1=c1,
                c2=c2,
                d_ij=d_ij,
                rho=rho,
                lead_id=lead_id,
                child_id=child_id,
                T_i=T_i,
                T_j=T_j,
                xi_s=xi_s,
                adp_prox=adp_prox,
                smoothness=None,
                merged=result is not None,
            )
        )
        step_counter[0] += 1
        return result

    DefaultProximity.compute = _patched_prox
    _alg_mod.compute_adaptive_threshold = _patched_thresh
    GaugingDelta._try_merge = _patched_tm
    try:
        model = GaugingDelta(preserve_labels=True)
        model.fit(X.copy())
    finally:
        DefaultProximity.compute = _orig_prox_compute
        _alg_mod.compute_adaptive_threshold = _orig_thresh
        GaugingDelta._try_merge = _orig_tm

    return np.asarray(model.labels_, dtype=int), logs


# ---------------------------------------------------------------------------
# Parity comparison helpers
# ---------------------------------------------------------------------------

COMPONENT_FIELDS = ["rho", "T_i", "T_j", "xi_s", "adp_prox", "merged"]
COMPONENT_TOLERANCE = {
    "rho": 1e-10,
    "T_i": 1e-10,
    "T_j": 1e-10,
    "xi_s": 1e-10,
    "adp_prox": 1e-10,
}


def _compare_components(
    legacy_logs: list[MergeDecisionLog],
    new_logs: list[MergeDecisionLog],
    dataset_name: str,
) -> list[Divergence]:
    """Compare component-level logs, return list of divergences."""
    divergences: list[Divergence] = []

    # Length mismatch
    if len(legacy_logs) != len(new_logs):
        divergences.append(
            Divergence(
                dataset=dataset_name,
                level="component",
                divergent_field="log_count",
                legacy_value=len(legacy_logs),
                new_value=len(new_logs),
                delta=abs(len(legacy_logs) - len(new_logs)),
            )
        )
        # Compare up to the shorter length
        n = min(len(legacy_logs), len(new_logs))
    else:
        n = len(legacy_logs)

    for i in range(n):
        lg = legacy_logs[i]
        nw = new_logs[i]

        # Check cluster pair alignment
        if lg.c1 != nw.c1 or lg.c2 != nw.c2:
            divergences.append(
                Divergence(
                    dataset=dataset_name,
                    level="component",
                    merge_step=i,
                    divergent_field="cluster_pair",
                    legacy_value=f"({lg.c1}, {lg.c2})",
                    new_value=f"({nw.c1}, {nw.c2})",
                    context={"legacy_d_ij": lg.d_ij, "new_d_ij": nw.d_ij},
                )
            )
            break  # Pairs diverged, further comparison meaningless

        for fld in COMPONENT_FIELDS:
            lv = getattr(lg, fld)
            nv = getattr(nw, fld)

            if lv is None or nv is None:
                continue

            if fld == "merged":
                if lv != nv:
                    divergences.append(
                        Divergence(
                            dataset=dataset_name,
                            level="component",
                            merge_step=i,
                            divergent_field="merged",
                            legacy_value=lv,
                            new_value=nv,
                            context={
                                "c1": lg.c1,
                                "c2": lg.c2,
                                "d_ij": lg.d_ij,
                                "rho_legacy": lg.rho,
                                "rho_new": nw.rho,
                                "T_i_legacy": lg.T_i,
                                "T_i_new": nw.T_i,
                                "T_j_legacy": lg.T_j,
                                "T_j_new": nw.T_j,
                            },
                        )
                    )
            else:
                tol = COMPONENT_TOLERANCE.get(fld, 1e-10)
                if abs(lv - nv) > tol:
                    divergences.append(
                        Divergence(
                            dataset=dataset_name,
                            level="component",
                            merge_step=i,
                            divergent_field=fld,
                            legacy_value=lv,
                            new_value=nv,
                            delta=abs(lv - nv),
                            context={
                                "c1": lg.c1,
                                "c2": lg.c2,
                                "d_ij": lg.d_ij,
                            },
                        )
                    )

    return divergences


# ---------------------------------------------------------------------------
# Level runners
# ---------------------------------------------------------------------------


def run_holistic(
    datasets: dict[str, np.ndarray], sizes: list[int]
) -> tuple[dict, list[Divergence]]:
    """Level 1: End-to-end ARI + timing + memory."""
    results = {"datasets": {}, "synthetic": {}}
    divergences: list[Divergence] = []

    # Benchmark datasets
    for name, X in datasets.items():
        print(f"  [{name}] N={len(X)}...", end=" ", flush=True)
        l_labels, l_time, l_mem = _run_legacy_holistic(X)
        n_labels, n_time, n_mem = _run_new_holistic(X)
        ari = adjusted_rand_score(l_labels, n_labels)

        results["datasets"][name] = {
            "n": len(X),
            "ari": ari,
            "legacy_time_s": l_time,
            "new_time_s": n_time,
            "speedup": l_time / n_time if n_time > 0 else float("inf"),
            "legacy_peak_bytes": l_mem,
            "new_peak_bytes": n_mem,
            "mem_ratio": l_mem / n_mem if n_mem > 0 else float("inf"),
        }

        if ari < 1.0 - 1e-9:
            divergences.append(
                Divergence(
                    dataset=name,
                    level="holistic",
                    divergent_field="ari",
                    legacy_value=1.0,
                    new_value=ari,
                    delta=1.0 - ari,
                )
            )

        print(f"ARI={ari:.6f}  speedup={l_time / n_time:.2f}x  mem_ratio={l_mem / n_mem:.2f}x")

    # Synthetic blobs
    for n in sizes:
        print(f"  [synthetic] N={n}...", end=" ", flush=True)
        X = make_blobs(n)
        l_labels, l_time, l_mem = _run_legacy_holistic(X)
        n_labels, n_time, n_mem = _run_new_holistic(X)
        ari = adjusted_rand_score(l_labels, n_labels)

        results["synthetic"][str(n)] = {
            "n": n,
            "ari": ari,
            "legacy_time_s": l_time,
            "new_time_s": n_time,
            "speedup": l_time / n_time if n_time > 0 else float("inf"),
            "legacy_peak_bytes": l_mem,
            "new_peak_bytes": n_mem,
            "mem_ratio": l_mem / n_mem if n_mem > 0 else float("inf"),
        }

        if ari < 1.0 - 1e-9:
            divergences.append(
                Divergence(
                    dataset=f"synthetic_N{n}",
                    level="holistic",
                    divergent_field="ari",
                    legacy_value=1.0,
                    new_value=ari,
                    delta=1.0 - ari,
                )
            )

        print(f"ARI={ari:.6f}  speedup={l_time / n_time:.2f}x")

    return results, divergences


def run_stage(
    datasets: dict[str, np.ndarray], sizes: list[int]
) -> tuple[dict, list[Divergence]]:
    """Level 2: Per-stage timing + memory + cluster state comparison."""
    results = {"datasets": {}, "synthetic": {}}
    divergences: list[Divergence] = []

    all_sources = list(datasets.items())
    for n in sizes:
        all_sources.append((f"synthetic_N{n}", make_blobs(n)))

    for name, X in all_sources:
        print(f"  [{name}] N={len(X)}...", flush=True)

        l_labels, l_stages = _run_legacy_staged(X)
        n_labels, n_stages = _run_new_staged(X)
        ari = adjusted_rand_score(l_labels, n_labels)

        stage_data = {}
        for ls, ns in zip(l_stages, n_stages, strict=True):
            stage_data[ls.name] = {
                "legacy_time_s": ls.time_s,
                "new_time_s": ns.time_s,
                "speedup": ls.time_s / ns.time_s if ns.time_s > 0 else float("inf"),
                "legacy_peak_bytes": ls.peak_bytes,
                "new_peak_bytes": ns.peak_bytes,
            }

        entry = {"n": len(X), "ari": ari, "stages": stage_data}
        if name.startswith("synthetic"):
            results["synthetic"][name] = entry
        else:
            results["datasets"][name] = entry

        if ari < 1.0 - 1e-9:
            divergences.append(
                Divergence(
                    dataset=name, level="stage", divergent_field="ari",
                    legacy_value=1.0, new_value=ari, delta=1.0 - ari,
                )
            )

        # Print stage breakdown
        print(f"    ARI={ari:.6f}")
        print(f"    {'Stage':<25s} {'Legacy (s)':>12s} {'New (s)':>12s} {'Speedup':>10s}")
        print(f"    {'-' * 59}")
        for ls, ns in zip(l_stages, n_stages, strict=True):
            spd = f"{ls.time_s / ns.time_s:.2f}x" if ns.time_s > 0 else "inf"
            print(f"    {ls.name:<25s} {ls.time_s:>12.4f} {ns.time_s:>12.4f} {spd:>10s}")

    return results, divergences


def run_component(
    datasets: dict[str, np.ndarray], sizes: list[int]
) -> tuple[dict, list[Divergence]]:
    """Level 3: Per-merge-decision component parity."""
    results = {"datasets": {}, "synthetic": {}}
    all_divergences: list[Divergence] = []

    all_sources = list(datasets.items())
    for n in sizes:
        all_sources.append((f"synthetic_N{n}", make_blobs(n)))

    for name, X in all_sources:
        print(f"  [{name}] N={len(X)}...", end=" ", flush=True)

        l_labels, l_logs = _run_legacy_component(X)
        n_labels, n_logs = _run_new_component(X)
        ari = adjusted_rand_score(l_labels, n_labels)

        divs = _compare_components(l_logs, n_logs, name)
        all_divergences.extend(divs)

        n_merges_l = sum(1 for lg in l_logs if lg.merged)
        n_merges_n = sum(1 for lg in n_logs if lg.merged)

        entry = {
            "n": len(X),
            "ari": ari,
            "legacy_decisions": len(l_logs),
            "new_decisions": len(n_logs),
            "legacy_merges": n_merges_l,
            "new_merges": n_merges_n,
            "divergences": len(divs),
        }
        if name.startswith("synthetic"):
            results["synthetic"][name] = entry
        else:
            results["datasets"][name] = entry

        status = "PASS" if len(divs) == 0 else f"FAIL ({len(divs)} divergences)"
        print(f"ARI={ari:.6f}  decisions={len(l_logs)}/{len(n_logs)}  merges={n_merges_l}/{n_merges_n}  {status}")

        if divs:
            d = divs[0]
            print(f"    First divergence: step={d.merge_step} field={d.divergent_field} "
                  f"legacy={d.legacy_value} new={d.new_value}")

    return results, all_divergences


def run_decision(
    datasets: dict[str, np.ndarray],
    sizes: list[int],
    decision_idx: int,
) -> tuple[dict, list[Divergence]]:
    """Level 4: Dump all details for a specific merge decision."""
    # Run component level first
    results = {}
    all_divergences: list[Divergence] = []

    all_sources = list(datasets.items())
    for n in sizes:
        all_sources.append((f"synthetic_N{n}", make_blobs(n)))

    for name, X in all_sources:
        print(f"\n  [{name}] N={len(X)} — dumping decision #{decision_idx}...", flush=True)

        _l_labels, l_logs = _run_legacy_component(X)
        _n_labels, n_logs = _run_new_component(X)

        if decision_idx < len(l_logs):
            lg = l_logs[decision_idx]
            print(f"    Legacy decision #{decision_idx}:")
            for k, v in asdict(lg).items():
                print(f"      {k}: {v}")
        else:
            print(f"    Legacy: decision #{decision_idx} does not exist (only {len(l_logs)} decisions)")

        if decision_idx < len(n_logs):
            nw = n_logs[decision_idx]
            print(f"    New decision #{decision_idx}:")
            for k, v in asdict(nw).items():
                print(f"      {k}: {v}")
        else:
            print(f"    New: decision #{decision_idx} does not exist (only {len(n_logs)} decisions)")

        # Compare if both exist
        if decision_idx < len(l_logs) and decision_idx < len(n_logs):
            lg = l_logs[decision_idx]
            nw = n_logs[decision_idx]
            print("    Comparison:")
            for fld in COMPONENT_FIELDS:
                lv = getattr(lg, fld)
                nv = getattr(nw, fld)
                match = "MATCH" if lv == nv else f"DIFF (delta={abs(lv - nv) if isinstance(lv, float) else 'N/A'})"
                print(f"      {fld}: legacy={lv}  new={nv}  {match}")

        results[name] = {
            "decision_idx": decision_idx,
            "legacy_total_decisions": len(l_logs),
            "new_total_decisions": len(n_logs),
        }

    return results, all_divergences


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_stage_breakdown(results: dict, out_dir: Path) -> None:
    """Stacked bar chart: time per stage, legacy vs new."""
    use_science_style()

    # Collect all stage data
    all_entries = []
    for section in ["datasets", "synthetic"]:
        for name, data in results.get(section, {}).items():
            if "stages" in data:
                all_entries.append((name, data))

    if not all_entries:
        return

    stage_names = [
        "init_clusters", "distance_matrix", "sorted_pairs",
        "merge_loop_total", "post_processing", "label_building",
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax_idx, version in enumerate(["legacy", "new"]):
        ax = axes[ax_idx]
        bottoms = np.zeros(len(all_entries))
        for s_idx, sname in enumerate(stage_names):
            vals = []
            for _ename, edata in all_entries:
                key = f"{version}_time_s"
                vals.append(edata["stages"].get(sname, {}).get(key, 0.0))
            vals = np.array(vals)
            ax.barh(
                range(len(all_entries)), vals, left=bottoms,
                color=colors[s_idx % len(colors)], label=sname,
                height=0.6,
            )
            bottoms += vals

        ax.set_yticks(range(len(all_entries)))
        ax.set_yticklabels([e[0] for e in all_entries], fontsize=6)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{'Legacy' if version == 'legacy' else 'New'} Stage Breakdown")
        ax.legend(fontsize=5, loc="lower right")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"benchmark_stages_breakdown.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved benchmark_stages_breakdown.png/pdf")


def plot_stage_scaling(results: dict, out_dir: Path) -> None:
    """Line plots: top stages vs N."""
    use_science_style()

    synthetic = results.get("synthetic", {})
    if not synthetic:
        return

    # Collect (N, stage_times) pairs
    entries = sorted(
        [(d["n"], d["stages"]) for d in synthetic.values() if "stages" in d],
        key=lambda x: x[0],
    )
    if len(entries) < 2:
        return

    stage_names = ["distance_matrix", "merge_loop_total", "sorted_pairs"]
    ns = [e[0] for e in entries]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax_idx, version in enumerate(["legacy", "new"]):
        ax = axes[ax_idx]
        key = f"{version}_time_s"
        for sname in stage_names:
            vals = [e[1].get(sname, {}).get(key, 0.0) for e in entries]
            ax.plot(ns, vals, marker="o", markersize=3, label=sname, linewidth=1)
        ax.set_xlabel("N")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"{'Legacy' if version == 'legacy' else 'New'} Stage Scaling")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"benchmark_stages_scaling.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved benchmark_stages_scaling.png/pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Gauging-delta Parity Benchmark (binary search levels)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--level",
        choices=["holistic", "stage", "component", "decision"],
        default="holistic",
        help="Parity level: holistic (default), stage, component, decision",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[100, 250, 500],
        help="Synthetic dataset sizes (default: 100 250 500)",
    )
    parser.add_argument(
        "--decision",
        type=int,
        default=0,
        help="Decision index for --level decision (default: 0)",
    )
    parser.add_argument(
        "--datasets-only",
        action="store_true",
        help="Only run on benchmark datasets, skip synthetic",
    )
    args = parser.parse_args()

    # Load benchmark datasets
    datasets = {}
    for name, path in BENCHMARK_DATASETS.items():
        if path.exists():
            datasets[name] = load_dataset(path)

    sizes = [] if args.datasets_only else args.sizes

    print(f"\n{'=' * 70}")
    print(f"Gauging-delta Parity Benchmark — Level: {args.level}")
    print(f"Datasets: {list(datasets.keys())}")
    print(f"Synthetic sizes: {sizes}")
    print(f"{'=' * 70}\n")

    # Run selected level
    if args.level == "holistic":
        results, divergences = run_holistic(datasets, sizes)
    elif args.level == "stage":
        results, divergences = run_stage(datasets, sizes)
    elif args.level == "component":
        results, divergences = run_component(datasets, sizes)
    elif args.level == "decision":
        results, divergences = run_decision(datasets, sizes, args.decision)
    else:
        raise ValueError(f"Unknown level: {args.level}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "benchmark_parity.json"
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "level": args.level,
        "sizes": sizes,
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved results to {json_path}")

    # Save divergences
    if divergences:
        div_path = RESULTS_DIR / "divergences.json"
        div_data = [asdict(d) for d in divergences]
        with open(div_path, "w") as f:
            json.dump(div_data, f, indent=2, default=str)
        print(f"  Saved {len(divergences)} divergence(s) to {div_path}")
    else:
        print("  No divergences found — perfect parity!")

    # Generate plots for stage level
    if args.level == "stage":
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        plot_stage_breakdown(results, ASSETS_DIR)
        plot_stage_scaling(results, ASSETS_DIR)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Summary: {len(divergences)} divergence(s) found")
    if divergences:
        for d in divergences[:5]:
            print(f"  - {d.dataset}: {d.divergent_field} "
                  f"(step={d.merge_step}, delta={d.delta})")
        if len(divergences) > 5:
            print(f"  ... and {len(divergences) - 5} more")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
