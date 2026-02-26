"""
Maximum-optimized Gauging-δ clustering algorithm.

Subclass of GaugingDelta that trades bit-exact parity for maximum performance:

Time:  O(N · k · log N)  vs  O(N²) in the base class
Space: O(N · k)          vs  O(N²) in the base class

Key optimisations over the base class:
  1. NO dense N×N distance matrix   → lightweight edge cache (dict)
  2. NO N²/2 ref_points dict        → ref points live in edge cache
  3. Heap-driven merge loop          → O(log E) per candidate extraction
  4. Sparse neighbor collection      → graph adjacency + KDTree discovery
  5. KDTree-based continuity search  → O(log N) per spatial query

The merge logic (proximity, threshold, continuity math) is identical.
FP differences from different distance-computation order may cascade into
different tie-breaking on edge cases, so ARI vs original may be <1.0.
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np

from gauging_delta.core.algorithm import GaugingDelta, MIN_BTN_CLUSTER_DIST
from gauging_delta.core.cluster import Cluster, MergeabilityResult
from gauging_delta.core.neighbor_graph import NeighborGraph
from gauging_delta.mergeability.continuity import (
    _compute_angle_transition_original,
    _compute_max_angle,
    _compute_transition_smoothness_kdtree,
    _find_local_points_kdtree,
    _remove_outliers,
)
from gauging_delta.mergeability.proximity import compute_proximity_rho
from gauging_delta.mergeability.threshold import (
    compute_adaptive_threshold_T,
    compute_beta_ij,
    compute_xi_s,
)


class GaugingDeltaFast(GaugingDelta):
    """
    Maximum-optimized Gauging-δ clustering.

    Drops the O(N²) dense distance matrix entirely and drives the merge
    loop from the sparse neighbour-graph heap instead.

    Overrides almost the entire hot path while keeping the mergeability
    mathematics (ρ, T, β, ξ, continuity) identical to the base class.

    Parameters are identical to GaugingDelta.
    """

    def __init__(
        self,
        k: Optional[int] = None,
        threshold_continuity: float = 0.15,
        n_neighbors: int = 5,
        graph_neighbors: int = 50,
        preserve_labels: bool = False,
        capture_merges: bool = False,
    ):
        super().__init__(
            k=k,
            threshold_continuity=threshold_continuity,
            n_neighbors=n_neighbors,
            graph_neighbors=graph_neighbors,
            legacy_point_dists=False,  # Skip point_dists; use KDTree instead
            preserve_labels=preserve_labels,
            capture_merges=capture_merges,
        )
        # Lightweight edge cache: (min_id, max_id) -> (dist, p_i, p_j)
        # where p_i is the ref point in the cluster with the smaller id.
        self._edge_cache: dict[tuple[int, int], tuple[float, int, int]] = {}

    # ==================================================================
    # Override 1: Sparse initialisation (no N×N matrix)
    # ==================================================================

    def _init_distances_combined(self) -> None:
        """Compute only fallback_dist from KDTree.  No matrix, no ref_points."""
        n_total = len(self._X)
        self._dist_matrix = None
        self._ref_points = {}
        self._point_dists = None

        if n_total <= 1:
            self._fallback_dist = MIN_BTN_CLUSTER_DIST
            return

        # Approximate the base class's "1st percentile of ALL pairwise
        # distances" using the k-nearest-neighbour distances already in the
        # heap.  The heap contains the O(N·k) smallest edges, so the
        # base class's index  N(N-1)/200  is usually reachable directly.
        heap_dists = np.array(
            [d for d, _, _ in self._graph.merge_heap], dtype=np.float64
        )
        if len(heap_dists) == 0:
            self._fallback_dist = MIN_BTN_CLUSTER_DIST
            return
        heap_dists.sort()
        n_full_pairs = n_total * (n_total - 1) // 2
        idx = max(0, min(int(n_full_pairs / 100), len(heap_dists) - 1))
        self._fallback_dist = float(heap_dists[idx])

    # ==================================================================
    # Override 2: Heap-driven fit()
    # ==================================================================

    def fit(self, X: np.ndarray) -> np.ndarray:
        """Heap-driven Gauging-δ clustering — O(N·k·log N) typical."""
        self._X = np.asarray(X, dtype=np.float64)
        n_samples = len(self._X)

        if self.capture_merges:
            self.merge_log = []

        # ── initialise graph + clusters ──────────────────────────────
        self._graph = NeighborGraph(X=self._X, n_neighbors=self.graph_neighbors)
        self._graph.initialize()

        self.clusters_ = {}
        for i in range(n_samples):
            self.clusters_[i] = Cluster(
                label=i,
                point_indices=[i],
                center=self._X[i].copy(),
            )

        self._edge_cache.clear()
        self._init_distances_combined()

        # ── main merge loop (rounds) ─────────────────────────────────
        while True:
            active = self._graph.get_active_clusters()
            pre_count = len(active)

            if self.k is not None and pre_count <= self.k:
                break

            merged_any = self._process_merge_round()

            if not merged_any:
                break

        # ── force-merge if k specified ───────────────────────────────
        if self.k is not None and len(self._graph.get_active_clusters()) > self.k:
            self._force_merge_to_k()

        self._build_labels()
        return self.labels_

    # ------------------------------------------------------------------

    def _process_merge_round(self) -> bool:
        """Drain the heap for one round; return True if any merge happened."""
        merged_any = False

        while True:
            pair = self._graph.pop_closest_pair()
            if pair is None:
                break

            c_i, c_j, heap_dist = pair

            if np.isinf(heap_dist):
                break

            if self.k is not None and len(self._graph.get_active_clusters()) <= self.k:
                break

            # Recompute actual inter-cluster distance (may have changed
            # since this heap entry was created).
            d_ij, p_i, p_j = self._get_cached_distance(c_i, c_j)

            C_i = self.clusters_[c_i]
            C_j = self.clusters_[c_j]

            merge_result = self._compute_mergeability(C_i, C_j, d_ij)

            if merge_result.is_mergeable:
                if len(C_i) >= len(C_j):
                    lead_id, child_id = c_i, c_j
                    lead_cluster, child_cluster = C_i, C_j
                else:
                    lead_id, child_id = c_j, c_i
                    lead_cluster, child_cluster = C_j, C_i

                if self.capture_merges and self.merge_log is not None:
                    self.merge_log.append({
                        "pair": (c_i, c_j),
                        "lead_id": lead_id,
                        "child_id": child_id,
                        "d_ij": d_ij,
                        "p_i": p_i,
                        "p_j": p_j,
                        "rho": merge_result.rho,
                        "T_i": merge_result.T_i,
                        "T_j": merge_result.T_j,
                        "beta_ij": merge_result.beta_ij,
                        "xi_s": merge_result.xi_s,
                        "continuity": merge_result.continuity,
                    })

                self._merge_cluster_data(lead_cluster, child_cluster, d_ij)
                self._graph.merge_clusters(lead_id, child_id)

                self._graph.cluster_points[lead_id] = np.array(
                    lead_cluster.point_indices, dtype=np.int64
                )
                if lead_cluster.center is not None:
                    self._graph.cluster_centers[lead_id] = lead_cluster.center.copy()

                # Invalidate stale cache entries for both the dead child
                # AND the lead (whose point set has grown).
                self._invalidate_cache(child_id)
                self._invalidate_cache(lead_id)

                merged_any = True

        # Prepare heap for the next round: re-push existing neighbour
        # edges + discover potential new neighbours via KDTree.
        if merged_any:
            self._refill_heap()
            # Replicate base class's fallback update:
            # _fallback_dist = max(matrix_min, _fallback_dist)
            if self._graph.merge_heap:
                heap_min = self._graph.merge_heap[0][0]  # min-heap root
                if np.isfinite(heap_min):
                    self._fallback_dist = max(heap_min, self._fallback_dist)

        return merged_any

    # ==================================================================
    # Override 3: Lightweight edge cache (replaces N×N matrix)
    # ==================================================================

    def _get_cached_distance(self, c_i: int, c_j: int) -> tuple[float, int, int]:
        """Return (distance, ref_point_in_c_i, ref_point_in_c_j)."""
        key = (min(c_i, c_j), max(c_i, c_j))
        cached = self._edge_cache.get(key)
        if cached is not None:
            d, pa, pb = cached          # pa belongs to key[0], pb to key[1]
            return (d, pa, pb) if c_i <= c_j else (d, pb, pa)

        d, pa, pb = self._graph.get_cluster_distance(c_i, c_j)
        # Store with (smaller_id's point, larger_id's point)
        if c_i <= c_j:
            self._edge_cache[key] = (d, pa, pb)
            return d, pa, pb
        self._edge_cache[key] = (d, pb, pa)
        return d, pb, pa

    def _invalidate_cache(self, dead_id: int) -> None:
        """Remove all cache entries involving a dead cluster."""
        to_delete = [k for k in self._edge_cache if dead_id in k]
        for k in to_delete:
            del self._edge_cache[k]

    def _update_distance_cache_after_merge(self, lead_id: int, child_id: int) -> None:
        """No-op — heap + edge cache handle everything."""

    # ==================================================================
    # Override 4: Heap refill + neighbour discovery
    # ==================================================================

    def _refill_heap(self) -> None:
        """Re-push edges for known neighbours; discover new ones via KDTree."""
        active = self._graph.get_active_clusters()
        seen: set[tuple[int, int]] = set()
        new_heap: list[tuple[float, int, int]] = []

        for c_id in active:
            # ── existing graph neighbours ────────────────────────────
            for n_id in self._graph.neighbors.get(c_id, set()):
                if n_id in self._graph.dead_clusters:
                    continue
                edge = (min(c_id, n_id), max(c_id, n_id))
                if edge in seen:
                    continue
                seen.add(edge)
                d, _, _ = self._get_cached_distance(c_id, n_id)
                heapq.heappush(new_heap, (d, edge[0], edge[1]))

            # ── KDTree discovery for new neighbours ──────────────────
            center = self._graph.cluster_centers[c_id]
            k_query = min(self.graph_neighbors + 1, len(self._X))
            _, pt_indices = self._graph.kdtree.query(center, k=k_query)
            pt_indices = np.atleast_1d(pt_indices)

            for pt_idx in pt_indices:
                n_id = int(self._graph.point_to_cluster[pt_idx])
                if n_id == c_id or n_id in self._graph.dead_clusters:
                    continue
                edge = (min(c_id, n_id), max(c_id, n_id))
                if edge in seen:
                    continue
                seen.add(edge)
                # Register as graph neighbours so future merges transfer them
                self._graph.neighbors.setdefault(c_id, set()).add(n_id)
                self._graph.neighbors.setdefault(n_id, set()).add(c_id)
                d, _, _ = self._get_cached_distance(c_id, n_id)
                heapq.heappush(new_heap, (d, edge[0], edge[1]))

        self._graph.merge_heap = new_heap

    # ==================================================================
    # Override 5: _get_sorted_cluster_pairs / _get_nearest_cluster
    #             (not used in heap-driven loop, but kept for
    #              _force_merge_to_k and any other callers)
    # ==================================================================

    def _get_sorted_cluster_pairs(self) -> list[tuple[int, int, float]]:
        """Not used in the heap-driven loop; kept for _force_merge_to_k."""
        active = self._graph.get_active_clusters()
        if len(active) < 2:
            return []
        pairs: list[tuple[int, int, float]] = []
        active_list = sorted(active)
        for i, c_i in enumerate(active_list):
            for c_j in active_list[i + 1:]:
                d, _, _ = self._get_cached_distance(c_i, c_j)
                pairs.append((c_i, c_j, d))
        pairs.sort(key=lambda x: x[2])
        return pairs

    def _get_nearest_cluster(self, cluster_id: int) -> tuple[int, float] | None:
        """Find nearest active cluster using graph neighbours + KDTree."""
        if cluster_id in self._graph.dead_clusters:
            return None

        best_id: int | None = None
        best_dist = float("inf")

        # Check graph neighbours first (cheap)
        for n_id in self._graph.neighbors.get(cluster_id, set()):
            if n_id in self._graph.dead_clusters:
                continue
            d, _, _ = self._get_cached_distance(cluster_id, n_id)
            if d < best_dist:
                best_dist = d
                best_id = n_id

        # KDTree fallback for discovery
        center = self._graph.cluster_centers.get(cluster_id)
        if center is not None:
            k_q = min(self.graph_neighbors + 1, len(self._X))
            _, pt_idx = self._graph.kdtree.query(center, k=k_q)
            pt_idx = np.atleast_1d(pt_idx)
            for pi in pt_idx:
                n_id = int(self._graph.point_to_cluster[pi])
                if n_id == cluster_id or n_id in self._graph.dead_clusters:
                    continue
                d, _, _ = self._get_cached_distance(cluster_id, n_id)
                if d < best_dist:
                    best_dist = d
                    best_id = n_id

        return (best_id, best_dist) if best_id is not None else None

    # ==================================================================
    # Override 6: Mergeability with sparse neighbour collection
    # ==================================================================

    def _collect_neighbors_sparse(
        self, cluster_id: int, k_neighbors: int,
    ) -> list[tuple[int, float]]:
        """Find k nearest active clusters via graph adjacency + KDTree.

        O(k · (log N + |C|)) instead of O(N · |C|).
        """
        if k_neighbors <= 0:
            return []

        candidates: dict[int, float] = {}  # cid -> distance

        # 1) Graph neighbours (O(degree))
        for n_id in self._graph.neighbors.get(cluster_id, set()):
            if n_id in self._graph.dead_clusters:
                continue
            d, _, _ = self._get_cached_distance(cluster_id, n_id)
            candidates[n_id] = d

        # 2) KDTree discovery if we need more (O(k·log N))
        if len(candidates) < k_neighbors:
            center = self._graph.cluster_centers[cluster_id]
            k_q = min(k_neighbors * 10 + 1, len(self._X))
            _, pt_indices = self._graph.kdtree.query(center, k=k_q)
            pt_indices = np.atleast_1d(pt_indices)
            for pi in pt_indices:
                n_id = int(self._graph.point_to_cluster[pi])
                if n_id == cluster_id or n_id in self._graph.dead_clusters:
                    continue
                if n_id not in candidates:
                    d, _, _ = self._get_cached_distance(cluster_id, n_id)
                    candidates[n_id] = d
                if len(candidates) >= k_neighbors * 2:
                    break

        sorted_cands = sorted(candidates.items(), key=lambda x: x[1])
        return sorted_cands[:k_neighbors]

    def _compute_mergeability(
        self,
        C_i: Cluster,
        C_j: Cluster,
        d_ij: float,
    ) -> MergeabilityResult:
        """
        Evaluate mergeability — same logic as base class but uses:
          • sparse _collect_neighbors_sparse  (graph + KDTree)
          • KDTree-based continuity           (O(log N) spatial queries)
        """
        # Step 1: Compute proximity ρ
        fallback = getattr(self, "_fallback_dist", MIN_BTN_CLUSTER_DIST)
        rho, _, lead_id, child_id = compute_proximity_rho(d_ij, C_i, C_j, fallback)

        # Step 2: Compute environmental factor β and shape factor ξ
        C_lead = C_i if lead_id == C_i.label else C_j
        C_child = C_j if lead_id == C_i.label else C_i

        k_neighbors = min(
            self.K_neighbors,
            max(len(self._graph.get_active_clusters()) - 1, 0),
        )

        # Local distance cache for this mergeability evaluation
        local_dcache: dict[tuple[int, int], float] = {
            (min(C_i.label, C_j.label), max(C_i.label, C_j.label)): d_ij
        }

        def _get_near_distance_by_id(a_id: int, b_id: int) -> float:
            key = (min(a_id, b_id), max(a_id, b_id))
            if key in local_dcache:
                return local_dcache[key]
            dist, _, _ = self._get_cached_distance(a_id, b_id)
            local_dcache[key] = dist
            return dist

        def _get_near_distance(C_a: Cluster, C_b: Cluster) -> float:
            return _get_near_distance_by_id(C_a.label, C_b.label)

        def _get_mix_distance(C_a: Cluster, C_b: Cluster) -> float:
            near_dist = _get_near_distance(C_a, C_b)
            if C_a.center is not None and C_b.center is not None:
                center_dist = float(np.linalg.norm(C_a.center - C_b.center))
            else:
                center_dist = near_dist
            return (near_dist + center_dist) / 2.0

        # Sparse neighbour collection (key optimisation)
        neighbor_i = self._collect_neighbors_sparse(C_lead.label, k_neighbors) if k_neighbors else []
        neighbor_j = self._collect_neighbors_sparse(C_child.label, k_neighbors) if k_neighbors else []

        neighbor_clusters = [self.clusters_[cid] for cid, _ in neighbor_i]
        neighbor_distances = [dist for _, dist in neighbor_i]
        neighbor_clusters_j = [self.clusters_[cid] for cid, _ in neighbor_j]
        neighbor_distances_j = [dist for _, dist in neighbor_j]

        beta_ij = compute_beta_ij(
            C_lead,
            C_child,
            d_ij,
            neighbor_clusters,
            neighbor_distances,
            neighbor_clusters_j=neighbor_clusters_j,
            neighbor_distances_j=neighbor_distances_j,
            get_near_distance=_get_near_distance,
            get_mix_distance=_get_mix_distance,
        )
        xi_s = compute_xi_s(C_lead, C_child)

        # Step 3: Compute adaptive thresholds
        T_i = compute_adaptive_threshold_T(C_lead, beta_ij, xi_s)
        T_j = compute_adaptive_threshold_T(C_child, beta_ij, xi_s)

        # Step 4: Check proximity threshold
        if rho > T_i or rho > T_j:
            return MergeabilityResult(
                is_mergeable=False,
                rho=rho,
                T_i=T_i,
                T_j=T_j,
                beta_ij=beta_ij,
                xi_s=xi_s,
                continuity=0.0,
                lead_cluster=lead_id,
                child_cluster=child_id,
                d_ij=d_ij,
            )

        # Step 5: Compute continuity using KDTree
        if lead_id == C_i.label:
            lead_cluster, child_cluster = C_i, C_j
        else:
            lead_cluster, child_cluster = C_j, C_i

        _, p_lead, p_child = self._get_cached_distance(
            lead_cluster.label, child_cluster.label
        )

        size_lead = len(lead_cluster)
        size_child = len(child_cluster)
        total_size = size_lead + size_child
        T_adaptive = T_i * size_lead / total_size + T_j * size_child / total_size
        shape_diff = xi_s if xi_s > 0 else 1.0
        mean_historical = d_ij / rho if rho > 0 else d_ij

        continuity = _compute_continuity_kdtree(
            lead_cluster,
            child_cluster,
            p_lead,
            p_child,
            self._X,
            self.T_continuity / shape_diff,
            mean_historical,
            T_adaptive,
            kdtree=self._graph.kdtree,
        )

        # Step 6: Final decision
        is_mergeable = continuity > self.T_continuity / shape_diff

        return MergeabilityResult(
            is_mergeable=is_mergeable,
            rho=rho,
            T_i=T_i,
            T_j=T_j,
            beta_ij=beta_ij,
            xi_s=xi_s,
            continuity=continuity,
            lead_cluster=lead_id,
            child_cluster=child_id,
            d_ij=d_ij,
        )


# ======================================================================
# KDTree-based continuity (module-level helper)
# ======================================================================

def _compute_continuity_kdtree(
    C_i: Cluster,
    C_j: Cluster,
    p_i: int,
    p_j: int,
    X: np.ndarray,
    T_continuity: float,
    d_ij: float,
    T_adaptive: float,
    kdtree=None,
) -> float:
    """
    Compute continuity score using KDTree for spatial queries.

    Same algorithm as ``continuity.compute_continuity`` but uses
    ``_find_local_points_kdtree`` and ``_compute_transition_smoothness_kdtree``
    for O(log N) spatial lookups instead of O(N) linear scans.
    """
    points_dist = np.linalg.norm(X[p_i] - X[p_j])
    middle_point = (X[p_i] + X[p_j]) / 2.0

    past_dists_i = C_i.merge_history[-5:] if C_i.merge_history else []
    past_dists_j = C_j.merge_history[-5:] if C_j.merge_history else []
    past_dists = past_dists_i + past_dists_j

    size_i, size_j = len(C_i), len(C_j)

    if size_i > 2 and C_i.sigma_history and C_i.merge_history:
        compact_i = (C_i.sigma_history[-1] / np.mean(C_i.merge_history) / np.log(size_i)) * size_i / (size_i + size_j)
    else:
        compact_i = 0.5

    if size_j > 2 and C_j.sigma_history and C_j.merge_history:
        compact_j = (C_j.sigma_history[-1] / np.mean(C_j.merge_history) / np.log(size_j)) * size_j / (size_i + size_j)
    else:
        compact_j = 0.5

    compact = compact_i + compact_j

    c_past_dists = [max(past_dists), points_dist] if past_dists else [points_dist]

    if C_i.center is not None and C_j.center is not None:
        center_dist = np.linalg.norm(C_i.center - C_j.center) / compact
    else:
        center_dist = points_dist

    rate = max(center_dist / points_dist, 3) - 3
    enlarge_rate = 2 / (1 + math.exp(-rate / 20))
    base_length = np.mean(c_past_dists) * enlarge_rate

    explore_range = [2.0, 2.5, 3.0]

    min_radius_idx = None
    min_radius_smoothness = 1.0
    surrounded = False
    prev_smoothness = 1.0
    prev_mass_smoothness = 1.0
    prev_is_boundary = False
    last_smoothness = 1.0

    for i, radius_mult in enumerate(explore_range):
        radius = radius_mult * base_length

        local_points_i = _find_local_points_kdtree(
            middle_point, p_i, p_j, radius, X, C_i.point_indices, kdtree
        )
        local_points_j = _find_local_points_kdtree(
            middle_point, p_j, p_i, radius, X, C_j.point_indices, kdtree
        )

        N_i = len(local_points_i) + 1
        N_j = len(local_points_j) + 1

        local_points_i = _remove_outliers(local_points_i, middle_point, X)
        local_points_j = _remove_outliers(local_points_j, middle_point, X)

        angle_smoothness = 1.0
        mass_smoothness = 1.0
        transition_smoothness = 1.0
        is_boundary = False

        if len(local_points_i) > 1 and len(local_points_j) > 1:
            if not surrounded:
                count_in_i = sum(1 for p, _ in local_points_j if int(p) in C_i.point_indices)
                surrounded = count_in_i >= len(C_j) and count_in_i > N_j / 2

            max_angle_i = _compute_max_angle(local_points_i, p_i, X)
            max_angle_j = _compute_max_angle(local_points_j, p_j, X)

            area_i = max_angle_i[2] if max_angle_i else 0.0
            area_j = max_angle_j[2] if max_angle_j else 0.0

            if min(area_i, area_j) == 0 and max(area_i, area_j) <= 0.2:
                mass_i = mass_j = 1.0
            else:
                mass_i = N_i * area_i
                mass_j = N_j * area_j

            if max(mass_i, mass_j) > 0:
                mass_smoothness = min(mass_i, mass_j) / max(mass_i, mass_j)

            if max_angle_i and max_angle_j:
                angle_smoothness = _compute_angle_transition_original(
                    p_i, p_j, max_angle_i, max_angle_j, local_points_i, local_points_j, X
                )

            transition_smoothness = _compute_transition_smoothness_kdtree(
                p_i, p_j, middle_point, N_i, N_j, radius,
                radius_mult - 1, X, C_i, C_j, kdtree,
            )

        orientation_smoothness = math.sqrt(mass_smoothness * angle_smoothness)
        smoothness = min(transition_smoothness * orientation_smoothness, 1.0)

        if surrounded:
            smoothness = 1.0

        if transition_smoothness > 2:
            is_boundary = True

        if smoothness <= T_continuity:
            min_radius_idx = i
            min_radius_smoothness = smoothness

        if i > 0:
            if is_boundary and prev_is_boundary:
                return 1.0
            if prev_smoothness == 1.0 and smoothness == 1.0:
                return smoothness
            if prev_mass_smoothness > 0 and mass_smoothness / prev_mass_smoothness > math.e:
                return smoothness

        prev_smoothness = smoothness
        prev_mass_smoothness = mass_smoothness
        prev_is_boundary = is_boundary
        last_smoothness = smoothness

    if min_radius_idx is not None:
        return min_radius_smoothness
    else:
        return last_smoothness
