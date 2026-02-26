"""
Main Gauging-δ clustering algorithm implementation.

Implements Algorithm 1 from the paper:
    Yao, Pan, Zeng. "Gauging-δ: A Non-Parametric Hierarchical Clustering Algorithm"
    IEEE TPAMI, Vol. 47, No. 6, June 2025
"""

from typing import Optional

import numpy as np

from gauging_delta.core.cluster import Cluster, MergeabilityResult
from gauging_delta.core.neighbor_graph import NeighborGraph
from gauging_delta.mergeability.continuity import compute_continuity
from gauging_delta.mergeability.proximity import compute_proximity_rho
from gauging_delta.mergeability.threshold import (
    compute_adaptive_threshold_T,
    compute_beta_ij,
    compute_xi_s,
)


# Constants matching legacy code
MIN_BTN_CLUSTER_DIST = 0.0001


class GaugingDelta:
    """
    Gauging-δ: Non-Parametric Hierarchical Clustering Algorithm.

    This algorithm performs hierarchical clustering by iteratively merging
    clusters based on an adaptive mergeability function that considers
    both proximity (ρ) and continuity statistics.

    Parameters:
        k: Optional target number of clusters (default: None, auto-determine)
        threshold_continuity: Continuity threshold T_c (default: 0.15)
        n_neighbors: Number of nearest clusters for context (default: 5)
        preserve_labels: Preserve legacy cluster IDs in output labels (default: False)
        capture_merges: Record merge sequence details to merge_log (default: False)

    Attributes:
        labels_: Cluster labels for each data point after fitting
        n_clusters_: Number of clusters found
        clusters_: Dictionary of final Cluster objects

    Example:
        >>> from gauging_delta import GaugingDelta
        >>> import numpy as np
        >>> X = np.random.randn(100, 2)
        >>> model = GaugingDelta()
        >>> labels = model.fit(X)
    """

    def __init__(
        self,
        k: Optional[int] = None,
        threshold_continuity: float = 0.15,
        n_neighbors: int = 5,
        graph_neighbors: int = 50,
        legacy_point_dists: bool = True,
        preserve_labels: bool = False,
        capture_merges: bool = False,
    ):
        self.k = k
        self.T_continuity = threshold_continuity
        self.K_neighbors = n_neighbors
        self.graph_neighbors = graph_neighbors
        self.legacy_point_dists = legacy_point_dists
        self.preserve_labels = preserve_labels
        self.capture_merges = capture_merges

        self.merge_log = [] if capture_merges else None

        self.labels_: Optional[np.ndarray] = None
        self.n_clusters_: int = 0
        self.clusters_: dict[int, Cluster] = {}

        # Internal state
        self._graph: Optional[NeighborGraph] = None
        self._X: Optional[np.ndarray] = None
        self._point_dists: list[list[tuple[int, float]]] | None = None
        self._dist_matrix: Optional[np.ndarray] = None
        self._ref_points: dict[tuple[int, int], tuple[int, int]] = {}

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Perform Gauging-δ clustering on data X.

        Implements Algorithm 1 from the paper:
        1. Initialize n clusters (each point is a cluster)
        2. Sort distance matrix D to get ascending list D'
        3. While D' not empty:
            - For each pair (C_i, C_j) in D':
                - Check mergeability using proximity and continuity
                - If mergeable, merge clusters
            - If no merges occurred, break
            - Update D' for new cluster configuration

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            labels: Cluster labels for each sample, shape (n_samples,)
        """
        self._X = np.asarray(X, dtype=np.float64)
        n_samples = len(self._X)

        if self.capture_merges:
            self.merge_log = []

        # Initialize neighbor graph
        self._graph = NeighborGraph(X=self._X, n_neighbors=self.graph_neighbors)
        self._graph.initialize()

        # Initialize cluster objects
        self.clusters_ = {}
        for i in range(n_samples):
            self.clusters_[i] = Cluster(
                label=i,
                point_indices=[i],
                center=self._X[i].copy(),
            )

        # Initialize distance cache AND point_dists in a single O(N²) pass
        self._init_distances_combined()

        # Main merging loop: process sorted pairs in rounds
        while True:
            active_count = len(self._graph.get_active_clusters())
            pre_length = active_count

            if self.k is not None and active_count <= self.k:
                break

            sorted_pairs = self._get_sorted_cluster_pairs()
            if not sorted_pairs:
                break

            i = 0
            last_merged_cluster = None

            while i < len(sorted_pairs):
                c_i, c_j, dist_candidate = sorted_pairs[i]

                if self.k is not None and len(self._graph.get_active_clusters()) <= self.k:
                    break

                if c_i == c_j or np.isinf(dist_candidate):
                    i += 1
                    continue

                if c_i in self._graph.dead_clusters or c_j in self._graph.dead_clusters:
                    i += 1
                    continue

                # After a merge, check if the merged cluster's nearest neighbor
                # is closer than the next pair in the sorted list
                if last_merged_cluster is not None:
                    nearest = self._get_nearest_cluster(last_merged_cluster)
                    if nearest is not None:
                        nearest_id, nearest_dist = nearest
                        current_dist, _, _ = self._get_cached_distance(c_i, c_j)
                        if nearest_dist < current_dist:
                            c_i, c_j = last_merged_cluster, nearest_id
                        else:
                            i += 1
                    else:
                        i += 1
                    last_merged_cluster = None
                else:
                    i += 1

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
                        self.merge_log.append(
                            {
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
                            }
                        )

                    self._merge_cluster_data(lead_cluster, child_cluster, d_ij)
                    self._graph.merge_clusters(lead_id, child_id)

                    self._graph.cluster_points[lead_id] = np.array(
                        lead_cluster.point_indices, dtype=np.int64
                    )
                    if lead_cluster.center is not None:
                        self._graph.cluster_centers[lead_id] = lead_cluster.center.copy()

                    self._update_distance_cache_after_merge(lead_id, child_id)

                    last_merged_cluster = lead_id

            if len(self._graph.get_active_clusters()) == pre_length:
                break

        # Force merge to k if specified and still have too many clusters
        if self.k is not None and len(self._graph.get_active_clusters()) > self.k:
            self._force_merge_to_k()

        # Build final labels
        self._build_labels()

        return self.labels_

    def _update_min_cluster_dist(self) -> None:
        """
        Update MIN_BTN_CLUSTER_DIST after merge (matches original's update_clusters).
        
        Original: self.MIN_BTN_CLUSTER_DIST = max(DIST_MATRIX.min(), self.MIN_BTN_CLUSTER_DIST)
        """
        if self._dist_matrix is None:
            return

        finite = self._dist_matrix[np.isfinite(self._dist_matrix)]
        if finite.size == 0:
            return

        min_dist = float(finite.min())
        self._fallback_dist = max(min_dist, self._fallback_dist)

    def _get_indices_of_k_smallest(
        self,
        matrix: np.ndarray,
        k: int,
        *,
        sorted: bool = False,
    ) -> np.ndarray:
        """Legacy get_indices_of_k_smallest (argpartition + unravel_index)."""
        idx = np.argpartition(matrix.ravel(), k)
        ind = np.array(np.unravel_index(idx, matrix.shape))[:, range(min(k, 0), max(k, 0))]
        if sorted:
            values = matrix[tuple(ind)]
            xx = np.argsort(values)
            return ind[:, xx]
        return ind

    def _init_distances_combined(self) -> None:
        """Initialize DIST_MATRIX, ref_points, point_dists, and fallback in ONE O(N²) pass.

        Previously _build_point_dists and _init_distance_cache each computed all
        N² pairwise distances independently.  This fused version halves the work.

        NOTE: We use per-pair np.linalg.norm (not scipy pdist) because pdist uses
        different FP accumulation that causes ~6% of distances to differ at the
        last bit, which cascades into different merge orderings.
        """
        n_total = len(self._X)
        self._dist_matrix = np.full((n_total, n_total), np.inf, dtype=np.float64)
        self._ref_points = {}

        build_point_dists = self.legacy_point_dists
        if build_point_dists:
            point_dists: list[list[tuple[int, float]]] = [[] for _ in range(n_total)]

        if n_total <= 1:
            self._fallback_dist = MIN_BTN_CLUSTER_DIST
            self._point_dists = point_dists if build_point_dists else None
            return

        all_dists: list[float] = []
        for i in range(n_total):
            for j in range(i + 1, n_total):
                dist = float(np.linalg.norm(self._X[i] - self._X[j]))
                # Populate DIST_MATRIX
                self._dist_matrix[i, j] = dist
                self._dist_matrix[j, i] = dist
                self._ref_points[(i, j)] = (i, j)
                all_dists.append(dist)
                # Populate point_dists (matching legacy insertion order: i<j)
                if build_point_dists:
                    point_dists[i].append((j, dist))
                    point_dists[j].append((i, dist))

        all_dists.sort()
        idx = max(0, int(len(all_dists) / 100))
        self._fallback_dist = all_dists[idx] if all_dists else MIN_BTN_CLUSTER_DIST

        if build_point_dists:
            # Legacy uses Python stable sort by distance only
            for i in range(n_total):
                point_dists[i].sort(key=lambda x: x[1])
            self._point_dists = point_dists
        else:
            self._point_dists = None

    def _get_cached_distance(self, c_i: int, c_j: int) -> tuple[float, int, int]:
        """Return cached distance and reference points between clusters."""
        if self._dist_matrix is None:
            return self._graph.get_cluster_distance(c_i, c_j)

        dist = float(self._dist_matrix[c_i, c_j])
        key = (min(c_i, c_j), max(c_i, c_j))
        ref = self._ref_points.get(key)

        if np.isinf(dist) or ref is None:
            dist, p_i, p_j = self._graph.get_cluster_distance(c_i, c_j)
            self._dist_matrix[c_i, c_j] = dist
            self._dist_matrix[c_j, c_i] = dist
            if c_i <= c_j:
                self._ref_points[key] = (int(p_i), int(p_j))
                return float(dist), int(p_i), int(p_j)
            self._ref_points[key] = (int(p_j), int(p_i))
            return float(dist), int(p_j), int(p_i)

        if c_i <= c_j:
            p_i, p_j = ref
            return dist, int(p_i), int(p_j)

        p_j, p_i = ref
        return dist, int(p_i), int(p_j)

    def _update_distance_cache_after_merge(self, lead_id: int, child_id: int) -> None:
        """Update cached distances after a merge (legacy update_clusters)."""
        if self._dist_matrix is None:
            return

        self._dist_matrix[:, child_id] = np.inf
        self._dist_matrix[child_id, :] = np.inf

        # Clean up ref_points for dead child cluster
        if self._ref_points:
            dead_keys = [
                key for key in self._ref_points
                if key[0] == child_id or key[1] == child_id
            ]
            for key in dead_keys:
                del self._ref_points[key]

        active = self._graph.get_active_clusters() - {lead_id}
        for c_id in active:
            dist, p_i, p_j = self._graph.get_cluster_distance(lead_id, c_id)
            self._dist_matrix[lead_id, c_id] = dist
            self._dist_matrix[c_id, lead_id] = dist
            key = (min(lead_id, c_id), max(lead_id, c_id))
            if lead_id <= c_id:
                self._ref_points[key] = (int(p_i), int(p_j))
            else:
                self._ref_points[key] = (int(p_j), int(p_i))

        self._dist_matrix[lead_id, lead_id] = np.inf

        # Legacy: self.MIN_BTN_CLUSTER_DIST = max(self.DIST_MATRIX.min(), self.MIN_BTN_CLUSTER_DIST)
        mat_min = float(self._dist_matrix.min())
        if np.isfinite(mat_min):
            self._fallback_dist = max(mat_min, self._fallback_dist)

    def _get_sorted_cluster_pairs(self) -> list[tuple[int, int, float]]:
        """Get cluster pairs sorted by distance.

        Legacy behavior:
        - Builds a full DIST_MATRIX sized to the original dataset (N x N)
        - Uses get_indices_of_k_smallest(k=N) on the full matrix
        - Includes BOTH orderings (a,b) and (b,a)

        We reproduce that logic exactly, then pre-filter dead/inf entries
        so the inner loop doesn't waste time skipping them.
        """
        if self._dist_matrix is None:
            return []

        active = self._graph.get_active_clusters()
        if len(active) < 2:
            return []

        dead = self._graph.dead_clusters

        # Legacy: get_indices_of_k_smallest(k=N, sorted=True)
        k = len(self._X)
        ind = self._get_indices_of_k_smallest(self._dist_matrix, k, sorted=True)
        values = self._dist_matrix[tuple(ind)]

        # Pre-filter: skip inf values and dead clusters (avoids inner-loop churn)
        result = []
        for pos in range(ind.shape[1]):
            v = values[pos]
            if np.isinf(v):
                break  # All subsequent values are >= this one (sorted)
            ci = int(ind[0, pos])
            cj = int(ind[1, pos])
            if ci == cj or ci in dead or cj in dead:
                continue
            result.append((ci, cj, float(v)))
        return result
    
    def _get_nearest_cluster(self, cluster_id: int) -> tuple[int, float] | None:
        """Get nearest active cluster to given cluster."""
        if cluster_id in self._graph.dead_clusters:
            return None
            
        active = self._graph.get_active_clusters() - {cluster_id}
        if not active:
            return None

        if self._dist_matrix is None:
            n_total = len(self._X)
            dist_row = np.full(n_total, np.inf)
            for c_id in active:
                dist_row[c_id] = self._get_cached_distance(cluster_id, c_id)[0]
        else:
            dist_row = self._dist_matrix[cluster_id].copy()
            dist_row[cluster_id] = np.inf
            if self._graph.dead_clusters:
                dead = np.fromiter(self._graph.dead_clusters, dtype=int)
                dist_row[dead] = np.inf

        ind = self._get_indices_of_k_smallest(dist_row, 1, sorted=True)
        nearest_id = int(ind[:, 0][0]) if ind.size else None
        if nearest_id is None or np.isinf(dist_row[nearest_id]):
            return None
        return (nearest_id, float(dist_row[nearest_id]))

    def _compute_cluster_distance(self, c_i: int, c_j: int) -> float:
        """Compute minimum distance between two clusters' points.
        
        Matches legacy's compute_2_clusters_dist behavior.
        """
        points_i = self._graph.cluster_points.get(c_i, [c_i])
        points_j = self._graph.cluster_points.get(c_j, [c_j])
        
        min_dist = float('inf')
        for pi in points_i:
            for pj in points_j:
                d = np.linalg.norm(self._X[pi] - self._X[pj])
                if d < min_dist:
                    min_dist = d
        return min_dist

    def _merge_cluster_data(
        self,
        lead: Cluster,
        child: Cluster,
        d_ij: float,
    ) -> None:
        """Merge child cluster data into lead cluster (matches original merge_2_cluster)."""
        # Store reference points for traces (matches original)
        # lead.merge_edges.append((p_i, p_j))  # Would need to pass these in
        lead.merge_edges.extend(child.merge_edges)
        
        # Merge point indices
        lead.point_indices = list(lead.point_indices) + list(child.point_indices)

        # Update center (matches original)
        lead.center = np.mean(self._X[lead.point_indices], axis=0)
        
        # Compute distances from center for stats (matches original)
        dist_data = np.linalg.norm(self._X[lead.point_indices] - lead.center, axis=1)
        lead.mu_dist = float(np.mean(dist_data))
        lead.sigma_dist = float(np.std(dist_data))
        
        # Update sigma_history (past_std in original)
        lead.sigma_history.append(lead.sigma_dist)
        
        # Legacy order (from call flow):
        #   1. vision_generic: past_dists.append(distance) - happens first
        #   2. merge_2_cluster: past_dists.extend(child's history) - happens after
        lead.merge_history.append(d_ij)
        lead.merge_history.extend(child.merge_history)
        lead.merging_dists.append(d_ij)

    def _force_merge_to_k(self) -> None:
        """Force merge clusters until we reach k clusters.

        Used when k is specified but mergeability checks prevent
        natural merging. Merges closest pairs regardless of mergeability.
        """
        while True:
            active = self._graph.get_active_clusters()
            if len(active) <= self.k:
                break

            # Find closest pair among active clusters
            min_dist = float("inf")
            best_pair = None

            active_list = list(active)
            for i, c_i in enumerate(active_list):
                for c_j in active_list[i + 1 :]:
                    dist, _, _ = self._graph.get_cluster_distance(c_i, c_j)
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (c_i, c_j)

            if best_pair is None:
                break

            c_i, c_j = best_pair
            C_i = self.clusters_[c_i]
            C_j = self.clusters_[c_j]

            # Determine lead and child
            if len(C_i) >= len(C_j):
                lead_id, child_id = c_i, c_j
                lead_cluster, child_cluster = C_i, C_j
            else:
                lead_id, child_id = c_j, c_i
                lead_cluster, child_cluster = C_j, C_i

            # Force merge
            self._merge_cluster_data(lead_cluster, child_cluster, min_dist)
            self._graph.merge_clusters(lead_id, child_id)

    def _build_labels(self) -> None:
        """Build labels array from final cluster configuration."""
        n_samples = len(self._X)
        self.labels_ = np.zeros(n_samples, dtype=np.int32)

        active_clusters = self._graph.get_active_clusters()

        if self.preserve_labels:
            for cid in active_clusters:
                points = self._graph.cluster_points[cid]
                for p in points:
                    self.labels_[p] = cid
            self.n_clusters_ = len(active_clusters)
            return

        # Relabel to contiguous integers
        label_map = {cid: new_label for new_label, cid in enumerate(sorted(active_clusters))}

        for cid in active_clusters:
            points = self._graph.cluster_points[cid]
            new_label = label_map[cid]
            for p in points:
                self.labels_[p] = new_label

        self.n_clusters_ = len(active_clusters)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience method: fit and return labels."""
        return self.fit(X)

    def _compute_mergeability(
        self,
        C_i: Cluster,
        C_j: Cluster,
        d_ij: float,
    ) -> MergeabilityResult:
        """
        Evaluate mergeability of two clusters.

        Paper Section II: Mergeability Function
        1. Compute proximity ρ (Eq. 3)
        2. Compute adaptive thresholds T_i, T_j (Eq. 4)
        3. If ρ > T_i or ρ > T_j, not mergeable
        4. Else compute continuity score
        5. If continuity > T_c, mergeable

        Args:
            C_i: First cluster
            C_j: Second cluster
            d_ij: Distance between clusters

        Returns:
            MergeabilityResult with decision and statistics
        """
        # Step 1: Compute proximity ρ
        # Use adaptive fallback based on data scale
        fallback = getattr(self, "_fallback_dist", MIN_BTN_CLUSTER_DIST)
        rho, _, lead_id, child_id = compute_proximity_rho(d_ij, C_i, C_j, fallback)

        # Step 2: Compute environmental factor β and shape factor ξ
        # Legacy computes thresholds using lead/child ordering.
        C_lead = C_i if lead_id == C_i.label else C_j
        C_child = C_j if lead_id == C_i.label else C_i

        # Legacy selects k nearest clusters from dense DIST_MATRIX rows.
        active_ids = list(self._graph.get_active_clusters())
        k_neighbors = min(self.K_neighbors, max(len(active_ids) - 1, 0))

        distance_cache: dict[tuple[int, int], float] = {
            (min(C_i.label, C_j.label), max(C_i.label, C_j.label)): d_ij
        }

        def _get_near_distance_by_id(a_id: int, b_id: int) -> float:
            key = (min(a_id, b_id), max(a_id, b_id))
            if key in distance_cache:
                return distance_cache[key]
            dist, _, _ = self._get_cached_distance(a_id, b_id)
            distance_cache[key] = dist
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

        def _collect_neighbors(cluster_id: int) -> list[tuple[int, float]]:
            if k_neighbors <= 0:
                return []
            n_total = len(self._X)
            if self._dist_matrix is not None:
                dist_row = self._dist_matrix[cluster_id].copy()
                dist_row[cluster_id] = np.inf
                if self._graph.dead_clusters:
                    dead = np.fromiter(self._graph.dead_clusters, dtype=int)
                    dist_row[dead] = np.inf
            else:
                dist_row = np.full(n_total, np.inf)
                for other_id in active_ids:
                    if other_id == cluster_id:
                        continue
                    dist_row[other_id] = _get_near_distance_by_id(cluster_id, other_id)

            k = min(k_neighbors, n_total - 1)
            if k <= 0:
                return []

            ind = self._get_indices_of_k_smallest(dist_row, k, sorted=True)
            indices = ind[:, range(min(k, 0), max(k, 0))][0]

            return [
                (int(cid), float(dist_row[cid]))
                for cid in indices
                if not np.isinf(dist_row[cid])
            ]

        neighbor_i = _collect_neighbors(C_lead.label) if k_neighbors else []
        neighbor_j = _collect_neighbors(C_child.label) if k_neighbors else []

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

        # Step 5: Compute continuity
        # CRITICAL: Use lead/child ordering (larger cluster first) to match legacy
        # This affects which cluster's reference point has fewer local points
        if lead_id == C_i.label:
            lead_cluster, child_cluster = C_i, C_j
        else:
            lead_cluster, child_cluster = C_j, C_i
        
        # Find reference points (closest points between clusters)
        _, p_lead, p_child = self._get_cached_distance(lead_cluster.label, child_cluster.label)

        # Legacy: adp_prox = T1 * size1/(size1+size2) + T2 * size2/(size1+size2)
        size_lead = len(lead_cluster)
        size_child = len(child_cluster)
        total_size = size_lead + size_child
        T_adaptive = T_i * size_lead / total_size + T_j * size_child / total_size
        shape_diff = xi_s if xi_s > 0 else 1.0
        
        # Legacy passes distance/proximity = mean_past_dist, not raw d_ij
        # distance/proximity = d_ij / (d_ij / mean_historical) = mean_historical
        mean_historical = d_ij / rho if rho > 0 else d_ij

        continuity = compute_continuity(
            lead_cluster,
            child_cluster,
            p_lead,
            p_child,
            self._X,
            self.T_continuity / shape_diff,
            mean_historical,
            T_adaptive,
            kdtree=self._graph.kdtree,
            point_dists=self._point_dists,
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
