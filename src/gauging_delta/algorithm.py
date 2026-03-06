"""Gauging-delta hierarchical clustering algorithm.

sklearn-compatible API with swappable proximity, continuity, and linkage metrics.
Port of ``Perception.fit`` (perception.py L48-146) and supporting methods.
"""

from __future__ import annotations

import math

import numpy as np

from gauging_delta._types import ContinuityMetric, LinkageMetric, ProximityMetric
from gauging_delta.cluster import Cluster
from gauging_delta.config import GaugingDeltaConfig
from gauging_delta.continuity import DefaultContinuity
from gauging_delta.linkage import DefaultLinkage
from gauging_delta.proximity import DefaultProximity
from gauging_delta.threshold import compute_adaptive_threshold


class GaugingDelta:
    """Gauging-delta clustering with sklearn-compatible API.

    Parameters
    ----------
    n_clusters : int or None
        Target number of clusters.  ``None`` = let the algorithm decide.
    mode : {'full', 'lite'}, default='full'
        Algorithm variant.

        - ``'full'``: Single-linkage distances with angle-based continuity
          analysis.  Highest quality; O(N²) memory for 4 dense arrays.
        - ``'lite'``: Centroid-linkage distances, no continuity gate.
          ~4× lower memory (1 dense array); suitable for larger datasets.

        When ``mode='lite'``, the *continuity* and *linkage* parameters
        are ignored.
    config : GaugingDeltaConfig or None
        All magic numbers.  Defaults are the paper's values.
    proximity, continuity, linkage : Protocol-compatible objects or None
        Swappable algorithm components (used in ``mode='full'`` only).
    preserve_labels : bool
        If *True*, ``labels_`` keeps the internal cluster IDs instead
        of renumbering to ``0 … n_clusters_-1``.
    """

    def __init__(
        self,
        *,
        n_clusters: int | None = None,
        mode: str = "full",
        config: GaugingDeltaConfig | None = None,
        proximity: ProximityMetric | None = None,
        continuity: ContinuityMetric | None = None,
        linkage: LinkageMetric | None = None,
        preserve_labels: bool = False,
    ) -> None:
        self.n_clusters = n_clusters
        self.mode = mode
        self.config = config or GaugingDeltaConfig()
        self.proximity = proximity or DefaultProximity()
        if mode == "full":
            self.continuity = continuity or DefaultContinuity(self.config)
            self.linkage = linkage or DefaultLinkage()
        else:
            self.continuity = continuity  # type: ignore[assignment]
            self.linkage = linkage  # type: ignore[assignment]
        self.preserve_labels = preserve_labels

    # ----- sklearn-compatible interface ------------------------------------

    def fit(self, X: np.ndarray) -> GaugingDelta:
        """Run Gauging-delta on data matrix *X* (n_samples, n_features)."""
        if self.mode not in ("full", "lite"):
            raise ValueError(f"mode must be 'full' or 'lite', got {self.mode!r}")

        self._X = np.asarray(X, dtype=float)
        n = len(self._X)

        # --- Initialise singleton clusters (perception.py L56-69) ---
        self._clusters: dict[int, Cluster] = {
            i: Cluster(
                label=i,
                point_indices=[i],
                center=self._X[i].copy(),
            )
            for i in range(n)
        }

        # --- Pairwise distances (perception.py L72, L176-220) ---
        if self.mode == "full":
            self._dist_matrix = np.full((n, n), np.inf)
            self._near_ref = np.empty((n, n), dtype=int)
        self._init_distances()

        # Precompute rho rejection bound for lite mode (max possible threshold)
        if self.mode == "lite":
            cfg = self.config
            self._rho_reject_bound = (
                (cfg.vision_scale_coeff / 2 + cfg.vision_scale_offset)
                * max(cfg.t_stat_numerator / 2 + cfg.t_stat_offset, cfg.t_stat_fallback)
            )

        # --- Main merge loop (perception.py L75-127) ---
        early_stop = False
        while len(self._clusters) > 1:
            pre_length = len(self._clusters)

            k_nearest = self._get_sorted_pairs()
            last_merged: int | None = None
            i = 0

            while i < k_nearest.shape[1]:
                c1, c2 = int(k_nearest[0, i]), int(k_nearest[1, i])

                # Early stop on target cluster count
                if self.n_clusters is not None and len(self._clusters) == self.n_clusters:
                    early_stop = True
                    break

                # Skip dead pairs (perception.py L97)
                if self.mode == "lite":
                    if c1 not in self._clusters or c2 not in self._clusters:
                        i += 1
                        continue
                elif np.isinf(self._dist_matrix[c1, c2]):
                    i += 1
                    continue

                # last_merged heuristic (perception.py L104-113)
                if last_merged is not None:
                    _c = self._get_nearest_cluster(last_merged)
                    if _c is not None:
                        if self.mode == "lite":
                            _diff = self._clusters[last_merged].center - self._clusters[_c].center
                            d_lm = math.sqrt(float(_diff @ _diff))
                            _diff = self._clusters[c1].center - self._clusters[c2].center
                            d_pair = math.sqrt(float(_diff @ _diff))
                            closer = d_lm < d_pair
                        else:
                            closer = self._dist_matrix[last_merged, _c] < self._dist_matrix[c1, c2]
                        if closer:
                            c1, c2 = last_merged, _c
                        else:
                            i += 1
                    else:
                        i += 1
                    last_merged = None
                else:
                    i += 1

                # Try mergeability pipeline (perception.py L259-342)
                # NOTE: legacy does NOT increment i on merge failure.
                # The else:i+=1 at perception.py L122-123 pairs with
                # if ~np.isinf (L97), not with if is_complete (L116).
                merged_id = self._try_merge(c1, c2)
                if merged_id is not None:
                    last_merged = merged_id

            if pre_length == len(self._clusters) or early_stop:
                break

        # Post-processing: force to n_clusters if needed (perception.py L128-139)
        if self.n_clusters is not None and self.n_clusters < len(self._clusters):
            self._post_processing()

        # --- Build output labels ---
        self._build_labels()
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        return self.fit(X).labels_

    # ----- Distance initialisation -----------------------------------------

    def _init_distances(self) -> None:
        """Initialise distance matrix and supporting structures."""
        if self.mode == "lite":
            self._init_distances_lite()
        else:
            self._init_distances_full()

    def _init_distances_full(self) -> None:
        """Full mode: single-linkage distances + sorted neighbor arrays."""
        from scipy.spatial.distance import cdist

        n = len(self._X)

        # Vectorized pairwise distances — O(N²·D) in one C call
        pairwise = cdist(self._X, self._X)  # (N, N)
        self._dist_matrix = pairwise.copy()
        np.fill_diagonal(self._dist_matrix, np.inf)

        # For singletons: nearest point in cluster i to cluster j = point i itself
        # _near_ref[i, j] = i for all j (vectorized init)
        idx = np.arange(n)
        self._near_ref[:] = idx[:, np.newaxis]

        # Build sorted neighbor arrays from pairwise matrix (replaces _point_dists dict)
        # _pd_indices[i] = neighbor indices sorted by distance from point i
        # _pd_dists[i] = corresponding sorted distances
        # kind="stable" matches legacy Python list.sort tie-breaking (lower index first)
        # Mask self-distances to inf so self never appears in sorted neighbors
        # (simple [:, 1:] fails for duplicate points where pairwise[i,j]=0=pairwise[i,i])
        pw_no_self = pairwise.copy()
        np.fill_diagonal(pw_no_self, np.inf)
        full_order = np.argsort(pw_no_self, axis=1, kind="stable")[:, : n - 1]
        self._pd_indices = full_order
        self._pd_dists = np.take_along_axis(pairwise, full_order, axis=1)

        # MIN_BTN_CLUSTER_DIST (perception.py L213)
        # Row-wise minimum tracking: per-row argmin column + value arrays
        self._row_argmins = np.argmin(self._dist_matrix, axis=1).astype(int)
        self._row_mins = self._dist_matrix[np.arange(n), self._row_argmins].copy()
        upper = pairwise[np.triu_indices(n, k=1)]
        upper.sort()
        self._fallback_dist = float(upper[int(len(upper) * self.config.min_dist_percentile)])

    def _init_distances_lite(self) -> None:
        """Lite mode: build KDTree over initial singleton centroids."""
        from scipy.spatial import cKDTree

        n = len(self._X)
        D = self._X.shape[1] if self._X.ndim > 1 else 1

        if n <= 1:
            self._fallback_dist = 0.0
            self._lite_tree = None
            self._lite_ids = list(self._clusters.keys())
            self._lite_centers = self._X.copy() if n == 1 else np.empty((0, D))
            self._lite_id_to_pos = {cid: i for i, cid in enumerate(self._lite_ids)}
            self._lite_K = n
            return

        # Build arrays: _lite_centers is (N, D) with room for shrinking via _lite_K
        self._lite_ids = list(range(n))
        self._lite_centers = self._X.copy()  # singletons → centers = data points
        self._lite_id_to_pos = {i: i for i in range(n)}
        self._lite_K = n
        self._lite_tree = cKDTree(self._lite_centers)

        # Compute fallback_dist from nearest-neighbor distances
        nn_dists, _ = self._lite_tree.query(self._lite_centers, k=2)
        nn_sorted = np.sort(nn_dists[:, 1])
        self._fallback_dist = float(
            nn_sorted[max(0, int(len(nn_sorted) * self.config.min_dist_percentile))]
        )

    # ----- Sorted pairs ----------------------------------------------------

    def _get_sorted_pairs(self) -> np.ndarray:
        """Port of ``get_indices_of_k_smallest`` (perception.py L80, L247-255).

        Returns (2, k) array of (row, col) pairs sorted by distance.
        """
        if self.mode == "lite":
            return self._get_sorted_pairs_lite()
        k = len(self._X)
        mat = self._dist_matrix
        flat = mat.ravel()
        k = min(k, flat.size)
        idx = np.argpartition(flat, k)[:k]
        ind = np.array(np.unravel_index(idx, mat.shape))
        values = mat[tuple(ind)]
        order = np.argsort(values)
        return ind[:, order]

    def _get_sorted_pairs_lite(self) -> np.ndarray:
        """Sorted candidate pairs via KDTree (rebuilt once per outer loop)."""
        from scipy.spatial import cKDTree

        K = self._lite_K
        if K <= 1:
            return np.empty((2, 0), dtype=int)

        # Rebuild KDTree from active portion of centers array
        centers = self._lite_centers[:K]
        self._lite_tree = cKDTree(centers)

        # Adaptive k: match original's coverage
        entries_per_cluster = min(K - 1, max(1, len(self._X) // K))
        k_query = entries_per_cluster + 1  # +1 for self
        dists_arr, idxs_arr = self._lite_tree.query(centers, k=min(k_query, K))

        # Ensure 2D
        if dists_arr.ndim == 1:
            dists_arr = dists_arr.reshape(1, -1)
            idxs_arr = idxs_arr.reshape(1, -1)

        # Update fallback_dist
        if K >= 2:
            current_min = float(np.min(dists_arr[:, 1]))
            if not np.isinf(current_min):
                self._fallback_dist = max(current_min, self._fallback_dist)

        # Build unique pairs sorted by distance
        # Skip self-pairs (nn == i) which can occur when KDTree returns the
        # queried point itself due to tie-breaking with duplicate points.
        seen: set[tuple[int, int]] = set()
        pairs_list: list[tuple[int, int, float]] = []
        for i in range(K):
            for j in range(1, idxs_arr.shape[1]):
                nn = int(idxs_arr[i, j])
                if nn == i:  # skip self-pair from KDTree tie-breaking
                    continue
                d = float(dists_arr[i, j])
                c1, c2 = self._lite_ids[i], self._lite_ids[nn]
                pair_key = (min(c1, c2), max(c1, c2))
                if pair_key not in seen:
                    seen.add(pair_key)
                    pairs_list.append((c1, c2, d))

        pairs_list.sort(key=lambda x: x[2])
        if not pairs_list:
            return np.empty((2, 0), dtype=int)
        return np.array([[p[0] for p in pairs_list], [p[1] for p in pairs_list]])

    def _get_nearest_cluster(self, cid: int) -> int | None:
        """Nearest active cluster to *cid*."""
        if self.mode == "lite":
            return self._get_nearest_cluster_lite(cid)
        # Full mode: O(1) lookup from cached row mins
        if np.isinf(self._row_mins[cid]):
            return None
        return int(self._row_argmins[cid])

    def _get_nearest_cluster_lite(self, cid: int) -> int | None:
        """Nearest active cluster via brute-force centroid scan (O(K·D))."""
        K = self._lite_K
        if K < 2:
            return None
        if cid not in self._lite_id_to_pos:
            return None
        center = self._clusters[cid].center
        pos = self._lite_id_to_pos[cid]
        dists = np.linalg.norm(self._lite_centers[:K] - center, axis=1)
        dists[pos] = np.inf
        nearest_pos = int(np.argmin(dists))
        return self._lite_ids[nearest_pos]

    # ----- Mergeability pipeline -------------------------------------------

    def _try_merge(self, c1: int, c2: int) -> int | None:
        """Run mergeability pipeline. Dispatches based on *mode*."""
        if self.mode == "lite":
            return self._try_merge_lite(c1, c2)
        return self._try_merge_full(c1, c2)

    def _try_merge_full(self, c1: int, c2: int) -> int | None:
        """Full mode: proximity → threshold → continuity → merge."""
        C_i = self._clusters[c1]
        C_j = self._clusters[c2]
        d_ij: float = float(self._dist_matrix[c1, c2])

        # Step 1: Proximity (perception.py L310)
        prox = self.proximity.compute(C_i, C_j, d_ij, self._fallback_dist)
        rho = prox.rho
        lead_id, child_id = prox.lead_id, prox.child_id
        lead = self._clusters[lead_id]
        child = self._clusters[child_id]

        # Step 2: Adaptive threshold (perception.py L312-313)
        thr = compute_adaptive_threshold(
            lead,
            child,
            d_ij,
            rho,
            self._clusters,
            self._dist_matrix,
            self.config,
        )

        # Gate: proximity > threshold → reject (perception.py L318)
        if rho > thr.T_i or rho > thr.T_j:
            return None

        # Step 3: Continuity (perception.py L323-325)
        # Set reference points for continuity
        lead.ref_point = int(self._near_ref[lead_id, child_id])
        child.ref_point = int(self._near_ref[child_id, lead_id])

        cont_threshold = self.config.threshold_continuity / thr.xi_s
        d_ij_norm = d_ij / rho if rho != 0 else d_ij
        smoothness = self.continuity.compute(
            lead,
            child,
            cont_threshold,
            d_ij_norm,
            thr.adp_prox,
            self._X,
            (self._pd_indices, self._pd_dists),
        )

        # Gate: smoothness > threshold → accept merge (perception.py L328)
        if smoothness > cont_threshold:
            # Record history BEFORE merge (perception.py L332-336)
            lead.density_history.append(smoothness)
            lead.merge_history.append(d_ij)
            lead.merging_dists.append(d_ij)

            self._do_merge(lead_id, child_id)
            return lead_id

        return None

    def _try_merge_lite(self, c1: int, c2: int) -> int | None:
        """Lite mode: proximity → threshold → merge (no continuity)."""
        from gauging_delta.threshold import compute_adaptive_threshold_lite

        C_i = self._clusters[c1]
        C_j = self._clusters[c2]
        _diff = C_i.center - C_j.center
        d_ij = math.sqrt(float(_diff @ _diff))

        # Step 1: Proximity
        prox = self.proximity.compute(C_i, C_j, d_ij, self._fallback_dist)
        rho = prox.rho

        # Short-circuit: rho exceeds maximum possible threshold → skip threshold
        if rho > self._rho_reject_bound:
            return None

        lead_id, child_id = prox.lead_id, prox.child_id
        lead = self._clusters[lead_id]

        # Step 2: Adaptive threshold (brute-force centers array, no KDTree)
        thr = compute_adaptive_threshold_lite(
            lead,
            self._clusters[child_id],
            d_ij,
            rho,
            self._clusters,
            self._lite_centers[: self._lite_K],
            self._lite_ids,
            self._lite_id_to_pos,
            self.config,
        )

        # Gate: proximity > threshold → reject
        if rho > thr.T_i or rho > thr.T_j:
            return None

        # No continuity gate — merge directly
        lead.merge_history.append(d_ij)
        lead.merging_dists.append(d_ij)

        self._do_merge(lead_id, child_id)
        return lead_id

    # ----- Merge execution -------------------------------------------------

    def _do_merge(self, lead_id: int, child_id: int) -> None:
        """Merge *child* into *lead* and update distances."""
        if self.mode == "lite":
            self._do_merge_lite(lead_id, child_id)
        else:
            self._do_merge_full(lead_id, child_id)

    def _do_merge_full(self, lead_id: int, child_id: int) -> None:
        """Full mode: Lance-Williams single-linkage update."""
        lead = self._clusters[lead_id]
        child = self._clusters[child_id]

        # --- Merge data (perception.py L298-306) ---
        p1 = int(self._near_ref[lead_id, child_id])
        p2 = int(self._near_ref[child_id, lead_id])
        lead.merge_edges.append((p1, p2))
        lead.merge_edges.extend(child.merge_edges)

        lead.point_indices.extend(child.point_indices)
        lead.center = np.mean(self._X[lead.point_indices], axis=0)

        dist_data = np.linalg.norm(self._X[lead.point_indices] - lead.center, axis=1)
        lead.mu_dist = float(np.mean(dist_data))
        lead.sigma_dist = float(np.std(dist_data))
        lead.sigma_history.append(lead.sigma_dist)

        # Extend history from child (perception.py L306)
        lead.merge_history.extend(child.merge_history)

        # --- Update clusters (perception.py L277-291) ---

        # Snapshot child's distances and refs BEFORE deletion
        # Lance-Williams: near_dist(A∪B, C) = min(near_dist(A,C), near_dist(B,C))
        child_near_row = self._dist_matrix[child_id, :].copy()
        child_ref_self = self._near_ref[child_id, :].copy()
        child_ref_other = self._near_ref[:, child_id].copy()

        del self._clusters[child_id]
        self._dist_matrix[:, child_id] = np.inf
        self._dist_matrix[child_id, :] = np.inf

        active = np.argwhere(~np.isinf(self._dist_matrix[lead_id, :])).reshape(-1)

        # Vectorized Lance-Williams update: only update where child was closer
        # Strict < so lead wins on exact tie (matches legacy argmin first-occurrence)
        child_dists = child_near_row[active]
        lead_dists = self._dist_matrix[lead_id, active]
        mask = child_dists < lead_dists
        winning = active[mask]

        self._dist_matrix[lead_id, winning] = child_near_row[winning]
        self._dist_matrix[winning, lead_id] = child_near_row[winning]
        self._near_ref[lead_id, winning] = child_ref_self[winning]
        self._near_ref[winning, lead_id] = child_ref_other[winning]

        # Update MIN_BTN_CLUSTER_DIST (perception.py L290)
        # Row-wise minimum tracking: maintain per-row min value + position.
        # Only rescan rows that actually changed instead of full O(N²) scan.

        # 1. Identify stale rows (argmin pointed to the now-dead child)
        stale_mask = self._row_argmins == child_id

        # 2. Invalidate child row
        self._row_mins[child_id] = np.inf

        # 3. Rescan lead row (always needed: lead distances changed)
        lead_min_col = int(np.argmin(self._dist_matrix[lead_id, :]))
        self._row_argmins[lead_id] = lead_min_col
        self._row_mins[lead_id] = self._dist_matrix[lead_id, lead_min_col]

        # 4. Non-stale winning rows: lead column decreased, might be new row min
        if len(winning) > 0:
            non_stale_winning = winning[~stale_mask[winning]]
            if len(non_stale_winning) > 0:
                new_dists = self._dist_matrix[non_stale_winning, lead_id]
                improved = new_dists < self._row_mins[non_stale_winning]
                update_idx = non_stale_winning[improved]
                self._row_mins[update_idx] = new_dists[improved]
                self._row_argmins[update_idx] = lead_id

        # 5. Stale rows: full row rescan (expected ~1-5 rows)
        stale_mask[child_id] = False
        stale_mask[lead_id] = False
        stale_rows = np.where(stale_mask)[0]
        for c in stale_rows:
            c_min_col = int(np.argmin(self._dist_matrix[c, :]))
            self._row_argmins[c] = c_min_col
            self._row_mins[c] = self._dist_matrix[c, c_min_col]

        # 6. Global min → update fallback_dist
        current_min = float(np.min(self._row_mins))
        if not np.isinf(current_min):
            self._fallback_dist = max(current_min, self._fallback_dist)

    def _do_merge_lite(self, lead_id: int, child_id: int) -> None:
        """Lite mode: merge cluster data, swap-and-pop centers array (O(D))."""
        lead = self._clusters[lead_id]
        child = self._clusters[child_id]

        # --- Merge data ---
        lead.merge_edges.extend(child.merge_edges)
        lead.point_indices.extend(child.point_indices)
        lead.center = np.mean(self._X[lead.point_indices], axis=0)

        dist_data = np.linalg.norm(self._X[lead.point_indices] - lead.center, axis=1)
        lead.mu_dist = float(np.mean(dist_data))
        lead.sigma_dist = float(np.std(dist_data))
        lead.sigma_history.append(lead.sigma_dist)
        lead.merge_history.extend(child.merge_history)

        # --- Delete child ---
        del self._clusters[child_id]

        # --- Swap-and-pop child from arrays (O(D)) ---
        child_pos = self._lite_id_to_pos[child_id]
        self._lite_K -= 1
        last_pos = self._lite_K

        if child_pos != last_pos:
            last_id = self._lite_ids[last_pos]
            self._lite_ids[child_pos] = last_id
            self._lite_centers[child_pos] = self._lite_centers[last_pos]
            self._lite_id_to_pos[last_id] = child_pos

        self._lite_ids.pop()
        del self._lite_id_to_pos[child_id]

        # Update lead's center in-place
        lead_pos = self._lite_id_to_pos[lead_id]
        self._lite_centers[lead_pos] = lead.center

        # Update fallback_dist from lead's nearest neighbor (O(K·D))
        K = self._lite_K
        if K >= 2:
            dists = np.linalg.norm(self._lite_centers[:K] - lead.center, axis=1)
            dists[lead_pos] = np.inf
            current_min = float(np.min(dists))
            if not np.isinf(current_min):
                self._fallback_dist = max(current_min, self._fallback_dist)

        # Mark tree as stale (will be rebuilt in _get_sorted_pairs_lite)
        self._lite_tree = None

    # ----- Post-processing -------------------------------------------------

    def _post_processing(self) -> None:
        """Force merge down to n_clusters (perception.py L161-173)."""
        if self.n_clusters is None:
            return
        ranking = sorted(self._clusters.items(), key=lambda x: len(x[1]))
        top_k = ranking[-self.n_clusters :]
        remaining = ranking[: len(ranking) - self.n_clusters]

        top_k_ids = {c[0] for c in top_k}
        merging_pairs: list[list[int]] = []

        for cid, cluster in remaining:
            if self.mode == "lite":
                c_center = cluster.center
                dists = []
                for tid in top_k_ids:
                    _d = self._clusters[tid].center - c_center
                    dists.append((tid, math.sqrt(float(_d @ _d))))
            else:
                dists = [(tid, self._dist_matrix[tid, cid]) for tid in top_k_ids]
            nearest = min(dists, key=lambda x: x[1])
            merging_pairs.append([nearest[0], cid])

        for lead_id, child_id in merging_pairs:
            self._force_merge(lead_id, child_id)

    def _force_merge(self, lead_id: int, child_id: int) -> None:
        """Unconditional merge for post-processing (perception.py L172)."""
        lead = self._clusters[lead_id]
        child = self._clusters[child_id]

        if self.mode == "full" and not np.isinf(self._dist_matrix[lead_id, child_id]):
            p1 = int(self._near_ref[lead_id, child_id])
            p2 = int(self._near_ref[child_id, lead_id])
            lead.merge_edges.append((p1, p2))
        lead.merge_edges.extend(child.merge_edges)

        lead.point_indices.extend(child.point_indices)
        lead.center = np.mean(self._X[lead.point_indices], axis=0)

        dist_data = np.linalg.norm(self._X[lead.point_indices] - lead.center, axis=1)
        lead.mu_dist = float(np.mean(dist_data))
        lead.sigma_dist = float(np.std(dist_data))
        lead.sigma_history.append(lead.sigma_dist)
        lead.merge_history.extend(child.merge_history)

        del self._clusters[child_id]
        if self.mode == "full":
            self._dist_matrix[:, child_id] = np.inf
            self._dist_matrix[child_id, :] = np.inf

    # ----- Label construction ----------------------------------------------

    def _build_labels(self) -> None:
        """Assign final labels to each point."""
        n = len(self._X)
        labels = np.full(n, -1, dtype=int)

        if self.preserve_labels:
            for cid, cluster in self._clusters.items():
                for p in cluster.point_indices:
                    labels[p] = cid
        else:
            for new_label, (_, cluster) in enumerate(sorted(self._clusters.items())):
                for p in cluster.point_indices:
                    labels[p] = new_label

        self.labels_ = labels
        self.n_clusters_ = len(self._clusters)
